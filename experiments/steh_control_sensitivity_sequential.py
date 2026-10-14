"""Sequential numerical recovery of caliper-control sensitivity of already frozen STEH circuits.

No model training or circuit selection. Vectorized v2 records are preserved separately. Original nearest-degree controls remain
the primary protocol. Two fixed calipers are reported separately, never selected
using test scores. Conditional confirmation excludes the early review basins.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from revision.common import digest,write_json,metrics,finite_population_interval


def draw_group(pool, chosen, degrees, rng, caliper):
    """Randomize within fixed robust degree calipers, without replacement."""
    values=np.asarray([degrees[i] for i in pool],float)
    scale=np.maximum(1,np.median(abs(values-np.median(values,axis=0)),axis=0))
    available=list(pool);selected=[];distances=[]
    for i in rng.permutation(chosen):
        distances_now=np.sum(abs(np.asarray([degrees[z] for z in available])-degrees[i])/scale,axis=1)
        nearest=float(distances_now.min())
        candidates=[z for z,d in zip(available,distances_now) if d<=nearest+caliper+1e-12]
        z=int(rng.choice(candidates));selected.append(z);available.remove(z)
        distances.append(float(np.sum(abs(np.asarray(degrees[z])-degrees[i])/scale)))
    return selected,distances


def predict_many(model,split,cfg,masks,chunk=1):
    """Sequential per-mask reference; stations remain parallel workers."""
    from revision.models import predict
    return np.stack([predict(model,split,cfg,mask) for mask in masks])


def work(payload):
    job,protocol,source,circuits,output=payload
    from revision.runner import job_path
    from revision.models import transformer,phase_switch,predict,seed_all
    from revision.data import load_basin
    from revision.analysis import binary_overrides,clone_masks
    from experiments.refine_steh import unit_list
    import torch
    f=job_path(Path(output),job);f.mkdir(parents=True,exist_ok=True)
    identity=digest({'job':job,'protocol':protocol});stamp=f/'result.json'
    if stamp.exists():
        result=json.loads(stamp.read_text());assert result['identity_sha256']==identity;return result
    started=time.time()
    try:
        torch.set_num_threads(1);seed_all(job['seed'])
        cfg=protocol['training_config']|job|{'device':'cuda'}
        splits,meta=load_basin(cfg['data_root'],job['station'],cfg)
        old=job_path(Path(source),job);c=job_path(Path(circuits),job)
        checkpoint=torch.load(old/'trained.pt',map_location='cpu',weights_only=False)
        model=transformer(len(meta['feature_names']),cfg);model.load_state_dict(checkpoint['state_dict']);model.to('cuda')
        phase_switch(model,'mask',cfg['target_activation_sparsity']);base=binary_overrides(model,'cuda')
        frozen=json.loads((old/'validation_selection.json').read_text())
        assert not frozen['test_used_for_selection']
        prior_selection=json.loads((c/'validation_selection.json').read_text())
        assert frozen['core_units']==prior_selection['core_units']
        core=clone_masks(base)
        for key in ['head_mask_overrides','neuron_mask_overrides']:
            for layer,flags in core[key].items():
                flags.zero_();indices=frozen['core_units'][f'{key}:{layer}']
                if indices:flags[indices]=1
        pools={};degrees={};chosen={}
        for key,layer,i in unit_list(base):pools.setdefault((key,layer),[]).append(i)
        for key,layer in pools:
            module=model.layers[layer]
            if key.startswith('neuron'):
                degree=torch.stack([(module.mlp.fc1.weight!=0).sum(1),(module.mlp.fc2.weight!=0).sum(0)],1).cpu().tolist()
            else:
                att=module.attn;degree=[]
                for h in range(att.n_heads):
                    ids=torch.arange(h*att.d_head,(h+1)*att.d_head,device='cuda')
                    rows=torch.cat([ids,ids+att.d_model,ids+2*att.d_model])
                    degree.append([int((att.qkv.weight[rows]!=0).sum()),int((att.out_proj.weight[:,ids]!=0).sum())])
            degrees[(key,layer)]=degree
            chosen[(key,layer)]=[i for k,l,i in unit_list(core) if k==key and l==layer]
        # Both control banks are fixed from graph structure and RNG before test scores.
        banks=[];bank_masks=[]
        for index,caliper in enumerate(protocol['calipers']):
            rng=np.random.default_rng(job['seed']+cfg['analysis_seed']+10000*(index+1));bank=[]
            for repeat in range(protocol['controls_per_caliper']):
                mask=clone_masks(base);overlap=0;distances=[];members={}
                for group,pool in pools.items():
                    key,layer=group;selected,distance=draw_group(pool,chosen[group],degrees[group],rng,caliper)
                    mask[key][layer].zero_()
                    if selected:mask[key][layer][selected]=1
                    members[f'{key}:{layer}']=sorted(selected);distances.extend(distance)
                    overlap+=len(set(selected)&set(chosen[group]))
                total=len(unit_list(core));same=overlap==total
                bank.append({'repeat':repeat,'caliper':caliper,'members':members,'mask_sha256':digest(members),
                             'core_overlap_fraction':overlap/max(1,total),'identical_core':same,
                             'mean_standardized_degree_distance':float(np.mean(distances)) if distances else None})
                bank_masks.append(mask)
            banks.append(bank)
        write_json(f/'control_design.json',{'selection_data':'graph degrees and RNG only; no test performance',
                                           'banks':banks,'core_source':frozen['core_units']})
        test=splits['test'];target=test['y']*meta['q_std']+meta['q_mean']
        prior=pd.read_csv(c/'predictions.csv');assert test['dates'].tolist()==prior.target_date.tolist()
        np.testing.assert_allclose(target,prior.observed)
        # Sequential reference reproducibility checked per model, including two controls.
        raw=predict_many(model,test,cfg,[base,core,*bank_masks]);scaled=raw*meta['q_std']+meta['q_mean']
        np.testing.assert_allclose(scaled[0],prior.full,rtol=3e-5,atol=max(3e-5,3e-6*meta['q_std']))
        np.testing.assert_allclose(scaled[1],prior.core,rtol=3e-5,atol=max(3e-5,3e-6*meta['q_std']))
        parity=[]
        for index in [0,protocol['controls_per_caliper']]:
            reference=predict(model,test,cfg,bank_masks[index])
            np.testing.assert_allclose(raw[index+2],reference,rtol=3e-5,atol=3e-6)
            parity.append(float(abs(raw[index+2]-reference).max()))
        reference_core=predict(model,test,cfg,core)*meta['q_std']+meta['q_mean']
        scores=[];core_nse=metrics(reference_core,target)['nse']
        primary_result=json.loads((c/'result.json').read_text())
        offset=0;tie_rechecks=0
        for bank in banks:
            for control in bank:
                value=metrics(scaled[offset+2],target)['nse']
                # Convert the verified prediction tolerance to an NSE error bound.
                # Low-variance targets can amplify tiny normalized forecast differences.
                epsilon=(3e-6+3e-5*float(abs(raw[offset+2]).max()))*meta['q_std']
                denominator=float(np.sum((target.astype(float)-float(target.mean()))**2))
                bound=(2*epsilon*float(abs(scaled[offset+2]-target).sum())+len(target)*epsilon**2)/denominator
                control['nse_numerical_comparison_bound']=bound
                if False:  # Already sequential for every mask; no approximate ordering.
                    scaled[offset+2]=predict(model,test,cfg,bank_masks[offset])*meta['q_std']+meta['q_mean']
                    value=metrics(scaled[offset+2],target)['nse'];tie_rechecks+=1
                control['nse']=value;offset+=1
            valid=[r for r in bank if not r['identical_core'] and r['nse'] is not None]
            unique={r['mask_sha256']:r for r in valid}
            win=float(np.mean([core_nse>r['nse'] for r in valid])) if valid else None
            scores.append({'caliper':bank[0]['caliper'],'nonidentical_draws':len(valid),'unique_control_masks':len(unique),
                           'random_win_fraction':win,
                           'unique_mask_win_fraction':float(np.mean([core_nse>r['nse'] for r in unique.values()])) if unique else None,
                           'criterion_pass':bool(primary_result['core_delta_nse']<=.05 and primary_result['remove_delta_nse']>=.02 and win is not None and win>=.8)})
        assert np.isfinite(scaled).all()
        np.savez_compressed(f/'control_predictions.npz',target_date=test['dates'].astype('U10'),observed=target,
                            full=prior.full.to_numpy(),core=reference_core,controls=scaled[2:])
        write_json(f/'control_membership.json',{'banks':banks,'selection_data':'graph degrees and RNG only; no test performance',
                                               'core_source':frozen['core_units']})
        result=job|{'status':'complete','identity_sha256':identity,'caliper_results':scores,
                    'control_design_sha256':hashlib.sha256((f/'control_design.json').read_bytes()).hexdigest(),
                    'source_checkpoint_sha256':hashlib.sha256((old/'trained.pt').read_bytes()).hexdigest(),
                    'test_dates_sha256':digest(test['dates'].tolist()),'parity_max_abs_normalized_prediction':parity,
                    'near_tie_sequential_rechecks':tie_rechecks,'seconds':time.time()-started}
        write_json(stamp,result);return result
    except Exception as error:
        failure=job|{'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        write_json(f/'failure.json',failure);return failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/steh_population_20261012')
    parser.add_argument('--circuits',default='outputs/steh_population_20261012_circuits')
    parser.add_argument('--output',default='outputs/steh_control_sensitivity_20261014_sequential')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-jobs',type=int)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(args.source)/'manifest.json').read_text());early=sorted({j['station'] for j in manifest['jobs'][:24]})
    eligible=sorted(set(manifest['confirmation_population'])-set(early))
    assert not set(early)&set(manifest['confirmation_stations'])
    jobs=manifest['jobs'][:args.max_jobs] if args.max_jobs else manifest['jobs']
    protocol={'method':'robust-degree caliper random structural controls; sensitivity only',
              'source_manifest_sha256':digest(manifest),'training_config':manifest['config'],
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'jobs':jobs,'calipers':[1.,2.],'controls_per_caliper':20,'vmap_chunk':1,
              'forecast_backend':'sequential per-mask reference; parallel stations',
              'method_source':'experiments/steh_control_sensitivity_sequential.py',
              'numerical_recovery_of':'34bac78acc39f7712f79674ec9dfa240d653f567981728a6b42efa28f802ddc3',
              'near_tie_policy':'Every control is computed sequentially; additional sequential tie rechecks are unnecessary.',
              'degree_distance':'sum absolute in/out degree difference divided by max(1,pool median absolute deviation)',
              'sampling':'without replacement within each group; random among distance <= nearest + caliper',
              'circuit_selection':'reuse exact validation-frozen core; no model retraining',
              'caliper_selection':'both prespecified; no selection by test performance',
              'early_method_development_basins':early,'confirmation_stations':manifest['confirmation_stations'],
              'conditional_confirmation_population':eligible,
              'confirmation_design':'Original SRS conditioned on avoiding fixed first12 development basins; N492, n12.',
              'confidence':.9875,'interval_family':'Bonferroni four marginal bounds: 2calipers x 2Qsettings, joint coverage >=95%',
              'limits':'Original nearest-degree protocol preserved as primary. No p-value or globally minimal/causal circuit claim.'}
    p=out/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:write_json(p,protocol)
    if args.prepare_only:print(json.dumps({'prepared':len(jobs),'confirmation_population':len(eligible)}));return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
    started=time.time();results=[]
    with mp.get_context('spawn').Pool(min(args.workers,len(jobs))) as pool:
        for result in pool.imap_unordered(work,[(j,protocol,args.source,args.circuits,args.output) for j in jobs]):
            results.append(result);write_json(out/'progress.json',results)
            print(f"CALIPER {result['station']} seed={result['seed']} q={result['with_q']} {result['status']} {len(results)}/{len(jobs)}",flush=True)
    rows=[]
    for r in results:
        if r['status']=='complete':
            for s in r['caliper_results']:rows.append({k:r[k] for k in ['station','seed','with_q']}|s)
    table=pd.DataFrame(rows);table.to_csv(out/'caliper_summary.csv',index=False)
    confirmation=[]
    if not args.max_jobs and len(results)==len(jobs) and all(r['status']=='complete' for r in results):
        fixed=table[table.station.isin(protocol['confirmation_stations'])].groupby(['station','with_q','caliper']).criterion_pass.all().reset_index()
        for (q,c),g in fixed.groupby(['with_q','caliper']):
            assert len(g)==12
            k=int(g.criterion_pass.sum());confirmation.append({'with_q':bool(q),'caliper':float(c),'successes':k,'basins':12,
                'interval':finite_population_interval(k,12,len(eligible),confidence=protocol['confidence'])})
    write_json(out/'confirmation.json',{'basins_are_independent_units':True,'interval_family':protocol['interval_family'],'results':confirmation})
    completion={'status':'PASS' if all(r['status']=='complete' for r in results) else 'FAILED',
                'completed':sum(r['status']=='complete' for r in results),'planned':len(jobs),'seconds':time.time()-started,
                'limits':protocol['limits']}
    write_json(out/'completion.json',completion);print(json.dumps(completion))
    if completion['status']!='PASS':raise RuntimeError('Control sensitivity failures retained')


if __name__=='__main__':main()

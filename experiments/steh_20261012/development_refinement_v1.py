"""Validation-only joint circuit reduction of frozen STEH checkpoints.

This is method development on the original 12 basins, not a new independent
confirmation sample. Original training results and 25% circuits remain intact.
"""
import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from revision.common import digest, write_json, metrics


def unit_list(masks):
    return [(key, layer, i) for key in ('head_mask_overrides', 'neuron_mask_overrides')
            for layer, flags in masks[key].items()
            for i, flag in enumerate(flags.detach().cpu().tolist()) if flag > 0]


def backward_reduce(base, score, tolerance=.05, rounds=12):
    """Jointly verify every accepted deletion against one fixed validation budget."""
    from revision.analysis import clone_masks
    current = clone_masks(base)
    full = score(current)
    if full is None or not np.isfinite(full):
        raise ValueError('Undefined full validation NSE')
    trace = []
    for iteration in range(rounds):
        candidates = unit_list(current)
        if len(candidates) <= 1:
            break
        ranked = []
        for key, layer, index in candidates:
            trial = clone_masks(current)
            trial[key][layer][index] = 0
            value = score(trial)
            if value is not None and np.isfinite(value):
                ranked.append((value, key, layer, index))
        ranked.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        accepted = False
        count = min(8, max(1, len(ranked)//10))
        while ranked and count >= 1:
            trial = clone_masks(current)
            for _, key, layer, index in ranked[:count]:
                trial[key][layer][index] = 0
            value = score(trial)
            valid = value is not None and np.isfinite(value)
            accepted = bool(valid and full-value <= tolerance and unit_list(trial))
            trace.append({'round': iteration, 'proposed_removed': count,
                          'remaining': len(unit_list(trial)), 'validation_nse': value,
                          'accepted': accepted})
            if accepted:
                current = trial
                break
            if count == 1:
                break
            count = max(1, count//2)
        if not accepted:
            break
    return current, {'full_validation_nse': full, 'core_validation_nse': score(current),
                     'active_units': len(unit_list(base)), 'core_units': len(unit_list(current)),
                     'trace': trace, 'search_round_limit': rounds}


def linear_cka(a, b):
    """Compare functions without equating arbitrary hidden-neuron identifiers."""
    a = a-a.mean(axis=0); b = b-b.mean(axis=0)
    denominator = np.linalg.norm(a.T@a, 'fro')*np.linalg.norm(b.T@b, 'fro')
    return float(np.linalg.norm(a.T@b, 'fro')**2/denominator) if denominator else None


def work(payload):
    job, protocol, source, output = payload
    from revision.runner import job_path
    from revision.models import transformer, phase_switch, predict, seed_all
    from revision.data import load_basin
    from revision.analysis import binary_overrides, clone_masks, subset
    import torch
    folder = job_path(Path(output), job); folder.mkdir(parents=True, exist_ok=True)
    stamp = digest({'job': job, 'protocol': protocol})
    complete = folder/'result.json'
    if complete.exists():
        result = json.loads(complete.read_text())
        if result['identity_sha256'] != stamp: raise ValueError('Frozen result differs')
        return result
    started = time.time()
    try:
        with (folder/'worker.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            torch.set_num_threads(1); seed_all(job['seed'])
            cfg = protocol['training_config']|job|{'device':'cuda', 'gpu_resident_data':True}
            splits, meta = load_basin(cfg['data_root'], job['station'], cfg)
            old = job_path(Path(source), job)
            checkpoint = torch.load(old/'trained.pt', map_location='cpu', weights_only=False)
            model = transformer(len(meta['feature_names']), cfg)
            model.load_state_dict(checkpoint['state_dict']); model.to('cuda')
            phase_switch(model, 'mask', cfg['target_activation_sparsity'])
            base = binary_overrides(model, 'cuda')
            validation = subset(splits['val'], protocol['validation_days'], cfg['analysis_seed'])
            def score(mask): return metrics(predict(model, validation, cfg, mask), validation['y'])['nse']
            core, search = backward_reduce(base, score, protocol['validation_tolerance'], protocol['rounds'])
            # Circuit, budgets and input-effect ranking are frozen before reading test scores.
            frozen = {'core_units': {f'{key}:{layer}':[i for k,l,i in unit_list(core) if k==key and l==layer]
                                    for key in core if isinstance(core[key], dict) for layer in core[key]},
                      'validation_search': search, 'validation_dates_sha256':digest(validation['dates'].tolist())}
            write_json(folder/'validation_selection.json', frozen)
            feature_rows = []
            for index, feature in enumerate(meta['feature_names']):
                mask = clone_masks(base); mask['input_mask_override'][index] = 0
                feature_rows.append({'feature':feature, 'active':bool(base['input_mask_override'][index]),
                                     'validation_delta_nse':search['full_validation_nse']-score(mask)})
            write_json(folder/'feature_effects.json', feature_rows)
            _, states = predict(model, validation, cfg, base, hidden=True)
            np.save(folder/'validation_hidden.npy', states)
            removed = clone_masks(base)
            for key, layer, index in unit_list(core): removed[key][layer][index] = 0
            test = splits['test']; target = test['y']*meta['q_std']+meta['q_mean']
            predictions = {name:predict(model, test, cfg, masks)*meta['q_std']+meta['q_mean']
                           for name,masks in [('full',base),('core',core),('removed',removed)]}
            scores = {name:metrics(values,target) for name,values in predictions.items()}
            prior = json.loads((old/'summary.json').read_text())['metrics']
            for name in ['nse','rmse','mae']:
                np.testing.assert_allclose(scores['full'][name], prior[name], rtol=3e-6, atol=3e-5)
            pd.DataFrame({'target_date':test['dates'],'observed':target,**predictions}).to_csv(folder/'predictions.csv',index=False)
            # Exact layer/type/count; degrees approximately matched, overlap disclosed.
            pools = {}
            for key,layer,index in unit_list(base): pools.setdefault((key,layer),[]).append(index)
            degrees = {}
            for key,layer in pools:
                module = model.layers[layer]
                if key.startswith('neuron'):
                    d = torch.stack([(module.mlp.fc1.weight!=0).sum(1),(module.mlp.fc2.weight!=0).sum(0)],1).cpu().tolist()
                else:
                    att = module.attn; d=[]
                    for h in range(att.n_heads):
                        ids = torch.arange(h*att.d_head,(h+1)*att.d_head,device='cuda')
                        rows = torch.cat([ids,ids+att.d_model,ids+2*att.d_model])
                        d.append([int((att.qkv.weight[rows]!=0).sum()),int((att.out_proj.weight[:,ids]!=0).sum())])
                degrees[(key,layer)] = d
            chosen = {(key,layer):[i for k,l,i in unit_list(core) if k==key and l==layer] for key,layer in pools}
            rng = np.random.default_rng(job['seed']+cfg['analysis_seed']); controls=[]
            for repeat in range(protocol['random_controls']):
                mask=clone_masks(base); overlaps=0; mismatch=[]
                for group,pool in pools.items():
                    key,layer=group; mask[key][layer].zero_(); available=list(pool)
                    for i in rng.permutation(chosen[group]):
                        differences=[sum(abs(a-b) for a,b in zip(degrees[group][z],degrees[group][i])) for z in available]
                        best=min(differences); z=int(rng.choice([z for z,d in zip(available,differences) if d==best]))
                        available.remove(z); mask[key][layer][z]=1; overlaps+=int(z in chosen[group]);mismatch.append(best)
                m=metrics(predict(model,test,cfg,mask)*meta['q_std']+meta['q_mean'],target)
                controls.append({'repeat':repeat,'nse':m['nse'],'core_overlap_fraction':overlaps/search['core_units'],
                                 'mean_degree_mismatch':float(np.mean(mismatch)) if mismatch else None,
                                 'identical_core':overlaps==search['core_units']})
            independent=[r for r in controls if not r['identical_core'] and r['nse'] is not None]
            full,core_score,remove=[scores[n]['nse'] for n in ['full','core','removed']]
            win=float(np.mean([core_score>r['nse'] for r in independent])) if independent else None
            result={**job,'identity_sha256':stamp,'status':'complete','metrics':scores,
                    'validation_search':{k:v for k,v in search.items() if k!='trace'},
                    'core_delta_nse':full-core_score,'remove_delta_nse':full-remove,
                    'random_win_fraction':win,'distinct_control_count':len(independent),
                    'pass':bool(full-core_score<=.05 and full-remove>=.02 and win is not None and win>=.8),
                    'seconds':time.time()-started,'test_dates_sha256':digest(test['dates'].tolist()),
                    'source_checkpoint_sha256':hashlib.sha256((old/'trained.pt').read_bytes()).hexdigest()}
            write_json(folder/'random_controls.json',controls);write_json(complete,result)
            return result
    except Exception as error:
        failure={**job,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        write_json(folder/'failure.json',failure);return failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='outputs/revision_3090')
    parser.add_argument('--output',default='outputs/steh_refined_20261012')
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--max-jobs',type=int)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(args.source)/'manifest.json').read_text())
    jobs=[job for job in manifest['jobs'] if job['model']=='steh']
    if args.max_jobs: jobs=jobs[:args.max_jobs]
    protocol={'method':'validation-guided joint backward unit reduction','source_manifest_sha256':digest(manifest),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'training_config':manifest['config'],'jobs':jobs,'validation_days':512,'rounds':12,
              'validation_tolerance':.05,'random_controls':20,'selection_data':'validation only',
              'test_used_for_selection':False,'status':'method-development; original 12 basins already evaluated',
              'comparison':'same frozen checkpoint and all test dates; no retraining or raw-neuron ID matching'}
    path=out/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol
    else: write_json(path,protocol)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    started=time.time();results=[]
    with mp.get_context('spawn').Pool(min(args.workers,len(jobs))) as pool:
        for result in pool.imap_unordered(work,[(j,protocol,args.source,args.output) for j in jobs]):
            results.append(result);write_json(out/'progress.json',results)
            print(f"REFINEMENT {result['station']} seed={result['seed']} q={result['with_q']} {result['status']} {len(results)}/{len(jobs)}",flush=True)
    good=[r for r in results if r['status']=='complete'];rows=[]
    for r in good:
        rows.append({k:r[k] for k in ['station','seed','with_q','core_delta_nse','remove_delta_nse','random_win_fraction','distinct_control_count','pass','seconds']}|
                    {'full_nse':r['metrics']['full']['nse'],'core_nse':r['metrics']['core']['nse'],**r['validation_search']})
    pd.DataFrame(rows).to_csv(out/'circuit_summary.csv',index=False)
    from revision.runner import job_path
    stability=[]
    for aidx,a in enumerate(good):
        for b in good[aidx+1:]:
            if a['station']!=b['station'] or a['with_q']!=b['with_q']:continue
            fa,fb=[job_path(out,r) for r in [a,b]]
            ea,eb=[pd.DataFrame(json.loads((f/'feature_effects.json').read_text())).set_index('feature') for f in [fa,fb]]
            x=ea.validation_delta_nse.rank();y=eb.loc[x.index].validation_delta_nse.rank()
            corr=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
            stability.append({'station':a['station'],'with_q':a['with_q'],'seed_a':a['seed'],'seed_b':b['seed'],
                              'validation_input_effect_spearman':corr,'validation_hidden_linear_cka':linear_cka(np.load(fa/'validation_hidden.npy'),np.load(fb/'validation_hidden.npy')),
                              'core_delta_nse_a':a['core_delta_nse'],'core_delta_nse_b':b['core_delta_nse']})
    pd.DataFrame(stability).to_csv(out/'functional_stability.csv',index=False)
    completion={'status':'PASS' if len(good)==len(jobs) else 'FAILED','completed':len(good),'planned':len(jobs),
                'seconds':time.time()-started,'criterion_passes':sum(r['pass'] for r in good),
                'limits':'Completion PASS means computational integrity, not scientific circuit validity. No significance claim.'}
    write_json(out/'completion.json',completion);print(json.dumps(completion),flush=True)
    if len(good)!=len(jobs):raise RuntimeError('Refinement failures retained; inspect per-job failure.json')


if __name__=='__main__':main()

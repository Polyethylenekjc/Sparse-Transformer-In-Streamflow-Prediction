"""Fresh all-station STEH fits and prespecified new-basin seed confirmation."""
import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from revision.common import digest,write_json,metrics
from revision.runner import job_path


def train_job(payload):
    job,manifest,output=payload
    import torch
    from revision.data import load_basin
    from revision.models import seed_all,transformer,fit,predict,phase_switch
    from revision.analysis import subset,binary_overrides,probes
    from revision.runner import baselines
    from experiments.refine_steh import backward_reduce
    folder=job_path(Path(output),job);folder.mkdir(parents=True,exist_ok=True)
    identity=digest({'job':job,'manifest':manifest});complete=folder/'summary.json'
    if complete.exists():
        old=json.loads(complete.read_text());assert old['run_sha256']==identity
        return {'job':job,'status':'complete','resumed':True}
    started=time.time()
    try:
        with (folder/'worker.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            torch.set_num_threads(1);seed_all(job['seed']);torch.cuda.reset_peak_memory_stats()
            cfg=manifest['config']|job|{'device':'cuda'}
            splits,meta=load_basin(cfg['data_root'],job['station'],cfg)
            write_json(folder/'preprocessing.json',meta);write_json(folder/'run_config.json',cfg)
            model=transformer(len(meta['feature_names']),cfg)
            checkpoint=folder/'trained.pt'
            if checkpoint.exists():
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert saved['run_sha256']==identity;model.load_state_dict(saved['state_dict'])
                model.to('cuda');phase_switch(model,'mask',cfg['target_activation_sparsity'])
            else:
                model,logs=fit(model,splits,cfg)
                pd.DataFrame(logs).to_csv(folder/'training_log.csv',index=False)
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'config':cfg,'preprocessing':meta,'run_sha256':identity},folder/'trained.tmp')
                (folder/'trained.tmp').replace(checkpoint)
            # Freeze the circuit before any test scoring in this fresh-model job.
            base=binary_overrides(model,'cuda');validation=subset(splits['val'],512,cfg['analysis_seed'])
            def score(mask):return metrics(predict(model,validation,cfg,mask),validation['y'])['nse']
            core,search=backward_reduce(base,score,.05,12)
            frozen={'core_units':{f'{key}:{layer}':[i for i,v in enumerate(flags.cpu().tolist()) if v>0]
                                  for key in ['head_mask_overrides','neuron_mask_overrides'] for layer,flags in core[key].items()},
                    'validation_search':search,'validation_dates_sha256':digest(validation['dates'].tolist()),
                    'method_script_sha256':manifest['refinement_script_sha256'],
                    'selection_data':'validation only','test_used_for_selection':False}
            write_json(folder/'validation_selection.json',frozen)
            # References and final scores now read test labels; no selection follows.
            baselines(splits,meta,cfg,folder)
            predicted=predict(model,splits['test'],cfg)*meta['q_std']+meta['q_mean']
            observed=splits['test']['y']*meta['q_std']+meta['q_mean']
            if not np.isfinite(predicted).all():raise ValueError('Non-finite test prediction')
            pd.DataFrame({'target_date':splits['test']['dates'],'observed':observed,'predicted':predicted}).to_csv(folder/'predictions.csv',index=False)
            write_json(folder/'probes.json',probes(model,splits,meta,cfg))
            versions={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                      'gpu':torch.cuda.get_device_name(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                      'float32_matmul_precision':torch.get_float32_matmul_precision()}
            write_json(folder/'performance.json',{'total_seconds':time.time()-started,
                       'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'pid':os.getpid()})
            write_json(complete,{'job':job,'run_sha256':identity,'metrics':metrics(predicted,observed),
                       'target_status':meta['target_status'],'training_versions':versions,
                       'test_dates_sha256':digest(splits['test']['dates'].tolist()),'selected_hyperparameters':manifest['selected_hyperparameters']})
            return {'job':job,'status':'complete','seconds':time.time()-started}
    except Exception as error:
        failure={'job':job,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        write_json(folder/'failure.json',failure);return failure


def freeze_plan(selection,output,all_seeds=False):
    selected=json.loads(Path(selection).read_text())
    assert selected['selection_data']=='validation only' and not selected['test_used_for_selection']
    reference=json.loads(Path('outputs/lstm_global_3090/tuning_protocol.json').read_text())
    original=json.loads(Path('outputs/revision_3090/manifest.json').read_text())
    population=reference['final_stations']
    unseen=sorted(set(population)-set(original['sample'])-set(reference['tuning_stations']))
    confirmation=sorted(np.random.default_rng(20261013).choice(unseen,12,replace=False).tolist())
    cfg=json.loads(Path('configs/revision_server_3090.json').read_text())|selected['hyperparameters']
    cfg.update({'output_root':output,'sample_size':len(population),'seeds':[42,43,44] if all_seeds else [42],
                'record_structural_diagnostics':True,'optional_window_lengths':[]})
    jobs=[{'station':station,'seed':seed,'model':'steh','with_q':q,'seq_len':15}
          for station in population for seed in cfg['seeds'] for q in [True,False]]
    if not all_seeds:
        jobs += [{'station':station,'seed':seed,'model':'steh','with_q':q,'seq_len':15}
                 for station in confirmation for seed in [43,44] for q in [True,False]]
    sources={}
    for station in population:
        for group in ['Forcing','Streamflow']:
            p=Path(cfg['data_root'])/group/f'{station}.csv';value=hashlib.sha256(p.read_bytes()).hexdigest()
            assert value==reference['basin_source_hashes'][station][f'{station}.csv:{group}']
            sources[f'{group}/{station}.csv']=value
    files=list((ROOT/'revision').glob('*.py'))+[ROOT/'src/flood_transformer/model.py',ROOT/'src/flood_transformer/sparsity.py',Path(__file__),ROOT/'experiments/refine_steh.py']
    code={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    plan={'config':cfg,'config_sha256':digest(cfg),'code_sha256':code,'data_sha256':sources,
          'population':population,'sample':population,'jobs':jobs,'selected_hyperparameters':selected['hyperparameters'],
          'selection_sha256':hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
          'refinement_script_sha256':code['experiments/refine_steh.py'],
          'primary_seed':42,'confirmation_stations':confirmation,'confirmation_population':unseen,'confirmation_sampling_seed':20261013,
          'confirmation_selection':'SRS without replacement before final STEH performance; exclude original12 and tuning6',
          'scientific_criteria':{'core_max_delta_nse':.05,'remove_min_delta_nse':.02,'random_min_win_fraction':.8},
          'scope':'all521 primary seed42 with-Q/no-Q; 12 new basins with seeds42/43/44' if not all_seeds else 'all521 with-Q/no-Q at seeds42/43/44',
          'limits':'One main forecast per station, no seed ensemble; remaining seeds assess functional sensitivity.'}
    out=Path(output);out.mkdir(parents=True,exist_ok=True);path=out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Frozen plan changed; use a new output'
    else:write_json(path,plan);write_json(out/'config_resolved.json',cfg)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',default='outputs/steh_tuning_20261012/selected_hyperparameters.json')
    parser.add_argument('--output',default='outputs/steh_population_20261012')
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--all-seeds',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--skip-circuits',action='store_true')
    args=parser.parse_args();plan=freeze_plan(args.selection,args.output,args.all_seeds)
    print(f"FROZEN: {len(plan['jobs'])} jobs; confirmation {plan['confirmation_stations']}",flush=True)
    if args.prepare_only:return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[variable]='1'
    jobs=plan['jobs'][:args.max_jobs] if args.max_jobs else plan['jobs'];started=time.time();results=[]
    with mp.get_context('spawn').Pool(min(args.workers,len(jobs))) as pool:
        for result in pool.imap_unordered(train_job,[(j,plan,args.output) for j in jobs]):
            results.append(result);write_json(Path(args.output)/'progress.json',{'results':results,'planned':len(jobs),'elapsed_seconds':time.time()-started})
            print(f"POPULATION {result['job']} {result['status']} {len(results)}/{len(jobs)}",flush=True)
    complete=[r for r in results if r['status']=='complete']
    rows=[]
    for r in complete:
        data=json.loads((job_path(Path(args.output),r['job'])/'summary.json').read_text())
        rows.append(r['job']|data['metrics'])
    pd.DataFrame(rows).to_csv(Path(args.output)/'summary.csv',index=False)
    write_json(Path(args.output)/'training_completion.json',{'status':'PASS' if len(complete)==len(jobs) else 'FAILED',
               'completed':len(complete),'planned':len(jobs),'seconds':time.time()-started,'smoke_only':bool(args.max_jobs)})
    if len(complete)!=len(jobs):raise RuntimeError('Training failures retained; inspect failure.json')
    if not args.max_jobs and not args.skip_circuits:
        import subprocess
        subprocess.run([sys.executable,str(ROOT/'experiments/refine_steh.py'),'--source',args.output,
                        '--output',args.output+'_circuits','--workers',str(args.workers)],check=True)
    print('STEH POPULATION FINISHED',flush=True)


if __name__=='__main__':main()

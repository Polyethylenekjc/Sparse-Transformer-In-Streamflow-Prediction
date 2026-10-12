"""Shared Bayesian LSTM tuning, followed by one freshly trained model per station."""
import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from revision.common import digest,write_json,metrics
from experiments.lstm_core import SPACE,train_one,save_state

CACHE={}

def choose_global(rows,stations):
    panels={}
    for row in rows:panels.setdefault(row['trial'],{})[row['station']]=row['validation_metrics']['nse']
    if not panels or any(set(v)!=set(stations) for v in panels.values()):
        raise ValueError('Incomplete tuning-station panel')
    values={t:float(np.mean([p[s] for s in stations])) for t,p in panels.items()}
    if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite validation NSE')
    return max(values,key=lambda t:(values[t],-t)),values

def worker(payload):
    phase,station,hp,trial,protocol,output=payload
    folder=Path(output)/phase/(f'trial_{trial:03d}' if phase=='search' else 'models')/station
    folder.mkdir(parents=True,exist_ok=True)
    try:
        with (folder/'worker.log').open('a',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            return execute(phase,station,hp,trial,protocol,folder)
    except Exception:
        record={'status':'failed','phase':phase,'station':station,'trial':trial,'traceback':traceback.format_exc()}
        write_json(folder/'failure.json',record)
        return record

def execute(phase,station,hp,trial,protocol,folder):
    import torch
    from revision.data import load_basin
    from revision.models import LSTM,predict
    from revision.runner import baselines
    cfg=protocol['training_config']|{'device':'cuda' if torch.cuda.is_available() else 'cpu','with_q':True}
    torch.set_num_threads(cfg['torch_threads'])
    key=(station,digest(cfg))
    if key not in CACHE:CACHE.clear();CACHE[key]=load_basin(cfg['data_root'],station,cfg)
    splits,meta=CACHE[key]
    assert meta['source_sha256']==protocol['basin_source_hashes'][station]
    write_json(folder/'preprocessing.json',meta)
    seed=protocol['seed']
    # Search and final outputs are different directories: final fits are fresh.
    record=train_one(folder,hp,seed,splits,meta,cfg)
    record|={'status':'complete','phase':phase,'station':station,'trial':trial}
    write_json(folder/'trial_result.json',record)
    if phase=='final':
        checkpoint=torch.load(folder/'trained.pt',map_location='cpu',weights_only=False)
        model=LSTM(len(meta['feature_names']),hp['hidden_size'],hp['num_layers'],hp['dropout']).to(cfg['device'])
        model.load_state_dict(checkpoint['state_dict'])
        local=checkpoint['config']
        pred=predict(model,splits['test'],local)*meta['q_std']+meta['q_mean']
        observed=splits['test']['y']*meta['q_std']+meta['q_mean']
        pd.DataFrame({'target_date':splits['test']['dates'],'observed':observed,'predicted':pred}).to_csv(folder/'predictions.csv',index=False)
        baselines(splits,meta,local,folder)
        write_json(folder/'run_config.json',local)
        write_json(folder/'summary.json',{'station':station,'seed':seed,'hyperparameters':hp,
                   'metrics':metrics(pred,observed),'selected_trial':trial,'protocol_sha256':digest(protocol),
                   'test_dates_sha256':digest(splits['test']['dates'].tolist()),'target_status':meta['target_status'],
                   'training_versions':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                    'gpu':torch.cuda.get_device_name(0) if cfg['device']=='cuda' else None,
                    'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                    'float32_matmul_precision':torch.get_float32_matmul_precision()}})
    return record

def make_protocol(cfg,reference,args):
    population=reference['population']
    stations=reference['sample'] if args.scope=='sample' else population
    tuning=sorted(np.random.default_rng(args.sampling_seed).choice(population,args.tuning_stations,replace=False).tolist())
    hashes={station:{f'{station}.csv:{name}':hashlib.sha256((Path(cfg['data_root'])/name/f'{station}.csv').read_bytes()).hexdigest()
                        for name in ['Forcing','Streamflow']} for station in sorted(set(stations+tuning))}
    files=['experiments/global_lstm.py','experiments/lstm_core.py','revision/common.py','revision/data.py','revision/models.py','revision/runner.py']
    return {'version':'global-lstm-bayesian-v1','training_config':cfg,'scope':args.scope,
            'final_stations':stations,'tuning_stations':tuning,'population_count':len(population),
            'sampling_seed':args.sampling_seed,'seed':args.seed,'search_seed':20261012,'search_space':SPACE,
            'max_trials':args.trials,'search_seconds':args.search_seconds,'candidate_batch':2,
            'objective':'maximize unweighted mean validation NSE over the same randomly sampled tuning stations',
            'test_policy':'test outcomes never enter Bayesian search or selection; evaluate only frozen global winner',
            'final_policy':'fresh initialization; one model per station; one seed; one input window',
            'initial_candidates':[hp|{'dropout':cfg['dropout'] if hp['num_layers']>1 else 0.,'weight_decay':cfg['weight_decay']} for hp in cfg['lstm_trials']],
            'basin_source_hashes':hashes,'reference_manifest_sha256':digest(reference),
            'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
            'limits':'finite Bayesian search on six sampled validation basins; no global-optimum or all-basin superiority guarantee'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/revision_server_3090.json')
    p.add_argument('--reference',default='outputs/revision_3090/manifest.json')
    p.add_argument('--output',default='outputs/lstm_global_3090')
    p.add_argument('--scope',choices=['sample','all'],default='sample')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sampling-seed',type=int,default=20261012)
    p.add_argument('--tuning-stations',type=int,default=6)
    p.add_argument('--trials',type=int,default=24)
    p.add_argument('--search-seconds',type=int,default=1800)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text());reference=json.loads(Path(args.reference).read_text())
    cfg=cfg|{'seeds':[args.seed],'optional_window_lengths':[]}
    assert args.trials>=4 and args.workers>0 and 1<=args.tuning_stations<=len(reference['population'])
    protocol=make_protocol(cfg,reference,args);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    file=out/'tuning_protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol,'Frozen global tuning protocol changed'
    else:write_json(file,protocol)
    print(f"FROZEN: {args.trials} shared candidates, {len(protocol['tuning_stations'])} random tuning stations, {len(protocol['final_stations'])} final stations, seed {args.seed}",flush=True)
    if args.prepare_only:return
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]=str(cfg['torch_threads'])
    sampler_file=out/'sampler.pkl'
    sampler=optuna.samplers.TPESampler(seed=protocol['search_seed'],n_startup_trials=8)
    if sampler_file.exists():
        with sampler_file.open('rb') as f:sampler=pickle.load(f)
    study=optuna.create_study(study_name='shared_lstm',direction='maximize',sampler=sampler,
                             storage=f"sqlite:///{(out/'study.sqlite3').resolve()}",load_if_exists=True)
    if not study.trials:
        for hp in protocol['initial_candidates']:study.enqueue_trial(hp)
    # Finish a saved complete panel after an interrupted tell; incomplete fits can resume.
    for t in study.trials:
        if t.state==optuna.trial.TrialState.RUNNING:
            files=[out/'search'/f'trial_{t.number:03d}'/s/'trial_result.json' for s in protocol['tuning_stations']]
            if all(f.exists() for f in files):
                records=[json.loads(f.read_text()) for f in files];_,values=choose_global(records,protocol['tuning_stations']);study.tell(t.number,values[t.number])
            else:study.tell(t.number,state=optuna.trial.TrialState.FAIL)
    timing_file=out/'timing.json'
    timing=json.loads(timing_file.read_text()) if timing_file.exists() else {'started_unix':time.time()}
    write_json(timing_file,timing)
    deadline=timing['started_unix']+args.search_seconds
    decision_file=out/'selected_hyperparameters.json'
    with mp.get_context('spawn').Pool(args.workers) as pool:
        while not decision_file.exists() and sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)<args.trials:
            completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
            if time.time()>=deadline and completed>=4:break
            batch=[]; payloads=[]
            for _ in range(min(2,args.trials-completed)):
                t=study.ask()
                hp={'hidden_size':t.suggest_categorical('hidden_size',SPACE['hidden_size']),
                    'num_layers':t.suggest_categorical('num_layers',SPACE['num_layers']),
                    'learning_rate':t.suggest_float('learning_rate',1e-4,3e-3,log=True),
                    'weight_decay':t.suggest_categorical('weight_decay',SPACE['weight_decay'])}
                hp['dropout']=t.suggest_categorical('dropout',SPACE['dropout']) if hp['num_layers']>1 else 0.
                batch.append((t,hp))
                write_json(out/'search'/f'trial_{t.number:03d}'/'hyperparameters.json',hp)
                payloads.extend(('search',s,hp,t.number,protocol,str(out)) for s in protocol['tuning_stations'])
            save_state(sampler_file,study.sampler)
            records=pool.map(worker,payloads,chunksize=1)
            if any(r['status']!='complete' for r in records):raise RuntimeError('Search station failed; inspect failure.json')
            _,values=choose_global(records,protocol['tuning_stations'])
            for t,hp in batch:
                study.tell(t,values[t.number])
                print(f"TRIAL {t.number} mean validation NSE={values[t.number]:.6g} best={study.best_value:.6g}",flush=True)
            save_state(sampler_file,study.sampler)
            write_json(out/'search_trials.json',[{'trial':t.number,'state':t.state.name,'mean_validation_nse':t.value,'params':t.params} for t in study.trials])
        if not decision_file.exists():
            best=study.best_trial;hp=json.loads((out/'search'/f'trial_{best.number:03d}'/'hyperparameters.json').read_text())
            decision={'selected_trial':best.number,'hyperparameters':hp,'mean_validation_nse':best.value,
                      'completed_trials':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),
                      'selection_data':'validation only','test_evaluated_for_selection':False,'protocol_sha256':digest(protocol)}
            write_json(decision_file,decision)
        else:decision=json.loads(decision_file.read_text())
        timing['search_finished_unix']=time.time();write_json(timing_file,timing)
        print('GLOBAL WINNER '+json.dumps(decision),flush=True)
        pending=[s for s in protocol['final_stations'] if not (out/'final/models'/s/'summary.json').exists()]
        progress=[]
        for r in pool.imap_unordered(worker,[('final',s,decision['hyperparameters'],decision['selected_trial'],protocol,str(out)) for s in pending],chunksize=1):
            progress.append(r);write_json(out/'final_progress.json',progress)
            print(f"FINAL {r['station']} {r['status']} {len(progress)}/{len(pending)}",flush=True)
        if any(r['status']!='complete' for r in progress):raise RuntimeError('Some final models failed')
    final=[json.loads((out/'final/models'/s/'summary.json').read_text()) for s in protocol['final_stations']]
    assert len(final)==len(protocol['final_stations'])
    assert all(s['hyperparameters']==decision['hyperparameters'] and s['seed']==args.seed for s in final)
    pd.DataFrame([{'station':s['station'],'seed':s['seed'],**s['metrics'],**s['hyperparameters']} for s in final]).to_csv(out/'final_summary.csv',index=False)
    timing['finished_unix']=time.time();timing['total_seconds']=timing['finished_unix']-timing['started_unix'];write_json(timing_file,timing)
    write_json(out/'completion.json',{'status':'PASS','final_models':len(final),'completed_candidates':decision['completed_trials'],
                                     'single_shared_hyperparameters':True,'single_seed':args.seed,'total_seconds':timing['total_seconds']})
    print(f"GLOBAL LSTM COMPLETE: {len(final)} fresh station models in {timing['total_seconds']:.1f}s",flush=True)

if __name__=='__main__':main()

"""Shared STEH Bayesian tuning; validation only, fixed 30-epoch phase budget."""
import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from revision.common import digest,write_json,metrics


def fit_candidate(payload):
    trial,station,with_q,hp,protocol,output=payload
    import torch
    from revision.data import load_basin
    from revision.models import seed_all,transformer,fit,predict
    folder=Path(output)/'search'/f'trial_{trial:03d}'/station/('with_q' if with_q else 'no_q')
    folder.mkdir(parents=True,exist_ok=True)
    identity=digest({'trial':trial,'station':station,'with_q':with_q,'hp':hp,'protocol':protocol})
    complete=folder/'result.json'
    if complete.exists():
        result=json.loads(complete.read_text());assert result['identity_sha256']==identity;return result
    started=time.time()
    try:
        with (folder/'worker.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            torch.set_num_threads(1);seed_all(42)
            cfg=protocol['base_config']|hp|{'station':station,'with_q':with_q,'seed':42,'device':'cuda'}
            splits,meta=load_basin(cfg['data_root'],station,cfg)
            model=transformer(len(meta['feature_names']),cfg)
            model,logs=fit(model,splits,cfg)
            # No test prediction, circuit score or test metric is computed here.
            prediction=predict(model,splits['val'],cfg)*meta['q_std']+meta['q_mean']
            target=splits['val']['y']*meta['q_std']+meta['q_mean']
            score=metrics(prediction,target)
            if score['nse'] is None or not np.isfinite(score['nse']):raise ValueError('Undefined validation NSE')
            pd.DataFrame({'target_date':splits['val']['dates'],'observed':target,'predicted':prediction}).to_csv(folder/'validation_predictions.csv',index=False)
            pd.DataFrame(logs).to_csv(folder/'training_log.csv',index=False)
            write_json(folder/'preprocessing.json',meta);write_json(folder/'run_config.json',cfg)
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,
                        'preprocessing':meta,'identity_sha256':identity},folder/'trained.pt')
            result={'trial':trial,'station':station,'with_q':with_q,'seed':42,'hyperparameters':hp,
                    'validation_metrics':score,'status':'complete','seconds':time.time()-started,'identity_sha256':identity}
            write_json(complete,result);return result
    except Exception as error:
        result={'trial':trial,'station':station,'with_q':with_q,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        write_json(folder/'failure.json',result);return result


def suggest(trial):
    return {'d_model':trial.suggest_categorical('d_model',[64,128]),
            'num_layers':trial.suggest_categorical('num_layers',[2,3]),
            'd_ff':trial.suggest_categorical('d_ff',[128,256]),
            'learning_rate':trial.suggest_float('learning_rate',1e-4,3e-3,log=True),
            'dropout':trial.suggest_categorical('dropout',[0.,.1]),
            'target_weight_sparsity':trial.suggest_categorical('target_weight_sparsity',[.8,.9]),
            'target_activation_sparsity':trial.suggest_categorical('target_activation_sparsity',[.5,.8]),
            'lambda_mask_l1':trial.suggest_categorical('lambda_mask_l1',[1e-5,1e-4]),
            'lambda_input_mask_l1':trial.suggest_categorical('lambda_input_mask_l1',[1e-6,1e-5])}


def main():
    import optuna
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/steh_tuning_20261012')
    parser.add_argument('--workers',type=int,default=14)
    parser.add_argument('--max-trials',type=int,default=12)
    parser.add_argument('--seconds',type=int,default=1800)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    reference=json.loads(Path('outputs/lstm_global_3090/tuning_protocol.json').read_text())
    base=json.loads(Path('configs/revision_server_3090.json').read_text())
    stations=reference['tuning_stations']
    code={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'revision').glob('*.py'))+[ROOT/'src/flood_transformer/model.py',ROOT/'src/flood_transformer/sparsity.py',Path(__file__)]}
    protocol={'base_config':base,'tuning_stations':stations,'population':reference['final_stations'],
              'sampling_seed':reference['sampling_seed'],'tpe_seed':20261012,'training_seed':42,
              'epochs':{'dense':10,'sparse':10,'mask':10},'max_trials':args.max_trials,'time_budget_seconds':args.seconds,
              'objective':'equal-weight mean validation NSE across six stations and both with-Q/no-Q settings',
              'selection_data':'validation only','test_used_for_selection':False,'code_sha256':code,
              'limitations':'No test-driven tuning; one training seed during tuning; separate settings equally weighted.'}
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:write_json(path,protocol)
    sampler=optuna.samplers.TPESampler(seed=20261012,n_startup_trials=6)
    study=optuna.create_study(storage=f'sqlite:///{(out/"study.sqlite3").resolve()}',study_name='shared-steh',direction='maximize',sampler=sampler,load_if_exists=True)
    if (out/'sampler.pkl').exists():study.sampler=pickle.loads((out/'sampler.pkl').read_bytes())
    if not study.trials:
        initial={k:base[k] for k in ['d_model','num_layers','d_ff','learning_rate','dropout','target_weight_sparsity','target_activation_sparsity','lambda_mask_l1','lambda_input_mask_l1']}
        study.enqueue_trial(initial)
        study.enqueue_trial(initial|{'d_model':64,'num_layers':2,'d_ff':128,'target_activation_sparsity':.5})
    unfinished=[t for t in study.trials if t.state==optuna.trial.TrialState.RUNNING]
    if unfinished:raise RuntimeError('Running trials in database; recover them explicitly before resuming')
    started=time.time();records=[]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    with mp.get_context('spawn').Pool(args.workers) as pool:
        while len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.FAIL]])<args.max_trials:
            if time.time()-started>args.seconds:break
            completed=len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.FAIL]])
            batch=[]
            for _ in range(min(2,args.max_trials-completed)):
                trial=study.ask();hp=suggest(trial);batch.append((trial,hp))
                write_json(out/'search'/f'trial_{trial.number:03d}'/'hyperparameters.json',hp)
            payloads=[(trial.number,station,q,hp,protocol,args.output) for trial,hp in batch for station in stations for q in [True,False]]
            results=[]
            for result in pool.imap_unordered(fit_candidate,payloads):
                results.append(result);records.append(result);write_json(out/'progress.json',records)
                print(f"TUNING trial={result['trial']} station={result['station']} q={result['with_q']} {result['status']}",flush=True)
            for trial,hp in batch:
                panel=[r for r in results if r['trial']==trial.number]
                if len(panel)!=len(stations)*2 or any(r['status']!='complete' for r in panel):
                    study.tell(trial,state=optuna.trial.TrialState.FAIL)
                else:
                    value=float(np.mean([r['validation_metrics']['nse'] for r in panel]));study.tell(trial,value)
                    print(f'CANDIDATE {trial.number} validation NSE={value:.6f}',flush=True)
            (out/'sampler.pkl').write_bytes(pickle.dumps(study.sampler))
    best=study.best_trial
    selected={'selected_trial':best.number,'hyperparameters':best.params,'mean_validation_nse':best.value,
              'selection_data':'validation only','test_used_for_selection':False,'protocol_sha256':digest(protocol),
              'completed_trials':len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]),
              'seconds':time.time()-started}
    write_json(out/'selected_hyperparameters.json',selected)
    write_json(out/'completion.json',{'status':'PASS','selected':selected})
    print('STEH TUNING COMPLETE '+json.dumps(selected),flush=True)


if __name__=='__main__':main()

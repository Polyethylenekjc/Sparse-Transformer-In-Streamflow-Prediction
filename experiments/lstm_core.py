"""One LSTM fit, with validation checkpointing and complete saved records."""
import contextlib
import json
import pickle
import time
import numpy as np
import pandas as pd
from revision.common import digest,metrics,write_json

SPACE={'hidden_size':[32,64,128,256],'num_layers':[1,2],
       'learning_rate':{'low':1e-4,'high':3e-3,'log':True},
       'dropout':[0.,.1,.3],'weight_decay':[0.,1e-5,1e-4]}

def save_state(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:pickle.dump(value,f)
    temporary.replace(path)

def train_one(folder,hp,seed,splits,meta,cfg):
    import torch
    from revision.models import LSTM,fit,predict,seed_all
    folder.mkdir(parents=True,exist_ok=True)
    stamp=digest({'hyperparameters':hp,'seed':seed,'protocol':cfg,'preprocessing':meta})
    record_file=folder/'trial_result.json'
    if record_file.exists():
        record=json.loads(record_file.read_text())
        assert record['identity_sha256']==stamp and (folder/'trained.pt').exists()
        return record
    seed_all(seed)
    local=cfg|{'seed':seed,'with_q':True,'dropout':hp['dropout'],'weight_decay':hp['weight_decay']}
    model=LSTM(len(meta['feature_names']),hp['hidden_size'],hp['num_layers'],hp['dropout'])
    started=time.perf_counter()
    with (folder/'training.log').open('w',buffering=1) as f,contextlib.redirect_stdout(f):
        model,log=fit(model,splits,local,'lstm',hp)
    prediction=predict(model,splits['val'],local)
    loss=float(np.mean((prediction-splits['val']['y'])**2))
    if not np.isfinite(loss):raise ValueError('Nonfinite validation objective')
    best=min(log,key=lambda r:r['val_mse'])
    np.testing.assert_allclose(loss,best['val_mse'],rtol=2e-6,atol=1e-7)
    mean,std=meta['q_mean'],meta['q_std']
    pd.DataFrame({'target_date':splits['val']['dates'],'observed':splits['val']['y']*std+mean,
                  'predicted':prediction*std+mean}).to_csv(folder/'validation_predictions.csv',index=False)
    pd.DataFrame(log).to_csv(folder/'training_log.csv',index=False)
    temporary=folder/'trained.tmp'
    torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'config':local,'preprocessing':meta,'hyperparameters':hp,'identity_sha256':stamp,
                'best_epoch':best['epoch']},temporary)
    temporary.replace(folder/'trained.pt')
    record={'seed':seed,'hyperparameters':hp,'validation_mse':loss,'best_epoch':best['epoch'],
            'validation_metrics':metrics(prediction*std+mean,splits['val']['y']*std+mean),
            'seconds':time.perf_counter()-started,'identity_sha256':stamp,
            'epochs':cfg['epochs_lstm'],'training_samples':meta['counts']['train']}
    write_json(record_file,record)
    return record

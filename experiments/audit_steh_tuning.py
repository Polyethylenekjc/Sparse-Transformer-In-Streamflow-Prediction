"""Recompute all shared STEH tuning scores from validation predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision.common import metrics,digest,clean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    decision=json.loads((root/'selected_hyperparameters.json').read_text(encoding='utf-8'))
    assert protocol['selection_data']==decision['selection_data']=='validation only'
    assert not protocol['test_used_for_selection'] and not decision['test_used_for_selection']
    rows=[];dates={};parameters={}
    for p in sorted((root/'search').rglob('result.json')):
        r=json.loads(p.read_text(encoding='utf-8'));assert r['status']=='complete' and r['seed']==42
        assert r['identity_sha256']==digest({'trial':r['trial'],'station':r['station'],
            'with_q':r['with_q'],'hp':r['hyperparameters'],'protocol':protocol})
        prediction=pd.read_csv(p.parent/'validation_predictions.csv')
        assert not prediction.target_date.duplicated().any()
        score=metrics(prediction.predicted,prediction.observed)
        for name in ['nse','kge','rmse','mae','bias']:
            if r['validation_metrics'][name] is None:assert score[name] is None
            else:np.testing.assert_allclose(score[name],r['validation_metrics'][name],rtol=3e-6,atol=3e-5)
        log=pd.read_csv(p.parent/'training_log.csv')
        assert log.groupby('phase').size().to_dict()=={'dense':10,'sparse':10,'mask':10}
        meta=json.loads((p.parent/'preprocessing.json').read_text(encoding='utf-8'))
        normalized_mse=np.mean((prediction.predicted-prediction.observed)**2)/meta['q_std']**2
        np.testing.assert_allclose(normalized_mse,log[log.phase=='mask'].val_mse.min(),rtol=3e-6,atol=1e-7)
        if r['station'] in dates:assert dates[r['station']]==prediction.target_date.tolist()
        else:dates[r['station']]=prediction.target_date.tolist()
        if r['trial'] in parameters:assert parameters[r['trial']]==r['hyperparameters']
        else:parameters[r['trial']]=r['hyperparameters']
        rows.append({k:r[k] for k in ['trial','station','with_q','seed','seconds']}|
                    {'validation_nse':score['nse'],'validation_kge':score['kge'],**r['hyperparameters']})
    frame=pd.DataFrame(rows);assert len(frame)==12*len(protocol['tuning_stations'])*2
    assert not frame.duplicated(['trial','station','with_q']).any()
    for trial,g in frame.groupby('trial'):
        assert set(g.station)==set(protocol['tuning_stations']) and len(g)==12
    ranking=frame.groupby('trial',as_index=False).agg(mean_validation_nse=('validation_nse','mean'))
    winner=int(ranking.sort_values(['mean_validation_nse','trial'],ascending=[False,True]).iloc[0].trial)
    assert winner==decision['selected_trial'] and parameters[winner]==decision['hyperparameters']
    np.testing.assert_allclose(ranking.set_index('trial').loc[winner,'mean_validation_nse'],decision['mean_validation_nse'],rtol=3e-6)
    ranking['selected']=ranking.trial==winner
    frame.to_csv(out/'tuning_station_records.csv',index=False)
    ranking.to_csv(out/'tuning_candidates.csv',index=False)
    audit={'status':'PASS','candidate_fits':len(frame),'candidates':len(ranking),
           'tuning_stations':protocol['tuning_stations'],'selected_trial':winner,
           'selected_mean_validation_nse':decision['mean_validation_nse'],
           'base_mean_validation_nse':float(ranking.set_index('trial').loc[0,'mean_validation_nse']),
           'protocol_sha256':digest(protocol),'test_used_for_selection':False,
           'scope':'Best of 12 evaluated shared candidates; not a global optimality claim.'}
    (out/'tuning-audit.json').write_text(json.dumps(clean(audit),indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':main()

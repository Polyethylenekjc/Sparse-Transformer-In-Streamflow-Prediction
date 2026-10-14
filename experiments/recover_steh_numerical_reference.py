"""Recover numerical-reference failures without changing any scientific method.

The frozen implementation compared binary-gated metrics to native STE metrics.
This runner changes only that verification call in memory. It independently
checks reloaded native forecasts against the original native records and keeps
the binary full-model NSE comparison. Cores, weights, controls and thresholds
are unchanged. Original failures and completion records are archived first.
"""
import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from revision.common import metrics,write_json,digest
from revision.runner import job_path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_native(model,test,cfg,meta,target,scores,prior,old,folder):
    from revision.models import predict
    native=predict(model,test,cfg)*meta['q_std']+meta['q_mean']
    saved=pd.read_csv(old/'predictions.csv')
    assert saved.target_date.tolist()==test['dates'].tolist()
    np.testing.assert_allclose(saved.observed,target,rtol=3e-6,atol=3e-5)
    a=(native-meta['q_mean'])/meta['q_std']
    b=(saved.predicted.to_numpy()-meta['q_mean'])/meta['q_std']
    np.testing.assert_allclose(a,b,rtol=3e-5,atol=3e-6)
    native_scores=metrics(native,target)
    for name in ['nse','kge','rmse','mae','bias']:
        if prior[name] is None:assert native_scores[name] is None
        else:np.testing.assert_allclose(native_scores[name],prior[name],rtol=3e-6,atol=3e-5)
    np.testing.assert_allclose(scores['full']['nse'],prior['nse'],rtol=3e-6,atol=3e-5)
    write_json(folder/'numerical_reference_recovery.json',{
        'status':'PASS','native_metrics':native_scores,'binary_metrics':scores['full'],
        'maximum_reloaded_native_difference_train_scaled':float(abs(a-b).max()),
        'checkpoint_sha256':sha(old/'trained.pt'),'recovery_script_sha256':sha(Path(__file__)),
        'scope':'Only verification reference corrected; original validation core, control RNG and scientific thresholds retained.'})


def recovered_worker(payload):
    import experiments.refine_steh as frozen
    tree=ast.parse(Path(frozen.__file__).read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='work')
    class CorrectReference(ast.NodeTransformer):
        count=0
        def visit_For(self,node):
            if ast.unparse(node.iter)=="['nse', 'rmse', 'mae']":
                assert len(node.body)==1 and "scores['full'][name]" in ast.unparse(node.body[0])
                self.count+=1
                replacement=ast.parse('_verify_native(model,test,cfg,meta,target,scores,prior,old,folder)').body[0]
                return ast.copy_location(replacement,node)
            return self.generic_visit(node)
    correct=CorrectReference();function=correct.visit(function);assert correct.count==1
    module=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(module)
    frozen.__dict__['_verify_native']=verify_native
    exec(compile(module,str(Path(__file__)), 'exec'),frozen.__dict__)
    return frozen.work(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/steh_population_20261012'))
    parser.add_argument('--circuits',type=Path,default=Path('outputs/steh_population_20261012_circuits'))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();source=args.source;out=args.circuits
    plan=json.loads((source/'manifest.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    assert sha(ROOT/'experiments/refine_steh.py')==protocol['script_sha256']==plan['refinement_script_sha256']
    assert protocol['source_manifest_sha256']==digest(plan)
    assert all(sha(ROOT/p)==h for p,h in plan['code_sha256'].items())
    missing=[j for j in protocol['jobs'] if not (job_path(out,j)/'result.json').exists()]
    assert len(missing)==3 and not set(j['station'] for j in missing)&set(plan['confirmation_stations'])
    history=out/'recovery_20261014';history.mkdir(exist_ok=True)
    for name in ['completion.json','progress.json','circuit_summary.csv','functional_stability.csv']:
        target=history/('original_'+name)
        if not target.exists():shutil.copyfile(out/name,target)
    failures=[]
    for job in missing:
        folder=job_path(out,job);failed=json.loads((folder/'failure.json').read_text())
        assert 'assert_allclose(scores' in failed['traceback']
        target=history/f"{job['station']}_seed{job['seed']}_q{int(job['with_q'])}_failure.json"
        if not target.exists():shutil.copyfile(folder/'failure.json',target)
        failures.append(job|{'failure_sha256':sha(target)})
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');started=time.time()
    with mp.get_context('spawn').Pool(min(args.workers,len(missing))) as pool:
        recovered=list(pool.imap_unordered(recovered_worker,[(j,protocol,str(source),str(out)) for j in missing]))
    ledger={'status':'PASS' if all(r['status']=='complete' for r in recovered) else 'FAILED',
            'original_failures':failures,'recovered_results':recovered,'seconds':time.time()-started,
            'recovery_script_sha256':sha(Path(__file__)),'frozen_scientific_method_sha256':protocol['script_sha256'],
            'limits':'Numerical-reference repair only; no retraining, core reselection or changed scientific pass criterion.'}
    write_json(history/'recovery.json',ledger)
    if ledger['status']!='PASS':raise RuntimeError('Recovery failure preserved')
    results=[]
    for j in protocol['jobs']:
        r=json.loads((job_path(out,j)/'result.json').read_text())
        assert r['status']=='complete' and r['identity_sha256']==digest({'job':j,'protocol':protocol})
        results.append(r)
    rows=[{k:r[k] for k in ['station','seed','with_q','core_delta_nse','remove_delta_nse','random_win_fraction','distinct_control_count','pass','seconds']}|
          {'full_nse':r['metrics']['full']['nse'],'core_nse':r['metrics']['core']['nse'],**r['validation_search']} for r in results]
    pd.DataFrame(rows).to_csv(out/'circuit_summary.csv',index=False);write_json(out/'progress.json',results)
    old=json.loads((history/'original_completion.json').read_text())
    write_json(out/'completion.json',old|{'status':'PASS','completed':len(results),'planned':len(results),
        'criterion_passes':sum(r['pass'] for r in results),'seconds':old['seconds']+ledger['seconds'],
        'numerical_recovery':'recovery_20261014/recovery.json','recovered_jobs':len(recovered)})
    print(json.dumps({'completion':'PASS','models':len(results),'recovered':len(recovered),
                      'scientific_passes':sum(r['pass'] for r in results),'recovery_seconds':ledger['seconds']}))


if __name__=='__main__':main()

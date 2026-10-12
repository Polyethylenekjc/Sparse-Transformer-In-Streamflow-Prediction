"""One-off export of completed experiment records and model backups.

No polling, schedule, network upload or shutdown is performed by this script.
Numerical records and checkpoint files are exported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def archive(paths,target):
    members=[]
    with tarfile.open(target,'w:gz',compresslevel=3) as tar:
        for path in paths:
            assert path.is_file() and not path.is_symlink(),path
            members.append({'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
            tar.add(path,arcname=path.as_posix(),recursive=False)
    return {'archive':target.name,'bytes':target.stat().st_size,'sha256':sha(target),'members':members}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/root/autodl-tmp/steh-export/final'))
    parser.add_argument('--population',type=Path,default=Path('outputs/steh_population_20261012'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.time();population=args.population;circuits=Path(str(population)+'_circuits')
    for path in [population/'training_completion.json',circuits/'completion.json']:
        completion=json.loads(path.read_text());assert completion['status']=='PASS',path
    plan=json.loads((population/'manifest.json').read_text())
    weights=sorted(population.rglob('trained.pt'))
    assert len(weights)==len(plan['jobs'])
    for job in plan['jobs']:
        f=population/job['station']/f"seed_{job['seed']}"/'L15'/('steh_with_q' if job['with_q'] else 'steh_no_q')
        assert all((f/name).exists() for name in ['trained.pt','summary.json','validation_selection.json'])
    roots=[population,circuits,Path('outputs/revision_3090'),Path('outputs/steh_population_early_review'),
           Path('outputs/steh_tuning_20261012'),Path('outputs/steh_refined_20261012'),
           Path('outputs/steh_control_sensitivity_20261012_v2')]
    caliper_completion=json.loads((roots[-1]/'completion.json').read_text())
    assert caliper_completion['status']=='PASS' and caliper_completion['completed']==len(plan['jobs'])
    roots.extend([Path('outputs/steh_control_sensitivity_20261012'),
                  Path('outputs/steh_control_caliper_smoke'),Path('outputs/steh_control_caliper_v2_smoke')])
    numerical=[]
    for root in roots:
        numerical.extend(p for p in sorted(root.rglob('*')) if p.is_file() and
                         not p.is_symlink() and p.suffix not in ['.pt','.tmp'] and '__pycache__' not in p.parts)
    allowed_telemetry={'steh-gpu-telemetry.csv','steh-cpu-memory-telemetry.csv',
                       'vmap-benchmark.json','vmap-benchmark.log','benchmark_vmap.py',
                       'early-review.log','population.log','control-smoke.log','control-v2-smoke.log',
                       'control-sensitivity.log','refine_steh_v1.py','monitor_steh_system.py'}
    allowed_telemetry.update({'ste-roundoff-diagnostic.json','ste-roundoff-diagnostic-confirmed.json','architecture-counts.json'})
    numerical.extend(p for p in Path('/root/autodl-tmp/steh-performance').iterdir() if p.name in allowed_telemetry)
    # Absolute monitoring paths get a relative reproducible archive name.
    numerical=[p for p in numerical if not p.is_absolute()]
    records=archive(numerical,out/'numerical-records.tar.gz')
    write(out/'numerical-manifest.json',records)
    parts=[];size=10*1024*1024
    with (out/'numerical-records.tar.gz').open('rb') as file:
        index=0
        while block:=file.read(size):
            index+=1;part=out/f'numerical-records.part-{index:03d}';part.write_bytes(block)
            parts.append({'path':part.name,'bytes':len(block),'sha256':sha(part)})
    write(out/'numerical-parts.json',{'format':'concatenate parts in order, verify archive SHA256, extract gzip tar',
                                    'archive':{k:v for k,v in records.items() if k!='members'},'parts':parts})
    for root in [Path('outputs/revision_3090'),Path('outputs/steh_tuning_20261012')]:
        weights.extend(sorted(root.rglob('trained.pt')))
    assert len(weights)==len(set(weights))
    checkpoints=archive(weights,out/'checkpoint-backup.tar.gz')
    write(out/'checkpoint-manifest.json',checkpoints)
    # Telemetry is kept outside the rooted record tar rather than storing unsafe absolute paths.
    monitor_files=sorted(p for p in Path('/root/autodl-tmp/steh-performance').iterdir() if p.name in allowed_telemetry)
    with tarfile.open(out/'performance-records.tar.gz','w:gz') as tar:
        for p in monitor_files:tar.add(p,arcname=p.name,recursive=False)
    performance={'archive':'performance-records.tar.gz',
                 'bytes':(out/'performance-records.tar.gz').stat().st_size,
                 'sha256':sha(out/'performance-records.tar.gz'),
                 'members':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in monitor_files]}
    write(out/'performance-manifest.json',performance)
    summary={'status':'PASS','population_models':len(plan['jobs']),'all_checkpoint_members':len(weights),
             'numerical_files':len(numerical),'numerical_parts':len(parts),
             'numerical_archive_sha256':records['sha256'],'checkpoint_archive_sha256':checkpoints['sha256'],
             'performance_archive_sha256':performance['sha256'],
             'seconds':time.time()-start,
             'limits':'Export integrity only. Public model-weight upload requires its separate user authorization.'}
    write(out/'export-completion.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()

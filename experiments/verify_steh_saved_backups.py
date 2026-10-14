"""One-off local backup verification, without network or model deserialization."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile
import time


def hash_stream(stream):
    digest=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--extract-numerical',type=Path)
    args=parser.parse_args();root=args.directory;start=time.time();audits=[]
    completion=json.loads((root/'export-completion.json').read_text(encoding='utf-8'))
    assert completion['status']=='PASS' and completion['population_models']==1090
    for kind in ['numerical','checkpoint','performance']:
        manifest=json.loads((root/(kind+'-manifest.json')).read_text(encoding='utf-8'))
        archive=root/manifest['archive'];assert archive.stat().st_size==manifest['bytes']
        with archive.open('rb') as file:assert hash_stream(file)==manifest['sha256']
        expected={r['path']:r for r in manifest['members']}
        assert len(expected)==len(manifest['members']);seen=set();total_bytes=0
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                path=PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                assert member.name in expected and member.name not in seen
                record=expected[member.name];assert member.size==record['bytes']
                file=tar.extractfile(member);assert file is not None
                with file:
                    if kind=='numerical' and args.extract_numerical:
                        destination=args.extract_numerical.resolve()/Path(*path.parts)
                        assert destination.resolve().is_relative_to(args.extract_numerical.resolve())
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        digest=hashlib.sha256()
                        with destination.open('wb') as target:
                            for block in iter(lambda:file.read(1024*1024),b''):
                                digest.update(block);target.write(block)
                        assert digest.hexdigest()==record['sha256'],member.name
                    else:assert hash_stream(file)==record['sha256'],member.name
                seen.add(member.name);total_bytes+=member.size
        assert seen==set(expected)
        if kind=='checkpoint':
            population=[p for p in seen if p.startswith('outputs/steh_population_20261012/')]
            assert len(population)==1090 and all(p.endswith('/trained.pt') for p in population)
        audits.append({'kind':kind,'archive':archive.name,'sha256':manifest['sha256'],
                       'members_verified':len(seen),'uncompressed_bytes':total_bytes})
        print(json.dumps(audits[-1]),flush=True)
    report={'status':'PASS','population_models':1090,'archives':audits,
            'seconds':time.time()-start,'scope':'Compressed and every member SHA256; no model deserialization or public weight upload.'}
    (root/'local-backup-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

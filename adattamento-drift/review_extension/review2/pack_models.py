"""Package immutable shuffled initial states in hash-checked 8 MiB gzip parts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(data):return hashlib.sha256(data).hexdigest()


def pack(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    raw=(source/'model.pkl').read_bytes()
    zipped=gzip.compress(raw,compresslevel=6,mtime=0)
    meta={'format':'gzip parts; concatenate in listed order, then decompress',
          'uncompressed_sha256':sha(raw),'uncompressed_bytes':len(raw),
          'compressed_sha256':sha(zipped),'parts':[]}
    size=8*1024*1024
    for i,start in enumerate(range(0,len(zipped),size)):
        name=f'model.pkl.gz.part{i:03d}';content=zipped[start:start+size]
        (destination/name).write_bytes(content)
        meta['parts'].append({'name':name,'bytes':len(content),'sha256':sha(content)})
    shutil.copy2(source/'ready.json',destination/'ready.json')
    (destination/'state_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'models')
    a=p.parse_args()
    for ds in ('ton','unsw'):
        for seed in range(42,47):
            source=a.cache_root/ds/f'arf_shuffled_{seed}'
            if (source/'ready.json').exists():
                pack(source,a.out/ds/source.name);print(ds,seed,flush=True)

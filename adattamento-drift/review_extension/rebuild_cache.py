"""Rebuild immutable inference caches from the supplied fitted initial states."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import experiment as e


def state_bytes(folder):
    """Read plain historical states or hash-checked chunked gzip states."""
    if (folder/'model.pkl').exists():return (folder/'model.pkl').read_bytes()
    if (folder/'model.pkl.gz').exists():return gzip.decompress((folder/'model.pkl.gz').read_bytes())
    meta=json.loads((folder/'state_manifest.json').read_text())
    chunks=[]
    for part in meta['parts']:
        content=(folder/part['name']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=part['sha256']:raise ValueError('State part hash mismatch: '+part['name'])
        chunks.append(content)
    compressed=b''.join(chunks)
    if hashlib.sha256(compressed).hexdigest()!=meta['compressed_sha256']:raise ValueError('Compressed state hash mismatch')
    raw=gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest()!=meta['uncompressed_sha256']:raise ValueError('Model state hash mismatch')
    return raw


def rebuild(data,models,cache,provenance):
    with np.load(data/'D.npz') as f:
        arrays={k:f[k] for k in f.files}
    expected=json.loads((provenance/'D.json').read_text())['array_sha256']
    for key,value in arrays.items():
        digest=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if digest!=expected[key]:raise ValueError('D array mismatch: '+key)
    X=arrays['X']
    for folder in sorted(models.iterdir()):
        source=folder/'model.pkl'
        compressed=folder/'model.pkl.gz'
        if not source.is_file() and not compressed.is_file() and not (folder/'state_manifest.json').is_file():continue
        destination=cache/folder.name;destination.mkdir(parents=True,exist_ok=True)
        # Only load the trusted fitted states supplied with this versioned package.
        model_bytes=state_bytes(folder)
        state=pickle.loads(model_bytes)
        model,transform=state['model'],state['transform']
        if folder.name.startswith('arf_'):
            score=e.predict_river(model,transform(X))
        else:
            R=np.lib.format.open_memmap(destination/'representation.npy',mode='w+',
                                       dtype=np.float64,shape=(len(X),8))
            for a in range(0,len(X),200000):
                R[a:a+200000]=model.rappresentazione(transform(X[a:a+200000]))
            R.flush();p=model.parametri();score=R@p[:-1]+p[-1];del R
        np.save(destination/'frozen_score.npy',score)
        (destination/'model.pkl').write_bytes(model_bytes)
        shutil.copy2(folder/'ready.json',destination/'ready.json')
        print('restored',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--provenance',type=Path,required=True);a=p.parse_args()
    rebuild(a.data,a.models,a.cache,a.provenance)

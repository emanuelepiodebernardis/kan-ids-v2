"""Verify reconstructed data against original D block identities and parser A."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ton_temporal'))
import replay
from replay_pesi_piu_soglia import sha_float64
from prepare_data import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--original-parser-a',type=Path,required=True)
    p.add_argument('--historical-csv',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    report=json.loads((a.original/'D_seme42.json').read_text())
    with np.load(a.data/'ton/D.npz') as f:
        rid,y=f['row_id'],f['y']
    checks=[]
    for b in report['per_blocco']:
        start=b['blocco']*10000;end=start+b['righe']
        assert replay.impronta_row_id(rid[start:end])==b['row_id_blocco_sha256']
        assert sha_float64(y[start:end])==b['etichette_blocco_sha256']
        checks.extend([f"block {b['blocco']} row identities",f"block {b['blocco']} labels"])
    with np.load(a.original_parser_a) as old,np.load(a.data/'ton/A.npz') as new:
        assert set(old.files)==set(new.files)
        for key in old.files:
            assert old[key].dtype==new[key].dtype
            assert np.array_equal(old[key],new[key]),key
    assert sha(a.original_parser_a)==sha(a.data/'ton/A.npz')
    historical=sha(a.historical_csv)
    assert historical=='26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974'
    arrays={}
    for dataset in ('ton','unsw'):
        for path in sorted((a.data/dataset).glob('[ABCD].npz')):
            meta=json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as f:
                for key in f.files:
                    actual=hashlib.sha256(np.ascontiguousarray(f[key]).tobytes()).hexdigest()
                    assert actual==meta['array_sha256'][key]
            assert sha(path)==meta['sha256']
            arrays[dataset+'/'+path.name]=meta['array_sha256']
    result={'original_D_digest_checks':len(checks),'all_passed':True,
            'original_parser_A_arrays_identical':True,'original_parser_A_npz_identical':True,
            'A_npz_sha256':sha(a.original_parser_a),'historical_csv_sha256':historical,
            'historical_csv_url':'https://huggingface.co/datasets/codymlewis/TON_IoT_network/resolve/main/train_test_network.csv',
            'old_windows_npz_byte_identity_established':False,'arrays':arrays}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='arrays'}))


if __name__=='__main__':main()

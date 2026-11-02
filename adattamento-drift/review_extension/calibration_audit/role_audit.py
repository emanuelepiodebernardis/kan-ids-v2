"""Occurrence-weighted normal-feature role allocation, without model fitting."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import experiment as e

SALTS={'default':'ADAPT04-C-seed42-v1','salt1':'review2-salt-1','salt2':'review2-salt-2'}

def allocation(X, salt):
    unique, counts=np.unique(np.asarray(X),axis=0,return_counts=True)
    previous=e.cr.SALT
    try:
        e.cr.SALT=salt.encode()
        roles,digests=e.cr.assegna_ruoli(unique)
    finally:
        e.cr.SALT=previous
    order=np.argsort(-counts,kind='stable')
    cal=roles=='calibration'
    summary=dict(normal_rows=int(counts.sum()),unique_normal_vectors=len(unique),
                 cal_unique_fraction=float(cal.mean()),cal_occurrence_fraction=float(counts[cal].sum()/counts.sum()),
                 repeated_row_fraction=float(counts[counts>1].sum()/counts.sum()),
                 largest_vector_fraction=float(counts.max()/counts.sum()),
                 top10_fraction=float(counts[order[:10]].sum()/counts.sum()),
                 top10_fit_fraction_of_all_normal=float(counts[order[:10]][~cal[order[:10]]].sum()/counts.sum()))
    top=[dict(rank=j+1,count=int(counts[i]),role=str(roles[i]),digest=digests[i]) for j,i in enumerate(order[:20])]
    return summary,top

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[];top=[];inputs={}
    for ds in ['ton','unsw']:
        path=a.data_root/ds/'D.npz';inputs[ds]=e.sha(path)
        with np.load(path) as f:X=f['X'][f['y']==0]
        for label,salt in SALTS.items():
            summary,details=allocation(X,salt)
            rows.append(dict(dataset=ds,salt=label,**summary))
            top.extend(dict(dataset=ds,salt=label,**r) for r in details)
    pd.DataFrame(rows).to_csv(a.out/'normal_role_coverage.csv',index=False)
    pd.DataFrame(top).to_csv(a.out/'normal_top_vectors.csv',index=False)
    (a.out/'role_audit_provenance.json').write_text(json.dumps({'input_sha256':inputs,'source_sha256':e.sha(Path(__file__)),'salts':SALTS},indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

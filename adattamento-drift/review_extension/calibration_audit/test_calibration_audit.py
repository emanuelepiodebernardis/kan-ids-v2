import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from calibration_audit import role_audit as a
from calibration_audit import runner as r

def test_repeated_vector_has_one_role_and_occurrence_weighted_mass():
    X=np.vstack([np.zeros((999,8)),np.ones((1,8))])
    summary,top=a.allocation(X,a.SALTS['default'])
    assert summary['unique_normal_vectors']==2
    assert summary['largest_vector_fraction']==.999
    expected=sum(t['count'] for t in top if t['role']=='calibration')/1000
    assert summary['cal_occurrence_fraction']==expected

def test_role_audit_restores_global_salt():
    before=a.e.cr.SALT
    a.allocation(np.zeros((3,8)),a.SALTS['salt2'])
    assert a.e.cr.SALT==before

def test_all_label_audit_has_identical_retention_to_real_controller():
    real=r.e.replay.MemoriaReplay(256,'per_classe');audit=r.e.cr.BufferRuolo(256)
    rng=np.random.default_rng(5)
    for k in range(30):
        X=rng.normal(size=(80,8));y=rng.integers(0,2,80) if k%3 else np.ones(80,dtype=int)
        real.aggiungi(X,y,k);audit.aggiungi(X,y,k,[f'{80*k+i:064x}' for i in range(80)])
        for one,two in zip(real.array_ammissibile(k+1,10),audit.array_ammissibile(k+1,10)):
            np.testing.assert_array_equal(one,two)

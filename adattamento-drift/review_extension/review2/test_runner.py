import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runner as r


@pytest.mark.parametrize('alpha',[0,.05,.1,.2,.75])
def test_empirical_fpr_cap_handles_ties(alpha):
    scores=np.array([0,0,1,1,1,2,3,3,3,4,50.])
    y=np.array([0]*10+[1])
    t=r.fpr_threshold(scores,y,alpha)
    assert np.mean(scores[y==0]>t)<=alpha+1e-12
    lower=np.unique(scores[y==0]);lower=lower[lower<t]
    if len(lower):assert np.mean(scores[y==0]>lower[-1])>alpha


def test_fixed_target_uses_only_negatives_and_is_monotone():
    s=np.arange(30,dtype=float);y=np.r_[np.zeros(20),np.ones(10)]
    t=[r.fpr_threshold(s,y,a) for a in r.ALPHAS]
    assert t==sorted(t,reverse=True)
    s[y==1]*=-100
    assert t==[r.fpr_threshold(s,y,a) for a in r.ALPHAS]
    with pytest.raises(ValueError):r.fpr_threshold(s,np.ones(30),.1)
    assert r.ap(np.ones(5),np.arange(5)) is None


@pytest.mark.parametrize('family',['spline','lr','arf'])
def test_default_reproduces_original_and_delay(tmp_path,family):
    data=tmp_path/'synthetic';data.mkdir();rng=np.random.default_rng(813)
    for name,n in [('A',500),('B',200),('D',400)]:
        X=rng.uniform(.1,10,(n,8)).astype('float32');y=(X[:,0]>5).astype('uint8')
        np.savez(data/(name+'.npz'),X=X,y=y,row_id=np.arange(n),frt=np.arange(n,dtype=float),
                 tipo=y.astype('int16'),tipi=np.array(['normal','attack']))
    np.save(data/'unseen.npy',np.ones(400,dtype=bool))
    cache=tmp_path/'cache';r.e.prepare(data,cache/data.name,family,42)
    old=r.e.run(data,cache/data.name,tmp_path/'old.json.gz',family,42,budget=.5,block_size=100)
    new=r.run(data,cache,tmp_path/'new.json.gz',family,42,budget=.5,block_size=100,audit=False)
    common=set(old['summary']['all'])&set(new['summary']['all'])
    for a,b in zip(old['per_block'],new['per_block']):
        assert a['query_sha256']==b['query_sha256']
        for p in common:
            assert a['thresholds'][p]==b['thresholds'][p]
            assert a['cm']['all'][p]==b['cm']['all'][p]
    with np.load(data/'D.npz') as f:d={k:f[k] for k in f.files}
    d['y'][:100]=1-d['y'][:100];np.savez(data/'D.npz',**d)
    changed=r.run(data,cache,tmp_path/'changed.json.gz',family,42,budget=.5,block_size=100,audit=False)
    for k in (0,1):assert changed['per_block'][k]['thresholds']==new['per_block'][k]['thresholds']
    if family=='arf':
        fit=r.run(data,cache,tmp_path/'fit.json.gz',family,42,'fit_only',budget=.5,block_size=100,audit=False)
        expected=0
        for k in range(3):
            ids=k*100+r.e.query_indices(100,42,k,.5)
            roles,_=r.e.cr.assegna_ruoli(d['X'][ids]);expected+=int((roles=='fit').sum())
        assert fit['updates']['learning_labels']==expected
        assert 'arf_cal256' not in fit['summary']['all']


def test_historical_runner_unchanged():
    assert r.e.sha(r.HERE.parent/'experiment.py')=='0417f5772371a507dde81263227c6c22be432f454a1bff06bf4c391f8ab9f73d'


def test_ap_validity_counts_uint8_labels_without_overflow():
    y=np.r_[np.ones(256,dtype=np.uint8),np.zeros(256,dtype=np.uint8)]
    assert r.ap(y,y)==1.0
    assert r.ap(np.ones(10000,dtype=np.uint8),np.arange(10000)) is None


def test_matched_error_interpolation_refuses_extrapolation_and_handles_ties():
    import importlib.util
    spec=importlib.util.spec_from_file_location('review2_analysis',r.HERE/'analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    interpolate=module.interpolate
    points=[{'fpr':.1,'recall':.7},{'fpr':.1,'recall':.8},{'fpr':.2,'recall':.9}]
    assert interpolate(points,.15)==pytest.approx(.85)
    assert interpolate(points,.05) is None
    assert interpolate(points,.25) is None
    assert interpolate(points,.85,'recall')==pytest.approx(.15)


def test_packaged_state_round_trip_and_corruption_detection(tmp_path):
    import importlib.util
    from rebuild_cache import state_bytes
    spec=importlib.util.spec_from_file_location('review2_pack',r.HERE/'pack_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';source.mkdir()
    raw=b'fixed model payload'*200
    (source/'model.pkl').write_bytes(raw);(source/'ready.json').write_text('{}')
    target=tmp_path/'packed';module.pack(source,target)
    assert state_bytes(target)==raw
    part=target/'model.pkl.gz.part000';part.write_bytes(part.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='part hash mismatch'):state_bytes(target)

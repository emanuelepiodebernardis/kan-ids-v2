"""Tests of scientific invariants, rather than snapshots of implementation."""
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
import experiment as e


def test_query_pairing_and_cross_seed_independence():
    a=e.query_indices(10000,42,1)
    assert np.array_equal(a,e.query_indices(10000,42,1))
    assert not np.array_equal(a,e.query_indices(10000,43,0))
    assert np.array_equal(e.query_indices(10000,42,1,legacy=True),
                          e.query_indices(10000,43,0,legacy=True))
    assert len(e.query_indices(9973,42,354))==99


def test_exact_threshold_and_cost_monotonicity():
    rng=np.random.default_rng(6);s=rng.normal(size=400);y=rng.integers(2,size=400)
    ts=e.thresholds(s,y)
    assert ts[1.]==e.sb.scegli_soglia(s,y)[0]
    fp=[e.cm(y,s>ts[v])[1] for v in e.LAMBDAS]
    assert all(a>=b for a,b in zip(fp,fp[1:]))


def test_intercept_cannot_change_threshold_decisions_or_auc():
    rng=np.random.default_rng(9);s=rng.normal(size=500);y=rng.integers(2,size=500)
    t=e.thresholds(s,y)[1.]
    for offset in (-8.,.2,13.):
        shifted=e.thresholds(s+offset,y)[1.]
        assert np.array_equal(s>t,s+offset>shifted)
        assert e.auc(y,s)==e.auc(y,s+offset)


def test_prior_penalty_is_not_warm_start_at_zero():
    rng=np.random.default_rng(10);R=rng.normal(size=(20,8));y=rng.integers(2,size=20)
    prior=np.r_[np.full(8,3.),0.]
    p=e.prior_refit(R,y,prior)
    zero=e.prior_refit(R,y,np.zeros(9))
    assert np.linalg.norm(p[:-1]-prior[:-1])<np.linalg.norm(zero[:-1]-prior[:-1])
    grad=R.T@(e.expit(R@p[:-1]+p[-1])-y)+(p[:-1]-prior[:-1])
    assert np.linalg.norm(grad)<1e-3


def test_fixed_model_duplicate_prediction_is_exact():
    from river import forest
    rng=np.random.default_rng(2);X=rng.normal(size=(30,8));y=rng.integers(2,size=30)
    model=forest.ARFClassifier(n_models=2,seed=4)
    for x,yy in zip(X,y):model.learn_one(dict(enumerate(x)),int(yy))
    X=np.r_[X,X]
    direct_model=e.copy.deepcopy(model)
    before_metrics=[m.get() for m in model._metrics]
    batched=e.predict_river(model,X)
    direct=np.array([direct_model.predict_proba_one(dict(enumerate(x))).get(1,0.) for x in X])
    assert np.array_equal(batched,direct)
    # River's metric reads may insert zero-valued defaultdict keys; they must
    # not change weights or subsequent learning/prediction behavior.
    assert [m.get() for m in model._metrics]==before_metrics
    for x,yy in zip(X[:30],y):
        model.learn_one(dict(enumerate(x)),int(yy))
        direct_model.learn_one(dict(enumerate(x)),int(yy))
    assert np.array_equal(e.predict_river(model,X),e.predict_river(direct_model,X))


@pytest.mark.parametrize('family',['spline','lr','arf'])
def test_future_feedback_does_not_change_emitted_predictions(tmp_path,family):
    rng=np.random.default_rng(4);data=tmp_path/'synthetic';data.mkdir()
    for name,n in [('A',500),('B',200),('D',400)]:
        X=rng.uniform(0,10,size=(n,8)).astype(np.float32);y=(X[:,0]>5).astype(np.uint8)
        np.savez(data/(name+'.npz'),X=X,y=y,row_id=np.arange(n),frt=np.arange(n,dtype=float),
                 tipo=y.astype(np.int16),tipi=np.array(['normal','attack']))
    np.save(data/'unseen.npy',np.ones(400,dtype=bool));cache=tmp_path/'cache'
    e.prepare(data,cache,family,42)
    first=e.run(data,cache,tmp_path/'r1.json.gz',family,42,delay=1,budget=.5,block_size=100)
    with np.load(data/'D.npz') as f:arrays={k:f[k] for k in f.files}
    arrays['y'][:100]=1-arrays['y'][:100]
    np.savez(data/'D.npz',**arrays)
    second=e.run(data,cache,tmp_path/'r2.json.gz',family,42,delay=1,budget=.5,block_size=100)
    # Block 0's changed labels arrive only after predicting block 1. Its
    # evaluation counts may change, but block 1 outcomes and thresholds cannot.
    assert first['per_block'][1]['cm']==second['per_block'][1]['cm']
    assert first['per_block'][1]['thresholds']==second['per_block'][1]['thresholds']
    assert first['per_block'][1]['first_effect_block']==2
    assert first['query_sequence_sha256']==second['query_sequence_sha256']
    for report in (first,second):
        assert report['queries']==200 and report['pending_labels']==50
        for policy in report['summary']['all']:
            actual=np.sum([r['cm']['all'][policy] for r in report['per_block']],axis=0)
            summary=report['summary']['all'][policy]
            assert actual.tolist()==[summary[k] for k in ('tn','fp','fn','tp')]

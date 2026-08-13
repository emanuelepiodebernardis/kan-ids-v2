"""Test che il protocollo sia leakage-free.

Non sono test cosmetici: il terzo (`test_permuted_labels...`) riproduce
esattamente il difetto che c'era nella pipeline precedente — ranking per
mutual information calcolato prima dello split — e mostra che produce
performance sopra il caso su etichette permutate, cioe' su dati in cui
per costruzione non c'e' nulla da imparare.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kanids import LeakageFreePreprocessor, cv_splits, outer_split  # noqa: E402
from kanids.datasets import encode_targets, make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return make_synthetic(n=6000, seed=0)


# ── 1. transform() non dipende dai dati che riceve ────────────────────
def test_transform_is_row_independent(df):
    """Trasformare un sottoinsieme == trasformare tutto e poi affettare.

    Se una statistica venisse ricalcolata dentro transform (scaler
    ri-fittato, vocabolario ricostruito), questa uguaglianza salterebbe.
    """
    yb, ym, _ = encode_targets(df)
    tr, te = outer_split(ym, seed=42)
    prep = LeakageFreePreprocessor(k_numeric=10).fit(df.iloc[tr], ym[tr])

    Xa, Ca = prep.transform(df.iloc[te])
    sub = te[:100]
    Xb, Cb = prep.transform(df.iloc[sub])

    np.testing.assert_allclose(Xa[:100], Xb, rtol=0, atol=0)
    np.testing.assert_array_equal(Ca[:100], Cb)


# ── 2. il fit non vede il test ────────────────────────────────────────
def test_fit_depends_only_on_training_rows(df):
    """Stesso training + test diversi -> preprocessor identico."""
    yb, ym, _ = encode_targets(df)
    tr, te = outer_split(ym, seed=42)

    p1 = LeakageFreePreprocessor(k_numeric=10).fit(df.iloc[tr], ym[tr])
    # ricostruisco un dataframe in cui il test e' stato alterato in modo grossolano
    df2 = df.copy()
    df2.loc[df2.index[te], "src_bytes"] = 10 ** 9
    df2.loc[df2.index[te], "proto"] = "sctp"          # categoria nuova
    p2 = LeakageFreePreprocessor(k_numeric=10).fit(df2.iloc[tr], ym[tr])

    assert p1.numeric_features_ == p2.numeric_features_
    assert p1.cardinalities_ == p2.cardinalities_
    assert p1.vocabularies_ == p2.vocabularies_


# ── 3. permutation test: il difetto vecchio contro quello nuovo ───────
def _cv_score(select_on_full: bool, seed: int = 0) -> float:
    """AUC media in CV su dati puro rumore con etichette casuali.

    select_on_full=True replica la selezione fatta prima dello split.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.RandomState(seed)
    n, p, k = 400, 80, 5
    X = rng.randn(n, p)                    # nessun segnale
    y = rng.randint(0, 2, n)               # etichette casuali

    if select_on_full:
        mi = mutual_info_classif(X, y, random_state=seed)
        sel_global = np.argsort(mi)[::-1][:k]

    aucs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sel = sel_global if select_on_full else \
            np.argsort(mutual_info_classif(X[tr], y[tr], random_state=seed))[::-1][:k]
        m = LogisticRegression(max_iter=500).fit(X[tr][:, sel], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X[va][:, sel])[:, 1]))
    return float(np.mean(aucs))


def test_permuted_labels_give_chance_performance():
    leaky = np.mean([_cv_score(True, s) for s in range(5)])
    clean = np.mean([_cv_score(False, s) for s in range(5)])

    # la selezione pre-split gonfia l'AUC su dati senza segnale
    assert leaky > clean, f"leaky={leaky:.3f} clean={clean:.3f}"
    # la selezione per-fold resta al caso
    assert abs(clean - 0.5) < 0.06, f"clean={clean:.3f} non e' al caso"


# ── 4. categorie mai viste ────────────────────────────────────────────
def test_unseen_category_maps_to_unk(df):
    yb, ym, _ = encode_targets(df)
    tr, te = outer_split(ym, seed=42)
    prep = LeakageFreePreprocessor().fit(df.iloc[tr], ym[tr])

    alien = df.iloc[te].copy()
    alien["proto"] = "quic"                       # mai vista in training
    _, C = prep.transform(alien)
    j = prep.categorical_.index("proto")
    assert (C[:, j] == 0).all(), "categoria nuova non mappata su UNK"
    assert prep.unseen_rate(alien)["proto"] == 1.0
    # l'indice UNK e' dentro la tabella: nessun accesso fuori range su MCU
    assert C.max() < max(prep.cardinalities_)


def test_cardinalities_cover_all_indices(df):
    yb, ym, _ = encode_targets(df)
    tr, te = outer_split(ym, seed=42)
    prep = LeakageFreePreprocessor().fit(df.iloc[tr], ym[tr])
    _, C = prep.transform(df)
    for j, card in enumerate(prep.cardinalities_):
        assert C[:, j].max() < card
        assert C[:, j].min() >= 0


# ── 5. determinismo e stratificazione ─────────────────────────────────
def test_splits_are_deterministic(df):
    yb, ym, _ = encode_targets(df)
    a = [(s["seed"], s["fold"], s["val_idx"].tolist()) for s in cv_splits(ym)]
    b = [(s["seed"], s["fold"], s["val_idx"].tolist()) for s in cv_splits(ym)]
    assert a == b
    assert len(a) == 15


def test_folds_partition_the_data(df):
    yb, ym, _ = encode_targets(df)
    per_seed = {}
    for s in cv_splits(ym):
        per_seed.setdefault(s["seed"], []).append(s["val_idx"])
        assert len(np.intersect1d(s["train_idx"], s["val_idx"])) == 0
    for seed, vals in per_seed.items():
        allv = np.sort(np.concatenate(vals))
        np.testing.assert_array_equal(allv, np.arange(len(ym)))


def test_rare_class_present_in_every_fold(df):
    yb, ym, classes = encode_targets(df)
    rare = classes.index("mitm")
    for s in cv_splits(ym):
        assert (ym[s["train_idx"]] == rare).sum() > 0
        assert (ym[s["val_idx"]] == rare).sum() > 0


def test_binary_and_multiclass_share_the_same_folds(df):
    """Stratificando su ym, i fold del binario sono quelli del multiclass."""
    yb, ym, _ = encode_targets(df)
    folds = list(cv_splits(ym))
    assert all((yb[f["val_idx"]] == (ym[f["val_idx"]] != 0).astype(int)).all()
               for f in folds)


# ── cross-domain: il target non deve entrare in NIENTE del training ───
def _harmonized_synth(n, seed, shift=0.0):
    """Frame sintetico nello schema armonizzato, per i test cross-domain."""
    from kanids.harmonized import HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC
    rng = np.random.RandomState(seed)
    d = {c: np.abs(rng.randn(n) * 10 + 20 + shift) for c in HARMONIZED_NUMERIC}
    df = pd.DataFrame(d)
    df["proto_h"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["state_h"] = rng.choice(["established", "closed", "incomplete", "reset"], n)
    for c in HARMONIZED_CATEGORICAL:
        if c not in df:
            df[c] = "other"
    df["label"] = rng.randint(0, 2, n)
    return df


def _fit_on(source, target, seed=42):
    sys.path.insert(0, str(REPO / "scripts"))
    from cross_domain import fit_eval
    info = {"exp": "a->b", "fold": 0, "variant": "cat", "ratio": 50.0,
            "train_pos_rate": 0.0, "test_pos_rate": 0.0}
    _, prep = fit_eval(source, target, seed, 5, True, "DecisionTree", False, info)
    return prep


def test_crossdomain_target_does_not_influence_training():
    """Cambiare completamente il target non deve cambiare nulla di appreso.

    E' il vincolo del punto 3: nel cross-domain il dataset target non puo'
    entrare in feature selection, normalizzazione, vocabolari o tuning.
    Qui lo si verifica per costruzione: si fitta due volte sullo stesso
    source con due target radicalmente diversi e si confronta tutto cio'
    che il preprocessor ha imparato, piu' la trasformazione di un terzo
    insieme di riferimento fissato.
    """
    source = _harmonized_synth(4000, seed=1)
    target_a = _harmonized_synth(3000, seed=2)
    target_b = _harmonized_synth(3000, seed=3, shift=5000.0)   # scala del tutto diversa
    target_b["proto_h"] = "sctp"                               # categoria mai vista
    target_b["state_h"] = "bizarre"

    p1 = _fit_on(source, target_a)
    p2 = _fit_on(source, target_b)

    assert p1.numeric_features_ == p2.numeric_features_, "la selezione feature e' cambiata"
    assert p1.vocabularies_ == p2.vocabularies_, "i vocabolari categorici sono cambiati"
    assert p1.cardinalities_ == p2.cardinalities_, "le cardinalita' sono cambiate"

    # i quantili appresi devono essere identici bit per bit
    np.testing.assert_array_equal(p1.quantile_.quantiles_, p2.quantile_.quantiles_)

    # e la trasformazione di un insieme di riferimento fissato non deve muoversi
    probe = _harmonized_synth(500, seed=99)
    X1, C1 = p1.transform(probe)
    X2, C2 = p2.transform(probe)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(C1, C2)


def test_crossdomain_unseen_target_categories_go_to_unk():
    """Le categorie del target assenti dal source finiscono in UNK, in range."""
    source = _harmonized_synth(2000, seed=1)
    target = _harmonized_synth(1000, seed=2)
    target["proto_h"] = "sctp"
    p = _fit_on(source, target)
    _, C = p.transform(target)
    j = p.categorical_.index("proto_h")
    assert (C[:, j] == 0).all(), "categoria nuova non mappata su UNK"
    assert p.unseen_rate(target)["proto_h"] == 1.0
    assert C.max() < max(p.cardinalities_), "indice fuori dalla tabella"

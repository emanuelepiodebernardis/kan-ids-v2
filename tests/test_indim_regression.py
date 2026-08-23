"""Regressione: `in_dim` delle KAN deve seguire il numero REALE di feature
numeriche selezionate dal preprocessor, non la costante K_NUMERIC.

Il difetto: `build_models(..., in_dim=K_NUMERIC)` fissava sempre 10, che
coincide con `len(prep.numeric_features_)` solo perche' lo spazio armonizzato
ricco ha 13 candidate (min(10, 13) = 10). Con meno di 10 candidate — lo
spazio ridotto usato per CIC-IoT-2023, 6 numeriche — il preprocessor seleziona
correttamente 6 feature ma le KAN venivano costruite per leggerne 10,
indicizzando fuori dall'array (`IndexError` in `kanids/models.py`, dentro
`chebyshev_basis`). Emerso in `scripts/joint_training.py --spazio ridotto`,
corretto li' e per difesa anche in `scripts/cross_domain.py` (radice e
adattamento-drift), che oggi gira solo nello spazio ricco e quindi non ha mai
manifestato il difetto — ma lo conteneva.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kanids import LeakageFreePreprocessor  # noqa: E402
from kanids.harmonized import HARMONIZED_CATEGORICAL  # noqa: E402


def _synth_poche_candidate(n, seed, n_numeric=4):
    """Frame armonizzato con MENO candidate numeriche di K_NUMERIC (10):
    riproduce esattamente la geometria dello spazio ridotto senza dipendere
    da RIDOTTO_NUMERIC, cosi' il test resta valido anche se quella lista
    cambia."""
    rng = np.random.RandomState(seed)
    cols = [f"num_{i}" for i in range(n_numeric)]
    df = pd.DataFrame({c: np.abs(rng.randn(n) * 10 + 20) for c in cols})
    df["proto_h"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["state_h"] = rng.choice(["established", "closed", "incomplete"], n)
    df["label"] = rng.randint(0, 2, n)
    return df, cols


def _fit_prep(df, cols, seed):
    return LeakageFreePreprocessor(
        k_numeric=10,  # K_NUMERIC: piu' delle candidate disponibili
        random_state=seed,
        numeric_candidates=cols,
        categorical=HARMONIZED_CATEGORICAL,
        skewed=set(),
        selection_target="binary",
    ).fit(df, df["label"].to_numpy())


def test_preprocessor_selects_fewer_than_k_numeric_when_candidates_are_scarce():
    """Precondizione del test dei due script: con 4 candidate e k_numeric=10,
    il preprocessor deve selezionarne 4, non 10 — altrimenti il resto del
    test non esercita la geometria del difetto."""
    df, cols = _synth_poche_candidate(2000, seed=1, n_numeric=4)
    prep = _fit_prep(df, cols, seed=42)
    assert len(prep.numeric_features_) == 4


def test_cross_domain_build_models_uses_real_feature_count():
    from cross_domain import build_models as build_models_root

    df, cols = _synth_poche_candidate(2000, seed=2, n_numeric=4)
    prep = _fit_prep(df, cols, seed=42)
    Xtr, Ctr = prep.transform(df)
    ytr = df["label"].to_numpy()

    for name, model in build_models_root(prep.cardinalities_, seed=42,
                                         in_dim=len(prep.numeric_features_)).items():
        model.fit(Xtr, Ctr, ytr)
        pred = model.predict(Xtr, Ctr)
        assert len(pred) == len(ytr), name


def test_joint_training_build_models_uses_real_feature_count():
    from joint_training import build_models as build_models_joint

    df, cols = _synth_poche_candidate(2000, seed=3, n_numeric=4)
    prep = _fit_prep(df, cols, seed=42)
    Xtr, Ctr = prep.transform(df)
    ytr = df["label"].to_numpy()

    for name, model in build_models_joint(prep.cardinalities_, seed=42,
                                          in_dim=len(prep.numeric_features_)).items():
        model.fit(Xtr, Ctr, ytr)
        pred = model.predict(Xtr, Ctr)
        assert len(pred) == len(ytr), name


def test_build_models_default_in_dim_would_break_with_scarce_candidates():
    """Controllo negativo: se qualcuno reintroduce `in_dim=K_NUMERIC` fisso
    (il default dei due `build_models`, lasciato per compatibilita' con le
    chiamate di enumerazione nomi che non fittano nulla), fittare con meno
    candidate deve fallire. Se questo test smette di fallire, il difetto e'
    rientrato."""
    from cross_domain import build_models as build_models_root

    df, cols = _synth_poche_candidate(500, seed=4, n_numeric=4)
    prep = _fit_prep(df, cols, seed=42)
    Xtr, Ctr = prep.transform(df)
    ytr = df["label"].to_numpy()

    model = build_models_root(prep.cardinalities_, seed=42,
                              wanted="KAN(cat,1L)", multilayer=False)["KAN(cat,1L)"]
    try:
        model.fit(Xtr, Ctr, ytr)
    except IndexError:
        return
    raise AssertionError(
        "in_dim=K_NUMERIC fisso non e' piu' fallito con candidate scarse: "
        "verificare che nessuno abbia tolto in_dim=len(prep.numeric_features_) "
        "dal punto di chiamata reale in fit_eval()"
    )

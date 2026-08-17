"""Preparazione dati condivisa dagli script storici del repository.

Perche' esiste
--------------
Sei script (`kan14_binary.py`, `kan_categorical_mc.py`, `kan14_ml_binary.py`,
`kan14_cv_driver.py`, ...) avevano ognuno la propria copia della stessa
preparazione dati, e ogni copia conteneva lo stesso difetto: il ranking per
mutual information era calcolato su un campione dell'INTERO dataset prima
dello split, quindi la scelta delle 10 feature numeriche vedeva le etichette
di test. Anche i vocabolari categorici erano costruiti su train+test.

Qui c'e' una sola preparazione, leakage-free, e gli script la importano.
Cosi' il difetto non puo' ripresentarsi in una delle copie.

Nota: sul dataset TON_IoT la correzione non cambia le feature scelte
(coincidono in 15 fold su 15, vedi results/leakage_audit_stability.csv) e
quindi non cambia i numeri pubblicati. Il protocollo va comunque corretto:
la sua validita' non puo' dipendere da quanto grande sia risultato l'effetto.
"""
from __future__ import annotations

import numpy as np

from .cache import cached_npz, fingerprint
from .config import CATEGORICAL, CLIP, K_NUMERIC, NUMERIC_RAW, SEEDS, TEST_SIZE
from .datasets import encode_targets, load_ton_iot
from .preprocessing import LeakageFreePreprocessor
from .splits import outer_split


def prepare14(seed: int = 42, k_numeric: int = K_NUMERIC,
              selection_target: str = "multiclass", verbose: bool = True):
    """Split holdout + preprocessing a 14 feature, fittato SOLO sul train.

    Restituisce lo stesso tuple che si aspettavano i vecchi `load14()`:

        Xtr, Xte, ybtr, ybte, ymtr, ymte, CTtr, CTte, cards, feats

    dove Xtr/Xte sono le 10 numeriche normalizzate e clippate a +/-CLIP,
    CTtr/CTte gli indici categorici (0 = categoria mai vista in training) e
    `cards` le cardinalita' delle tabelle, UNK incluso.
    """
    key = fingerprint(seed=seed, k=k_numeric, target=selection_target,
                      test_size=TEST_SIZE, clip=CLIP,
                      numeric=NUMERIC_RAW, categorical=CATEGORICAL,
                      exposes_indices=True)

    def build():
        df = load_ton_iot(verbose=verbose)
        yb, ym, classes = encode_targets(df)
        # split stratificato sulle 10 classi: binario e multiclass condividono
        # lo stesso partizionamento, come nel protocollo storico
        tr, te = outer_split(ym, seed=seed, test_size=TEST_SIZE)

        y_sel = ym if selection_target == "multiclass" else yb
        prep = LeakageFreePreprocessor(
            k_numeric=k_numeric, random_state=seed,
            selection_target=selection_target,
        ).fit(df.iloc[tr], y_sel[tr])

        Xtr, CTtr = prep.transform(df.iloc[tr])
        Xte, CTte = prep.transform(df.iloc[te])
        if verbose:
            print(f"[legacy] feature: {prep.numeric_features_}")
            print(f"[legacy] categoriche: "
                  f"{dict(zip(prep.categorical_, prep.cardinalities_))} (indice 0 = UNK)")
        return {
            "Xtr": Xtr, "Xte": Xte,
            "ybtr": yb[tr], "ybte": yb[te],
            "ymtr": ym[tr], "ymte": ym[te],
            "CTtr": CTtr, "CTte": CTte,
            "cards": np.array(prep.cardinalities_),
            "feats": np.array(prep.numeric_features_),
            "classes": np.array(classes),
            # indici dello split, indispensabili a chi deve riallineare
            # altre colonne dello stesso dataframe (per esempio i valori
            # grezzi nella pipeline end-to-end): outer_split restituisce
            # indici ORDINATI, quindi ricostruire lo split altrove con
            # train_test_split darebbe gli stessi elementi in ordine diverso
            "itr": tr, "ite": te,
        }

    d = cached_npz("legacy_prepare14.npz", key, build, verbose=verbose)
    return (d["Xtr"], d["Xte"], d["ybtr"], d["ybte"], d["ymtr"], d["ymte"],
            d["CTtr"], d["CTte"], list(d["cards"]), [str(f) for f in d["feats"]])


def prepare_cv_folds(seed: int, n_splits: int = 5, k_numeric: int = K_NUMERIC,
                     selection_target: str = "multiclass"):
    """Generatore di fold con preprocessing rifittato DENTRO ogni fold.

    Sostituisce lo schema dei vecchi driver di cross-validation, che
    selezionavano le feature una volta sola fuori dal ciclo.
    """
    from .splits import cv_splits

    df = load_ton_iot(verbose=False)
    yb, ym, _ = encode_targets(df)
    y_sel = ym if selection_target == "multiclass" else yb

    for sp in cv_splits(ym, n_splits=n_splits, seeds=(seed,)):
        tr, va = sp["train_idx"], sp["val_idx"]
        prep = LeakageFreePreprocessor(
            k_numeric=k_numeric, random_state=sp["seed"],
            selection_target=selection_target,
        ).fit(df.iloc[tr], y_sel[tr])
        Xtr, CTtr = prep.transform(df.iloc[tr])
        Xva, CTva = prep.transform(df.iloc[va])
        yield {
            "seed": sp["seed"], "fold": sp["fold"],
            "Xtr": Xtr, "Xva": Xva, "CTtr": CTtr, "CTva": CTva,
            "ybtr": yb[tr], "ybva": yb[va],
            "ymtr": ym[tr], "ymva": ym[va],
            "cards": prep.cardinalities_, "feats": prep.numeric_features_,
        }


def prepare14_dict(seed: int = 42, **kw) -> dict:
    """Stessi dati di prepare14(), come dizionario.

    Sostituisce il vecchio `np.load(artifacts/kcat14_bin.npz)` che gli
    script di compilazione ed export usavano come cache condivisa. Quella
    cache era prodotta come effetto collaterale di kan14_binary.py: se non
    lo avevi lanciato prima, gli export fallivano con un FileNotFoundError
    e nulla nel repository diceva quale script andasse eseguito per primo.
    Chiamando la preparazione direttamente, ogni script e' autosufficiente
    e l'ordine di esecuzione smette di essere una conoscenza implicita.
    """
    key = fingerprint(seed=seed, k=kw.get("k_numeric", K_NUMERIC),
                      target=kw.get("selection_target", "multiclass"),
                      test_size=TEST_SIZE, clip=CLIP,
                      numeric=NUMERIC_RAW, categorical=CATEGORICAL,
                      exposes_indices=True)
    prepare14(seed, **kw)                      # garantisce la cache
    d = cached_npz("legacy_prepare14.npz", key, lambda: {}, verbose=False)
    return {k: d[k] for k in d}

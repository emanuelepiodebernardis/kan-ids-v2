"""Protocollo di validazione unico per tutti i modelli.

Scelte, tutte deliberate e da riportare nel paper:

* La stratificazione usa SEMPRE l'etichetta a 10 classi (`type`), anche
  quando il task e' binario. Cosi' il fold di un modello binario e quello
  di un modello multiclass contengono esattamente gli stessi flussi e i
  due task sono confrontabili riga per riga. In particolare le classi rare
  (MITM) restano rappresentate in ogni fold.
* 5-fold x 3 seed = 15 fit per modello. Si riporta media +/- std sui 15.
* L'held-out esterno (20%) e' generato con lo stesso stratificatore e non
  viene mai toccato durante tuning o selezione: serve solo al numero finale.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import N_SPLITS, SEEDS, TEST_SIZE


def outer_split(y_strat: np.ndarray, seed: int = 42, test_size: float = TEST_SIZE):
    """Held-out esterno stratificato. Restituisce (train_idx, test_idx)."""
    idx = np.arange(len(y_strat))
    tr, te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y_strat
    )
    return np.sort(tr), np.sort(te)


def cv_splits(
    y_strat: np.ndarray,
    n_splits: int = N_SPLITS,
    seeds: Sequence[int] = SEEDS,
    subset: np.ndarray | None = None,
) -> Iterator[dict]:
    """Genera i 5x3 fold.

    `subset` limita la CV a un sottoinsieme di indici (tipicamente il
    training dell'held-out esterno); gli indici restituiti sono sempre
    riferiti all'array originale.
    """
    base = np.arange(len(y_strat)) if subset is None else np.asarray(subset)
    ys = np.asarray(y_strat)[base]

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(ys)), ys), start=1):
            yield {
                "seed": int(seed),
                "fold": int(fold),
                "train_idx": base[tr],
                "val_idx": base[va],
            }


def n_runs(n_splits: int = N_SPLITS, seeds: Sequence[int] = SEEDS) -> int:
    return n_splits * len(seeds)


def describe_protocol(n_splits: int = N_SPLITS, seeds: Sequence[int] = SEEDS) -> str:
    return (
        f"StratifiedKFold(n_splits={n_splits}, shuffle=True) ripetuta su "
        f"seeds={list(seeds)} -> {n_splits * len(seeds)} fit per modello; "
        f"stratificazione sull'etichetta a 10 classi; media +/- std sui fold. "
        f"Feature selection, encoding e normalizzazione sono rifittati "
        f"dentro ogni fold sul solo training."
    )

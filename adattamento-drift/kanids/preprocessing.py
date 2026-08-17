"""Preprocessing leakage-free.

Regola unica, verificata dai test in tests/test_leakage.py:

    fit(df_train, y_train)  ->  impara TUTTO (ranking MI, vocabolari
                                categorici, quantili) solo dal training split
    transform(df)           ->  non riceve y, non guarda le statistiche di df

Cosa cambia rispetto alla versione precedente della pipeline
--------------------------------------------------------------
1. Il ranking per mutual information era calcolato su un campione
   dell'intero dataset PRIMA dello split: la scelta delle 10 feature
   numeriche vedeva le etichette di test. Ora la MI e' calcolata dentro
   fit(), quindi per-fold.
2. I LabelEncoder delle categoriche erano fittati sull'intero dataframe:
   il vocabolario e le cardinalita' includevano categorie presenti solo
   nel test. Ora il vocabolario viene dal training split e ogni categoria
   mai vista finisce nello slot UNK (indice 0), che esiste sempre.
3. QuantileTransformer e log1p erano gia' fit-su-train: comportamento
   invariato, ora pero' e' esplicito e testato.

Lo slot UNK non e' un dettaglio difensivo: e' cio' che rende definito il
modello sul dominio target nel cross-domain (BoT-IoT ha valori di
protocollo/stato assenti da TON_IoT) e rende l'edge categorico una
funzione totale anche a runtime su MCU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import QuantileTransformer

from .config import CATEGORICAL, CLIP, K_NUMERIC, NUMERIC_RAW, SKEWED, UNK_INDEX


def rank_by_mi(X: np.ndarray, y: np.ndarray, seed: int = 42,
               sample: int | None = 40_000) -> np.ndarray:
    """Mutual information fra ogni colonna di X e y, su un campione di X.

    UNICO punto del repository in cui si calcola la mutual information per
    la selezione delle feature. Chiunque la chiami deve passare SOLO righe
    di training: il test tests/test_reproducibility.py vieta l'import
    diretto di sklearn.mutual_info_classif fuori da questo modulo, proprio
    per rendere impossibile ripetere l'errore del protocollo v1 (ranking
    calcolato sull'intero dataset, prima dello split).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    if sample and sample < n:
        idx = np.random.RandomState(seed).choice(n, sample, replace=False)
        X, y = X[idx], y[idx]
    return mutual_info_classif(X, y, random_state=seed)


@dataclass
class FeatureRanking:
    """Esito della selezione, da salvare in results/ per il report."""
    candidates: List[str]
    mi_scores: np.ndarray
    selected: List[str]
    n_used_for_mi: int

    def to_frame(self) -> pd.DataFrame:
        order = np.argsort(self.mi_scores)[::-1]
        return pd.DataFrame({
            "rank": np.arange(1, len(order) + 1),
            "feature": [self.candidates[i] for i in order],
            "mutual_information": np.round(self.mi_scores[order], 6),
            "selected": [self.candidates[i] in self.selected for i in order],
        })


class LeakageFreePreprocessor:
    """Selezione feature + normalizzazione + encoding categorico.

    Parameters
    ----------
    k_numeric : int
        Quante feature numeriche tenere (ordinate per MI sul training split).
    numeric_candidates : sequence of str
        Colonne numeriche candidate. Per il cross-domain si passa qui lo
        spazio armonizzato fra i due dataset.
    categorical : sequence of str
        Colonne categoriche da mappare su edge tabellari.
    selection_target : {"multiclass", "binary"}
        Su quale etichetta calcolare la MI. Il default resta multiclass
        (piu' informativo) anche quando il modello finale e' binario, ma
        deve essere lo stesso in tutti i fold ed e' registrato in
        `feature_ranking_`.
    impute : {"zero", "median"}
        "zero" replica la pipeline pubblicata (i missing di TON_IoT sono
        marcatori "-", non veri NaN numerici). "median" usa la mediana del
        training split: anch'essa leakage-free, ma cambia i numeri.
    """

    def __init__(
        self,
        k_numeric: int = K_NUMERIC,
        numeric_candidates: Sequence[str] | None = None,
        categorical: Sequence[str] | None = None,
        skewed: Iterable[str] | None = None,
        clip: float = CLIP,
        mi_sample: int = 40_000,
        n_quantiles: int = 1000,
        random_state: int = 42,
        selection_target: str = "multiclass",
        impute: str = "zero",
    ):
        self.k_numeric = k_numeric
        self.numeric_candidates = list(numeric_candidates if numeric_candidates is not None else NUMERIC_RAW)
        self.categorical = list(categorical if categorical is not None else CATEGORICAL)
        self.skewed = set(skewed if skewed is not None else SKEWED)
        self.clip = clip
        self.mi_sample = mi_sample
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.selection_target = selection_target
        self.impute = impute
        self._fitted = False

    # ── helpers ────────────────────────────────────────────────
    def _numeric_matrix(self, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"colonne numeriche assenti dal dataframe: {missing}")
        X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        if self.impute == "median" and self._fitted:
            X = X.fillna(pd.Series(self.impute_values_, index=list(cols)))
        elif self.impute == "median":
            X = X.fillna(X.median())
        else:
            X = X.fillna(0.0)
        return X.to_numpy(np.float64)

    # ── fit ────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LeakageFreePreprocessor":
        """Impara tutto dal solo training split. `y` e' usato SOLO qui."""
        y = np.asarray(y)
        if len(y) != len(df):
            raise ValueError(f"len(y)={len(y)} != len(df)={len(df)}")

        cands = [c for c in self.numeric_candidates if c in df.columns]
        if not cands:
            raise ValueError("nessuna delle feature numeriche candidate e' presente")
        self.candidates_ = cands

        Xc = df[cands].apply(pd.to_numeric, errors="coerce")
        self.impute_values_ = (Xc.median().to_dict() if self.impute == "median"
                               else {c: 0.0 for c in cands})
        X = Xc.fillna(pd.Series(self.impute_values_)).to_numpy(np.float64)

        # 1) ranking per mutual information — SOLO sul training split
        n = len(X)
        n_mi = min(self.mi_sample, n) if self.mi_sample else n
        mi = rank_by_mi(X, y, seed=self.random_state, sample=self.mi_sample)

        k = min(self.k_numeric, len(cands))
        order = np.argsort(mi)[::-1][:k]
        self.numeric_features_ = [cands[i] for i in order]
        self.feature_ranking_ = FeatureRanking(
            candidates=cands, mi_scores=mi,
            selected=list(self.numeric_features_), n_used_for_mi=int(n_mi),
        )

        # 2) log1p sulle selezionate asimmetriche + quantile fit su train
        Xs = X[:, order].copy()
        self._log_mask_ = np.array([f in self.skewed for f in self.numeric_features_])
        Xs[:, self._log_mask_] = np.log1p(np.clip(Xs[:, self._log_mask_], 0, None))
        self.quantile_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=int(min(self.n_quantiles, Xs.shape[0])),
            random_state=self.random_state,
        ).fit(Xs)

        # 3) vocabolari categorici — solo dal training split, indice 0 = UNK
        self.categorical_ = [c for c in self.categorical if c in df.columns]
        self.vocabularies_ = {}
        for c in self.categorical_:
            vals = sorted(df[c].astype(str).unique().tolist())
            self.vocabularies_[c] = {v: i + 1 for i, v in enumerate(vals)}  # 0 = UNK
        self.cardinalities_ = [len(self.vocabularies_[c]) + 1 for c in self.categorical_]

        self._fitted = True
        return self

    # ── transform ──────────────────────────────────────────────
    def transform(self, df: pd.DataFrame):
        """Applica la trasformazione appresa. Non riceve mai `y`."""
        self._check_fitted()
        X = self._numeric_matrix(df, self.numeric_features_)
        X[:, self._log_mask_] = np.log1p(np.clip(X[:, self._log_mask_], 0, None))
        Xn = np.clip(self.quantile_.transform(X), -self.clip, self.clip)

        cats = np.empty((len(df), len(self.categorical_)), dtype=np.int64)
        for j, c in enumerate(self.categorical_):
            voc = self.vocabularies_[c]
            cats[:, j] = df[c].astype(str).map(voc).fillna(UNK_INDEX).to_numpy(np.int64)
        return Xn, cats

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray):
        return self.fit(df, y).transform(df)

    # ── diagnostica ────────────────────────────────────────────
    def unseen_rate(self, df: pd.DataFrame) -> dict:
        """Frazione di righe che cade in UNK per ogni categorica.

        Nel cross-domain e' una diagnosi diretta del degrado: se `service`
        va al 90% UNK, l'edge corrispondente e' di fatto una costante.
        """
        self._check_fitted()
        out = {}
        for c in self.categorical_:
            voc = self.vocabularies_[c]
            out[c] = float((~df[c].astype(str).isin(voc)).mean())
        return out

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("preprocessor non fittato: chiamare fit(df_train, y_train)")

    # ── persistenza ────────────────────────────────────────────
    def save(self, path):
        self._check_fitted()
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path) -> "LeakageFreePreprocessor":
        return joblib.load(path)

    def summary(self) -> dict:
        self._check_fitted()
        return {
            "numeric_features": list(self.numeric_features_),
            "categorical_features": list(self.categorical_),
            "cardinalities": list(self.cardinalities_),
            "k_numeric": self.k_numeric,
            "clip": self.clip,
            "selection_target": self.selection_target,
            "random_state": self.random_state,
            "n_used_for_mi": self.feature_ranking_.n_used_for_mi,
        }

"""Metriche uniformi per tutti i modelli e tutti i dataset.

Include PR-AUC ovunque (richiesta esplicita: sotto forte sbilanciamento —
BoT-IoT ha 477 flussi normali su 3.67 M — la ROC-AUC e' ottimistica e la
PR-AUC e' la metrica onesta) e sempre la confusion matrix, che nel
cross-domain e' l'oggetto da leggere piu' del numero aggregato.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def binary_metrics(y_true, y_pred, y_score=None, prefix: str = "") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        m["roc_auc"] = np.nan
        m["pr_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = cm
    m.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    m["fpr"] = float(fp / max(fp + tn, 1))
    m["fnr"] = float(fn / max(fn + tp, 1))
    return {f"{prefix}{k}": v for k, v in m.items()} if prefix else m


def multiclass_metrics(y_true, y_pred, y_proba=None, labels: Sequence | None = None,
                       class_names: Sequence[str] | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        yb = label_binarize(y_true, classes=labels)
        if yb.shape[1] == 1:
            yb = np.hstack([1 - yb, yb])
        try:
            m["roc_auc_macro"] = roc_auc_score(yb, y_proba, average="macro", multi_class="ovr")
            m["pr_auc_macro"] = average_precision_score(yb, y_proba, average="macro")
        except ValueError:
            m["roc_auc_macro"] = np.nan
            m["pr_auc_macro"] = np.nan

    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    names = list(class_names) if class_names is not None else [str(l) for l in labels]
    for name, v in zip(names, per_class):
        m[f"f1_{name}"] = float(v)
    return m


def confusion_frame(y_true, y_pred, labels, class_names=None, normalize=False) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = list(class_names) if class_names is not None else [str(l) for l in labels]
    return pd.DataFrame(cm, index=pd.Index(names, name="true"),
                        columns=pd.Index(names, name="pred"))


def aggregate(rows: list[dict], by: Sequence[str] = ("model",)) -> pd.DataFrame:
    """Media +/- std sui fold. Restituisce colonne <metrica>_mean/_std."""
    df = pd.DataFrame(rows)
    by = [c for c in by if c in df.columns]
    metric_cols = [c for c in df.columns
                   if c not in set(by) | {"seed", "fold", "task", "dataset"}
                   and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(list(by), dropna=False)[metric_cols]
    out = g.agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size().values
    return out.reset_index()


def format_mean_std(df: pd.DataFrame, metric: str, decimals: int = 4) -> pd.Series:
    """'0.9837 +/- 0.0007' pronto per la tabella del report."""
    return (df[f"{metric}_mean"].round(decimals).astype(str) + " ± "
            + df[f"{metric}_std"].round(decimals).astype(str))

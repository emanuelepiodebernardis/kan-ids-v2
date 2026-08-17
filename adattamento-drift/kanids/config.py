"""Costanti, percorsi e seed. Unico posto in cui sono definiti.

Nessun percorso assoluto e nessun /tmp: tutti gli artefatti intermedi
finiscono in <repo>/artifacts (override con la variabile d'ambiente
KANIDS_ARTIFACTS), cosi' un clone pulito e' autosufficiente.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# PERCORSI
# ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = Path(os.environ.get("KANIDS_ARTIFACTS", REPO_ROOT / "artifacts")).resolve()
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR = Path(os.environ.get("KANIDS_DATA", REPO_ROOT / "data")).resolve()

for _d in (ARTIFACTS_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def artifact_path(name: str) -> Path:
    """Percorso di un artefatto intermedio (sostituisce i vecchi /tmp/...)."""
    p = ARTIFACTS_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────
# PROTOCOLLO SPERIMENTALE
# ─────────────────────────────────────────────────────────────
# Protocollo unico per TUTTI i modelli principali (KAN single-layer,
# KAN multi-layer, LightGBM, XGBoost, Decision Tree, MLP):
# 5-fold stratificata ripetuta su 3 seed -> 15 fit per modello,
# si riporta media +/- deviazione standard.
SEEDS = (42, 43, 44)
N_SPLITS = 5
TEST_SIZE = 0.2          # held-out esterno, usato solo per il report finale

# Dominio della base di Chebyshev: le feature normalizzate sono clippate qui.
CLIP = 3.5

# Numero di feature numeriche selezionate (giustificato dall'ablation
# feature_curve: il massimo e' a k=10 su entrambi i task).
K_NUMERIC = 10

# ─────────────────────────────────────────────────────────────
# SPAZIO DELLE FEATURE (TON_IoT)
# ─────────────────────────────────────────────────────────────
NUMERIC_RAW = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
    "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]

# Distribuzioni fortemente asimmetriche (conteggi/byte/durata): log1p
# prima del quantile, per comprimere le code lunghe.
SKEWED = {
    "duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts",
    "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "http_request_body_len", "http_response_body_len",
}

CATEGORICAL = ["proto", "service", "conn_state", "dns_rejected"]

# Slot 0 di ogni tabella categorica = categoria mai vista in training.
# Serve per il cross-domain (BoT-IoT avra' valori di `state`/`proto` assenti
# da TON_IoT) e rende l'edge categorico totale anche a runtime su MCU.
UNK_INDEX = 0


def set_global_seed(seed: int) -> None:
    """Fissa i seed di python/numpy (+torch se presente)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

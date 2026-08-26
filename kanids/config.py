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

# Numero di feature numeriche selezionate.
#
# NON e' un ottimo di accuratezza, ed e' importante non riscriverlo come
# tale: la curva feature_curve che sembrava avere il massimo a k=10 era
# misurata sullo stesso held-out poi riportato come risultato. La CV
# annidata (scripts/nested_cv.py), che riseleziona k dentro ogni fold,
# trova una curva monotona e sceglie k=16 in 15 fold su 15. Il prezzo di
# restare a 10 e' stato misurato ed e' 0,0009 di F1: k=10 e' una scelta di
# deployment (meno feature da calcolare a bordo), non di accuratezza.
# Vedi README, "Nested cross-validation".
K_NUMERIC = 10


# ─────────────────────────────────────────────────────────────
# Architettura della KAN: larghezza e grado.
#
# Questi due numeri NON vanno riscritti a mano qui dentro. Vengono da
# `scripts/select_architettura.py`, che li sceglie su una validation
# ritagliata dentro il training — la stessa disciplina del rapporto del
# joint training, e per la stessa ragione: le ablation che avevano dato
# hidden=16 e degree=8 (results/protocol_v1/ml_binary_real.csv e
# kan_ml_cat_deg4_real.csv) erano misurate sull'held-out, cioe' sullo
# stesso insieme poi riportato come risultato.
#
# Finche' la selezione non e' stata eseguita si usano i valori ereditati
# dalla fase 1, ed e' giusto che sia cosi': sono quelli con cui sono stati
# prodotti i numeri pubblicati. Quando l'artefatto compare, questi valori
# lo seguono, e un test verifica che i due coincidano — altrimenti i
# risultati committati sarebbero di un'architettura e il codice di
# un'altra, senza che nulla lo segnali.
ARCH_EREDITATA = {"KAN(cat,1L)": {"hidden": 0, "degree": 8},
                  "KAN(cat,ML)": {"hidden": 16, "degree": 8}}


def _selezione():
    """Cosa ha scelto la selezione su validation, se e' stata eseguita."""
    p = RESULTS_DIR / "arch_selection_scelta.json"
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text(encoding="utf-8"))["scelte"]


# Cosa la pipeline USA e cosa la selezione SCEGLIE sono due cose diverse, e
# dal 26 agosto 2026 non coincidono. Vanno tenute separate proprio per
# questo: far leggere alla pipeline la selezione avrebbe nascosto lo scarto
# dentro un file JSON, invece di costringere qualcuno a dichiararlo.
#
#   selezione su validation (1-SE, 5 seed):  KAN(cat,ML) h=32 grado=6
#                                            0,99632 di balanced accuracy
#                                            2.592 parametri
#   architettura deployata:                  KAN(cat,ML) h=16 grado=8
#                                            0,99600 sulla stessa validation
#                                            1.648 parametri
#
# Lo scarto e' -0,00032 di balanced accuracy per il 36% di parametri in meno.
# La 16/8 manca la soglia 1-SE per 0,00020: la selezione la esclude, ma di
# un margine che vale meno di un campione ogni tremila. Il t appaiato fra le
# due da' p = 0,067 su cinque seed.
#
# Il progetto deploya la 16/8, che e' ereditata dalla fase 1, e questo NON e'
# un risultato della selezione: e' il vincolo di dimensione di un
# microcontrollore, dichiarato come tale. La selezione resta agli atti come
# misura di quanto costa quel vincolo — che era il punto di eseguirla.
#
# Percio' ARCH_ORIGINE dice "ereditata" anche quando l'artefatto esiste, e un
# test pretende che lo scarto sia scritto nel README. Adottare la selezione
# significherebbe rigenerare CV, cross-domain, joint, header C, golden
# vector, footprint, figure, report e firmware.
ARCH = {k: dict(v) for k, v in ARCH_EREDITATA.items()}
ARCH_ORIGINE = "ereditata"
ARCH_SELEZIONATA = _selezione()
HIDDEN = ARCH["KAN(cat,ML)"]["hidden"]
DEGREE = ARCH["KAN(cat,ML)"]["degree"]
DEGREE_1L = ARCH["KAN(cat,1L)"]["degree"]


def scarto_dalla_selezione():
    """(modello -> {deployata, selezionata, coincidono}) o None.

    Serve ai test e all'audit: se un giorno qualcuno adottasse la selezione,
    o la selezione cambiasse, questo dice subito se le due sono allineate.
    """
    if ARCH_SELEZIONATA is None:
        return None
    fuori = {}
    for m, s in ARCH_SELEZIONATA.items():
        d = ARCH.get(m, {})
        fuori[m] = {
            "deployata": {"hidden": d.get("hidden"), "degree": d.get("degree")},
            "selezionata": {"hidden": int(s["hidden"]), "degree": int(s["degree"])},
            "coincidono": (d.get("hidden") == int(s["hidden"])
                           and d.get("degree") == int(s["degree"])),
        }
    return fuori

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

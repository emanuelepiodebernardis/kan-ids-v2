"""Caricamento dataset + generatore sintetico per smoke test.

Il generatore sintetico non serve a produrre risultati: serve a rendere il
repository eseguibile da un clone pulito SENZA aver prima scaricato
TON_IoT. `python reproduce.py --smoke` gira in meno di un minuto su dati
finti con lo stesso schema, e verifica che l'intera catena
(preprocessing -> CV -> metriche -> export) sia integra. Chi vuole i
numeri veri scarica il CSV e lancia gli stage reali.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL, DATA_DIR, NUMERIC_RAW, REPO_ROOT

TON_IOT_FILENAME = "train_test_network.csv"
TON_IOT_CLASSES = ["normal", "backdoor", "ddos", "dos", "injection",
                   "mitm", "password", "ransomware", "scanning", "xss"]

# Posizioni in cui cerchiamo il CSV, in ordine.
_SEARCH = [
    lambda: DATA_DIR / TON_IOT_FILENAME,
    lambda: REPO_ROOT / TON_IOT_FILENAME,
    lambda: Path.cwd() / TON_IOT_FILENAME,
]


def ton_iot_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for f in _SEARCH:
        p = f()
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{TON_IOT_FILENAME} non trovato. Scaricalo (vedi data/README.md) e "
        f"mettilo in {DATA_DIR}/ oppure nella root del repo, oppure lancia "
        f"con --smoke per usare dati sintetici."
    )


def load_ton_iot(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Carica TON_IoT e valida lo schema atteso."""
    p = ton_iot_path(path)
    df = pd.read_csv(p, low_memory=False)
    required = {"label", "type"} | set(CATEGORICAL)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"colonne mancanti in {p.name}: {sorted(missing)}")
    if not any(c in df.columns for c in NUMERIC_RAW):
        raise ValueError("nessuna feature numerica attesa presente")
    df["label"] = df["label"].astype(int)
    df["type"] = df["type"].astype(str)
    if verbose:
        print(f"[data] {p}  righe={len(df):,}  colonne={df.shape[1]}  "
              f"attacchi={df['label'].mean():.1%}")
    return df


def make_synthetic(n: int = 20_000, seed: int = 0, n_classes: int = 10) -> pd.DataFrame:
    """Dataframe con lo schema di TON_IoT, per smoke test e unit test.

    Le classi sono separabili ma non banalmente: serve solo che la
    pipeline abbia segnale da trovare.
    """
    rng = np.random.RandomState(seed)
    classes = TON_IOT_CLASSES[:n_classes]
    # distribuzione sbilanciata realistica (mitm raro)
    w = np.array([3.0] + [1.0] * (len(classes) - 1))
    w[classes.index("mitm")] = 0.05 if "mitm" in classes else w[-1]
    w = w / w.sum()
    y = rng.choice(len(classes), size=n, p=w)

    centers = rng.randn(len(classes), len(NUMERIC_RAW)) * 2.0
    data = {}
    for j, col in enumerate(NUMERIC_RAW):
        base = centers[y, j] + rng.randn(n) * 1.5
        if col in {"duration"}:
            data[col] = np.abs(base) * 0.5
        elif "bytes" in col or "pkts" in col or "len" in col:
            data[col] = np.abs(np.expm1(np.abs(base))).clip(0, 1e7).astype(np.int64)
        elif "port" in col:
            data[col] = rng.randint(0, 65536, n)
        else:
            data[col] = np.abs(base).astype(np.int64)
    df = pd.DataFrame(data)

    protos = ["tcp", "udp", "icmp"]
    services = ["-", "dns", "http", "ssl", "ftp", "smtp"]
    states = ["S0", "SF", "REJ", "RSTO", "RSTR", "OTH", "SH"]
    df["proto"] = [protos[i] for i in (y % len(protos))]
    df["service"] = rng.choice(services, n, p=[.45, .2, .15, .1, .05, .05])
    df["conn_state"] = [states[(c + r) % len(states)]
                        for c, r in zip(y, rng.randint(0, 2, n))]
    df["dns_rejected"] = rng.choice(["F", "T"], n, p=[.9, .1])

    df["type"] = [classes[i] for i in y]
    df["label"] = (df["type"] != "normal").astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# BoT-IoT (UNSW, versione 5% full-feature: 4 CSV, 3.668.522 flussi)
# ─────────────────────────────────────────────────────────────
BOT_IOT_GLOB = "UNSW_2018_IoT_Botnet_Full5pc_*.csv"

# Solo le colonne che servono allo spazio armonizzato: leggere tutte e 46
# le colonne di 1 GB di CSV non serve a niente.
BOT_IOT_USECOLS = ["proto", "state", "dur", "spkts", "dpkts",
                   "sbytes", "dbytes", "attack", "category"]


def bot_iot_paths(directory=None) -> list[Path]:
    import glob
    roots = [Path(directory)] if directory else [DATA_DIR, REPO_ROOT, Path.cwd()]
    for r in roots:
        found = sorted(Path(p) for p in glob.glob(str(r / BOT_IOT_GLOB)))
        if found:
            return found
    raise FileNotFoundError(
        f"{BOT_IOT_GLOB} non trovato. Scarica la versione 5% full-feature di "
        f"BoT-IoT (4 CSV) e mettila in {DATA_DIR}/. Attenzione: NON i file "
        f"'10_best', la cui selezione di feature e' stata fatta sull'intero "
        f"dataset (leakage) e non e' armonizzabile con TON_IoT."
    )


def load_bot_iot(directory=None, verbose: bool = True) -> pd.DataFrame:
    files = bot_iot_paths(directory)
    parts = [pd.read_csv(f, usecols=BOT_IOT_USECOLS, low_memory=False) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)
    if verbose:
        n_norm = int((df["attack"] == 0).sum())
        print(f"[data] BoT-IoT: {len(files)} file, {len(df):,} flussi, "
              f"normali={n_norm} ({n_norm/len(df):.5%})")
        print("       category:", df["category"].value_counts().to_dict())
    return df


UNSW_GLOB = "UNSW_NB15_*-set.csv"
UNSW_USECOLS = ["dur", "proto", "state", "spkts", "dpkts", "sbytes",
                "dbytes", "attack_cat", "label"]


def unsw_paths(directory=None) -> list[Path]:
    import glob
    roots = [Path(directory)] if directory else [DATA_DIR, REPO_ROOT, Path.cwd()]
    for r in roots:
        found = sorted(Path(p) for p in glob.glob(str(r / UNSW_GLOB)))
        if found:
            return found
    raise FileNotFoundError(
        f"{UNSW_GLOB} non trovato. Servono UNSW_NB15_training-set.csv e "
        f"UNSW_NB15_testing-set.csv (la versione a 45 colonne con "
        f"intestazione), da mettere in {DATA_DIR}/. NON i quattro file "
        f"UNSW-NB15_1..4.csv, che sono senza intestazione e con nomi di "
        f"colonna diversi."
    )


def load_unsw(directory=None, verbose: bool = True) -> pd.DataFrame:
    """UNSW-NB15. Stesso strumento (Argus) e stesso schema di BoT-IoT.

    I due file training-set e testing-set sono due partizioni della stessa
    cattura: qui si concatenano, perche' la nostra suddivisione la
    decidiamo noi e quella originale non ci vincola.
    """
    files = unsw_paths(directory)
    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        cols = [c for c in UNSW_USECOLS if c in d.columns]
        mancanti = set(UNSW_USECOLS) - set(cols)
        if mancanti - {"attack_cat"}:
            raise ValueError(f"colonne mancanti in {f.name}: {sorted(mancanti)}")
        parts.append(d[cols])
    df = pd.concat(parts, ignore_index=True)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if verbose:
        n_norm = int((df["label"] == 0).sum())
        print(f"[data] UNSW-NB15: {len(files)} file, {len(df):,} flussi, "
              f"normali={n_norm:,} ({n_norm/len(df):.4%})")
        if "attack_cat" in df.columns:
            print("       categorie:", df["attack_cat"].value_counts().head(12).to_dict())
    return df


CIC_GLOB = "*.csv"
CIC_DIRNAME = "cic"


def cic_paths(directory=None) -> list[Path]:
    import glob
    roots = ([Path(directory)] if directory
             else [DATA_DIR / CIC_DIRNAME, DATA_DIR, REPO_ROOT / CIC_DIRNAME])
    for r in roots:
        found = sorted(Path(p) for p in glob.glob(str(r / CIC_GLOB)))
        if found:
            return found
    raise FileNotFoundError(
        f"CSV di CIC-IoT-2023 non trovati. Mettili in {DATA_DIR / CIC_DIRNAME}/. "
        f"Bastano 10-20 dei 169 shard: sono partizioni della stessa cattura."
    )


def load_cic(directory=None, verbose: bool = True, max_files: int | None = None,
             max_attacchi: int = 400_000, seed: int = 42):
    """CIC-IoT-2023. Attenzione: NON ha i conteggi direzionali.

    Utilizzabile solo nello spazio ridotto (kanids.harmonized.RIDOTTO_NUMERIC),
    e il costo di quella riduzione e' misurato in scripts/spazio_ridotto.py.
    """
    files = cic_paths(directory)
    if max_files:
        files = files[:max_files]
    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        if "label" not in d.columns and "Label" not in d.columns:
            # in questa distribuzione l'etichetta e' nel NOME del file:
            # Benign_test.pcap.csv contro i file di attacco
            d["label"] = 0 if f.name.lower().startswith("benign") else 1
        elif "Label" in d.columns:
            d["label"] = (~d["Label"].astype(str).str.lower()
                          .str.startswith("benign")).astype(int)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    # gli attacchi sono milioni e i benigni decine di migliaia: si tiene
    # tutta la minoritaria e si sottocampiona la maggioritaria, come per
    # BoT-IoT. La valutazione resta su proporzioni dichiarate.
    if max_attacchi and int((df.label == 1).sum()) > max_attacchi:
        rng = np.random.RandomState(seed)
        att = np.flatnonzero(df.label.to_numpy() == 1)
        keep = np.concatenate([np.flatnonzero(df.label.to_numpy() == 0),
                               rng.choice(att, max_attacchi, replace=False)])
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
    if verbose:
        n_norm = int((df.label == 0).sum())
        print(f"[data] CIC (famiglia IoT/IoMT): {len(files)} file, {len(df):,} righe, "
              f"benigni={n_norm:,} ({n_norm/len(df):.4%})")
    return df


def encode_targets(df: pd.DataFrame):
    """(y_binary, y_multiclass, class_names) con ordine di classi FISSO.

    L'ordine e' quello di TON_IOT_CLASSES, non quello alfabetico appreso da
    un LabelEncoder: cosi' l'indice di classe e' stabile fra fold, fra
    macchine e nell'export C, e i golden vector restano validi.
    """
    present = [c for c in TON_IOT_CLASSES if c in set(df["type"])]
    extra = sorted(set(df["type"]) - set(present))
    classes = present + extra
    mapping = {c: i for i, c in enumerate(classes)}
    ym = df["type"].map(mapping).to_numpy(np.int64)
    yb = df["label"].to_numpy(np.int64)
    return yb, ym, classes

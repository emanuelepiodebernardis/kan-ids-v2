"""Spazio di feature armonizzato fra TON_IoT (Zeek) e BoT-IoT (Argus).

Il vincolo del punto 3 e' "solo feature realmente confrontabili". Qui
"confrontabile" significa due cose insieme:

  1. la grandezza fisica misurata e' la stessa nei due dataset;
  2. la feature e' calcolata con la STESSA formula a partire da colonne
     equivalenti, non ricopiata da una colonna gia' presente in uno dei due.

Cosa e' escluso, e perche'
--------------------------
* **Porte e indirizzi** (`src_port`, `dst_port`, `saddr`, ...). Sono
  identificatori del testbed: in TON_IoT la MI le mette al primo posto, ma
  la porta che identifica un attacco in un laboratorio non identifica nulla
  in un altro. Misurato in-domain: escluderle costa 0,0008 di F1 a LightGBM
  e alla KAN fa *guadagnare* 0,0025 (results/cv_leakagefree_summary_binary_real_nosrcdst.csv).
* **Aggregati a finestra di BoT-IoT** (`TnBPSrcIP`, `N_IN_Conn_P_SrcIP`,
  `AR_P_Proto_P_Sport`, ...). Non hanno alcun corrispettivo in TON_IoT e
  presuppongono uno stato globale che un MCU non mantiene.
* **Metadati applicativi di TON_IoT** (DNS/SSL/HTTP). BoT-IoT non li ha.

Semantica dei byte
------------------
`sbytes`/`dbytes` di Argus sono byte totali trasmessi, header inclusi:
l'equivalente Zeek e' `src_ip_bytes`/`dst_ip_bytes`, non `src_bytes`
(che in Zeek e' il solo payload). La mappatura usa quindi i byte a
livello IP su entrambi i lati.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Le 13 feature candidate. La selezione MI ne terra' 10, decisa solo sul
# source domain.
HARMONIZED_NUMERIC = [
    "duration",
    "bytes_src", "bytes_dst", "bytes_total",
    "pkts_src", "pkts_dst", "pkts_total",
    "byte_asymmetry", "pkt_asymmetry",
    "payload_mean_src", "payload_mean_dst",
    "flow_rate", "byte_rate",
]

HARMONIZED_CATEGORICAL = ["proto_h", "state_h"]

# Distribuzioni con code lunghe: log1p prima del quantile.
HARMONIZED_SKEWED = {
    "duration", "bytes_src", "bytes_dst", "bytes_total",
    "pkts_src", "pkts_dst", "pkts_total",
    "payload_mean_src", "payload_mean_dst", "flow_rate", "byte_rate",
}

_EPS = 1e-6

# ── protocollo: alfabeto comune ───────────────────────────────
_PROTO = {
    "tcp": "tcp", "udp": "udp", "icmp": "icmp",
    "ipv6-icmp": "icmp",          # stesso ruolo semantico
    "arp": "other", "ipv6": "other", "rarp": "other", "igmp": "other",
}

# ── stato della connessione: alfabeto comune ──────────────────
# Zeek e Argus usano vocabolari diversi per la stessa nozione. La mappa e'
# semantica e decisa a priori, NON tarata sui dati (tararla sul target
# violerebbe il vincolo del cross-domain).
_STATE_ZEEK = {
    "S0": "incomplete",   # tentativo senza risposta
    "SH": "incomplete", "SHR": "incomplete",
    "S1": "established", "S2": "established", "S3": "established",
    "SF": "closed",       # chiusa normalmente
    "REJ": "rejected",
    "RSTO": "reset", "RSTR": "reset", "RSTOS0": "reset", "RSTRH": "reset",
    "OTH": "other",
}
_STATE_ARGUS = {
    "INT": "incomplete",  # solo pacchetto iniziale
    "REQ": "incomplete",
    "CON": "established", "ACC": "established",
    "FIN": "closed", "CLO": "closed",
    "RST": "reset",
    "URP": "other", "ECO": "other", "ECR": "other",
    "TST": "other", "MAS": "other", "NRS": "other",
    # stati che compaiono in UNSW-NB15 ma non in BoT-IoT. Stesso strumento,
    # vocabolario leggermente piu' ampio: aggiunti qui, non tarati sui dati.
    "PAR": "other", "URN": "other", "no": "other", "NO": "other",
    "ECR ": "other",
}

COMMON_STATES = ["established", "closed", "rejected", "incomplete", "reset", "other"]


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float64)


def _derive(duration, b_src, b_dst, p_src, p_dst) -> pd.DataFrame:
    """Le formule derivate: identiche per i due dataset, per costruzione."""
    b_tot = b_src + b_dst
    p_tot = p_src + p_dst
    return pd.DataFrame({
        "duration": duration,
        "bytes_src": b_src,
        "bytes_dst": b_dst,
        "bytes_total": b_tot,
        "pkts_src": p_src,
        "pkts_dst": p_dst,
        "pkts_total": p_tot,
        "byte_asymmetry": (b_src - b_dst) / (b_tot + 1.0),
        "pkt_asymmetry": (p_src - p_dst) / (p_tot + 1.0),
        "payload_mean_src": b_src / (p_src + 1.0),
        "payload_mean_dst": b_dst / (p_dst + 1.0),
        "flow_rate": p_tot / (duration + _EPS),
        "byte_rate": b_tot / (duration + _EPS),
    })


def build_harmonized_ton(df: pd.DataFrame) -> pd.DataFrame:
    """TON_IoT (Zeek conn log) -> spazio armonizzato."""
    out = _derive(
        duration=_num(df, "duration"),
        b_src=_num(df, "src_ip_bytes"),
        b_dst=_num(df, "dst_ip_bytes"),
        p_src=_num(df, "src_pkts"),
        p_dst=_num(df, "dst_pkts"),
    )
    out["proto_h"] = df["proto"].astype(str).str.lower().map(_PROTO).fillna("other")
    out["state_h"] = df["conn_state"].astype(str).str.upper().map(_STATE_ZEEK).fillna("other")
    out["label"] = df["label"].astype(int).to_numpy()
    return out


def build_harmonized_bot(df: pd.DataFrame) -> pd.DataFrame:
    """BoT-IoT (Argus) -> spazio armonizzato."""
    out = _derive(
        duration=_num(df, "dur"),
        b_src=_num(df, "sbytes"),
        b_dst=_num(df, "dbytes"),
        p_src=_num(df, "spkts"),
        p_dst=_num(df, "dpkts"),
    )
    out["proto_h"] = df["proto"].astype(str).str.lower().map(_PROTO).fillna("other")
    out["state_h"] = df["state"].astype(str).str.upper().map(_STATE_ARGUS).fillna("other")
    out["label"] = df["attack"].astype(int).to_numpy()
    return out


def build_harmonized_unsw(df: pd.DataFrame) -> pd.DataFrame:
    """UNSW-NB15 (Argus) -> spazio armonizzato ricco.

    Stesso strumento di cattura di BoT-IoT, quindi la corrispondenza e'
    diretta e non richiede nessuna decisione nuova: sbytes/dbytes sono gia'
    byte a livello IP, header inclusi, come src_ip_bytes/dst_ip_bytes di
    Zeek. La mappa degli stati e' la stessa di Argus, estesa ai pochi valori
    in piu' che UNSW-NB15 usa.
    """
    out = _derive(
        duration=_num(df, "dur"),
        b_src=_num(df, "sbytes"),
        b_dst=_num(df, "dbytes"),
        p_src=_num(df, "spkts"),
        p_dst=_num(df, "dpkts"),
    )
    out["proto_h"] = df["proto"].astype(str).str.lower().map(_PROTO).fillna("other")
    out["state_h"] = df["state"].astype(str).str.upper().map(_STATE_ARGUS).fillna("other")
    out["label"] = df["label"].astype(int).to_numpy()
    return out


def coverage_report(h: pd.DataFrame, name: str) -> pd.DataFrame:
    """Distribuzione delle categoriche armonizzate: serve a leggere il
    degrado cross-domain prima ancora di addestrare qualcosa."""
    rows = []
    for c in HARMONIZED_CATEGORICAL:
        vc = h[c].value_counts(normalize=True)
        for k, v in vc.items():
            rows.append({"dataset": name, "feature": c, "valore": k, "frazione": round(float(v), 6)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# SPAZIO RIDOTTO — il massimo comune denominatore con CIC-IoT-2023
# ─────────────────────────────────────────────────────────────
# CIC-IoT-2023 non riporta i conteggi direzionali: non esistono
# src_bytes/dst_bytes ne' src_pkts/dst_pkts, e non c'e' uno stato di
# connessione. Le sue feature sono durata, statistiche sulla dimensione dei
# pacchetti, tassi e conteggi di flag TCP.
#
# Sette delle tredici feature numeriche ricche dipendono dalla direzione (le
# due asimmetrie, i due payload medi, i quattro conteggi per direzione) e
# vanno perse. Restano sei quantita' che TON_IoT, BoT-IoT, UNSW-NB15 e
# CIC-IoT-2023 misurano davvero tutti e quattro.
#
# NON si ricostruiscono le direzioni da Srate/Drate: sarebbe una stima
# spacciata per misura. Meglio uno spazio piu' povero ma onesto.
RIDOTTO_NUMERIC = [
    "duration", "bytes_total", "pkts_total",
    "payload_mean", "flow_rate", "byte_rate",
]

RIDOTTO_SKEWED = {
    "duration", "bytes_total", "pkts_total",
    "payload_mean", "flow_rate", "byte_rate",
}

_PROTO_CIC = ["TCP", "UDP", "ICMP", "ARP"]


def _state_da_flag(df: pd.DataFrame) -> np.ndarray:
    """Stato della connessione per CIC-IoT-2023, dalla stessa nozione
    semantica usata per Zeek e Argus: chi ha chiuso, chi ha resettato, chi
    non ha mai risposto. Decisione a priori, non tarata sui dati."""
    def col(*nomi):
        for n in nomi:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce").fillna(0.0).to_numpy(np.float64)
        return np.zeros(len(df))

    rst = col("rst_count", "rst_flag_number")
    fin = col("fin_count", "fin_flag_number")
    syn = col("syn_count", "syn_flag_number")
    ack = col("ack_count", "ack_flag_number")
    tcp = col("TCP")

    out = np.full(len(df), "other", dtype=object)
    non_tcp = tcp <= 0
    out[(syn > 0) & (ack <= 0)] = "incomplete"
    out[(syn > 0) & (ack > 0)] = "established"
    out[fin > 0] = "closed"
    out[rst > 0] = "reset"
    out[non_tcp] = "incomplete"
    return out


def _cic_label(df: pd.DataFrame) -> np.ndarray:
    """0/1 dalla colonna label della famiglia CIC (gia' normalizzata a 0/1
    da load_cic; robusta anche su un dataframe grezzo non passato di li')."""
    lab = df["label"] if "label" in df.columns else df["Label"]
    if lab.dtype == object:
        return (~lab.astype(str).str.lower().str.startswith("benign")).astype(int).to_numpy()
    return lab.astype(int).to_numpy()


def _derive_ridotto(duration, b_tot, p_tot) -> pd.DataFrame:
    return pd.DataFrame({
        "duration": duration,
        "bytes_total": b_tot,
        "pkts_total": p_tot,
        "payload_mean": b_tot / (p_tot + 1.0),
        "flow_rate": p_tot / (duration + _EPS),
        "byte_rate": b_tot / (duration + _EPS),
    })


def build_ridotto_da_ricco(h: pd.DataFrame) -> pd.DataFrame:
    """Proietta lo spazio ricco (TON_IoT, BoT-IoT, UNSW-NB15) su quello
    ridotto: stesse righe, stessi modelli, solo meno colonne, cosi' la
    differenza di risultato e' attribuibile solo alla riduzione."""
    out = _derive_ridotto(h["duration"].to_numpy(),
                          h["bytes_total"].to_numpy(),
                          h["pkts_total"].to_numpy())
    out["proto_h"] = h["proto_h"].to_numpy()
    out["state_h"] = h["state_h"].to_numpy()
    out["label"] = h["label"].to_numpy()
    return out


def build_ridotto_cic(df: pd.DataFrame) -> pd.DataFrame:
    """CIC-IoT-2023 -> spazio ridotto.

    Corrispondenze (nessuna e' una stima: tutte richiedono la colonna vera,
    sollevano KeyError se assente invece di riempire con un valore inventato):
      duration    <- flow_duration  (genuina in test.csv: mediana benigni
                     26,1 s contro 0,0 s per gli attacchi, non il TTL)
      pkts_total  <- Number          (numero di pacchetti del flusso)
      bytes_total <- Tot sum         (somma delle dimensioni dei pacchetti)
      proto_h     <- indicatori TCP/UDP/ICMP/ARP
      state_h     <- conteggi dei flag TCP (vedi _state_da_flag)
    """
    def col(*nomi):
        for n in nomi:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce").fillna(0.0).to_numpy(np.float64)
        raise KeyError(f"nessuna di {nomi} presente in CIC-IoT-2023")

    dur = col("flow_duration")
    pk = col("Number", "Tot size")
    bt = col("Tot sum")
    out = _derive_ridotto(dur, bt, pk)

    proto = np.full(len(df), "other", dtype=object)
    for nome in _PROTO_CIC:
        if nome in df.columns:
            m = pd.to_numeric(df[nome], errors="coerce").fillna(0.0).to_numpy() > 0
            proto[m] = _PROTO.get(nome.lower(), "other")
    out["proto_h"] = proto
    out["state_h"] = _state_da_flag(df)
    out["label"] = _cic_label(df)
    return out

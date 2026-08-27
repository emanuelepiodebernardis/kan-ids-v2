"""La KAN single-layer spiega le proprie decisioni per costruzione.

La richiesta (Prof. Kuznetsov, punto 7)
=======================================
"Sfruttare l'interpretabilita' della KAN single-layer: una figura semplice con
le funzioni apprese per feature e due o tre esempi di contributi locali al
logit — una spiegazione diretta, non post-hoc. Mantenere una formulazione piu'
prudente per il multi-layer."

Perche' qui "diretta" si puo' dire davvero
==========================================
Il kernel deployato, `mcu_pio/include/kan14_coeff_infer.h`, calcola

    logit = somma_i  ((acc_i * KC_MULT[i]) >> 15)          10 edge numerici
          + somma_j  (KC_CAT[off_j + c_j] * KC_CAT_MULT[j] * 6)   4 categorici

e non c'e' nient'altro: nessun termine di interazione, nessun bias residuo.
La scomposizione per feature non e' una *stima* del contributo — come sono
SHAP, LIME o le mappe di salienza, che approssimano una funzione opaca con un
modello locale — ma sono **gli addendi stessi della somma che il
microcontrollore esegue**. Sommandoli si riottiene il logit bit per bit, e
`tests/test_interpretabilita.py` lo verifica sui 200 vettori reali
confrontando con il kernel C compilato.

E' questa la differenza fra "spiegazione diretta" e "post-hoc", ed e' anche il
motivo per cui la stessa cosa NON si puo' dire del multi-layer: li' il secondo
strato vede combinazioni delle unita' nascoste, quindi il contributo di una
feature dipende dalle altre e una scomposizione additiva esatta non esiste.
Per quel modello questo file non produce niente, di proposito.

Tutto quello che serve sta negli header committati: i coefficienti e i 200
vettori di verifica. Non serve il dataset.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

Q12 = 1 << 12


# ─────────────────────────────────────────────────────────────
# lettura dell'header committato
# ─────────────────────────────────────────────────────────────
def _blocco(testo: str, nome: str) -> str:
    inizio = testo.index(f" {nome}[")
    inizio = testo.index("= {", inizio) + 2
    livello, i = 0, inizio
    while True:
        if testo[i] == "{":
            livello += 1
        elif testo[i] == "}":
            livello -= 1
            if livello == 0:
                break
        i += 1
    return testo[inizio:i + 1]


def _interi(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _define(testo: str, nome: str) -> int:
    return int(re.search(rf"#define\s+{nome}\s+(\d+)", testo).group(1))


def leggi_modello(header: Path) -> dict:
    """Coefficienti e moltiplicatori della KAN single-layer, dall'header C.

    Si legge l'header e non il checkpoint: il modello che spiega le proprie
    decisioni deve essere quello che gira sulla scheda, non un suo parente in
    virgola mobile addestrato con lo stesso script.
    """
    testo = Path(header).read_text(encoding="utf-8")
    nfeat = _define(testo, "KC_NFEAT")
    nseg = _define(testo, "KC_NSEG")
    ncat = _define(testo, "KC_NCAT")
    coef = np.array(_interi(_blocco(testo, "KC_COEF")), dtype=np.int64)
    coef = coef.reshape(nfeat, -1)
    return {
        "NFEAT": nfeat, "NSEG": nseg, "NCAT": ncat,
        "COEF": coef,
        "MULT": np.array(_interi(_blocco(testo, "KC_MULT")), dtype=np.int64),
        "CAT": np.array(_interi(_blocco(testo, "KC_CAT")), dtype=np.int64),
        "CAT_OFF": np.array(_interi(_blocco(testo, "KC_CAT_OFF")), dtype=np.int64),
        "CAT_MULT": np.array(_interi(_blocco(testo, "KC_CAT_MULT")), dtype=np.int64),
    }


def leggi_vettori(header: Path) -> dict:
    """I 200 flussi reali di verifica: ingressi Q12, codici, predizione attesa
    dalla simulazione bit-fedele ed etichetta vera."""
    testo = Path(header).read_text(encoding="utf-8")
    n = _define(testo, "KTV_N")
    return {
        "X": np.array(_interi(_blocco(testo, "KTV_X")), dtype=np.int64).reshape(n, -1),
        "CAT": np.array(_interi(_blocco(testo, "KTV_CAT")), dtype=np.int64).reshape(n, -1),
        "ATTESA": np.array(_interi(_blocco(testo, "KTV_EXPECTED")), dtype=np.int64),
        "LABEL": np.array(_interi(_blocco(testo, "KTV_LABEL")), dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────
# i contributi: gli addendi della somma che il kernel esegue
# ─────────────────────────────────────────────────────────────
def contributo_numerico(m: dict, i: int, xq) -> np.ndarray:
    """Il termine dell'edge numerico `i`, con l'aritmetica del kernel C.

    Riga per riga la traduzione di `kan14_coeff_logit`. Gli spostamenti a
    destra su interi Python sono aritmetici come in C, quindi il risultato e'
    lo stesso anche sui negativi.
    """
    xq = np.asarray(xq, dtype=np.int64)
    xi = np.clip(xq + 4096, 0, 8192)
    u = xi * m["NSEG"]
    seg = np.minimum(u >> 13, m["NSEG"] - 1)
    t = (u - (seg << 13)) << 2
    om = 32768 - t
    b0 = (((om * om) >> 15) * om) >> 15
    t2 = (t * t) >> 15
    t3 = (t2 * t) >> 15
    b1 = 3 * t3 - 6 * t2 + (4 << 15)
    b2 = -3 * t3 + 3 * t2 + 3 * t + (1 << 15)
    b3 = t3
    c = m["COEF"][i]
    acc = b0 * c[seg] + b1 * c[seg + 1] + b2 * c[seg + 2] + b3 * c[seg + 3]
    return (acc * m["MULT"][i]) >> 15


def contributo_categorico(m: dict, j: int, codice) -> np.ndarray:
    codice = np.asarray(codice, dtype=np.int64)
    return m["CAT"][m["CAT_OFF"][j] + codice] * m["CAT_MULT"][j] * 6


def contributi(m: dict, xq, cat) -> tuple[np.ndarray, np.ndarray]:
    """(contributi numerici, contributi categorici) per ogni riga di ingresso."""
    xq = np.atleast_2d(np.asarray(xq, dtype=np.int64))
    cat = np.atleast_2d(np.asarray(cat, dtype=np.int64))
    num = np.stack([contributo_numerico(m, i, xq[:, i])
                    for i in range(m["NFEAT"])], axis=1)
    ctg = np.stack([contributo_categorico(m, j, cat[:, j])
                    for j in range(m["NCAT"])], axis=1)
    return num, ctg


def logit(m: dict, xq, cat) -> np.ndarray:
    """Il logit come somma dei suoi addendi. Non e' una ricostruzione: e'
    la stessa somma, negli stessi interi, dello stesso ordine di grandezza."""
    num, ctg = contributi(m, xq, cat)
    return num.sum(axis=1) + ctg.sum(axis=1)


# ─────────────────────────────────────────────────────────────
# le funzioni apprese, sul dominio degli ingressi
# ─────────────────────────────────────────────────────────────
def curva(m: dict, i: int, punti: int = 401, clip: float = 3.5):
    """La funzione appresa sull'edge `i`: (valore normalizzato, contributo).

    L'ascissa e' la feature dopo la trasformazione quantile-normale e il clip
    a +/-clip, cioe' lo spazio in cui il modello lavora; l'ordinata e' il
    contributo al logit nelle unita' intere del kernel.
    """
    xq = np.linspace(-Q12, Q12, punti).round().astype(np.int64)
    return xq / Q12 * clip, contributo_numerico(m, i, xq)


def tabella_categorica(m: dict, j: int) -> np.ndarray:
    """I contributi di tutti i codici della feature categorica `j`.
    Lo slot 0 e' UNK, la categoria mai vista in training."""
    n = int(m["CAT_OFF"][j + 1] - m["CAT_OFF"][j]) if j + 1 < len(m["CAT_OFF"]) \
        else int(len(m["CAT"]) - m["CAT_OFF"][j])
    return contributo_categorico(m, j, np.arange(n))


def escursione(m: dict, xq, cat) -> list[dict]:
    """Quanto ciascun edge puo' muovere il logit sui dati osservati.

    Non e' una "feature importance" stimata: e' l'escursione effettiva del
    termine additivo su quei campioni. Ordina gli edge per quanto contano
    davvero nella somma, e si legge senza scomodare alcun modello surrogato.
    """
    num, ctg = contributi(m, xq, cat)
    fuori = []
    for i in range(num.shape[1]):
        v = num[:, i]
        fuori.append({"tipo": "numerica", "indice": i,
                      "min": int(v.min()), "max": int(v.max()),
                      "escursione": int(v.max() - v.min()),
                      "media": float(v.mean())})
    for j in range(ctg.shape[1]):
        v = ctg[:, j]
        fuori.append({"tipo": "categorica", "indice": j,
                      "min": int(v.min()), "max": int(v.max()),
                      "escursione": int(v.max() - v.min()),
                      "media": float(v.mean())})
    return fuori

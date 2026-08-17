"""Aggiornamento integer-only dei guadagni per edge.

Il kernel intero gia' deployato calcola

    z = somma_i  (acc_i * MULT[i]) >> 15

dove acc_i e' l'accumulatore della spline sull'edge i e MULT[i] e' un
moltiplicatore in Q15. L'adattamento al drift consiste esattamente nel
riscrivere quei moltiplicatori piu' un termine noto: **non serve cambiare
il firmware**, solo una tabella. Per il modello a 10 edge numerici e 2
categorici sono 13 valori int16, cioe' 26 byte.

Qui c'e' il riferimento bit-fedele di due cose:

  int_forward   il calcolo del logit con i guadagni applicati
  fit_gains_int la stima dei guadagni dalle etichette raccolte sul campo,
                anch'essa senza una sola operazione in virgola mobile

La stima float (regressione logistica) resta il termine di paragone: se
l'aritmetica intera perdesse accuratezza, si vedrebbe confrontando le due.
"""
from __future__ import annotations

import numpy as np

Q15 = 1 << 15
SIG_BITS = 8                 # 2^8 = 256 voci nella LUT della sigmoide
SIG_RANGE = 8 * Q15          # la LUT copre z in [-8, 8] in Q15

# sigmoide in Q15 su una griglia uniforme: unica tabella, 512 byte
_grid = (np.arange(1 << SIG_BITS) - (1 << (SIG_BITS - 1))) * (
    2 * SIG_RANGE / (1 << SIG_BITS)) / Q15
SIG_LUT = np.round(1.0 / (1.0 + np.exp(-_grid)) * Q15).astype(np.int64)


def isigmoid(z_q15):
    """Sigmoide in Q15 per argomento in Q15, via LUT a 256 voci.

    Fuori dall'intervallo [-8, 8] la sigmoide vale 0 o 1 a meno di 3e-4,
    quindi la saturazione non e' un'approssimazione osservabile.
    """
    z = np.asarray(z_q15, dtype=np.int64)
    idx = ((z + SIG_RANGE) * (1 << SIG_BITS)) // (2 * SIG_RANGE)
    idx = np.clip(idx, 0, (1 << SIG_BITS) - 1)
    return SIG_LUT[idx]


def int_forward(P, gains_q15, bias):
    """z = somma_i (P[:, i] * gains[i]) >> 15 + bias, tutto in int64.

    P sono i contributi interi degli edge, cioe' esattamente cio' che il
    kernel calcola prima di applicare MULT.
    """
    P = np.asarray(P, dtype=np.int64)
    g = np.asarray(gains_q15, dtype=np.int64)
    return ((P * g[None, :]) >> 15).sum(1) + np.int64(bias)


def _floor_shift(v, s):
    """Shift aritmetico a destra: divisione con arrotondamento verso -inf,
    che e' cio' che fa >> sugli interi con segno in C su ogni target reale."""
    return np.asarray(v, dtype=np.int64) >> s


def fit_gains_int(P, y, mult=None, iters: int = 2000, lr_shift: int | None = None,
                  scale_bits: int = 12, reg_shift: int | None = None,
                  reg_target: int = Q15, lr_decay: bool = False,
                  adaptive: bool = False, max_iters: int = 20000,
                  window: int = 100, patience: int = 5,
                  return_iters: bool = False):
    """Stima i 13 guadagni dalle etichette raccolte, in aritmetica intera.

    Discesa del gradiente sulla log-verosimiglianza pesata per classe,
    partendo da guadagni unitari (32768 = 1.0), cioe' dal modello com'e'.
    Se le etichette non dicono nulla, i guadagni restano dove sono: e'
    una proprieta' desiderabile su un dispositivo.

    `scale_bits` porta i contributi degli edge in un intervallo in cui il
    prodotto per il guadagno non trabocca int32; il fattore viene poi
    riassorbito nei guadagni finali, quindi non cambia il risultato.

    `reg_shift` aggiunge un termine di weight decay verso l'identita'
    (guadagno=Q15, cioe' "non toccare l'edge"): a ogni passo si somma al
    gradiente `(g - Q15) >> reg_shift`. La controparte float (sklearn
    LogisticRegression) ha L2 di default (C=1.0) e questa non ne aveva
    nessuna: misurato che non chiude il divario in TON->BoT, la causa vera
    era non convergenza (sezione 16.1). None disattiva la regolarizzazione
    (comportamento previo, per compatibilita').

    `adaptive=True` sostituisce il numero fisso di iterazioni e il passo
    fisso con due meccanismi (COMPITO 3a):

    1. **Passo adattivo**: il passo iniziale (`lr_shift`) resta scelto dal
       primo gradiente come prima, ma ogni `window` iterazioni si confronta
       la perdita pesata corrente con il minimo osservato finora. Se non e'
       migliorata, il passo si dimezza (`lr_shift += 1`). Prima diagnosi
       (sezione 16.1): con un passo fisso scelto una volta sola, il seed
       piu' lento misurato non era ancora convergente **a 20 000
       iterazioni** (perdita ancora in discesa, un edge saturato al clip e
       gli altri che si muovevano lentissimi) — non un problema di quante
       iterazioni fare, ma di un passo troppo piccolo per la curvatura di
       quel problema specifico. Il passo che si dimezza solo quando serve
       lascia i passi grandi dove la discesa e' ancora ripida.
    2. **Arresto sulla convergenza**: se un aggiornamento lascia g e b
       **esattamente invariati**, la ricorrenza (deterministica: stessi
       P, y, mult) si e' fermata su un punto fisso e ogni iterazione
       successiva ripeterebbe lo stesso risultato — non serve una
       tolleranza. Si ferma anche se la perdita non migliora per
       `patience` finestre consecutive anche dopo aver dimezzato il passo
       (arresto pratico quando il progresso e' diventato trascurabile,
       non necessariamente nullo).

    `max_iters` resta un tetto di sicurezza. Con `return_iters=True` la
    funzione restituisce anche il numero di iterazioni usate: e' la base
    del modello di costo (COMPITO 3c), perche' il costo in
    moltiplicazioni-accumulo dipende linearmente da questo numero e con il
    passo fisso era un parametro costante scelto a priori, ora e' una
    misura per seed.
    """
    P = np.asarray(P, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    n, d = P.shape

    # riscalatura per edge: |P| < 2^scale_bits, con shift interi
    amax = np.maximum(np.abs(P).max(0), 1)
    sh = np.maximum(np.array([int(a).bit_length() for a in amax]) - scale_bits, 0)
    Ps = P >> sh[None, :]

    # pesi di classe interi: 0.5/frequenza, in Q15
    # peso di classe (interi in Q15) x molteplicita' del pattern. La
    # molteplicita' e' quante righe dello stream collassano su quel pattern
    # dopo la quantizzazione: un contatore intero che il dispositivo tiene
    # gratis mentre deduplica.
    mult = np.ones(n, np.int64) if mult is None else np.asarray(mult, np.int64)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(n - y.sum()), 1)
    w_pos = (Q15 * n) // (2 * n_pos)
    w_neg = (Q15 * n) // (2 * n_neg)
    w = np.where(y == 1, w_pos, w_neg) * mult
    w = (w * Q15) // max(int(w.max()), 1)      # rinormalizzati in Q15

    g = np.full(d, Q15, dtype=np.int64)
    b = np.int64(0)
    lr0 = lr_shift
    n_loop = max_iters if adaptive else iters
    used = n_loop
    best_loss = None
    bad_windows = 0
    for it in range(n_loop):
        z = ((Ps * g[None, :]) >> 15).sum(1) + b
        p = isigmoid(z)                       # Q15
        err = ((p - y * Q15) * w) >> 15       # Q15, pesato
        grad = (Ps * err[:, None]).sum(0) // n
        if lr0 is None and lr_shift is None:
            # passo scelto una volta sola, in interi: il primo aggiornamento
            # non deve muovere i guadagni di piu' di ~Q15/8
            mx = max(int(np.abs(grad).max()), 1)
            lr_shift = max(int(mx).bit_length() - 12, 0)
        if reg_shift is not None:
            grad = grad + ((g - reg_target) >> reg_shift)
        step = lr_shift + 1 if (lr_decay and it >= iters // 2) else lr_shift
        g_new = np.clip(g - _floor_shift(grad, step), -(1 << 20), (1 << 20))
        b_new = b - _floor_shift(err.sum() // n, step)
        if adaptive and np.array_equal(g_new, g) and b_new == b:
            g, b = g_new, b_new
            used = it + 1
            break
        g, b = g_new, b_new
        if adaptive and (it + 1) % window == 0:
            loss = int((w * np.abs(err)).sum() // n)
            if best_loss is not None and loss >= best_loss:
                bad_windows += 1
                lr_shift += 1                  # dimezza il passo
            else:
                bad_windows = 0
            best_loss = loss if best_loss is None else min(best_loss, loss)
            if bad_windows >= patience:
                used = it + 1
                break

    if return_iters:
        return (g >> sh), int(b), used
    # riassorbe la riscalatura: il guadagno effettivo su P non riscalato
    return (g >> sh), int(b)


def gains_to_mult(mult_q15, gains_q15):
    """Nuovi moltiplicatori da scrivere in E2E_MULT: (MULT * gain) >> 15."""
    m = np.asarray(mult_q15, dtype=np.int64)
    g = np.asarray(gains_q15, dtype=np.int64)
    return (m * g) >> 15


# ─────────────────────────────────────────────────────────────
# MARTINGALA CONFORMAL, IN INTERI
# ─────────────────────────────────────────────────────────────
# scripts/drift_graduale.py misura che l'innesco a martingala di potenza
# (Vovk) recupera quasi tutto il valore del riadattamento continuo con
# molte meno etichette, ma gira in virgola mobile: log(), potenza frazionaria
# ed p-value randomizzati con un RNG float. Qui lo stesso principio in
# interi, con lo stesso trucco della sigmoide: log e potenza si calcolano
# UNA VOLTA, offline, per costruire una LUT; a runtime restano solo
# confronti fra interi e una somma.
MART_Q_BITS = 16
MART_Q = 1 << MART_Q_BITS


def build_martingale_lut(n_cal: int, eps: float = 0.9, n_bins: int = 256) -> np.ndarray:
    """LUT dell'incremento di log-martingala per rango binnato.

    Il rango r in [0, n_cal] (quante calibrazioni hanno conformita' >= alla
    riga corrente) si binna in n_bins prima del lookup -- 256 voci come la
    LUT della sigmoide, invece delle migliaia di righe di calibrazione.
    Il p-value e' "mid-p" deterministico, (r+0.5)/(n_cal+1): sostituisce la
    randomizzazione dei pareggi del rifermento float (che userebbe un RNG
    float a runtime) con una scelta fissa. E' una semplificazione dichiarata,
    non una fedelta' bit-esatta al float: si misura quanto costa.
    """
    lut = np.empty(n_bins, dtype=np.int64)
    for b in range(n_bins):
        # r_mid rango medio del bin: r grande = molte calibrazioni con
        # conformita' >= alla riga -> riga tipica -> p vicino a 1.
        # r piccolo = poche -> riga anomala -> p vicino a 0. Bug misurato:
        # la prima versione usava (n_cal - r_mid), invertendo tipico e
        # anomalo -- la martingala scendeva proprio quando la deriva
        # cresceva (verificato su ton->bot: d_logM sempre piu' negativo
        # da batch 0 a batch 19, il contrario di quello che deve fare).
        r_mid = (b + 0.5) * (n_cal + 1) / n_bins
        p = np.clip((r_mid + 0.5) / (n_cal + 1), 1e-9, 1.0)
        incr = np.log(eps) + (eps - 1.0) * np.log(p)
        lut[b] = int(round(incr * MART_Q))
    return lut


def conformal_scores_int(z_int, med_cal: int):
    """Conformita' intera: |z - mediana della calibrazione|. La mediana e'
    calcolata una volta sola in fase di calibrazione (int, nessun float)."""
    return np.abs(np.asarray(z_int, dtype=np.int64) - np.int64(med_cal))


def martingale_batch_int(s_batch, s_cal_sorted, lut, n_bins: int = 256) -> int:
    """Incremento di log-martingala per un intero batch, in Q16 interi.

    s_cal_sorted: conformita' di calibrazione, ordinate (int64), fissate a
    calibrazione. Il rango e' un confronto fra interi (searchsorted su int64
    e' esatto, a differenza del confronto fra punteggi quantizzati grezzi
    della sezione 6: qui il "pareggio" e' gia' assorbito dal binning).
    """
    n_cal = len(s_cal_sorted)
    r = n_cal - np.searchsorted(s_cal_sorted, np.asarray(s_batch, dtype=np.int64), side="left")
    b = np.clip((r.astype(np.int64) * n_bins) // (n_cal + 1), 0, n_bins - 1)
    return int(lut[b].sum())


def martingale_update_int(logM: int, d_logM: int, floor: int = 0) -> int:
    """Un passo della martingala: somma l'incremento del batch, pavimento a
    zero. Stesso accorgimento del riferimento float (sezione 9): senza il
    pavimento, una deriva negativa sotto scambiabilita' affonda la
    statistica cosi' in basso che nessuna deriva la recupera piu'."""
    return max(floor, logM + d_logM)


def martingale_threshold_int(evidenza: float = 100.0) -> int:
    """Soglia in Q16 interi, equivalente a log(evidenza) in float."""
    return int(round(np.log(evidenza) * MART_Q))

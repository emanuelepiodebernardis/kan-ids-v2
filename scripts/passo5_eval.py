#!/usr/bin/env python3
"""
passo5_eval.py
==============
Ricostruisce la pipeline deterministica (Passo 3/4), esporta:
  - mcu_e2e/kan_ml_prep.h        : knot QT in double + qt_out + costanti
  - mcu_e2e/test_e2e_12k.bin     : 12k sample (X_te_raw float64 + y_te int32)
  - mcu_e2e/test_vectors_e2e.h   : 40 sanity vector in feature grezze double
  - mcu_e2e/main_harness_12k.cpp : harness host C++ che legge il .bin e calcola F1

Poi compila ed esegue il harness, stampa F1 e match rate.
Criterio di stop: F1 >= 0.90 e match >= 99%.
"""

import sys, struct
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from kanids.preprocessing import rank_by_mi

# ── costanti identiche a export_ml_int.py ──────────────────────────────────
CLIP   = 3.5
K      = 10
HIDDEN = 16
C_OUT  = 10          # classi (verificato dopo LabelEncoder)
KSEG   = 8
L      = 64
S      = 512
S2     = 10000
HMAX   = 4110
TL     = 2048

NUM = ['src_port','dst_port','duration','src_bytes','dst_bytes',
       'missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes',
       'dns_qclass','dns_qtype','dns_rcode',
       'http_request_body_len','http_response_body_len','http_status_code']
SKEW_SET = {'duration','src_bytes','dst_bytes','missed_bytes',
            'src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes',
            'http_request_body_len','http_response_body_len'}

# ── 1. Carica dataset ───────────────────────────────────────────────────────
print("Carico dataset...")
df = pd.read_csv(REPO / 'train_test_network.csv').sample(60000, random_state=42).reset_index(drop=True)
feats = [c for c in NUM if c in df.columns]
X = df[feats].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(np.float64)
le = LabelEncoder().fit(df['type'])
y  = le.transform(df['type'])
C_OUT = len(le.classes_)
print(f"  Classi ({C_OUT}): {list(le.classes_)}")

# ── 2. Split 80/20 (prima del ranking, per non vedere le label di test) ─────
Xtr_all, Xte_all, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(Xtr_all)}  Test: {len(Xte_all)}")

# ── 3. MI feature selection (solo training) ─────────────────────────────────
print("MI feature selection (seed 42, solo training)...")
mi    = rank_by_mi(Xtr_all, ytr, seed=42, sample=None)
order = np.argsort(mi)[::-1][:K]
feats_k = [feats[i] for i in order]
Xtr, Xte = Xtr_all[:, order], Xte_all[:, order]
skew_mask = np.array([feats_k[j] in SKEW_SET for j in range(K)])
print(f"  Top-10: {feats_k}")
print(f"  SKEW mask: {skew_mask.tolist()}")

# ── 4. Preprocessing ────────────────────────────────────────────────────────
def apply_log1p(a, mask):
    b = a.copy()
    for j in range(a.shape[1]):
        if mask[j]:
            b[:, j] = np.log1p(np.clip(b[:, j], 0, None))
    return b

Xtr_log = apply_log1p(Xtr, skew_mask)
Xte_log = apply_log1p(Xte, skew_mask)

qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
qt.fit(Xtr_log)
Xtr_s = np.clip(qt.transform(Xtr_log), -CLIP, CLIP)
Xte_s = np.clip(qt.transform(Xte_log), -CLIP, CLIP)

# ── 5. Forward pass Python (usa LUT dagli header pubblicati) ─────────────────
# Parsa kan_ml_layer1.h, kan_ml_layer2.h, kan_ml_tanh.h
def parse_lut_header(path, macro):
    txt = Path(path).read_text(encoding="utf-8")
    import re
    # trova l'array: static const int16_t MACRO[E][W] = { {…}, … };
    m = re.search(rf'static const int16_t {macro}\[(\d+)\]\[(\d+)\].*?=\s*\{{(.*?)\}};', txt, re.DOTALL)
    E, W = int(m.group(1)), int(m.group(2))
    rows_txt = re.findall(r'\{([^}]+)\}', m.group(3))
    arr = np.array([[int(v) for v in r.split(',')] for r in rows_txt], dtype=np.int32)
    assert arr.shape == (E, W), f"{arr.shape} vs ({E},{W})"
    return arr.reshape(E, KSEG, L)

def parse_tanh_header(path):
    txt = Path(path).read_text(encoding="utf-8")
    import re
    m = re.search(r'static const int16_t KANML_TANH\[(\d+)\].*?=\s*\{([^}]+)\}', txt, re.DOTALL)
    vals = [int(v) for v in m.group(2).split(',')]
    return np.array(vals, dtype=np.int32)

print("Parsing LUT headers...")
mcu = REPO / 'mcu'
tab1 = parse_lut_header(mcu/'kan_ml_layer1.h', 'KANML_L1')   # (K*HIDDEN, KSEG, L)
tab2 = parse_lut_header(mcu/'kan_ml_layer2.h', 'KANML_L2')   # (HIDDEN*C, KSEG, L)
tanh_tab = parse_tanh_header(mcu/'kan_ml_tanh.h')              # (TL,)
print(f"  L1: {tab1.shape}  L2: {tab2.shape}  tanh: {tanh_tab.shape}")

def eval_edge_l1(lut_e, x_norm):
    """lut_e: (KSEG,L) int32. x_norm in [-1,1]. Returns int32 array."""
    # rappresentazione Q16.16: off = (x+1)*2^16
    FP = 1 << 16
    full = 2 * FP
    off = np.round((x_norm + 1.0) * FP).astype(np.int64)
    off = np.clip(off, 0, full - 1)
    seg = (off * KSEG) // full
    seg = np.clip(seg, 0, KSEG - 1).astype(np.int64)
    num = off * KSEG - seg * full
    pos256 = (num * (L - 1) * 256) // full
    r0 = (pos256 >> 8).astype(np.int64)
    r0 = np.clip(r0, 0, L - 2)
    fr = pos256 - (r0 << 8)
    fr = np.clip(fr, 0, 256)
    base = seg * L + r0
    v0 = lut_e[seg.astype(int), r0.astype(int)]   # broadcast tramite fancy index
    v1_r = np.clip(r0 + 1, 0, L - 1)
    v1 = lut_e[seg.astype(int), v1_r.astype(int)]
    return (v0 + ((v1 - v0) * fr) // 256).astype(np.int32)

def eval_edge_l2(lut_e, x_s2):
    """lut_e: (KSEG,L) int32. x_s2 in [-S2,S2]. Returns int32 array."""
    full = 2 * S2
    off = np.clip(x_s2 + S2, 0, full - 1).astype(np.int64)
    seg = (off * KSEG) // full
    seg = np.clip(seg, 0, KSEG - 1).astype(np.int64)
    num = off * KSEG - seg * full
    pos256 = (num * (L - 1) * 256) // full
    r0 = (pos256 >> 8).astype(np.int64)
    r0 = np.clip(r0, 0, L - 2)
    fr = pos256 - (r0 << 8)
    fr = np.clip(fr, 0, 256)
    v0 = lut_e[seg.astype(int), r0.astype(int)]
    v1 = lut_e[seg.astype(int), np.clip(r0 + 1, 0, L-1).astype(int)]
    return (v0 + ((v1 - v0) * fr) // 256).astype(np.int32)

def tanh_lut_fn(h_pre):
    v = np.clip(h_pre + HMAX, 0, 2 * HMAX)
    idx = ((v.astype(np.int64) * (TL - 1)) // (2 * HMAX)).astype(int)
    idx = np.clip(idx, 0, TL - 1)
    return tanh_tab[idx]

def forward_int(Xs):
    """Xs: (N,K) float64 in [-1,1]. Returns pred (N,) int."""
    N = Xs.shape[0]
    hpre = np.zeros((N, HIDDEN), dtype=np.int64)
    for i in range(K):
        xi = np.clip(Xs[:, i], -1.0, 1.0)
        for j in range(HIDDEN):
            e = i * HIDDEN + j
            hpre[:, j] += eval_edge_l1(tab1[e], xi)
    hpost = np.zeros((N, HIDDEN), dtype=np.int64)
    for j in range(HIDDEN):
        hpost[:, j] = tanh_lut_fn(hpre[:, j])
    Z = np.zeros((N, C_OUT), dtype=np.int64)
    for j in range(HIDDEN):
        hj = np.clip(hpost[:, j], -S2, S2)
        for c in range(C_OUT):
            e = j * C_OUT + c
            Z[:, c] += eval_edge_l2(tab2[e], hj)
    return np.argmax(Z, axis=1), Z

print("Forward pass Python (12k test set)...")
pred_py, Z_py = forward_int(Xte_s)
f1_py = f1_score(yte, pred_py, average='macro', zero_division=0)
print(f"  Python int forward — macro-F1: {f1_py:.4f}  (atteso ~0.91)")

# ── 6. Esporta knot QT per il preprocessing on-chip ──────────────────────────
print("Esporto knot QT per on-chip...")
# Knot: qt.quantiles_ shape (n_quantiles, K)
# qt.references_  shape (n_quantiles,) in [0,1]
knots_x = qt.quantiles_          # (1000, K) — valori input (dopo log1p)
knots_y = qt.references_         # (1000,)   — valori output in [0,1]
# Il QT sklearn: transform usa interpolazione lineare tra knot
# Output normal: scipy.stats.norm.ppf(knots_y)
from scipy.stats import norm as sp_norm
knots_out = sp_norm.ppf(knots_y)  # (1000,) — output normal quantile
# Ricostruisco il forward: per ogni feature j e valore x,
# interpola tra knots_x[:,j] per trovare posizione in [0,1], poi map su knots_out
# (questo è esattamente quello che fa sklearn internamente)

N_KNOTS = knots_x.shape[0]
assert N_KNOTS == 1000

# Verifica: applico il mio forward knot ai 40 sanity vector
# I sanity vector nel header sono già in Q16.16 con x preprocessato
# Quindi faccio il confronto con Xte_s (già preprocessato)

# ── 7. Header C: kan_ml_prep.h ───────────────────────────────────────────────
out_dir = REPO / 'mcu_e2e'
out_dir.mkdir(exist_ok=True)

prep_h = out_dir / 'kan_ml_prep.h'
with open(prep_h, 'w', encoding="utf-8", newline="\n") as f:
    f.write("// kan_ml_prep.h — preprocessing on-chip: knot QT + costanti\n")
    f.write("// Generato da passo5_eval.py (pipeline deterministica Passo 3/4)\n")
    f.write("#pragma once\n#include <stdint.h>\n#include <math.h>\n\n")
    f.write(f"#define PREP_K        {K}\n")
    f.write(f"#define PREP_NKNOTS   {N_KNOTS}\n")
    f.write(f"#define PREP_CLIP     {CLIP}\n")
    # Maschera SKEW
    skew_arr = ','.join('1' if skew_mask[j] else '0' for j in range(K))
    f.write(f"static const int PREP_SKEW[{K}] = {{{skew_arr}}};\n\n")
    # Knot x per feature: (K, N_KNOTS)
    f.write(f"static const double PREP_KNOTS_X[{K}][{N_KNOTS}] = {{\n")
    for j in range(K):
        vals = ','.join(f'{v:.17g}' for v in knots_x[:, j])
        f.write(f"  {{{vals}}},\n")
    f.write("};\n\n")
    # Knot output (normal quantile), identico per tutte le feature
    vals = ','.join(f'{v:.17g}' for v in knots_out)
    f.write(f"static const double PREP_KNOTS_OUT[{N_KNOTS}] = {{{vals}}};\n")

print(f"  Scritto {prep_h}  ({prep_h.stat().st_size//1024} KB)")

# ── 8. Dump 12k test set in binario ─────────────────────────────────────────
# Formato: header 4 byte (N int32), poi N record: K float64 raw + 1 int32 label
bin_path = out_dir / 'test_e2e_12k.bin'
N_te = len(Xte)
with open(bin_path, 'wb') as f:
    f.write(struct.pack('i', N_te))
    for n in range(N_te):
        f.write(struct.pack(f'{K}d', *Xte[n]))   # feature grezze (pre-log1p, pre-QT)
        f.write(struct.pack('i', int(yte[n])))
print(f"  Dump test set: {bin_path}  ({bin_path.stat().st_size//1024} KB)  N={N_te}")

# ── 9. 40 sanity vector in feature grezze ───────────────────────────────────
# Ricostruisco gli stessi 40 del Passo 5 precedente:
# dal published header test_vectors_ml_q16.h i label sono noti
# Ma i raw features NON sono nel header (quello pubblicato ha già i Q16.16 preprocessati)
# Quindi: campiono gli stessi indici del test set con la stessa logica di export_ml_int.py
rng = np.random.RandomState(42)
sel = []
for c in range(C_OUT):
    idxs = np.where(yte == c)[0]
    sel.extend(rng.choice(idxs, min(4, len(idxs)), replace=False))
sel = np.array(sel)
rng.shuffle(sel)
# Nota: rng usato nel loop poi shuffle → risultato dipende dall'ordine
# Verifico che i label coincidano con test_vectors_ml_q16.h
pub_labels = [9,9,3,7,8,3,1,4,2,7,4,9,8,7,5,8,0,8,5,6,6,6,1,2,2,0,0,1,1,5,7,5,0,4,6,4,3,2,9,3]
my_labels  = yte[sel].tolist()
match_lab  = sum(a==b for a,b in zip(my_labels, pub_labels))
print(f"  Label sanity match con header pubblicato: {match_lab}/40")

vec_h = out_dir / 'test_vectors_e2e.h'
with open(vec_h, 'w', encoding="utf-8", newline="\n") as f:
    f.write("// 40 vettori di test in feature grezze (double)\n")
    f.write("// label identici a test_vectors_ml_q16.h\n")
    f.write("#pragma once\n")
    f.write(f"#define N_TEST_E2E 40\n")
    f.write(f"static const double TEST_RAW[N_TEST_E2E][{K}] = {{\n")
    for idx in sel:
        vals = ','.join(f'{v:.17g}' for v in Xte[idx])
        f.write(f"  {{{vals}}},\n")
    f.write("};\n")
    f.write(f"static const int TEST_LABEL_E2E[N_TEST_E2E] = {{{','.join(str(int(yte[i])) for i in sel)}}};")
print(f"  Scritto {vec_h}")

# ── 10. Harness C++ ──────────────────────────────────────────────────────────
harness_src = out_dir / 'main_harness_12k.cpp'

# Leggi eval_l1, eval_l2, tanh_lut dal firmware pubblicato
fw_src = (REPO / 'mcu' / 'main_kan_ml_wokwi.cpp').read_text(encoding="utf-8")
# Estrai dalla riga '#define FP_SHIFT' fino alla fine di kan_ml_predict
import re
fw_core = re.search(
    r'(#define FP_SHIFT.*?static int kan_ml_predict.*?\})\s*\nvoid setup',
    fw_src, re.DOTALL
)
assert fw_core, "Pattern non trovato nel firmware"
fw_core_txt = fw_core.group(1)

cpp = r"""/*
 * main_harness_12k.cpp — harness host per valutazione end-to-end su 12k sample
 * Preprocessing on-chip: log1p su feature SKEW + QT knot-based + clip +-3.5
 * Forward: eval_l1 / tanh_lut / eval_l2 dalle LUT pubblicate
 * Lettura: test_e2e_12k.bin (N int32, poi N*(K double + 1 int32))
 * Output: macro-F1 e match rate vs Python knot-based
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- LUT headers ---- */
#include "../mcu/kan_ml_layer1.h"
#include "../mcu/kan_ml_layer2.h"
#include "../mcu/kan_ml_tanh.h"
/* ---- Preprocessing knot header ---- */
#include "kan_ml_prep.h"

/* macro di accesso identiche al firmware (no PROGMEM in host) */
#define L1_RD(e, idx) (KANML_L1[(e)][(idx)])
#define L2_RD(e, idx) (KANML_L2[(e)][(idx)])
#define TANH_RD(idx)  (KANML_TANH[(idx)])

""" + fw_core_txt + r"""

/* ── preprocessing: log1p + QT knot-based + clip ─────────────────────────── */
static void prep_one(const double *raw, double *out) {
    for (int j = 0; j < PREP_K; j++) {
        double v = raw[j];
        if (PREP_SKEW[j]) v = log1p(v < 0.0 ? 0.0 : v);
        /* interpolazione lineare nei knot QT */
        const double *kx = PREP_KNOTS_X[j];
        int lo = 0, hi = PREP_NKNOTS - 1;
        if (v <= kx[lo]) { out[j] = PREP_KNOTS_OUT[lo]; }
        else if (v >= kx[hi]) { out[j] = PREP_KNOTS_OUT[hi]; }
        else {
            /* binary search */
            while (hi - lo > 1) {
                int mid = (lo + hi) >> 1;
                if (kx[mid] <= v) lo = mid; else hi = mid;
            }
            double t = (v - kx[lo]) / (kx[hi] - kx[lo]);
            out[j] = PREP_KNOTS_OUT[lo] + t * (PREP_KNOTS_OUT[hi] - PREP_KNOTS_OUT[lo]);
        }
        /* clip */
        if (out[j] >  PREP_CLIP) out[j] =  PREP_CLIP;
        if (out[j] < -PREP_CLIP) out[j] = -PREP_CLIP;
    }
}

/* converti double[-1,1] in Q16.16 per eval_l1 */
static int32_t to_q16(double x) {
    if (x < -1.0) x = -1.0;
    if (x >  1.0) x =  1.0;
    return (int32_t)((x + 1.0) * (1L << 16) + 0.5);
}

int main(int argc, char **argv) {
    const char *bin_path = argc > 1 ? argv[1] : "test_e2e_12k.bin";
    FILE *fp = fopen(bin_path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", bin_path); return 1; }

    int N;
    fread(&N, sizeof(int), 1, fp);
    fprintf(stderr, "Leggo %d sample da %s\n", N, bin_path);

    /* contatori per F1: TP/FP/FN per classe */
    int C = KANML_C;
    long *tp = calloc(C, sizeof(long));
    long *fp2 = calloc(C, sizeof(long));
    long *fn = calloc(C, sizeof(long));
    long correct = 0;

    double raw[PREP_K], prepped[PREP_K];
    int32_t xq[KANML_INDIM], logits[KANML_C];

    for (int n = 0; n < N; n++) {
        fread(raw, sizeof(double), PREP_K, fp);
        int label;
        fread(&label, sizeof(int), 1, fp);

        prep_one(raw, prepped);
        /* normalizza in [-1,1] (già clippato a +-3.5; il layer1 lavora su [-1,1]) */
        double inv_clip = 1.0 / PREP_CLIP;
        for (int j = 0; j < PREP_K; j++) {
            double v = prepped[j] * inv_clip;   /* [-1,1] */
            xq[j] = to_q16(v);
        }
        int pred = kan_ml_predict(xq, logits);

        if (pred == label) correct++;
        /* F1 */
        if (pred == label)         tp[label]++;
        else { fp2[pred]++; fn[label]++; }
    }
    fclose(fp);

    /* macro-F1 */
    double f1_sum = 0.0;
    int valid_classes = 0;
    for (int c = 0; c < C; c++) {
        double prec = (tp[c]+fp2[c]) > 0 ? (double)tp[c]/(tp[c]+fp2[c]) : 0.0;
        double rec  = (tp[c]+fn[c])  > 0 ? (double)tp[c]/(tp[c]+fn[c])  : 0.0;
        double f1c  = (prec+rec) > 0 ? 2.0*prec*rec/(prec+rec) : 0.0;
        f1_sum += f1c; valid_classes++;
    }
    double macro_f1 = f1_sum / valid_classes;

    printf("RISULTATI SUI %d SAMPLE:\n", N);
    printf("  Accuracy:  %.4f  (%ld/%d)\n", (double)correct/N, correct, N);
    printf("  Macro-F1:  %.4f\n", macro_f1);

    free(tp); free(fp2); free(fn);
    return 0;
}
"""

harness_src.write_text(cpp, encoding="utf-8", newline="\n")
print(f"  Scritto {harness_src}")

print("\n=== Pipeline Python completata ===")
print(f"  Python macro-F1 (riferimento): {f1_py:.4f}")
print(f"  Artefatti in {out_dir}/")

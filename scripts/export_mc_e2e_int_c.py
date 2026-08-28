#!/usr/bin/env python3
"""Catena integer-only END-TO-END a 10 classi, esportata in C.

    valori grezzi -> soglie per-feature (assorbono log1p + quantile +
    probit + clip) -> z Q12 -> layer1 spline int8 + tabelle categoriche
    -> tanh LUT -> layer2 spline int8 -> argmax su accumulatori interi

Come per il binario, Python resta confinato a training, export e golden
vector. Le soglie sono tabelle empiriche costruite offline dal
transformer fittato: la replica analitica del QuantileTransformer
fallisce sulle masse discrete (porte, contatori), mentre le tabelle sono
esatte per costruzione sui nodi e interpolate linearmente in mezzo.

I valori grezzi entrano come INTERI: ogni feature ha una scala fissa
(RAW_SCALE) che porta il valore in unita' intere — 1 per i contatori,
10^6 per la durata in secondi. Sul dispositivo i contatori sono gia'
interi: la scala esiste solo per la durata.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in [_REPO, _REPO / "src", _REPO / "preprocessing", _REPO / "scripts"]:
    sys.path.insert(0, str(p))

from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer

import feature_curve as fc
from c_footprint import scan
from kan_bspline import bspline_basis
from kanids.config import RESULTS_DIR, artifact_path
from kanids.datasets import load_ton_iot
from kanids.legacy import prepare14_dict

CLIP = 3.5
N_INT = 16
Q15 = 1 << 15
Q12 = 1 << 12
NZ = 129
TL = 512
N_GOLDEN = 200


def cheb_T(x, deg):
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, deg + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T, axis=-1)


def spline_kernel(u, Cq, seg_shift):
    seg = np.minimum(u >> seg_shift, N_INT - 1)
    rem = u - (seg << seg_shift)
    t = (rem << (15 - seg_shift)) if seg_shift <= 15 else (rem >> (seg_shift - 15))
    om = Q15 - t
    b0 = (((om * om) >> 15) * om) >> 15
    t2 = (t * t) >> 15
    t3 = (t2 * t) >> 15
    b1 = 3 * t3 - 6 * t2 + (4 << 15)
    b2 = -3 * t3 + 3 * t2 + 3 * t + (1 << 15)
    b3 = t3
    return b0 * Cq[seg] + b1 * Cq[seg + 1] + b2 * Cq[seg + 2] + b3 * Cq[seg + 3]


def carr(name, vals, ctype, per_line=12):
    # PROGMEM: kan_mc_e2e_infer.h legge questi array con pgm_read_* su AVR.
    # Senza l'annotazione finiscono in SRAM e pgm_read su un indirizzo di
    # SRAM legge la Flash a quell'offset, cioe' dati arbitrari. Il
    # generatore non la emetteva e l'header committato ce l'ha: chi
    # rigenerava rompeva silenziosamente il firmware.
    out = [f"static const {ctype} {name}[{len(vals)}] PROGMEM = {{"]
    for i in range(0, len(vals), per_line):
        out.append("  " + ", ".join(str(int(v)) for v in vals[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def main():
    d = prepare14_dict()
    feats = [str(f) for f in d["feats"]]
    cards = [int(c) for c in d["cards"]]
    ite, itr = d["ite"], d["itr"]
    ymte, CTte, CTtr = d["ymte"], d["CTte"], d["CTtr"]
    Xtr_p, Xte_p = d["Xtr"], d["Xte"]
    classes = [str(c) for c in d["classes"]]

    # come sopra: cache, poi models/, poi un messaggio che dice quale script
    # produce lo stato invece di un FileNotFoundError su un percorso
    from kanids.checkpoint import motivo as _motivo, trova as _trova
    _stato = _trova("mlcat_state.pkl")
    if _stato is None:
        raise SystemExit(_motivo("mlcat_state.pkl"))
    print(f"[stato] {_stato}")
    st = pickle.load(open(_stato, "rb"))
    C1 = st["p"][0].astype(np.float64)
    C2 = st["p"][1].astype(np.float64)
    tabs = [t.astype(np.float64) for t in st["p"][2:]]
    K, HID = C1.shape[0], C1.shape[1]
    C = C2.shape[1]
    J = len(tabs)

    # ── riferimento float ────────────────────────────────────
    H = np.einsum("nid,ihd->nh", cheb_T(Xte_p / CLIP, 8), C1)
    for j in range(J):
        H += tabs[j][CTte[:, j]]
    A = np.tanh(H)
    Zf = np.einsum("nhd,hcd->nc", cheb_T(A, 8), C2)
    pf = Zf.argmax(1)
    f1f = f1_score(ymte, pf, average="macro", zero_division=0)
    print(f"riferimento float   macro-F1 = {f1f:.4f}")

    # ── valori grezzi, allineati allo split tramite gli indici ──
    df = load_ton_iot(verbose=False)
    Xraw = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
    Rtr, Rte = Xraw[itr], Xraw[ite]

    # Scala intera per feature: 1 per i contatori (gia' interi), 10^6 per
    # la durata in secondi. Le soglie restano a 64 bit perche' su TON_IoT
    # src_bytes e dst_bytes raggiungono 3,9e9, oltre il limite di int32:
    # non e' la durata a imporre i 64 bit, sono i contatori di byte.
    RAW_SCALE = []
    for i in range(K):
        col = np.abs(Xraw[:, i])
        if np.all(np.equal(np.mod(Xraw[:, i], 1), 0)):
            RAW_SCALE.append(1)
            continue
        RAW_SCALE.append(1_000_000)      # durata: microsecondi
    RAW_SCALE = np.array(RAW_SCALE, dtype=np.int64)
    print(f"scale intere dei grezzi: {dict(zip(feats, RAW_SCALE.tolist()))}")

    # ── tabelle di soglie (assorbono log1p + quantile + clip) ──
    Ltr = Rtr.copy()
    for i, nm in enumerate(feats):
        if nm in fc.SKEWED:
            Ltr[:, i] = np.log1p(np.clip(Ltr[:, i], 0, None))
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, Ltr.shape[0]),
                             random_state=fc.RANDOM_STATE).fit(Ltr)
    qs = np.linspace(0.0, 1.0, NZ)
    KN_INT, KN_Z = [], []
    for i, nm in enumerate(feats):
        vals, cnt = np.unique(Rtr[:, i], return_counts=True)
        freq = vals[np.argsort(cnt)[::-1][:16]]
        knots = np.unique(np.concatenate([np.quantile(Rtr[:, i], qs), freq]))
        gl = np.log1p(np.clip(knots, 0, None)) if nm in fc.SKEWED else knots.copy()
        Gm = np.tile(Ltr[:1], (len(knots), 1))
        Gm[:, i] = gl
        zk = np.clip(qt.transform(Gm)[:, i], -CLIP, CLIP)
        ki = np.round(knots * RAW_SCALE[i]).astype(np.int64)
        ki, uniq = np.unique(ki, return_index=True)
        KN_INT.append(ki)
        KN_Z.append(np.round(zk[uniq] / CLIP * Q12).astype(np.int64))

    # ── preprocessing intero: grezzi -> z Q12 ────────────────
    def to_z(R):
        out = np.empty((R.shape[0], K), dtype=np.int64)
        for i in range(K):
            v = np.round(R[:, i] * RAW_SCALE[i]).astype(np.int64)
            kr, kz = KN_INT[i], KN_Z[i]
            k = np.clip(np.searchsorted(kr, v, side="right") - 1, 0, len(kr) - 2)
            lo, hi = kr[k], kr[k + 1]
            span = np.maximum(hi - lo, 1)
            wq = np.clip(((v - lo) << 15) // span, 0, Q15)
            zi = kz[k] + (((kz[k + 1] - kz[k]) * wq) >> 15)
            out[:, i] = np.where(v <= kr[0], kz[0],
                                 np.where(v >= kr[-1], kz[-1], zi))
        return np.clip(out, -Q12, Q12)

    zq = to_z(Rte)

    # ── ri-fit int8 dei 320 edge ─────────────────────────────
    h1 = 2.0 / N_INT
    kn1 = np.arange(-1 - 3 * h1, 1 + 3 * h1 + h1 / 2, h1)
    rs = np.random.RandomState(0)
    sub = rs.choice(Xtr_p.shape[0], min(30000, Xtr_p.shape[0]), replace=False)
    Xs = Xtr_p[sub] / CLIP
    xa = np.linspace(-1, 1 - 1e-9, 200)
    Ba = bspline_basis(xa, kn1, 3)
    Ta = cheb_T(xa, 8)

    C1q, s1 = [], []
    for i in range(K):
        xi = np.clip(Xs[:, i], -1, 1 - 1e-9)
        A_ = np.vstack([bspline_basis(xi, kn1, 3), 0.1 * Ba])
        tgt = np.vstack([cheb_T(xi, 8) @ C1[i].T, 0.1 * (Ta @ C1[i].T)])
        coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
        sc_ = np.maximum(np.abs(coef).max(0) / 127.0, 1e-12)
        C1q.append(np.round(coef / sc_).astype(np.int64))
        s1.append(sc_)
    s1 = np.array(s1)

    Hs = np.einsum("nid,ihd->nh", cheb_T(Xs, 8), C1)
    for j in range(J):
        Hs += tabs[j][CTtr[sub, j]]
    As = np.tanh(Hs)
    C2q, s2 = [], []
    for hh in range(HID):
        ah = np.clip(As[:, hh], -1, 1 - 1e-9)
        A_ = np.vstack([bspline_basis(ah, kn1, 3), 0.1 * Ba])
        tgt = np.vstack([cheb_T(ah, 8) @ C2[hh].T, 0.1 * (Ta @ C2[hh].T)])
        coef, *_ = np.linalg.lstsq(A_, tgt, rcond=None)
        sc_ = np.maximum(np.abs(coef).max(0) / 127.0, 1e-12)
        C2q.append(np.round(coef / sc_).astype(np.int64))
        s2.append(sc_)
    s2 = np.array(s2)
    t8 = [(np.round(t / max(np.abs(t).max() / 127.0, 1e-12)).astype(np.int64),
           max(np.abs(t).max() / 127.0, 1e-12)) for t in tabs]

    # ── forward intero di riferimento ────────────────────────
    sref1 = s1.max()
    m1 = np.round(s1 / sref1 * Q15).astype(np.int64)
    tm = np.array([int(round(t8[j][1] / sref1 * Q15)) for j in range(J)], dtype=np.int64)
    sref2 = s2.max()
    m2 = np.round(s2 / sref2 * Q15).astype(np.int64)
    txs = np.linspace(-8, 8, TL)
    # Saturazione Q15: round(tanh(x)*32768) tocca esattamente 32768 alle
    # estremita' del dominio, che in int16 e' overflow. Si satura a 32767
    # QUI, nel riferimento, non solo nel C: altrimenti riferimento e
    # firmware divergerebbero di 1 LSB proprio sui valori saturi.
    tanh_q15 = np.clip(np.round(np.tanh(txs) * Q15), -Q15, Q15 - 1).astype(np.int64)
    idx_mult = int(round(sref1 / (6 * Q15) * (TL - 1) / 16 * (1 << 30)))

    def forward_int(zqx, CTx):
        n = zqx.shape[0]
        Hq = np.zeros((n, HID), dtype=np.int64)
        for i in range(K):
            u = (zqx[:, i] + Q12) * N_INT
            for hh in range(HID):
                Hq[:, hh] += (spline_kernel(u, C1q[i][:, hh], 13) * m1[i, hh]) >> 15
        for j in range(J):
            Hq += t8[j][0][CTx[:, j]] * tm[j] * 6
        idx = np.clip(((Hq * idx_mult) >> 30) + (TL // 2), 0, TL - 1)
        Aq = tanh_q15[idx]
        Zq = np.zeros((n, C), dtype=np.int64)
        for hh in range(HID):
            u = np.clip(Aq[:, hh] + Q15, 0, 2 * Q15 - 1) * N_INT
            for c in range(C):
                Zq[:, c] += (spline_kernel(u, C2q[hh][:, c], 16) * m2[hh, c]) >> 15
        return Zq

    Zq = forward_int(zq, CTte)
    pq = Zq.argmax(1)
    f1q = f1_score(ymte, pq, average="macro", zero_division=0)
    wq = f1_score(ymte, pq, average="weighted", zero_division=0)
    agree = float((pq == pf).mean())
    print(f"catena integer e2e  macro-F1 = {f1q:.4f} (delta {f1q - f1f:+.4f}), "
          f"weighted {wq:.4f}, agreement argmax {agree * 100:.2f}%")

    # 8 B per soglia (int64) + 2 B per il valore z (int16). int64 e' necessario:
    # src_bytes e dst_bytes arrivano a 3,9e9 e sforano int32 anche a scala 1.
    # Il totale in byte NON si calcola qui: si legge dall'header appena
    # scritto, piu' in basso, con la stessa funzione che conta tutti gli
    # altri modelli. Questi termini restano solo come scomposizione
    # informativa nel log.
    knot_bytes = sum(len(k) * 10 for k in KN_INT)
    coef_bytes = sum(c.size for c in C1q) + sum(c.size for c in C2q)
    cat_bytes = sum(t[0].size for t in t8)
    mult_bytes = (m1.size + m2.size) * 4 + J * 4
    print(f"scomposizione: soglie {knot_bytes} B + coeff {coef_bytes} B "
          f"+ cat {cat_bytes} B + tanh {TL*2} B + mult {mult_bytes} B")

    # ── header C ─────────────────────────────────────────────
    NK = max(len(k) for k in KN_INT)
    g = np.random.RandomState(42).choice(len(zq), N_GOLDEN, replace=False)

    H_ = [
        "// Generato da scripts/export_mc_e2e_int_c.py - NON modificare a mano.",
        "// Catena integer-only end-to-end a 10 classi: grezzi -> argmax.",
        "// Nessun tipo in virgola mobile: tutte le costanti sono intere.",
        "// STATO CANONICO: models/kan14_multiclass_multilayer.pkl. Questo header",
        "// e' la funzione deterministica di quel file versionato, non un",
        "// artefatto congelato: si riemette con",
        "//     python reproduce.py --stage integer-10classi",
        "// e tests/test_stato_multiclasse.py lo confronta byte per byte.",
        "#ifndef KAN_MC_E2E_INT_H", "#define KAN_MC_E2E_INT_H",
        "#include <stdint.h>",
        "#ifdef __AVR__",
        "#include <avr/pgmspace.h>",
        "#else",
        "#ifndef PROGMEM",
        "#define PROGMEM      /* su ESP32 lo definisce gia' pgmspace.h */",
        "#endif",
        "#endif", "",
        f"#define MC_K       {K}", f"#define MC_HID     {HID}",
        f"#define MC_C       {C}", f"#define MC_J       {J}",
        f"#define MC_NCOEF   {C1q[0].shape[0]}", f"#define MC_NSEG    {N_INT}",
        f"#define MC_NK      {NK}", f"#define MC_TL      {TL}",
        f"#define MC_IDXMULT {idx_mult}LL", f"#define MC_N_GOLDEN {N_GOLDEN}", "",
        carr("MC_RAW_SCALE", RAW_SCALE, "int32_t", 10), "",
        carr("MC_NKNOT", [len(k) for k in KN_INT], "int16_t", 10), "",
        f"static const int64_t MC_KNOT[MC_K][MC_NK] PROGMEM = {{",
    ]
    for k in KN_INT:
        pad = list(k) + [k[-1]] * (NK - len(k))
        H_.append("  {" + ", ".join(f"{int(v)}LL" for v in pad) + "},")
    H_.append("};")
    H_.append(f"static const int16_t MC_KNOTZ[MC_K][MC_NK] PROGMEM = {{")
    for kz in KN_Z:
        pad = list(kz) + [kz[-1]] * (NK - len(kz))
        H_.append("  {" + ", ".join(str(int(v)) for v in pad) + "},")
    H_.append("};")
    H_.append(f"static const int8_t MC_C1[MC_K][MC_HID][MC_NCOEF] PROGMEM = {{")
    for i in range(K):
        H_.append("  {" + ", ".join("{" + ", ".join(str(int(v)) for v in C1q[i][:, hh]) + "}"
                                    for hh in range(HID)) + "},")
    H_.append("};")
    H_.append(f"static const int8_t MC_C2[MC_HID][MC_C][MC_NCOEF] PROGMEM = {{")
    for hh in range(HID):
        H_.append("  {" + ", ".join("{" + ", ".join(str(int(v)) for v in C2q[hh][:, c]) + "}"
                                    for c in range(C)) + "},")
    H_.append("};")
    H_.append(carr("MC_M1", m1.ravel(), "int32_t", 16))
    H_.append(carr("MC_M2", m2.ravel(), "int32_t", 10))
    H_.append(carr("MC_TM", tm, "int32_t", 8))
    H_.append(carr("MC_CARD", cards, "int16_t", 8))
    H_.append(carr("MC_TANH", tanh_q15, "int16_t", 16))
    maxcard = max(cards)
    H_.append(f"static const int8_t MC_CAT[MC_J][{maxcard}][MC_HID] PROGMEM = {{")
    for j in range(J):
        rows = [list(t8[j][0][v]) if v < cards[j] else [0] * HID for v in range(maxcard)]
        H_.append("  {" + ", ".join("{" + ", ".join(str(int(x)) for x in r) + "}"
                                    for r in rows) + "},")
    H_.append("};")
    H_.append("")
    H_.append("typedef struct { int64_t raw[MC_K]; int16_t cat[MC_J]; "
              "int64_t z[MC_C]; uint8_t pred, label; } mc_golden_t;")
    H_.append(f"static const mc_golden_t MC_GOLDEN[MC_N_GOLDEN] PROGMEM = {{")
    Rg = np.round(Rte[g] * RAW_SCALE).astype(np.int64)
    for n, j in enumerate(g):
        raw = ", ".join(f"{int(v)}LL" for v in Rg[n])
        cat = ", ".join(str(int(v)) for v in CTte[j])
        zz = ", ".join(f"{int(v)}LL" for v in Zq[j])
        H_.append(f"  {{{{{raw}}}, {{{cat}}}, {{{zz}}}, {int(pq[j])}, {int(ymte[j])}}},")
    H_.append("};")
    H_.append("")
    H_.append("// classi: " + ", ".join(f"{i}={c}" for i, c in enumerate(classes)))
    H_.append("#endif // KAN_MC_E2E_INT_H")

    out = _REPO / "mcu_pio" / "include" / "kan_mc_e2e_int.h"
    out.write_text("\n".join(H_), encoding="utf-8", newline="\n")
    print(f"scritto {out.relative_to(_REPO)} ({out.stat().st_size/1024:.0f} KB)")

    mem, dettaglio = scan(out, "MC_")
    print(f"memoria del modello: {mem} B ({mem/1024:.2f} KB)")
    for nome, typ, count, nbytes in dettaglio:
        print(f"    {nome:<14} {typ:<8} x{count:<6} {nbytes:>6} B")

    pd.DataFrame([{
        "macro_f1_float": round(f1f, 4), "macro_f1_e2e_int": round(f1q, 4),
        "delta": round(f1q - f1f, 4), "weighted_f1": round(wq, 4),
        "agreement_pct": round(agree * 100, 2), "mem_bytes": mem,
        "mem_kb": round(mem / 1024, 2), "n_golden": N_GOLDEN,
    }]).to_csv(RESULTS_DIR / "mc_e2e_int_export.csv", index=False, lineterminator="\n")
    print("salvato results/mc_e2e_int_export.csv")


if __name__ == "__main__":
    main()

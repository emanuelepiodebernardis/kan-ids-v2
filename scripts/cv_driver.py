#!/usr/bin/env python3
"""Driver riprendibile per cv_multiseed con esecuzione a checkpoint.

Esegue unita' (model, seed, fold) pendenti finche' resta budget di tempo,
checkpointando dopo ogni unita' su results/cv_real_progress.csv.
Rilanciare finche' non stampa DONE.
"""

# --- percorsi artefatti (migrato da /tmp, vedi tools/migrate_tmp_paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from kanids.config import artifact_path as _ART
# ---------------------------------------------------------------------------
import argparse, os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv_multiseed as cvm
from sklearn.model_selection import StratifiedKFold

PROG = "results/cv_real_progress.csv"
CACHE = _ART("cv_cache.npz")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_test_network.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=35.0)
    a = ap.parse_args()
    t0 = time.time()

    # cache del preprocessing (costoso) su npz
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        X, y = d["X"], d["y"]
    else:
        X, y = cvm.load_unified(a.csv, synthetic=False, sample=None)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
        np.savez_compressed(CACHE, X=X, y=y)
    if time.time() - t0 > a.budget:
        print("CHECKPOINT (solo cache preprocessing)"); return

    done = set()
    rows = []
    if os.path.exists(PROG):
        prev = pd.read_csv(PROG)
        rows = prev.to_dict("records")
        done = {(r["model"], r["seed"], r["fold"]) for r in rows}

    models_order = list(cvm.build_baselines(0).keys()) + ["KAN Chebyshev"]
    units = []
    for seed in range(a.seeds):
        skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
        for fold, (tr, va) in enumerate(splits):
            for m in models_order:
                if (m, seed, fold) not in done:
                    units.append((m, seed, fold, tr, va))

    if not units:
        df = pd.DataFrame(rows)
        df.to_csv("results/cv_multiseed_results_real.csv", index=False)
        summ = cvm.summarize(df)
        summ.to_csv("results/cv_multiseed_summary_real.csv", index=False)
        cvm.print_table(summ)
        print("DONE"); return

    for (m, seed, fold, tr, va) in units:
        if time.time() - t0 > a.budget:
            break
        Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
        if m == "KAN Chebyshev":
            met = kan_chunked(Xtr, ytr, Xva, yva, seed, fold, t0, a.budget)
            if met is None:
                print(f"CHECKPOINT: KAN seed={seed} fold={fold} in corso", flush=True)
                return
        else:
            est = cvm.build_baselines(seed)[m]
            if hasattr(est, "n_jobs"):
                est.n_jobs = -1
            met = cvm.eval_baseline_fold(est, Xtr, ytr, Xva, yva)
        rows.append({"model": m, "fold": fold, "seed": seed, **met})
        pd.DataFrame(rows).to_csv(PROG, index=False)
        print(f"unit ok: {m} seed={seed} fold={fold} f1={met['f1']:.4f} t={time.time()-t0:.0f}s", flush=True)
    remaining = len(units) - sum(1 for r in rows if True) + len(done)
    print(f"CHECKPOINT: {len(rows)} unita' complete")



def kan_chunked(Xtr, ytr, Xva, yva, seed, fold, t0, budget, chunk=50):
    """Addestra la KAN a blocchi di epoche con checkpoint pickle.

    Ritorna le metriche quando raggiunge cvm.KAN_EPOCHS, altrimenti None
    (= budget esaurito, riprendere alla prossima invocazione)."""
    import pickle
    from sklearn.preprocessing import StandardScaler
    ck = _ART(f"kan_ckpt_{seed}_{fold}.pkl")
    scaler = StandardScaler().fit(Xtr)
    Xtr_k = np.clip(scaler.transform(Xtr), -cvm.KAN_CLIP, cvm.KAN_CLIP)
    if os.path.exists(ck):
        with open(ck, "rb") as f:
            kan, done_ep = pickle.load(f)
    else:
        kan = cvm.ChebyshevKANBinary(
            in_dim=Xtr_k.shape[1], degree=cvm.KAN_DEGREE,
            x_min=-cvm.KAN_CLIP, x_max=cvm.KAN_CLIP, seed=seed)
        done_ep = 0
    # basi Chebyshev precalcolate una volta sola (fit() le ricalcola a
    # ogni epoca: identita' matematica, solo piu' veloce)
    from kan_chebyshev import chebyshev_basis
    Xn = kan._norm(Xtr_k)
    B = np.stack([chebyshev_basis(Xn[:, i], kan.degree)
                  for i in range(kan.in_dim)])          # (in_dim, N, deg+1)
    yf = ytr.astype(np.float64)
    pos = yf.mean()
    w_pos, w_neg = 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6)
    sw = np.where(yf == 1, w_pos, w_neg)
    l2, lr = 1e-4, cvm.KAN_LR
    while done_ep < cvm.KAN_EPOCHS:
        if time.time() - t0 > budget:
            kan._bases = None
            with open(ck, "wb") as f:
                pickle.dump((kan, done_ep), f)
            return None
        ep = min(chunk, cvm.KAN_EPOCHS - done_ep)
        for _ in range(ep):
            z = np.einsum("ind,id->n", B, kan.coeffs)
            p = kan._sigmoid(z)
            g = sw * (p - yf)
            grads = np.einsum("ind,n->id", B, g) / B.shape[1] + l2 * kan.coeffs
            kan.coeffs -= lr * grads
        done_ep += ep
        kan._bases = None
        with open(ck, "wb") as f:
            pickle.dump((kan, done_ep), f)
    Xva_k = np.clip(scaler.transform(Xva), -cvm.KAN_CLIP, cvm.KAN_CLIP)
    met = cvm.compute_metrics(yva, kan.predict(Xva_k), kan.predict_proba(Xva_k))
    os.remove(ck)
    return met

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sceglie larghezza e grado della KAN su una validation interna al training.

PERCHE' QUESTO SCRIPT ESISTE
============================
La revisione ha corretto il rapporto del joint training, che era stato scelto
guardando TON_test e BoT_test. Ricontrollando il resto del progetto e' venuto
fuori che il rapporto era il caso piu' visibile, non l'unico: anche
`hidden = 16` e `degree = 8` vengono da ablation misurate sull'held-out.

    results/protocol_v1/ml_binary_real.csv     10-16-1 deg8 -> f1 0,9784
                                               10-32-1 deg8 -> f1 0,9778
    results/protocol_v1/kan_ml_cat_deg4_real.csv   grado 4 -> macro-F1 0,9374
                                                   grado 8 -> macro-F1 0,9409

Quegli F1 sono `f1_score(yte, ...)` sul 20% held-out, cioe' sullo stesso
insieme poi riportato come risultato. E' lo stesso difetto del rapporto, un
piano sotto. Qui la scelta viene rifatta dove va fatta.

COME
====
Identica per costruzione a `joint_training.py --select-ratio`:

    outer_split(seed)  ->  (train, test)      il test viene calcolato e MAI letto
    inner_split(train) ->  (fit, validation)  20% stratificato dentro il training
    preprocessor fittato SOLO su `fit`
    per ogni (hidden, degree) del reticolo: fit su `fit`, punteggio su `validation`

Ripetuto su piu' seed, che danno osservazioni APPAIATE: lo stesso seed
produce lo stesso split per tutte le configurazioni, quindi le differenze
si confrontano configurazione contro configurazione sullo stesso taglio.

LA REGOLA DI SCELTA, FISSATA PRIMA DI VEDERE I NUMERI
=====================================================
Non "vince la media piu' alta", ma la **regola a un errore standard** (1-SE,
quella di Breiman per gli alberi di potatura, la stessa che glmnet chiama
`lambda.1se`):

    fra tutte le configurazioni la cui balanced accuracy media in validation
    e' >= (media della migliore - 1 errore standard della migliore),
    si sceglie la piu' PICCOLA.

"Piu' piccola" = meno parametri: a parita' pratica di punteggio vince la
configurazione che occupa meno Flash, che per un progetto su microcontrollore
e' il criterio del progetto, non un ripiego.

Perche' la 1-SE e non "non significativamente peggiore con un t appaiato".
Con pochi seed un test appaiato ha potenza quasi nulla: qualunque cosa
risulterebbe "non significativamente peggiore", e la regola degenererebbe in
"scegli sempre la piu' piccola" travestita da statistica. La 1-SE non finge
di essere un test d'ipotesi: e' una soglia dichiarata sulla dispersione
osservata, e resta interpretabile anche con tre seed. I p appaiati vengono
comunque calcolati e scritti nell'artefatto, come informazione a corredo —
non e' su quelli che si decide.

La regola sta scritta qui, in `REGOLA`, e in `tests/test_architettura.py`.
Se un giorno il risultato non piacesse e la regola cambiasse, il diff di
questo file lo direbbe.

USO
===
    python scripts/select_architettura.py --stima      # cronometra un fit solo
    python scripts/select_architettura.py              # la selezione vera
    python scripts/select_architettura.py --smoke      # dati sintetici, ci mette 1'

Riprendibile: ogni run completata finisce nel checkpoint, `--max-seconds`
ferma e salva, rilanciare lo stesso comando riprende.

Produce:
    results/arch_selection_runs.csv     una riga per (seed, modello, h, grado)
    results/arch_selection.csv          medie e deviazioni per configurazione
    results/arch_selection_scelta.json  la scelta, con la regola che l'ha prodotta
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kanids import (  # noqa: E402
    ARTIFACTS_DIR, CLIP, K_NUMERIC, NUMERIC_RAW, RESULTS_DIR, SEEDS,
    LeakageFreePreprocessor, outer_split, set_global_seed,
)
from kanids.datasets import encode_targets, load_ton_iot, make_synthetic  # noqa: E402
from kanids.models import (  # noqa: E402
    CategoricalKANBinary, MultiLayerKANBinary,
)

# ── il reticolo ──────────────────────────────────────────────────────
# I due valori attuali (16, 8) sono dentro, altrimenti la selezione non
# potrebbe confermarli. 32 e' l'alternativa gia' provata nella fase 1; 8 e 4
# stanno sotto, dove il progetto non aveva mai guardato per la larghezza.
HIDDEN_CANDIDATI = (4, 8, 16, 32)
DEGREE_CANDIDATI = (4, 6, 8)

VAL_SIZE = 0.2          # come inner_split di joint_training.py
METRICA = "balanced_accuracy"

REGOLA = ("1-SE: fra le configurazioni con media >= (migliore - 1 errore "
          "standard della migliore), la piu' piccola per numero di parametri")


def n_parametri(modello: str, hidden: int, degree: int, n_cat: int = 4,
                in_dim: int = K_NUMERIC) -> int:
    """Parametri appresi, per ordinare le configurazioni per dimensione.

    Non e' il footprint in byte del modello compilato (quello lo conta
    c_footprint.scan sull'header): serve solo a dire quale configurazione e'
    piu' piccola, e per questo il conteggio dei coefficienti basta.
    """
    if modello == "KAN(cat,1L)":
        return in_dim * (degree + 1) + n_cat * 10
    return (in_dim * hidden * (degree + 1)         # primo strato
            + hidden * (degree + 1)                # secondo strato
            + n_cat * hidden)                      # edge categorici


def inner_split(y_train, seed, val_size=VAL_SIZE):
    """Ritaglia la validation DENTRO il training. Stessa funzione di
    joint_training.py, sullo stesso principio: la validation non e' mai
    stata nel test, perche' e' ritagliata da cio' che test non e'."""
    return outer_split(y_train, seed=seed, test_size=val_size)


def balanced_accuracy(y_vero, y_pred) -> float:
    """Media dei recall per classe. La stessa metrica su cui e' stato scelto
    il rapporto, per non introdurre un secondo criterio."""
    classi = np.unique(y_vero)
    return float(np.mean([(y_pred[y_vero == c] == c).mean() for c in classi]))


def configurazioni(solo_1l=False, solo_ml=False):
    """(modello, hidden, degree) da valutare.

    Per la single-layer la larghezza non esiste: si valuta una volta sola per
    grado, con hidden=0 come segnaposto. Valutarla quattro volte darebbe
    quattro copie identiche e gonfierebbe il conto delle run."""
    if not solo_ml:
        for d in DEGREE_CANDIDATI:
            yield ("KAN(cat,1L)", 0, d)
    if not solo_1l:
        for h, d in product(HIDDEN_CANDIDATI, DEGREE_CANDIDATI):
            yield ("KAN(cat,ML)", h, d)


def costruisci(modello, hidden, degree, cardinalities, seed, in_dim):
    """`in_dim` viene dal preprocessor, non da K_NUMERIC.

    Prima erano fissi a K_NUMERIC: con --k diverso dal default il modello si
    aspettava dieci colonne e ne riceveva sei, IndexError dentro la base di
    Chebyshev. Sarebbe emerso solo a chi avesse provato una k diversa —
    cioe' a nessuno, fino al primo che ci prova."""
    if modello == "KAN(cat,1L)":
        return CategoricalKANBinary(
            in_dim=in_dim, cardinalities=cardinalities, degree=degree,
            clip=CLIP, seed=seed)
    return MultiLayerKANBinary(
        in_dim=in_dim, cardinalities=cardinalities, hidden=hidden,
        degree=degree, clip=CLIP, seed=seed)


def una_run(df, y, seed, modello, hidden, degree, k):
    """Un (seed, configurazione): split, inner split, fit, punteggio.

    Il test set viene calcolato e non viene letto. E' la riga che rende vero
    tutto il resto, ed e' verificata da un test che cambia le etichette del
    test e pretende che il punteggio non si muova di un bit.
    """
    set_global_seed(seed)
    tr, _te = outer_split(y, seed=seed)          # `_te` non si usa: e' il punto
    pos_fit, pos_val = inner_split(y[tr], seed=seed)
    fit, val = tr[pos_fit], tr[pos_val]

    prep = LeakageFreePreprocessor(
        k_numeric=k, random_state=seed, numeric_candidates=list(NUMERIC_RAW),
        selection_target="binary")
    prep.fit(df.iloc[fit], y[fit])
    Xfit, Cfit = prep.transform(df.iloc[fit])
    Xval, Cval = prep.transform(df.iloc[val])

    in_dim = Xfit.shape[1]
    m = costruisci(modello, hidden, degree, prep.cardinalities_, seed, in_dim)
    t0 = time.time()
    m.fit(Xfit, Cfit, y[fit])
    dt = time.time() - t0
    pred = m.predict(Xval, Cval)
    return {
        "seed": seed, "model": modello, "hidden": hidden, "degree": degree,
        "split": "validation",               # marchio: non e' un numero di test
        METRICA: balanced_accuracy(y[val], pred),
        "n_parametri": n_parametri(modello, hidden, degree,
                                   len(prep.cardinalities_), in_dim),
        "n_fit": int(len(fit)), "n_val": int(len(val)),
        "secondi": round(dt, 1),
    }


def scegli(runs: pd.DataFrame, modello: str) -> dict:
    """Applica REGOLA e restituisce la scelta con l'evidenza che la sostiene."""
    d = runs[runs.model == modello]
    per_conf = (d.groupby(["hidden", "degree"])
                 .agg(media=(METRICA, "mean"), dev=(METRICA, "std"),
                      n=(METRICA, "size"), parametri=("n_parametri", "first"))
                 .reset_index())
    per_conf["errore_standard"] = per_conf["dev"] / np.sqrt(per_conf["n"])

    # Con un seed solo la deviazione e' NaN e la regola 1-SE non e' definita:
    # la soglia diventa NaN, nessuna configurazione la supera, e senza questo
    # controllo lo script morirebbe con un idxmin su una tabella vuota — o,
    # peggio, sceglierebbe per un motivo che nessuno saprebbe ricostruire.
    # Servono almeno tre seed perche' l'errore standard significhi qualcosa.
    if int(per_conf["n"].min()) < 3:
        raise SystemExit(
            f"{modello}: alcune configurazioni hanno solo "
            f"{int(per_conf['n'].min())} run. La regola 1-SE si appoggia alla "
            f"dispersione fra seed: servono almeno 3 seed completi. "
            f"Rilancia con --seeds 42,43,44 (o piu') e lascia finire.")

    best = per_conf.loc[per_conf["media"].idxmax()]
    soglia = float(best["media"] - best["errore_standard"])
    assert np.isfinite(soglia), "soglia 1-SE non finita: dispersione mancante"
    ammesse = per_conf[per_conf["media"] >= soglia]
    scelta = ammesse.loc[ammesse["parametri"].idxmin()]

    # p appaiati contro la scelta: informazione, non criterio
    pivot = d.pivot_table(index="seed", columns=["hidden", "degree"], values=METRICA)
    confronti = []
    chiave = (int(scelta["hidden"]), int(scelta["degree"]))
    for col in pivot.columns:
        if col == chiave:
            continue
        a, b = pivot[chiave].to_numpy(), pivot[col].to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        p = float("nan")
        if ok.sum() >= 3 and np.ptp((a - b)[ok]) > 0:
            from scipy.stats import ttest_rel
            p = float(ttest_rel(a[ok], b[ok]).pvalue)
        confronti.append({"hidden": int(col[0]), "degree": int(col[1]),
                          "delta": round(float(np.mean((a - b)[ok])), 5),
                          "p": None if np.isnan(p) else round(p, 6),
                          "vince_in": int((a[ok] > b[ok]).sum()), "su": int(ok.sum())})
    return {
        "modello": modello,
        "hidden": int(scelta["hidden"]), "degree": int(scelta["degree"]),
        "media_validation": round(float(scelta["media"]), 5),
        "parametri": int(scelta["parametri"]),
        "migliore_assoluta": {"hidden": int(best["hidden"]),
                              "degree": int(best["degree"]),
                              "media": round(float(best["media"]), 5),
                              "parametri": int(best["parametri"])},
        "soglia_1se": round(soglia, 5),
        "candidate_ammesse": len(ammesse),
        "confronti_appaiati": confronti,
        "per_configurazione": per_conf.round(5).to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                    help="piu' seed = piu' potenza sui confronti appaiati")
    ap.add_argument("--k", type=int, default=K_NUMERIC)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="dati sintetici: verifica la catena in un minuto")
    ap.add_argument("--stima", action="store_true",
                    help="cronometra UNA configurazione e stima il totale, "
                         "senza impegnare la macchina per ore")
    ap.add_argument("--solo-1l", action="store_true")
    ap.add_argument("--solo-ml", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    df = make_synthetic(20_000, seed=0) if args.smoke else load_ton_iot(args.csv)
    yb, _ym, _classi = encode_targets(df)
    out_dir = (ARTIFACTS_DIR / "smoke") if args.smoke else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    confs = list(configurazioni(args.solo_1l, args.solo_ml))

    if args.stima:
        print("cronometro la configurazione piu' cara del reticolo "
              f"(ML h={max(HIDDEN_CANDIDATI)} grado={max(DEGREE_CANDIDATI)})...")
        r = una_run(df, yb, seeds[0], "KAN(cat,ML)",
                    max(HIDDEN_CANDIDATI), max(DEGREE_CANDIDATI), args.k)
        # il costo scala circa con hidden*(degree+1); la piu' cara e' il tetto
        peso_max = max(HIDDEN_CANDIDATI) * (max(DEGREE_CANDIDATI) + 1)
        totale = sum((h or 1) * (d + 1) for _m, h, d in confs) / peso_max * r["secondi"]
        print(f"\n  un fit della piu' cara: {r['secondi']:.0f} s "
              f"({r['n_fit']:,} righe di fit, {r['n_val']:,} di validation)")
        print(f"  configurazioni: {len(confs)}   seed: {len(seeds)}")
        print(f"  stima del totale: ~{totale * len(seeds) / 60:.0f} minuti "
              f"({totale * len(seeds) / 3600:.1f} ore)")
        print("\n  e' una stima per eccesso: le configurazioni piccole costano meno.")
        print("  Se e' troppo: --solo-1l per iniziare, o --seeds 42,43 per meta'.")
        return

    ckpt = ARTIFACTS_DIR / ("arch_selection_smoke.jsonl" if args.smoke
                            else "arch_selection.jsonl")
    if args.fresh and ckpt.exists():
        ckpt.unlink()
    done, rows = set(), []
    if ckpt.exists():
        for riga in ckpt.read_text(encoding="utf-8").splitlines():
            if riga.strip():
                r = json.loads(riga)
                rows.append(r)
                done.add((r["seed"], r["model"], r["hidden"], r["degree"]))
        print(f"[ckpt] {len(rows)} run gia' completate")

    print("=" * 74)
    print(f"SELEZIONE DELL'ARCHITETTURA — {len(confs)} configurazioni x "
          f"{len(seeds)} seed, metrica {METRICA} su validation interna")
    print(f"regola: {REGOLA}")
    print("=" * 74)

    t0 = time.time()
    for seed in seeds:
        for modello, hidden, degree in confs:
            if (seed, modello, hidden, degree) in done:
                continue
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print("\n[ckpt] fermato per tempo. Rilancia lo stesso comando.")
                return
            r = una_run(df, yb, seed, modello, hidden, degree, args.k)
            rows.append(r)
            with ckpt.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(r) + "\n")
            eti = modello if hidden == 0 else f"{modello} h={hidden}"
            print(f"  seed={seed} {eti:<22} grado={degree}  "
                  f"{METRICA}={r[METRICA]:.4f}  ({r['secondi']:.0f} s)")

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "arch_selection_runs.csv", index=False)

    # Un reticolo incompleto non deve produrre una scelta: il file JSON viene
    # letto dalla pipeline, e una scelta fatta su meta' delle configurazioni
    # sarebbe indistinguibile da una fatta su tutte. Meglio fermarsi con le
    # run gia' salvate nel checkpoint e ripartire da li'.
    attese = {(m, h, d) for m, h, d in configurazioni()}
    presenti = {(r.model, r.hidden, r.degree) for r in runs.itertuples()}
    mancanti = attese - presenti
    incompiute = [c for c in attese - mancanti
                  if len(runs[(runs.model == c[0]) & (runs.hidden == c[1])
                              & (runs.degree == c[2])]) < len(seeds)]
    if mancanti or incompiute:
        print(f"\n[parziale] {len(mancanti)} configurazioni non iniziate, "
              f"{len(incompiute)} incomplete sui {len(seeds)} seed richiesti.")
        print("Le run fatte sono nel checkpoint e nei CSV; la SCELTA non viene "
              "scritta finche' il reticolo non e' completo — una scelta su "
              "meta' reticolo sarebbe indistinguibile da una su tutto.")
        return

    sintesi = (runs.groupby(["model", "hidden", "degree"])
                   .agg(**{f"{METRICA}_mean": (METRICA, "mean"),
                           f"{METRICA}_std": (METRICA, "std"),
                           "n_runs": (METRICA, "size"),
                           "n_parametri": ("n_parametri", "first")})
                   .reset_index())
    sintesi.to_csv(out_dir / "arch_selection.csv", index=False)

    scelte = {m: scegli(runs, m) for m in sorted(runs.model.unique())}
    payload = {"regola": REGOLA, "metrica": METRICA, "seeds": list(seeds),
               "val_size": VAL_SIZE, "hidden_candidati": list(HIDDEN_CANDIDATI),
               "degree_candidati": list(DEGREE_CANDIDATI), "scelte": scelte}
    (out_dir / "arch_selection_scelta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")

    print("\n" + "=" * 74)
    for m, s in scelte.items():
        b = s["migliore_assoluta"]
        print(f"{m}: hidden={s['hidden']} grado={s['degree']}  "
              f"({s['parametri']:,} parametri, {METRICA}={s['media_validation']:.4f})")
        if (b["hidden"], b["degree"]) != (s["hidden"], s["degree"]):
            print(f"    la media piu' alta era h={b['hidden']} grado={b['degree']} "
                  f"({b['media']:.4f}, {b['parametri']:,} parametri): "
                  f"dentro 1 errore standard, quindi vince la piu' piccola")
        print(f"    soglia 1-SE {s['soglia_1se']:.4f}, "
              f"{s['candidate_ammesse']} configurazioni ammesse")
    print("=" * 74)
    print(f"salvati arch_selection{{_runs,}}.csv e arch_selection_scelta.json "
          f"in {out_dir.name}/")


if __name__ == "__main__":
    main()

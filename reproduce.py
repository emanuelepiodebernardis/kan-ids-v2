#!/usr/bin/env python3
"""Punto di ingresso unico per riprodurre gli esperimenti.

Da un clone pulito:

    pip install -r requirements.txt
    python reproduce.py --stage smoke        # ~1 min, dati sintetici, nessun download
    python reproduce.py --stage all          # richiede data/train_test_network.csv

Ogni stage e' idempotente, scrive in results/ e non dipende da file
temporanei fuori dal repository: gli artefatti intermedi vivono in
artifacts/, ricostruibili e cancellabili con --stage clean.

I seed sono fissati in kanids/config.py (SEEDS = 42, 43, 44) e stampati
all'inizio di ogni run insieme alle versioni delle librerie, cosi' il log
di un esperimento contiene tutto quello che serve per ripeterlo.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from kanids import ARTIFACTS_DIR, RESULTS_DIR, SEEDS, describe_protocol  # noqa: E402
from kanids.cache import PIPELINE_VERSION, clean as clean_artifacts  # noqa: E402

PY = sys.executable

STAGES = {
    "smoke": (
        "verifica su dati sintetici che l'intera catena giri (nessun download)",
        [[PY, "scripts/cv_leakagefree.py", "--task", "binary", "--smoke",
          "--seeds", "42", "--folds", "3"],
         [PY, "-m", "pytest", "tests/", "-q"]],
    ),
    "audit": (
        "verifica meccanica dei sei punti della revisione, con le evidenze",
        [[PY, "tools/audit_richieste.py"]],
    ),
    "tests": (
        "solo i test di leakage e riproducibilita'",
        [[PY, "-m", "pytest", "tests/", "-q"]],
    ),
    "features": (
        "ablation sul numero di feature (giustifica k=10)",
        [[PY, "scripts/feature_curve_driver.py"]],
    ),
    "cv-binary": (
        "5-fold x 3 seed, task binario, KAN + tutte le baseline",
        [[PY, "scripts/cv_leakagefree.py", "--task", "binary"]],
    ),
    "cv-multiclass": (
        "5-fold x 3 seed, task a 10 classi, KAN + tutte le baseline",
        [[PY, "scripts/cv_leakagefree.py", "--task", "multiclass"]],
    ),
    "compile": (
        "compilazione ibrida Chebyshev -> B-spline e quantizzazione",
        [[PY, "scripts/kan14_compile.py"],
         [PY, "scripts/hybrid_compile.py"]],
    ),
    "integer": (
        "pipeline integer-only end-to-end: export in C + golden vectors",
        [[PY, "scripts/export_e2e_int_c.py"],
         [PY, "scripts/export_mc_e2e_int_c.py"],
         [PY, "scripts/e2e_int_pipeline.py"]],
    ),
    "nested-cv": (
        "cross-validation annidata: misura l'ottimismo della stima piatta",
        [[PY, "scripts/nested_cv.py", "--task", "binary",
          "--models", "KAN(cat,1L)|LightGBM"]],
    ),
    "models": (
        "salva i pesi in models/ con il manifest (protocollo, seed, metriche)",
        [[PY, "scripts/export_models.py"]],
    ),
    "footprint": (
        "ingombro dei parametri di tutti i modelli, regola di conteggio unica",
        [[PY, "scripts/footprint.py"]],
    ),
    "figures": (
        "figure del report a partire dai CSV in results/",
        [[PY, "scripts/make_figures.py"]],
    ),
    "report": (
        "report tecnico in PDF (richiede gli stage precedenti)",
        [[PY, "scripts/make_figures.py"],
         [PY, "scripts/make_report.py"]],
    ),
    "crossdomain": (
        "TON_IoT <-> BoT-IoT: 4 direzioni + ablation senza categoriche",
        [[PY, "scripts/cross_domain.py", "--exp", "all"],
         [PY, "scripts/cross_domain.py", "--exp", "all", "--no-cat"],
         [PY, "scripts/crossdomain_report.py"]],
    ),
    "leakage-audit": (
        "quantifica l'effetto del difetto del protocollo v1",
        [[PY, "scripts/leakage_audit.py"]],
    ),
    "conformal": (
        "calibrazione split-conformal ed estrazione della forma simbolica",
        [[PY, "scripts/kan14_conformal_symbolic.py"]],
    ),
}

ORDER = ["tests", "audit", "leakage-audit", "features", "cv-binary", "cv-multiclass",
         "crossdomain", "compile", "integer", "conformal", "nested-cv", "models", "footprint",
         "figures", "report"]


def env_report() -> str:
    lines = [
        f"python            {platform.python_version()} ({platform.system()} {platform.machine()})",
        f"pipeline version  {PIPELINE_VERSION}",
        f"seeds             {list(SEEDS)}",
        f"artifacts         {ARTIFACTS_DIR}",
        f"results           {RESULTS_DIR}",
    ]
    for mod in ["numpy", "pandas", "sklearn", "scipy", "lightgbm", "xgboost", "torch"]:
        try:
            m = __import__(mod)
            lines.append(f"{mod:<18}{getattr(m, '__version__', '?')}")
        except ImportError:
            lines.append(f"{mod:<18}(assente)")
    return "\n".join(lines)


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="smoke",
                    help="uno stage, 'all', o 'clean'. --list per l'elenco")
    ap.add_argument("--list", action="store_true", help="elenca gli stage")
    ap.add_argument("--dry-run", action="store_true", help="stampa i comandi senza eseguirli")
    args = ap.parse_args()

    if args.list:
        print("stage disponibili:\n")
        for k in ["smoke"] + ORDER:
            print(f"  {k:<16}{STAGES[k][0]}")
        print(f"\n  {'all':<16}esegue in ordine: {', '.join(ORDER)}")
        print(f"  {'clean':<16}svuota artifacts/")
        return

    print("=" * 74)
    print("KAN-IDS — riproduzione esperimenti")
    print("=" * 74)
    print(env_report())
    print("\n" + describe_protocol())
    print("=" * 74)

    if args.stage == "clean":
        clean_artifacts()
        return

    stages = ORDER if args.stage == "all" else [args.stage]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"stage sconosciuto: {unknown}. Usa --list.")

    t0 = time.time()
    failed = []
    for s in stages:
        desc, cmds = STAGES[s]
        print(f"\n{'─' * 74}\nSTAGE {s} — {desc}\n{'─' * 74}")
        for cmd in cmds:
            if args.dry_run:
                print(f"$ {' '.join(cmd)}")
                continue
            if run(cmd) != 0:
                failed.append((s, " ".join(cmd)))
                print(f"!! stage {s} fallito", file=sys.stderr)

    print(f"\n{'=' * 74}\ncompletato in {time.time() - t0:.0f}s")
    if failed:
        print("stage falliti:")
        for s, c in failed:
            print(f"  {s}: {c}")
        sys.exit(1)
    print(f"risultati in {RESULTS_DIR}")


if __name__ == "__main__":
    main()

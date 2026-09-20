#!/usr/bin/env python3
"""Punto di ingresso unico per riprodurre gli esperimenti.

Da un clone pulito:

    pip install -r requirements.txt
    python reproduce.py --stage smoke        # ~1 min, dati sintetici, nessun download
    python reproduce.py --stage all          # richiede data/train_test_network.csv

Ogni stage e' idempotente, scrive in results/ e non dipende da file
temporanei fuori dal repository: gli artefatti intermedi vivono in
artifacts/, ricostruibili e cancellabili con --stage clean.
Eccezione: 'lut' non rigioca artefatti gia' pronti, li RICOSTRUISCE dal CSV
di TON_IoT in cinque fasi legate — prepara il solo training, seleziona la LUT
e scrive il protocollo, confronta l'header rigenerato con quello misurato,
prepara il test vincolato a quel protocollo, valuta — e scrive tutto in una
directory nuova sotto artifacts/lut_reproduction/ (oppure --lut-output-dir).
Serve quindi il dataset: percorso predefinito del progetto, oppure --csv. Non
sovrascrive header, artefatti consegnati o risultati misurati, e ricostruisce
il preprocessing senza riaddestrare i classificatori. Si ferma al primo
errore di preparazione, selezione o confronto, senza valutare il test.

    python reproduce.py --stage lut --csv data/train_test_network.csv

I seed sono fissati in kanids/config.py (SEEDS = 42, 43, 44) e stampati
all'inizio di ogni run insieme alle versioni delle librerie, cosi' il log
di un esperimento contiene tutto quello che serve per ripeterlo. I blocchi
cross-domain e joint usano invece dieci seed (42-51), che e' il protocollo
richiesto per quella parte.

Le tabelle principali dell'articolo escono da qui:

    tabella di Pareto (byte/accuratezza)   stage 'footprint'
    degrado cross-domain a 4 direzioni     stage 'crossdomain'
    tabella finale a 7 colonne             stage 'joint' poi 'tabelle'

Lo stage 'joint' sceglie prima il rapporto attacco:normale sulla sola
validation interna al training, e solo dopo valuta una volta sui test:
l'ordine dei tre comandi non e' scambiabile.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from kanids import ARTIFACTS_DIR, RESULTS_DIR, SEEDS, describe_protocol  # noqa: E402
from kanids.cache import PIPELINE_VERSION, clean as clean_artifacts  # noqa: E402
from kanids.datasets import ton_iot_path                            # noqa: E402


def sha256_file(percorso: Path) -> str:
    """SHA-256 del CSV sorgente: identifica i dati su cui la catena ha girato."""
    import hashlib
    h = hashlib.sha256()
    with percorso.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()

PY = sys.executable
LUT_FROZEN_HEADER = Path("mcu_pio/include/kan14_lut_int16.h")
LUT_OUTPUT_DIR = Path("artifacts/lut_reproduction") / f"run_{time.time_ns()}"


def lut_commands(output_dir: Path, csv: Path) -> list[tuple[str, list[str]]]:
    """Le cinque fasi del replay LUT, nell'ordine che le lega l'una all'altra.

    1. preparazione dei SOLI ingressi di training, dal CSV di TON_IoT;
    2. selezione della LUT su quella calibrazione, e scrittura del protocollo;
    3. confronto byte per byte dell'header rigenerato con quello misurato
       (lo fa il chiamante, subito dopo la fase 2: se differisce ci si ferma);
    4. preparazione degli ingressi di test, vincolata a quel protocollo;
    5. valutazione sul test.

    L'ordine non e' una convenzione di questo file: lo impone
    `prepare_finalization_data.py`, che in fase `evaluation` rifiuta di
    partire senza un protocollo gia' scritto e ne verifica gli hash contro la
    calibrazione. Invertire due fasi non produce un risultato sbagliato,
    produce un errore.

    Tutto, calibrazione compresa, viene scritto DENTRO `output_dir`, che e'
    nuova a ogni esecuzione. Cosi' la ricostruzione non tocca
    `artifacts/finalization/`: un nuovo run ha una provenienza nuova, e gli
    artefatti consegnati restano quelli consegnati. Il preprocessing viene
    ricostruito, i classificatori non vengono riaddestrati.
    """
    exporter = "scripts/export_kan14_lut_c.py"
    prep = "scripts/prepare_finalization_data.py"
    dati = output_dir / "finalization"
    protocollo = output_dir / "lut_selection_protocol.json"
    return [
        ("prepara-train",
         [PY, prep, "train", "--csv", str(csv), "--out", str(dati)]),
        ("select",
         [PY, exporter, "select",
          "--calibration", str(dati / "train_calibration.npz"),
          "--metadata", str(dati / "train_calibration.json"),
          "--model", "mcu_pio/include/kan14_coeff_int8.h",
          "--header", str(output_dir / LUT_FROZEN_HEADER.name),
          "--output-dir", str(output_dir)]),
        ("prepara-test",
         [PY, prep, "evaluation", "--csv", str(csv), "--out", str(dati),
          "--lut-protocol", str(protocollo)]),
        ("evaluate",
         [PY, exporter, "evaluate", "--protocol", str(protocollo),
          "--test-data", str(dati / "test_evaluation.npz"),
          "--output-dir", str(output_dir)]),
    ]

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
    "multiclass-state": (
        "riaddestra la KAN multi-layer multiclasse (artifacts/mlcat_state.pkl). "
        "FUORI da 'all': i due header C a 10 classi sono congelati, vedi sotto",
        # Il budget e' un meccanismo di RIPRESA, non un criterio di arresto:
        # con un valore piccolo lo script salva un checkpoint parziale e
        # ritorna, e gli esportatori esporterebbero un modello mezzo
        # addestrato senza accorgersene. Qui gliene si da' abbastanza da
        # arrivare in fondo alle 300 epoche in una volta sola. Deve stampare
        # "DONE"; se stampa "CHECKPOINT" va rilanciato.
        [[PY, "scripts/kan_ml_cat_mc.py", "100000"]],
    ),
    "integer": (
        "pipeline integer-only end-to-end binaria: export in C + golden vectors "
        "(rigenera kan_e2e_int.h byte per byte)",
        [[PY, "scripts/export_e2e_int_c.py"],
         [PY, "scripts/e2e_int_pipeline.py"]],
    ),
    "firmware-size": (
        "compila ogni environment con PlatformIO e misura Flash e SRAM del "
        "binario flashato (results/firmware_size.csv + il blocco del README). "
        "FUORI da 'all': richiede PlatformIO e le sue toolchain",
        [[PY, "scripts/firmware_size.py", "--readme"]],
    ),
    "integer-10classi": (
        "rigenera i due header C a 10 classi. FUORI da 'all', e da usare solo "
        "sapendo che sovrascrive artefatti congelati con versioni equivalenti "
        "ma non identiche: richiede prima lo stage 'multiclass-state'",
        [[PY, "scripts/export_mc_e2e_int_c.py"],
         [PY, "scripts/export_kan14_mc_coeff_c.py"]],
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
    "baseline-c": (
        "esporta in C intero le due baseline non-KAN che vanno su scheda: "
        "albero profondo 5 e MLP a 16 nascosti. Sta PRIMA di 'footprint' "
        "perche' i byte di entrambe si leggono dagli header prodotti qui",
        # export_tree_c.py non era in nessuno stage: l'header dell'albero
        # esisteva nel repository ma nessuna riproduzione lo rigenerava, ed
        # e' il modello con cui la KAN viene confrontata piu' spesso.
        [[PY, "scripts/export_tree_c.py"],
         [PY, "scripts/export_mlp_int_c.py"]],
    ),
    "lut": (
        "cinque fasi legate: prepara il solo training dal CSV, seleziona la LUT "
        "e scrive il protocollo, confronta l'header rigenerato con quello "
        "misurato, prepara il test vincolandolo a quel protocollo, valuta. "
        "Scrive tutto in una directory nuova: 'footprint' continua a leggere "
        "l'header congelato, che non viene mai sovrascritto. Ricostruisce il "
        "preprocessing, non riaddestra i classificatori. Serve il CSV di "
        "TON_IoT: percorso predefinito del progetto, oppure --csv",
        [],   # i comandi dipendono da --lut-output-dir e --csv: li costruisce main()
    ),
    "footprint": (
        "ingombro dei parametri di tutti i modelli, regola di conteggio unica",
        [[PY, "scripts/footprint.py"]],
    ),
    "footprint-architettura": (
        "compila la configurazione che la selezione sceglie (h=32 grado=6) e "
        "ne misura l'ingombro accanto a quella deployata, per rendere il "
        "compromesso accuratezza/memoria un numero misurato invece che una "
        "affermazione. Richiede 'architettura' per sapere quale sia la scelta",
        [[PY, "scripts/footprint_architettura.py"]],
    ),
    "interpretabilita": (
        "funzioni apprese e contributi locali della KAN single-layer, letti "
        "dall'header deployato. Non serve il dataset: coefficienti e vettori "
        "di verifica stanno negli header committati",
        [[PY, "scripts/interpretabilita.py"]],
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
    "architettura": (
        "sceglie larghezza e grado della KAN su una validation interna al "
        "training (regola 1-SE). Sta PRIMA di cv-binary perche' tutto quello "
        "che viene dopo usa l'architettura che esce di qui. E' lo stage piu' "
        "lungo di 'all': se serve solo rifare le tabelle, si salta e valgono "
        "i valori gia' scelti in results/arch_selection_scelta.json.",
        [[PY, "scripts/select_architettura.py"]],
    ),
    "joint": (
        "joint training TON+BoT: scelta del rapporto su validation, poi "
        "valutazione unica sui test e su UNSW-NB15",
        # PRIMO comando: sceglie il rapporto guardando SOLO la validation
        # ritagliata dentro il training. I test set non vengono toccati.
        # SECONDO e TERZO: valutano una volta sola, al rapporto gia' scelto,
        # che joint_training.py rilegge da
        # results/joint_ratio_selection_scelta.json.
        [[PY, "scripts/joint_training.py", "--select-ratio"],
         [PY, "scripts/joint_training.py"],
         [PY, "scripts/joint_training.py", "--eval-extra", "unsw"]],
    ),
    "statistica": (
        "ricalcola i confronti fra modelli dai run archiviati, con il seed "
        "come unita' di analisi. Non riaddestra niente: si puo' rilanciare "
        "da solo ogni volta che i run cambiano",
        [[PY, "scripts/statistica_confronti.py"]],
    ),
    "tabelle": (
        "compone la tabella principale a sette colonne dai CSV, senza "
        "ricopiature a mano (richiede 'crossdomain' e 'joint')",
        [[PY, "scripts/tabella_finale.py"]],
    ),
    "leakage-audit": (
        "quantifica l'effetto del difetto del protocollo v1",
        [[PY, "scripts/leakage_audit.py"]],
    ),
    "pacchetto": (
        "costruisce l'archivio da consegnare (risultati, figure, header C, "
        "audit, indice). FUORI da 'all': si fa alla fine, quando tutto il "
        "resto e' aggiornato, con --firmware per includere i binari",
        [[PY, "scripts/pacchetto_finale.py"]],
    ),
    "conformal": (
        "calibrazione split-conformal ed estrazione della forma simbolica",
        [[PY, "scripts/kan14_conformal_symbolic.py"]],
    ),
}

ORDER = ["tests", "audit", "leakage-audit", "features", "architettura",
         "cv-binary", "cv-multiclass",
         "crossdomain", "joint", "statistica", "tabelle", "compile", "integer", "conformal",
         "nested-cv", "models", "baseline-c", "lut", "footprint",
         "footprint-architettura", "interpretabilita", "figures", "report"]

# Due stage restano FUORI da ORDER di proposito: "multiclass-state" e
# "integer-10classi". I due header C a 10 classi derivano da uno stato
# addestrato che non e' riproducibile bit per bit — 300 epoche di Adam
# amplificano l'ordine delle riduzioni BLAS fino a spostare un campione
# MITM su 208, e il macro-F1 passa da 0,9378 a 0,9384. Sono artefatti
# congelati, verificati bit-esatti dai check host; rigenerarli dentro
# `--stage all` sostituirebbe in silenzio un artefatto di deployment
# verificato con uno diverso, senza guadagnare riproducibilita'.
#
# Il relatore ha chiesto (rc3, punto 7) di salvare lo stato canonico da cui
# quegli header si rigenerano esattamente. La via e' questa, e va percorsa
# UNA volta, di proposito:
#
#     python reproduce.py --stage multiclass-state    # riaddestra
#     python scripts/export_models.py                 # copia in models/
#     python reproduce.py --stage integer-10classi    # riemette i due header
#
# Da li' in poi la differenza e' sostanziale: l'header smette di essere
# l'unica copia di una cosa perduta e diventa la funzione deterministica di
# un file versionato, che tests/test_stato_multiclasse.py riemette e
# confronta byte per byte. Il riaddestramento resta fuori da `all` — rifarlo
# darebbe un altro stato, non lo stesso — mentre l'export da quello stato e'
# deterministico. Il prezzo, da dichiarare quando si fa: le cifre del
# modello a 10 classi si spostano di 6 decimillesimi di macro-F1.


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


def ambiente_figlio() -> dict:
    """L'ambiente degli script lanciati da qui, con l'output in UTF-8.

    Gli script che importano kanids si riconfigurano da soli (vedi
    kanids/console.py), ma non tutti lo importano — export_lut_int.py e
    compagni no. Su Windows, se `reproduce.py` gira con l'output rediretto
    (`> riproduzione.log`, che e' il modo naturale di conservare una
    riproduzione lunga), quei figli ereditano una pipe, scelgono cp1252 e
    muoiono sulla prima delta o freccia che stampano. PYTHONIOENCODING
    copre anche loro, senza doverli modificare uno per uno.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=REPO, env=ambiente_figlio())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="smoke",
                    help="uno stage, 'all', o 'clean'. --list per l'elenco")
    ap.add_argument("--list", action="store_true", help="elenca gli stage")
    ap.add_argument("--dry-run", action="store_true", help="stampa i comandi senza eseguirli")
    ap.add_argument("--csv", type=Path, default=None,
                    help="CSV di TON_IoT per lo stage 'lut'. Senza questa "
                         "opzione si usa il percorso predefinito del progetto "
                         "(KANIDS_DATA o data/train_test_network.csv)")
    ap.add_argument("--lut-output-dir", type=Path, default=LUT_OUTPUT_DIR,
                    help="directory nuova per la riproduzione LUT; non riusare risultati congelati")
    args = ap.parse_args()

    if args.list:
        print("stage disponibili:\n")
        for k in ["smoke"] + ORDER:
            print(f"  {k:<18}{STAGES[k][0]}")
        fuori = [k for k in STAGES if k not in ORDER and k != "smoke"]
        if fuori:
            print("\nfuori da 'all', da lanciare solo esplicitamente:\n")
            for k in fuori:
                print(f"  {k:<18}{STAGES[k][0]}")
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
        etichette = None
        if s == "lut":
            # `ton_iot_path` solleva se il file non c'e', e il suo messaggio
            # dice gia' dove metterlo: lo si riporta come fallimento dello
            # stage invece che come traccia di eccezione, perche' un
            # prerequisito mancante non e' un errore di programmazione.
            try:
                csv = Path(args.csv) if args.csv else ton_iot_path()
            except FileNotFoundError as e:
                failed.append((s, str(e)))
                print(f"!! {e}", file=sys.stderr)
                continue
            if not csv.is_file():
                failed.append((s, f"CSV di TON_IoT non trovato: {csv}"))
                print(f"!! CSV di TON_IoT non trovato: {csv}", file=sys.stderr)
                continue
            passi = lut_commands(args.lut_output_dir, csv)
            etichette = [e for e, _ in passi]
            cmds = [c for _, c in passi]
            print(f"LUT_REPRODUCTION_DIRECTORY: {REPO / args.lut_output_dir}")
            print(f"LUT_SOURCE_CSV: {csv}")
            if not args.dry_run:
                print(f"LUT_SOURCE_CSV_SHA256: {sha256_file(csv)}")
            if not args.dry_run and (REPO / args.lut_output_dir).exists():
                failed.append((s, "LUT output directory already exists; choose a new directory"))
                print("!! LUT output already exists; preserving it", file=sys.stderr)
                continue
        print(f"\n{'─' * 74}\nSTAGE {s} — {desc}\n{'─' * 74}")
        for i, cmd in enumerate(cmds):
            if args.dry_run:
                print(f"$ {' '.join(cmd)}")
                continue
            if run(cmd) != 0:
                failed.append((s, " ".join(cmd)))
                print(f"!! stage {s} fallito", file=sys.stderr)
                if s == "lut":
                    # Ci si ferma qui: nessuna fase successiva, e in
                    # particolare nessuna lettura dei dati di test dopo una
                    # preparazione, una selezione o un confronto falliti.
                    break
            elif s == "lut" and etichette[i] == "select":
                replay_header = REPO / args.lut_output_dir / LUT_FROZEN_HEADER.name
                if (not replay_header.is_file() or
                        replay_header.read_bytes() != (REPO / LUT_FROZEN_HEADER).read_bytes()):
                    failed.append((s, "LUT replay header differs from the frozen measured header"))
                    print("!! LUT header mismatch; test evaluation skipped", file=sys.stderr)
                    break
                print("LUT_FROZEN_HEADER_BYTE_MATCH: PASS")

    print(f"\n{'=' * 74}\ncompletato in {time.time() - t0:.0f}s")
    if failed:
        print("stage falliti:")
        for s, c in failed:
            print(f"  {s}: {c}")
        sys.exit(1)
    if args.stage == "lut":
        print(f"risultati LUT in {REPO / args.lut_output_dir}")
    else:
        print(f"risultati in {RESULTS_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rigenera i risultati di `adattamento-drift/` sotto il protocollo corretto.

Perche' esiste: `kanids/valutazione.py` ha introdotto la partizione
validation/test, e ogni script che valuta su un complemento delle righe
selezionate e' stato allineato. I risultati prodotti prima stanno in
`results/protocollo_v1/` e restano validi come valutazione -- il complemento
non era contaminato -- ma su un test set diverso, quindi non vanno mescolati
con i numeri nuovi. Questo script li rigenera, tutti sugli stessi 10 seed.

Gli script sono checkpointati: interrompere con Ctrl-C e rilanciare lo stesso
comando riprende da dove si era fermato, senza ricalcolare nulla.

Uso:

    python rigenera.py --dati "C:\\Users\\...\\kanids-data" --lista

oppure impostando la variabile d'ambiente, con la sintassi della shell in
uso -- in PowerShell `set` NON imposta una variabile d'ambiente:

    $env:KANIDS_DATA = "C:\\percorso\\ai\\dataset"   (PowerShell)
    set KANIDS_DATA=C:\\percorso\\ai\\dataset         (cmd.exe)
    export KANIDS_DATA=/percorso/ai/dataset       (bash/zsh)

    python rigenera.py --lista            # cosa farebbe, senza fare nulla
    python rigenera.py                    # gli stage del paper, in ordine
    python rigenera.py --stage int_adapt  # uno solo
    python rigenera.py --tutto            # anche gli stage secondari

Ordine: dal piu' economico al piu' caro, cosi' i primi risultati arrivano
subito e un'interruzione costa poco.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
SEED = "42,43,44,45,46,47,48,49,50,51"
SEI = "ton->bot,bot->ton,ton->unsw,unsw->ton,bot->unsw,unsw->bot"

PROTOCOLLO = Path(__file__).resolve().parent / "kanids" / "valutazione.py"


def checkpoint_obsoleto(argomenti) -> Path | None:
    """Un checkpoint scritto prima del cambio di protocollo contiene numeri
    che non sono confrontabili con quelli nuovi, e -- peggio -- verrebbe
    riusato in silenzio, perche' gli script saltano cio' che e' gia' fatto.
    E' esattamente il modo in cui un risultato vecchio rientra da una porta
    laterale, quindi qui si controlla invece di sperare."""
    script = Path(argomenti[0]).stem
    base = script.replace("drift_", "drift_") if script != "tre_domini" else "tre_domini"
    for ckpt in (Path(__file__).resolve().parent / "artifacts").glob(f"{base}*.jsonl"):
        if "ratio" in ckpt.stem or "iters" in ckpt.stem or "adaptive" in ckpt.stem:
            continue          # varianti con nome proprio, non toccate da questi stage
        if ckpt.stat().st_mtime < PROTOCOLLO.stat().st_mtime:
            return ckpt
    return None


# (nome, argomenti, nel_paper, cosa produce)
STAGE = [
    ("spazio_ridotto", ["scripts/spazio_ridotto.py", "--seeds", SEED], True,
     "costo della riduzione dello spazio armonizzato (sezione 10)"),
    ("senza_etichette", ["scripts/drift_senza_etichette.py", "--seeds", SEED], True,
     "EM, TENT, TENT filtrato, IM/SHOT (sezione 8)"),
    ("sampling", ["scripts/drift_sampling.py", "--seeds", SEED], True,
     "regole di selezione delle etichette (sezione 4)"),
    ("diagnosi", ["scripts/drift_diagnosi.py", "--seeds", SEED], True,
     "soglia contro rappresentazione, ROC-AUC target (sezione 1)"),
    ("baselines", ["scripts/drift_baselines.py", "--seeds", SEED], True,
     "aggiornamento minimo strutturale per modello (sezione 5)"),
    ("int_adapt", ["scripts/drift_int_adapt.py", "--seeds", SEED, "--exp", SEI], True,
     "catena integer-only sulle sei direzioni (sezioni 6 e 13)"),
    ("tre_domini_ricco", ["scripts/tre_domini.py", "--spazio", "ricco",
                          "--domini", "ton,bot,unsw", "--seeds", SEED], True,
     "sei direzioni nello spazio ricco (sezione 11)"),
    ("adapt", ["scripts/drift_adapt.py", "--seeds", SEED], False,
     "scala di interventi (sezione 3, superata dalla 11)"),
    ("trasferimenti", ["scripts/drift_trasferimenti.py", "--seeds", SEED], False,
     "Firth e k-center (sezione 9)"),
    ("tre_domini_ridotto_unsw", ["scripts/tre_domini.py", "--spazio", "ridotto",
                                 "--domini", "ton,bot,unsw", "--seeds", SEED], False,
     "terna UNSW nello spazio ridotto (sezione 15)"),
    ("tre_domini_ridotto_cic", ["scripts/tre_domini.py", "--spazio", "ridotto",
                                "--domini", "ton,bot,cic", "--seeds", SEED], False,
     "terna CIC nello spazio ridotto (sezione 15)"),
]


def durata(s: float) -> str:
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append",
                    help="esegui solo questo stage (ripetibile)")
    ap.add_argument("--tutto", action="store_true",
                    help="includi anche gli stage secondari")
    ap.add_argument("--lista", action="store_true",
                    help="stampa cosa verrebbe eseguito e termina")
    ap.add_argument("--dati", metavar="CARTELLA",
                    help="cartella dei dataset; alternativa a KANIDS_DATA, "
                         "utile perche' la sintassi per impostare una "
                         "variabile d'ambiente cambia fra cmd e PowerShell")
    args = ap.parse_args()

    if args.stage:
        ignoti = set(args.stage) - {s[0] for s in STAGE}
        if ignoti:
            print(f"stage sconosciuti: {sorted(ignoti)}", file=sys.stderr)
            print(f"disponibili: {[s[0] for s in STAGE]}", file=sys.stderr)
            return 2
        scelti = [s for s in STAGE if s[0] in args.stage]
    else:
        scelti = [s for s in STAGE if s[2] or args.tutto]

    if args.lista:
        print(f"{'stage':<26} {'nel paper':<10} cosa produce")
        for nome, _, paper, desc in STAGE:
            print(f"{nome:<26} {'si' if paper else 'no':<10} {desc}")
        print(f"\nseed: {SEED}")
        print("selezionati ora:", ", ".join(s[0] for s in scelti))
        return 0

    if args.dati:
        os.environ["KANIDS_DATA"] = str(Path(args.dati).expanduser().resolve())
    dati = os.environ.get("KANIDS_DATA")
    if not dati:
        print("Non so dove sono i dataset. Passa --dati, oppure imposta\n"
              "KANIDS_DATA con la sintassi della tua shell:\n\n"
              '  PowerShell   $env:KANIDS_DATA = "C:\\percorso\\ai\\dataset"\n'
              "  cmd.exe      set KANIDS_DATA=C:\\percorso\\ai\\dataset\n"
              "  bash/zsh     export KANIDS_DATA=/percorso/ai/dataset\n\n"
              "Servono i quattro dataset: train_test_network.csv,\n"
              "UNSW_2018_IoT_Botnet_Full5pc_1..4.csv, UNSW_NB15_*-set.csv,\n"
              "test.csv. Vedi README.md.", file=sys.stderr)
        return 2
    if not Path(dati).is_dir():
        print(f"la cartella dei dataset non esiste: {dati}", file=sys.stderr)
        return 2
    attesi = ["train_test_network.csv", "UNSW_NB15_training-set.csv",
              "UNSW_2018_IoT_Botnet_Full5pc_1.csv"]
    mancanti = [f for f in attesi if not (Path(dati) / f).exists()]
    if mancanti:
        print(f"in {dati} mancano: {', '.join(mancanti)}", file=sys.stderr)
        print("controlla che sia la cartella giusta.", file=sys.stderr)
        return 2
    print(f"dataset: {dati}")

    obsoleti = [(n, c) for n, a, _, _ in scelti
                if (c := checkpoint_obsoleto(a)) is not None]
    if obsoleti:
        print("Questi stage hanno un checkpoint scritto PRIMA del cambio di\n"
              "protocollo. Rilanciandoli verrebbe riusato in silenzio, e i\n"
              "numeri vecchi rientrerebbero mescolati ai nuovi:\n", file=sys.stderr)
        for nome, ckpt in obsoleti:
            print(f"  {nome:<24} {ckpt.relative_to(QUI)}", file=sys.stderr)
        print("\nSpostali in artifacts/protocollo_v1/ (insieme ai loro CSV in\n"
              "results/) e rilancia.", file=sys.stderr)
        return 2

    print(f"{len(scelti)} stage, seed {SEED}")
    print("gli script sono checkpointati: Ctrl-C e rilancio riprendono da qui.\n")
    t0 = time.time()
    esiti = []
    for i, (nome, argomenti, _, desc) in enumerate(scelti, 1):
        print(f"[{i}/{len(scelti)}] {nome} — {desc}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, *argomenti], cwd=QUI)
        esiti.append((nome, r.returncode, time.time() - t))
        print(f"    {'ok' if r.returncode == 0 else 'USCITA ' + str(r.returncode)}"
              f" in {durata(time.time() - t)}\n", flush=True)

    print("=" * 64)
    for nome, codice, sec in esiti:
        print(f"  {nome:<26} {'ok' if codice == 0 else 'FALLITO':<8} {durata(sec):>10}")
    print(f"  {'totale':<26} {'':<8} {durata(time.time() - t0):>10}")
    print("=" * 64)
    return 0 if all(c == 0 for _, c, _ in esiti) else 1


if __name__ == "__main__":
    raise SystemExit(main())

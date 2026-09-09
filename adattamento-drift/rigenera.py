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

    python rigenera.py --lista

I quattro CSV vengono cercati in `kanids-data/` accanto al repo; se stanno
altrove, `--dati <cartella>` la passa una volta sola e viene ricordata in
`.ultima_cartella_dati`.

oppure impostando la variabile d'ambiente, con la sintassi della shell in
uso -- in PowerShell `set` NON imposta una variabile d'ambiente:

    $env:KANIDS_DATA = "<cartella>"      (PowerShell)
    set KANIDS_DATA=<cartella>           (cmd.exe)
    export KANIDS_DATA=<cartella>        (bash/zsh)

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

# Dove i dataset sono stati trovati l'ultima volta. Non e' una comodita': la
# cartella e' fuori dal repo, il percorso e' lungo, e ridigitarlo a ogni
# rilancio e' un modo di sbagliarlo -- l'errore piu' probabile e' passare il
# segnaposto della documentazione, che nomina una cartella inesistente.
MEMO_DATI = Path(__file__).resolve().parent / ".ultima_cartella_dati"

# I quattro CSV vivono fuori da `data/` (che ne contiene uno solo) e sotto un
# nome che nessuno indovina: cercarli qui evita di dover ricordare il
# percorso, ed evita l'errore opposto -- passare il segnaposto della
# documentazione, che nomina una cartella inesistente.
CANDIDATE = ("kanids-data", "data", "../kanids-data", "../data")
ATTESI = ("train_test_network.csv", "UNSW_NB15_training-set.csv",
          "UNSW_2018_IoT_Botnet_Full5pc_1.csv")


def cerca_i_dataset() -> Path | None:
    radice = Path(__file__).resolve().parent
    for c in CANDIDATE:
        cartella = (radice / c).resolve()
        if all((cartella / f).exists() for f in ATTESI):
            return cartella
    return None


def checkpoint_obsoleto(ckpt_nomi) -> Path | None:
    """Un checkpoint scritto prima del cambio di protocollo contiene numeri
    che non sono confrontabili con quelli nuovi, e -- peggio -- verrebbe
    riusato in silenzio, perche' gli script saltano cio' che e' gia' fatto.
    E' esattamente il modo in cui un risultato vecchio rientra da una porta
    laterale, quindi qui si controlla invece di sperare.

    Ogni stage dichiara i propri checkpoint per nome: la versione precedente
    li cercava con una glob e saltava tutto cio' che conteneva "ratio", e
    proprio per quella scorciatoia i file della sezione 18 sarebbero passati
    inosservati."""
    for nome in ckpt_nomi:
        ckpt = QUI / "artifacts" / nome
        if ckpt.exists() and ckpt.stat().st_mtime < PROTOCOLLO.stat().st_mtime:
            return ckpt
    return None


# (nome, argomenti, nel_paper, cosa produce, checkpoint scritti)
STAGE = [
    ("spazio_ridotto", ["scripts/spazio_ridotto.py", "--seeds", SEED], "paper",
     "costo della riduzione dello spazio armonizzato (sezione 10)",
     ["spazio_ridotto.jsonl"]),
    ("senza_etichette", ["scripts/drift_senza_etichette.py", "--seeds", SEED,
                         "--exp", SEI], "paper",
     "EM, TENT, TENT filtrato, IM/SHOT sulle sei direzioni (sezioni 8 e 12)",
     ["drift_senza_etichette.jsonl"]),
    ("sampling", ["scripts/drift_sampling.py", "--seeds", SEED], "paper",
     "regole di selezione delle etichette (sezione 4)",
     ["drift_sampling.jsonl"]),
    ("diagnosi", ["scripts/drift_diagnosi.py", "--seeds", SEED], "paper",
     "soglia contro rappresentazione, ROC-AUC target (sezione 1)",
     ["drift_diagnosi.jsonl"]),
    ("baselines", ["scripts/drift_baselines.py", "--seeds", SEED], "paper",
     "aggiornamento minimo strutturale per modello (sezione 5)",
     ["drift_baselines.jsonl"]),
    ("int_adapt", ["scripts/drift_int_adapt.py", "--seeds", SEED, "--exp", SEI], "paper",
     "catena integer-only sulle sei direzioni (sezioni 6 e 13)",
     ["drift_int_adapt.jsonl"]),
    ("tre_domini_ricco", ["scripts/tre_domini.py", "--spazio", "ricco",
                          "--domini", "ton,bot,unsw", "--seeds", SEED], "paper",
     "sei direzioni nello spazio ricco (sezione 11)",
     ["tre_domini_ricco_tonbotunsw.jsonl"]),
    ("adapt", ["scripts/drift_adapt.py", "--seeds", SEED], "secondario",
     "scala di interventi (sezione 3, superata dalla 11)",
     ["drift_adapt.jsonl"]),
    ("trasferimenti", ["scripts/drift_trasferimenti.py", "--seeds", SEED,
                       "--exp", SEI], "paper",
     "Firth e k-center sulle sei direzioni (sezioni 9 e 12)",
     ["drift_trasferimenti.jsonl"]),
    ("tre_domini_ridotto_unsw", ["scripts/tre_domini.py", "--spazio", "ridotto",
                                 "--domini", "ton,bot,unsw", "--seeds", SEED],
     "secondario", "terna UNSW nello spazio ridotto (sezione 15)",
     ["tre_domini_ridotto_tonbotunsw.jsonl"]),
    ("tre_domini_ridotto_cic", ["scripts/tre_domini.py", "--spazio", "ridotto",
                                "--domini", "ton,bot,cic", "--seeds", SEED],
     "secondario", "terna CIC nello spazio ridotto (sezione 15)",
     ["tre_domini_ridotto_tonbotcic.jsonl"]),
]

# Sezione 18 --- sensibilita' al rapporto di undersampling.
#
# Non e' una ripetizione della griglia intera. `undersample()` tiene tutta la
# classe minoritaria e taglia la maggioritaria a `ratio` volte la minoritaria,
# con un `min()`: se il rapporto naturale del dominio sorgente e' gia' piu'
# equilibrato di `ratio`, la funzione e' un no-op e il training e' identico
# bit per bit. I rapporti naturali sono TON 3,22:1, UNSW 1,77:1, BoT 7 690:1,
# quindi vincolano solo queste combinazioni:
#
#   ratio 1     BoT, TON, UNSW      -> tutte e sei le direzioni
#   ratio 3     BoT, TON            -> quattro direzioni
#   ratio 20    BoT                 -> due direzioni
#   ratio 100   BoT                 -> due direzioni
#
# Tutto il resto e' gia' misurato a ratio 50 e ricalcolarlo produrrebbe per
# costruzione lo stesso numero. `drift_graduale.py` e `drift_graduale_int.py`
# non compaiono qui: sono prequenziali (valutano il batch prima di
# addestrarci sopra), non usano `kanids/valutazione.py` e quindi il cambio di
# protocollo non li tocca -- i loro numeri a ogni rapporto restano validi.
BOT_SRC = "bot->ton,bot->unsw"
QUATTRO = "ton->bot,ton->unsw,bot->ton,bot->unsw"

def _s18(nome, argomenti, desc, ckpt):
    return (nome, argomenti, "sezione18", desc, ckpt)


STAGE += [
    _s18("s18_tre_domini_r1",
         ["scripts/tre_domini.py", "--spazio", "ricco", "--domini",
          "ton,bot,unsw", "--seeds", SEED, "--ratio", "1"],
         "tre_domini a ratio 1: vincola tutte e tre le sorgenti (sezione 18)",
         ["tre_domini_ricco_tonbotunsw_ratio1.jsonl"]),
    _s18("s18_tre_domini_r3_ton",
         ["scripts/tre_domini.py", "--spazio", "ricco", "--domini",
          "ton,bot,unsw", "--src", "ton", "--seeds", SEED, "--ratio", "3"],
         "tre_domini a ratio 3, sorgente TON (sezione 18)",
         ["tre_domini_ricco_tonbotunsw_ratio3.jsonl"]),
    _s18("s18_tre_domini_r3_bot",
         ["scripts/tre_domini.py", "--spazio", "ricco", "--domini",
          "ton,bot,unsw", "--src", "bot", "--seeds", SEED, "--ratio", "3"],
         "tre_domini a ratio 3, sorgente BoT (sezione 18)",
         ["tre_domini_ricco_tonbotunsw_ratio3.jsonl"]),
    _s18("s18_tre_domini_r20",
         ["scripts/tre_domini.py", "--spazio", "ricco", "--domini",
          "ton,bot,unsw", "--src", "bot", "--seeds", SEED, "--ratio", "20"],
         "tre_domini a ratio 20, sorgente BoT (sezione 18)",
         ["tre_domini_ricco_tonbotunsw_ratio20.jsonl"]),
    _s18("s18_tre_domini_r100",
         ["scripts/tre_domini.py", "--spazio", "ricco", "--domini",
          "ton,bot,unsw", "--src", "bot", "--seeds", SEED, "--ratio", "100"],
         "tre_domini a ratio 100, sorgente BoT (sezione 18)",
         ["tre_domini_ricco_tonbotunsw_ratio100.jsonl"]),
    _s18("s18_int_adapt_r1",
         ["scripts/drift_int_adapt.py", "--seeds", SEED, "--exp", SEI,
          "--ratio", "1"],
         "catena intera a ratio 1, sei direzioni (sezione 18)",
         ["drift_int_adapt_ratio1.jsonl"]),
    _s18("s18_int_adapt_r3",
         ["scripts/drift_int_adapt.py", "--seeds", SEED, "--exp", QUATTRO,
          "--ratio", "3"],
         "catena intera a ratio 3, quattro direzioni (sezione 18)",
         ["drift_int_adapt_ratio3.jsonl"]),
    _s18("s18_int_adapt_r20",
         ["scripts/drift_int_adapt.py", "--seeds", SEED, "--exp", BOT_SRC,
          "--ratio", "20"],
         "catena intera a ratio 20, sorgente BoT (sezione 18)",
         ["drift_int_adapt_ratio20.jsonl"]),
    _s18("s18_int_adapt_r100",
         ["scripts/drift_int_adapt.py", "--seeds", SEED, "--exp", BOT_SRC,
          "--ratio", "100"],
         "catena intera a ratio 100, sorgente BoT (sezione 18)",
         ["drift_int_adapt_ratio100.jsonl"]),
]


# Sezione 19 --- la guardia sui batch a una classe sola.
#
# `stat_13x13_guardia` e' affiancata a `stat_13x13`, non la sostituisce:
# serve la stessa corsa per confrontarle. Le altre sette politiche devono
# uscire identiche a `results/prima_della_guardia/` (nessuna politica
# consuma il generatore condiviso), e il test lo verifica. Le direzioni per
# rapporto seguono la stessa identita' di `undersample()` della sezione 18.
STAGE += [
    ("guardia_r50", ["scripts/drift_graduale.py", "--seeds", SEED,
                     "--exp", SEI], "guardia",
     "deriva graduale a ratio 50, sei direzioni (sezioni 7, 16.2, 19)",
     ["drift_graduale.jsonl"]),
    ("guardia_r1", ["scripts/drift_graduale.py", "--seeds", SEED,
                    "--exp", SEI, "--ratio", "1"], "guardia",
     "deriva graduale a ratio 1, sei direzioni (sezione 19)",
     ["drift_graduale_ratio1.jsonl"]),
    ("guardia_r3", ["scripts/drift_graduale.py", "--seeds", SEED,
                    "--exp", QUATTRO, "--ratio", "3"], "guardia",
     "deriva graduale a ratio 3, quattro direzioni (sezione 19)",
     ["drift_graduale_ratio3.jsonl"]),
    ("guardia_r20", ["scripts/drift_graduale.py", "--seeds", SEED,
                     "--exp", BOT_SRC, "--ratio", "20"], "guardia",
     "deriva graduale a ratio 20, sorgente BoT (sezione 19)",
     ["drift_graduale_ratio20.jsonl"]),
    ("guardia_r100", ["scripts/drift_graduale.py", "--seeds", SEED,
                      "--exp", BOT_SRC, "--ratio", "100"], "guardia",
     "deriva graduale a ratio 100, sorgente BoT (sezione 19)",
     ["drift_graduale_ratio100.jsonl"]),
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
                    help="includi gli stage secondari e quelli della sezione 18")
    ap.add_argument("--sezione18", action="store_true",
                    help="solo la griglia dei rapporti di undersampling "
                         "(sezione 18)")
    ap.add_argument("--guardia", action="store_true",
                    help="solo la deriva graduale con la guardia sui batch "
                         "monoclasse (sezione 19), ~50 minuti")
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
        gruppi = {"paper"}
        if args.sezione18:
            gruppi = {"sezione18"}
        if args.guardia:
            gruppi = {"guardia"}
        if args.tutto:
            gruppi = {"paper", "secondario", "sezione18", "guardia"}
        scelti = [s for s in STAGE if s[2] in gruppi]

    if args.lista:
        print(f"{'stage':<26} {'gruppo':<12} cosa produce")
        for nome, _, gruppo, desc, _ck in STAGE:
            print(f"{nome:<26} {gruppo:<12} {desc}")
        print(f"\nseed: {SEED}")
        print("selezionati ora:", ", ".join(s[0] for s in scelti))
        return 0

    if args.dati:
        os.environ["KANIDS_DATA"] = str(Path(args.dati).expanduser().resolve())
    dati = os.environ.get("KANIDS_DATA")
    if not dati and (trovata := cerca_i_dataset()) is not None:
        dati = str(trovata)
        os.environ["KANIDS_DATA"] = dati
        print(f"dataset trovati accanto al repo: {dati}")
    if not dati and MEMO_DATI.exists():
        ricordata = MEMO_DATI.read_text(encoding="utf-8").strip()
        if Path(ricordata).is_dir():
            dati = ricordata
            os.environ["KANIDS_DATA"] = dati
            print(f"cartella dei dataset dell'ultimo run: {dati}")
            print("(passa --dati per usarne un'altra)")
    if not dati:
        print("Non so dove sono i dataset. Passa --dati, oppure imposta\n"
              "KANIDS_DATA con la sintassi della tua shell:\n\n"
              '  PowerShell   $env:KANIDS_DATA = "<cartella>"\n'
              "  cmd.exe      set KANIDS_DATA=<cartella>\n"
              "  bash/zsh     export KANIDS_DATA=<cartella>\n\n"
              "Servono i quattro dataset: train_test_network.csv,\n"
              "UNSW_2018_IoT_Botnet_Full5pc_1..4.csv, UNSW_NB15_*-set.csv,\n"
              "test.csv. Vedi README.md.", file=sys.stderr)
        return 2
    if not Path(dati).is_dir():
        print(f"la cartella dei dataset non esiste: {dati}", file=sys.stderr)
        if "percorso" in dati:
            print("\nQuello e' il segnaposto della documentazione, non un "
                  "percorso vero:\nsostituiscilo con la cartella dove stanno "
                  "davvero i quattro CSV\n(quella usata per la rigenerazione "
                  "principale). Dopo la prima volta\nnon serve piu' "
                  "ripeterlo: viene ricordata qui accanto, in "
                  f"{MEMO_DATI.name}.", file=sys.stderr)
        return 2
    mancanti = [f for f in ATTESI if not (Path(dati) / f).exists()]
    if mancanti:
        print(f"in {dati} mancano: {', '.join(mancanti)}", file=sys.stderr)
        print("controlla che sia la cartella giusta.", file=sys.stderr)
        return 2
    print(f"dataset: {dati}")
    MEMO_DATI.write_text(dati, encoding="utf-8", newline="\n")

    obsoleti = [(n, c) for n, _a, _g, _d, ck in scelti
                if (c := checkpoint_obsoleto(ck)) is not None]
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
    for i, (nome, argomenti, _g, desc, _ck) in enumerate(scelti, 1):
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

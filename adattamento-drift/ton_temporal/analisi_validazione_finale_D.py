"""Rigenera gli artefatti leggeri della validazione finale su D.

Legge soltanto i cinque rendiconti e il riepilogo gia' prodotti dalla corsa
ufficiale. Non carica D, non addestra modelli e non modifica i raw esterni.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


RADICE = Path(__file__).resolve().parent
ARTICOLO = RADICE / "articolo_02" / "validazione_finale_D"
RAW_PREDEFINITI = (
    RADICE.parent
    / "archivio_esterno"
    / "articolo_02_validazione_finale"
    / "campagna_D_ea21691"
)
SEMI = (42, 43, 44, 45, 46)
AMBITI = ("D_completo", "D_non_visto_ABC")
TRACCE = ("frozen_b", "frozen_cal64", "fit192_b", "fit192_cal64")
ETICHETTE_TRACCE = {
    "frozen_b": "frozen+B",
    "frozen_cal64": "frozen+CAL64",
    "fit192_b": "FIT192+B",
    "fit192_cal64": "FIT192+CAL64",
}
CAMPI_TABELLA = (
    "ambito",
    "traccia",
    "semi",
    "fn_media",
    "fn_sd",
    "fp_media",
    "fp_sd",
    "recall_pp_media",
    "recall_pp_sd",
    "fpr_pp_media",
    "fpr_pp_sd",
    "balanced_accuracy_pp_media",
    "balanced_accuracy_pp_sd",
    "auroc_media_blocco_media",
    "auroc_media_blocco_sd",
    "denominatore_auroc_per_seme",
)


def sha256_file(percorso: Path) -> str:
    h = hashlib.sha256()
    with percorso.open("rb") as f:
        for blocco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blocco)
    return h.hexdigest()


def _leggi_json(percorso: Path) -> dict:
    return json.loads(percorso.read_text(encoding="utf-8"))


def _scrivi_json(percorso: Path, dati: dict) -> None:
    percorso.write_text(
        json.dumps(dati, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _statistiche(valori) -> dict:
    valori = [float(v) for v in valori]
    return {
        "per_seme": valori,
        "media": statistics.mean(valori),
        "deviazione_standard": statistics.stdev(valori),
    }


def carica_fonti(raw_dir: Path, manifest_path: Path) -> tuple[list[dict], dict, dict]:
    manifest = _leggi_json(manifest_path)
    attesi = {int(v["seme"]): v for v in manifest["rendiconti_completi"]}
    if set(attesi) != set(SEMI):
        raise ValueError("il manifest non identifica esattamente i semi 42--46")

    rapporti = []
    for seme in SEMI:
        voce = attesi[seme]
        percorso = raw_dir / voce["file"]
        if not percorso.is_file():
            raise FileNotFoundError(percorso)
        if percorso.stat().st_size != voce["dimensione_byte"]:
            raise ValueError("dimensione inattesa per " + percorso.name)
        if sha256_file(percorso) != voce["sha256"]:
            raise ValueError("SHA-256 inatteso per " + percorso.name)
        rapporto = _leggi_json(percorso)
        if rapporto["seme"] != seme:
            raise ValueError("seme interno inatteso in " + percorso.name)
        if [b["blocco"] for b in rapporto["per_blocco"]] != list(range(355)):
            raise ValueError("blocchi incompleti o non ordinati in " + percorso.name)
        if rapporto["per_blocco"][-1]["righe"] != 9973:
            raise ValueError("ultimo blocco non conforme in " + percorso.name)
        rapporti.append(rapporto)

    riepilogo_path = raw_dir / "riepilogo.json"
    riepilogo_atteso = manifest["archivio_esterno"]["riepilogo_originale"]
    if riepilogo_path.stat().st_size != riepilogo_atteso["dimensione_byte"]:
        raise ValueError("dimensione inattesa per riepilogo.json")
    if sha256_file(riepilogo_path) != riepilogo_atteso["sha256"]:
        raise ValueError("SHA-256 inatteso per riepilogo.json")
    riepilogo = _leggi_json(riepilogo_path)
    if riepilogo["commit_eseguito"] != manifest["commit_eseguito"]:
        raise ValueError("commit eseguito non coerente")
    if riepilogo["D_usato_per_scegliere_il_disegno"] is not False:
        raise ValueError("D risulta usato per scegliere il disegno")
    return rapporti, riepilogo, manifest


def risultati_assoluti(rapporti: list[dict]) -> list[dict]:
    righe = []
    for ambito in AMBITI:
        for traccia in TRACCE:
            aggregati = [r["aggregati"][ambito][traccia] for r in rapporti]
            denominatori = {
                (a["auroc_blocchi_validi"], a["blocchi_totali"])
                for a in aggregati
            }
            if len(denominatori) != 1:
                raise ValueError("denominatori AUROC diversi fra semi")
            validi, totali = denominatori.pop()
            valori = {
                "fn": _statistiche(a["cm_globale"]["fn"] for a in aggregati),
                "fp": _statistiche(a["cm_globale"]["fp"] for a in aggregati),
                "recall_pp": _statistiche(
                    100.0 * a["recall_attacchi_globale"] for a in aggregati
                ),
                "fpr_pp": _statistiche(100.0 * a["fpr_globale"] for a in aggregati),
                "balanced_accuracy_pp": _statistiche(
                    100.0 * a["balanced_accuracy_globale"] for a in aggregati
                ),
                "auroc_media_blocco": _statistiche(
                    a["auroc_media_per_blocco"] for a in aggregati
                ),
            }
            righe.append(
                {
                    "ambito": ambito,
                    "traccia": traccia,
                    "etichetta": ETICHETTE_TRACCE[traccia],
                    "semi": list(SEMI),
                    "metriche": valori,
                    "denominatore_auroc_per_seme": f"{validi}/{totali}",
                }
            )
    return righe


def confronti(rapporti: list[dict]) -> dict:
    risultato = {}
    for ambito in AMBITI:
        risultato[ambito] = {}
        for nome in (
            "principale_pesi",
            "soglia_a_pesi_adattati",
            "soglia_a_pesi_frozen",
        ):
            campi = rapporti[0]["confronti"][ambito][nome]
            risultato[ambito][nome] = {
                campo: _statistiche(
                    r["confronti"][ambito][nome][campo] for r in rapporti
                )
                for campo in campi
            }

        principale = [
            r["confronti"][ambito]["principale_pesi"] for r in rapporti
        ]
        risultato[ambito]["principale_pesi"]["miglioramento_congiunto_fn_fp"] = {
            "semi": [
                seme
                for seme, delta in zip(SEMI, principale)
                if delta["delta_fn"] < 0 and delta["delta_fp"] < 0
            ]
        }

        esterno = []
        for r in rapporti:
            controllo = r["aggregati"][ambito]["frozen_b"]
            variante = r["aggregati"][ambito]["fit192_cal64"]
            esterno.append(
                {
                    "delta_fn": variante["cm_globale"]["fn"]
                    - controllo["cm_globale"]["fn"],
                    "delta_fp": variante["cm_globale"]["fp"]
                    - controllo["cm_globale"]["fp"],
                    "delta_recall_pp": 100.0
                    * (
                        variante["recall_attacchi_globale"]
                        - controllo["recall_attacchi_globale"]
                    ),
                    "delta_fpr_pp": 100.0
                    * (variante["fpr_globale"] - controllo["fpr_globale"]),
                }
            )
        risultato[ambito]["fit192_cal64_meno_frozen_b"] = {
            campo: _statistiche(d[campo] for d in esterno) for campo in esterno[0]
        }
    return risultato


def comportamento_temporale(rapporti: list[dict]) -> list[dict]:
    righe = []
    for rapporto in rapporti:
        decisioni = [
            b for b in rapporto["per_blocco"] if b["aggiornamento"] is not None
        ]
        if [b["blocco"] for b in decisioni] != list(range(1, 355)):
            raise ValueError("le decisioni non coprono esattamente i blocchi 1--354")
        skip = [b for b in decisioni if not b["aggiornamento"]["applicato"]]
        applicati = [b for b in decisioni if b["aggiornamento"]["applicato"]]
        primo_update = applicati[0]["blocco"]
        if [b["blocco"] for b in skip] != list(range(1, primo_update)):
            raise ValueError("uno skip compare dopo il primo update")
        if decisioni[-1]["aggiornamento"].get("attivo_dal_blocco") != 355:
            raise ValueError("la decisione finale non e' marcata attiva dal blocco 355")
        if len(applicati) != rapporto["aggiornamenti"]["applicati"]:
            raise ValueError("conteggio update incoerente")
        if len(skip) != rapporto["aggiornamenti"]["saltati"]:
            raise ValueError("conteggio skip incoerente")
        righe.append(
            {
                "seme": rapporto["seme"],
                "decisioni": len(decisioni),
                "aggiornamenti_applicati": len(applicati),
                "skip": len(skip),
                "blocchi_skip": [b["blocco"] for b in skip],
                "cause_skip": dict(
                    Counter(b["aggiornamento"]["motivo"] for b in skip)
                ),
                "primo_aggiornamento_blocco": primo_update,
                "skip_dopo_primo_aggiornamento": 0,
                "decisione_finale_blocco": 354,
                "decisione_finale_attiva_dal_blocco": 355,
                "decisione_finale_influenza_predizioni_D": False,
                "etichette_ancora_in_attesa": rapporto[
                    "richieste_in_attesa_fine"
                ]["esempi"],
            }
        )
    return righe


def riferimento_temporale_C(percorso: Path) -> dict:
    dati = _leggi_json(percorso)
    righe = {
        r["strato"]: r
        for r in dati["righe"]
        if r["fase"] == "ADAPT-04"
        and r["strato"] in {"aggiornamento", "skip", "tratto_242_664"}
    }
    if set(righe) != {"aggiornamento", "skip", "tratto_242_664"}:
        raise ValueError("strati temporali ADAPT-04 incompleti")
    return righe


def costruisci_analisi(rapporti: list[dict], riepilogo: dict, c_path: Path) -> dict:
    assoluti = risultati_assoluti(rapporti)
    comp = confronti(rapporti)
    temporale_d = comportamento_temporale(rapporti)
    temporale_c = riferimento_temporale_C(c_path)

    fp_completo = comp["D_completo"]["principale_pesi"]["delta_fp"]["media"]
    fp_non_visto = comp["D_non_visto_ABC"]["principale_pesi"]["delta_fp"]["media"]
    fp_gia_visto = fp_completo - fp_non_visto
    fpr_non_visto = next(
        r
        for r in assoluti
        if r["ambito"] == "D_non_visto_ABC"
        and r["traccia"] == "fit192_cal64"
    )["metriche"]["fpr_pp"]["media"]

    return {
        "schema": 1,
        "protocollo": riepilogo["protocollo"],
        "commit_eseguito": riepilogo["commit_eseguito"],
        "nota": "analisi dai rendiconti esistenti; nessun fit, replay o tuning",
        "risultati_assoluti": assoluti,
        "confronti": comp,
        "temporale_D": temporale_d,
        "temporale_C_ADAPT04": temporale_c,
        "analisi_descrittive_successive": {
            "delta_fp_medio_D_completo": fp_completo,
            "delta_fp_medio_D_non_visto_ABC": fp_non_visto,
            "delta_fp_medio_parte_gia_vista_ABC": fp_gia_visto,
            "quota_riduzione_fp_nella_parte_gia_vista_percento": (
                100.0 * abs(fp_gia_visto) / abs(fp_completo)
            ),
            "fpr_fit192_cal64_non_visto_percento": fpr_non_visto,
            "interpretazione": "decomposizione descrittiva, non prova causale dei duplicati",
        },
        "costi": {
            "secondi_totali": riepilogo["secondi_totali"],
            "per_seme": riepilogo["campagna"]["tempi_per_seme"],
            "confine_secondi_totali": (
                "da dopo gli SHA-256 di A/B/C a prima della serializzazione del "
                "riepilogo finale; include SHA-256/caricamento di D, preparazione "
                "A/B, cinque replay e scrittura dei cinque JSON per seme"
            ),
        },
        "etichette_D": {
            "uso_online": "etichette ritardate secondo la regola preregistrata",
            "uso_per_scegliere_disegno_o_iperparametri": False,
        },
    }


def _fmt_it(valore: float, cifre: int = 3) -> str:
    testo = f"{valore:,.{cifre}f}"
    return testo.replace(",", "X").replace(".", ",").replace("X", ".")


def _pm(stat: dict, cifre: int = 3) -> str:
    return f"{_fmt_it(stat['media'], cifre)} +/- {_fmt_it(stat['deviazione_standard'], cifre)}"


def _riga_assoluta(riga: dict) -> str:
    m = riga["metriche"]
    return "| {traccia} | {fn} | {fp} | {recall} | {fpr} | {ba} | {auc} |".format(
        traccia=riga["etichetta"],
        fn=_pm(m["fn"], 1),
        fp=_pm(m["fp"], 1),
        recall=_pm(m["recall_pp"], 3),
        fpr=_pm(m["fpr_pp"], 3),
        ba=_pm(m["balanced_accuracy_pp"], 3),
        auc=_pm(m["auroc_media_blocco"], 6),
    )


def _riga_delta(ambito: str, dati: dict) -> str:
    return (
        f"| {ambito} | {_pm(dati['delta_fn'], 1)} | {_pm(dati['delta_fp'], 1)} | "
        f"{_pm(dati['delta_recall_pp'], 3)} | {_pm(dati['delta_fpr_pp'], 3)} | "
        f"{_pm(dati['delta_balanced_accuracy_pp'], 3)} | "
        f"{_pm(dati['delta_auroc_media_per_blocco'], 6)} |"
    )


def rapporto_markdown(analisi: dict) -> str:
    assoluti = analisi["risultati_assoluti"]
    comp = analisi["confronti"]
    temporale = analisi["temporale_D"]
    costi = analisi["costi"]["per_seme"]

    sezioni_assolute = []
    for ambito, titolo, denom in (
        ("D_completo", "D completo", "352/355"),
        ("D_non_visto_ABC", "D non visto in A/B/C", "350/355"),
    ):
        righe = [r for r in assoluti if r["ambito"] == ambito]
        sezioni_assolute.append(
            f"### {titolo}\n\n"
            "| traccia | FN | FP | recall (%) | FPR (%) | BA globale (%) | AUROC media/blocco |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(_riga_assoluta(r) for r in righe)
            + f"\n\nLe metriche AUROC e BA calcolate per blocco hanno denominatore `{denom}` per seme."
        )

    principale = comp["D_completo"]["principale_pesi"]
    principale_nv = comp["D_non_visto_ABC"]["principale_pesi"]
    semi_ok = principale["miglioramento_congiunto_fn_fp"]["semi"]
    semi_ok_nv = principale_nv["miglioramento_congiunto_fn_fp"]["semi"]
    per_seme = list(
        zip(
            SEMI,
            principale["delta_fn"]["per_seme"],
            principale["delta_fp"]["per_seme"],
            principale_nv["delta_fn"]["per_seme"],
            principale_nv["delta_fp"]["per_seme"],
        )
    )
    righe_seme = "\n".join(
        f"| {s} | {_fmt_it(fn, 0)} | {_fmt_it(fp, 0)} | "
        f"{_fmt_it(fnnv, 0)} | {_fmt_it(fpnv, 0)} |"
        for s, fn, fp, fnnv, fpnv in per_seme
    )

    esterno = comp["D_completo"]["fit192_cal64_meno_frozen_b"]
    esterno_nv = comp["D_non_visto_ABC"]["fit192_cal64_meno_frozen_b"]
    soglia = comp["D_completo"]["soglia_a_pesi_adattati"]
    soglia_nv = comp["D_non_visto_ABC"]["soglia_a_pesi_adattati"]

    righe_tempi = "\n".join(
        "| {seme} | {up} | {skip} | {fit} | {cf} | {ca} | {replay} |".format(
            seme=t["seme"],
            up=t["aggiornamenti_misurati"],
            skip=next(r["skip"] for r in temporale if r["seme"] == t["seme"]),
            fit=_fmt_it(t["aggiornamenti_secondi_totali"]["fit"], 3),
            cf=_fmt_it(
                t["aggiornamenti_secondi_totali"]["calibrazione_frozen"], 3
            ),
            ca=_fmt_it(
                t["aggiornamenti_secondi_totali"]["calibrazione_adattata"], 3
            ),
            replay=_fmt_it(t["replay_D_secondi"], 3),
        )
        for t in costi
    )
    intervalli_skip = ", ".join(
        f"1--{r['skip']}" if r["skip"] > 1 else "1"
        for r in temporale
    )
    descr = analisi["analisi_descrittive_successive"]
    c = analisi["temporale_C_ADAPT04"]

    return f"""# Validazione finale: KAN su D, semi 42--46

## Esecuzione e perimetro

La campagna finale usa il commit congelato
`{analisi['commit_eseguito']}`. Prima di calcolare l'impronta di D o caricarlo, il
runner ha addestrato su A e calibrato sull'intero B tutti e cinque i modelli
iniziali. Sono poi stati eseguiti una sola volta i semi 42--46, senza tuning o
modifiche del disegno. La suite del commit eseguito ha superato 375 test.

Ogni replay contiene 3.549.973 predizioni in 355 blocchi: 354 da 10.000 righe
e l'ultimo da 9.973. Il sottoinsieme non visto in A/B/C contiene 2.228.436
righe. I due ambiti sono aggregazioni delle stesse predizioni; la maschera non
modifica richieste, memorie o stato. I blocchi monoclasse conservano metriche
nulle e non vengono sostituiti con zero.

D non e' stato usato per scegliere il disegno o gli iperparametri. Le etichette
richieste su D sono pero' entrate online, con il ritardo e la regola FIT/CAL
preregistrati, e hanno potuto modificare soltanto le predizioni successive.

## Risultati assoluti delle quattro tracce

I valori sono media +/- deviazione standard campionaria sui cinque semi. FN e
FP sono conteggi; recall, FPR e BA sono percentuali. La BA della tabella e'
calcolata dalla matrice di confusione globale e non e' una media per blocco.

{(chr(10) * 2).join(sezioni_assolute)}

## Confronto principale: contributo dei pesi

Il confronto preregistrato e' `FIT192+CAL64 - frozen+CAL64`: stessa
calibrazione recente e stesso calendario, con o senza aggiornamento dei pesi.
Per FN, FP e FPR un delta negativo favorisce FIT192+CAL64.

| ambito | delta FN | delta FP | delta recall (pp) | delta FPR (pp) | delta BA (pp) | delta AUROC media/blocco |
|---|---:|---:|---:|---:|---:|---:|
{_riga_delta('D completo', principale)}
{_riga_delta('D non visto in A/B/C', principale_nv)}

| seme | FN D completo | FP D completo | FN non visto | FP non visto |
|---:|---:|---:|---:|---:|
{righe_seme}

Su D completo i FP diminuiscono in 5/5 semi e BA e AUROC media per blocco
migliorano in 5/5; i FN diminuiscono nei semi {', '.join(map(str, semi_ok))}.
Sul sottoinsieme non visto FN e FP migliorano insieme nei semi
{', '.join(map(str, semi_ok_nv))}; il seme 44 peggiora entrambi. Il criterio
preregistrato di superiorita' generale, almeno 4/5 senza peggiorare FN o FP,
non e' quindi soddisfatto.

## Soglia e riferimento frozen+B

A pesi adattati, `CAL64 - B` riduce in media i FN di
{_fmt_it(-soglia['delta_fn']['media'], 1)} ma aggiunge
{_fmt_it(soglia['delta_fp']['media'], 1)} FP su D completo; sul non visto
riduce i FN di {_fmt_it(-soglia_nv['delta_fn']['media'], 1)} e aggiunge
{_fmt_it(soglia_nv['delta_fp']['media'], 1)} FP. Recall e FPR aumentano
insieme, mentre l'AUROC resta invariata perche' gli score a pesi comuni sono
identici.

Rispetto al riferimento esterno `frozen+B`, `FIT192+CAL64` riduce in media i
FN di {_fmt_it(-esterno['delta_fn']['media'], 1)} e aumenta i FP di
{_fmt_it(esterno['delta_fp']['media'], 1)} su D completo. Sul non visto i
corrispondenti valori sono {_fmt_it(-esterno_nv['delta_fn']['media'], 1)} FN
in meno e {_fmt_it(esterno_nv['delta_fp']['media'], 1)} FP in piu'. Neppure
questo confronto indica dominanza: il recupero di attacchi ha un costo in
falsi allarmi.

## Temporalita' su C e D

In ADAPT-04 su C/seed42 sono stati osservati {c['aggiornamento']['blocchi']}
blocchi successivi a un update e {c['skip']['blocchi']} successivi a uno skip.
Il tratto 242--664, collegato a 423 skip consecutivi, ha prodotto `+807 FP` e
`+219.608 FN` rispetto a frozen+CAL64.

Su D gli update/skip per i semi 42--46 sono rispettivamente 337/17, 351/3,
347/7, 341/13 e 350/4. Tutti i 44 skip sono iniziali, nei blocchi
{intervalli_skip}; dopo il primo update non si osservano altri skip. C e D
hanno lunghezza e composizione diverse: il contrasto localizza periodi
sfavorevoli ma non identifica causalmente una singola ragione della differenza.

Ogni replay registra 354 decisioni. Quella presa nel blocco 354 e' marcata
attiva dal blocco 355, che non esiste, e non influenza quindi altre predizioni
di D. Le ultime 99 etichette richieste restano in attesa a fine stream.

## Costi e confini dei timer

| seme | update | skip | fit online (s) | cal. frozen (s) | cal. adattata (s) | replay D (s) |
|---:|---:|---:|---:|---:|---:|---:|
{righe_tempi}

`secondi_totali` vale {_fmt_it(analisi['costi']['secondi_totali'], 3)} s. Il
timer parte dopo il calcolo degli SHA-256 di A/B/C e termina prima della
serializzazione del riepilogo finale. Include SHA-256 e caricamento di D,
caricamenti A/B/C, preparazione A/B dei cinque modelli, cinque replay,
scrittura dei cinque JSON per seme e costruzione del riepilogo. I timer di
fit e delle due calibrazioni sono sottoinsiemi del replay, non costi ulteriori.

## Analisi descrittive successive e limiti

La riduzione dei FP del confronto principale e' uniforme in 5/5 semi soltanto
su D completo; sul non visto il seme 44 aumenta i FP. Sottraendo il non visto
dal completo, il {_fmt_it(descr['quota_riduzione_fp_nella_parte_gia_vista_percento'], 1)}%
della riduzione media aggregata dei FP ricade nelle righe con vettori gia'
comparsi in A/B/C. Questa decomposizione e' successiva e descrittiva: non prova
che le ripetizioni causino il beneficio. Inoltre la maschera non elimina le
ripetizioni interne a D. Il FPR assoluto di FIT192+CAL64 sul non visto resta
{_fmt_it(descr['fpr_fit192_cal64_non_visto_percento'], 3)}%.

I cinque semi rappresentano inizializzazioni sullo stesso stream, non cinque
dataset indipendenti; la deviazione standard non e' un intervallo di
confidenza. L'aggiornamento dei pesi migliora stabilmente FP, BA e AUROC su D
completo, mentre la soglia recente recupera recall pagando piu' falsi allarmi.
La variabilita' dei FN e il risultato sul non visto impediscono una conclusione
di superiorita' generale.
"""


def scrivi_tabella(percorso: Path, righe: list[dict]) -> None:
    with percorso.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPI_TABELLA, lineterminator="\n")
        writer.writeheader()
        for r in righe:
            m = r["metriche"]
            writer.writerow(
                {
                    "ambito": r["ambito"],
                    "traccia": r["etichetta"],
                    "semi": "42-46",
                    "fn_media": m["fn"]["media"],
                    "fn_sd": m["fn"]["deviazione_standard"],
                    "fp_media": m["fp"]["media"],
                    "fp_sd": m["fp"]["deviazione_standard"],
                    "recall_pp_media": m["recall_pp"]["media"],
                    "recall_pp_sd": m["recall_pp"]["deviazione_standard"],
                    "fpr_pp_media": m["fpr_pp"]["media"],
                    "fpr_pp_sd": m["fpr_pp"]["deviazione_standard"],
                    "balanced_accuracy_pp_media": m["balanced_accuracy_pp"]["media"],
                    "balanced_accuracy_pp_sd": m["balanced_accuracy_pp"]["deviazione_standard"],
                    "auroc_media_blocco_media": m["auroc_media_blocco"]["media"],
                    "auroc_media_blocco_sd": m["auroc_media_blocco"]["deviazione_standard"],
                    "denominatore_auroc_per_seme": r[
                        "denominatore_auroc_per_seme"
                    ],
                }
            )


def costruisci(raw_dir: Path, uscita: Path, manifest_path: Path) -> dict:
    rapporti, riepilogo, _ = carica_fonti(raw_dir, manifest_path)
    analisi = costruisci_analisi(
        rapporti,
        riepilogo,
        RADICE / "articolo_02" / "analisi_aggiornamenti_skip.json",
    )
    uscita.mkdir(parents=True, exist_ok=True)
    _scrivi_json(uscita / "analisi_finale.json", analisi)
    scrivi_tabella(uscita / "tabella_risultati_assoluti.csv", analisi["risultati_assoluti"])
    (uscita / "rapporto.md").write_text(
        rapporto_markdown(analisi), encoding="utf-8", newline="\n"
    )
    return analisi


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rendiconti-dir", type=Path, default=RAW_PREDEFINITI)
    p.add_argument("--uscita-dir", type=Path, default=ARTICOLO)
    p.add_argument("--manifest", type=Path, default=ARTICOLO / "manifest.json")
    a = p.parse_args(argv)
    costruisci(a.rendiconti_dir.resolve(), a.uscita_dir.resolve(), a.manifest.resolve())


if __name__ == "__main__":
    main()

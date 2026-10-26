"""Rigenera i materiali documentali per il secondo articolo da ADAPT-01--04."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


RADICE = Path(__file__).resolve().parent
MODELLI = ("lr", "mlp", "kan")
SEMI = (42, 43, 44, 45, 46)
CAMPI_TABELLA = (
    "fase",
    "gruppo_comparabilita",
    "modello",
    "semi",
    "controllo",
    "variante",
    "delta_recall_pp",
    "delta_fpr_pp",
    "delta_auroc",
    "delta_fn",
    "delta_fp",
    "denominatore_auroc",
    "nota",
)
CAMPI_PERIODI = (
    "fase",
    "confronto",
    "modello",
    "semi",
    "periodo_determinato_da",
    "strato",
    "blocchi",
    "normali",
    "attacchi",
    "fp_controllo",
    "fp_variante",
    "delta_fp",
    "fn_controllo",
    "fn_variante",
    "delta_fn",
    "fpr_controllo",
    "fpr_variante",
    "delta_fpr_pp",
    "recall_controllo",
    "recall_variante",
    "delta_recall_pp",
)


def _carica(percorso: Path) -> dict:
    if not percorso.is_file():
        raise FileNotFoundError(f"fonte richiesta assente: {percorso}")
    return json.loads(percorso.read_text(encoding="utf-8"))


def _sha256(percorso: Path) -> str:
    h = hashlib.sha256()
    with percorso.open("rb") as f:
        for blocco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blocco)
    return h.hexdigest()


def _metrica_globale(voce: dict) -> dict:
    cm = voce.get("cm_globale", {})
    medie = voce.get("medie_per_blocco", {})
    return {
        "recall": voce.get("recall_attacchi_globale", voce.get("richiamo_attacchi")),
        "fpr": voce.get("fpr_globale", voce.get("fpr_complessivo")),
        "auroc": voce.get("auroc_media_per_blocco", voce.get("auroc", medie.get("auroc"))),
        "fn": cm.get("fn"),
        "fp": cm.get("fp"),
    }


def _riga_evidenza(
    fase: str,
    gruppo: str,
    modello: str,
    semi: str,
    controllo_nome: str,
    variante_nome: str,
    controllo: dict,
    variante: dict,
    denominatore: str,
    nota: str,
) -> dict:
    c = _metrica_globale(controllo)
    v = _metrica_globale(variante)
    return {
        "fase": fase,
        "gruppo_comparabilita": gruppo,
        "modello": modello.upper(),
        "semi": semi,
        "controllo": controllo_nome,
        "variante": variante_nome,
        "delta_recall_pp": None if c["recall"] is None else 100 * (v["recall"] - c["recall"]),
        "delta_fpr_pp": None if c["fpr"] is None else 100 * (v["fpr"] - c["fpr"]),
        "delta_auroc": None if c["auroc"] is None else v["auroc"] - c["auroc"],
        "delta_fn": None if c["fn"] is None else v["fn"] - c["fn"],
        "delta_fp": None if c["fp"] is None else v["fp"] - c["fp"],
        "denominatore_auroc": denominatore,
        "nota": nota,
    }


def _tabella_evidenze() -> list[dict]:
    a1 = _carica(RADICE / "replay_evidenza" / "confronto_politiche.json")
    a2 = _carica(RADICE / "adapt02_memoria_per_classe" / "analisi_campagna_C.json")
    a3 = _carica(RADICE / "adapt03_soglia_adattiva" / "analisi_checkpoint.json")
    a3p = _carica(
        RADICE
        / "adapt03_pesi_piu_soglia"
        / "replay_completo"
        / "riepilogo_replay_completo.json"
    )
    a4 = _carica(
        RADICE / "adapt04_calibrazione_recente" / "replay_completo" / "riepilogo.json"
    )
    righe: list[dict] = []

    for modello in MODELLI:
        t = a1["tabella"][modello]
        for controllo, variante, nota in (
            ("congelato", "ogni_blocco", "beneficio dell'aggiornamento continuo"),
            ("casuale", "evidenza", "calendari con uguale numero di aggiornamenti"),
            ("ogni_blocco", "evidenza", "momento selettivo contro aggiornamento continuo"),
        ):
            righe.append(
                _riga_evidenza(
                    "ADAPT-01",
                    "A1-C-cinque-semi",
                    modello,
                    "42-46",
                    controllo,
                    variante,
                    t[controllo],
                    t[variante],
                    "875/916 blocchi per seme",
                    nota,
                )
            )

    for politica in ("ogni_blocco", "evidenza_inversione"):
        for modello in MODELLI:
            d = a2["differenze_appaiate_per_classe_meno_fifo"][politica][modello]
            valori = {
                "recall_attacchi_globale": d["recall_attacchi"]["riassunto"]["media"],
                "fpr_globale": d["fpr_complessivo"]["riassunto"]["media"],
                "auroc_media_per_blocco": d["auroc"]["riassunto"]["media"],
                "cm_globale": {
                    "fn": d["fn"]["riassunto"]["media"],
                    "fp": d["fp"]["riassunto"]["media"],
                },
            }
            zero = {
                "recall_attacchi_globale": 0.0,
                "fpr_globale": 0.0,
                "auroc_media_per_blocco": 0.0,
                "cm_globale": {"fn": 0.0, "fp": 0.0},
            }
            righe.append(
                _riga_evidenza(
                    "ADAPT-02",
                    f"A2-{politica}-cinque-semi",
                    modello,
                    "42-46",
                    f"FIFO/{politica}",
                    f"per-classe/{politica}",
                    zero,
                    valori,
                    "875/916 blocchi per seme",
                    "media delle differenze appaiate per seme; FN e FP sono differenze medie",
                )
            )

    for modello in MODELLI:
        x = a3["configurazioni"]["per_classe"][modello]
        righe.append(
            _riga_evidenza(
                "ADAPT-03",
                "A3-sola-soglia-C-seed42",
                modello,
                "42",
                "frozen+soglia_B",
                "frozen+soglia_adattiva/per-classe",
                x["frozen_b"],
                x["soglia_adattiva"],
                "875/916 blocchi",
                "score e AUROC invariati; cambia soltanto la soglia",
            )
        )

    righe.append(
        _riga_evidenza(
            "ADAPT-03",
            "A3-pesi-piu-soglia-C-seed42",
            "kan",
            "42",
            "pesi_adattati+soglia_B",
            "pesi_adattati+soglia_B_ricalibrata",
            a3p["metriche_globali"]["controllo_soglia_b"],
            a3p["metriche_globali"]["variante_soglia_ricalibrata"],
            "875/916 blocchi",
            "unica traiettoria dei pesi; B e' archivio aggiuntivo",
        )
    )

    for chiave, nota in (
        ("pesi_e_soglia_meno_sola_soglia", "effetto congiunto dei pesi rispetto alla sola soglia"),
        ("soglia_meno_b_a_pesi_identici", "effetto della soglia a pesi identici"),
    ):
        confronto = a4["confronti_appaiati"][chiave]
        righe.append(
            _riga_evidenza(
                "ADAPT-04",
                "A4-192-64-C-seed42",
                "kan",
                "42",
                confronto["controllo"],
                confronto["variante"],
                a4["metriche_globali"][confronto["controllo"]],
                a4["metriche_globali"][confronto["variante"]],
                "875/916 blocchi",
                nota,
            )
        )
    return righe


def _cm_vuota() -> dict[str, int]:
    return {"vn": 0, "fp": 0, "fn": 0, "vp": 0}


def _accumula(destinazione: dict[str, int], metrica: dict) -> None:
    for chiave in destinazione:
        destinazione[chiave] += int(metrica[chiave])


def _stratifica(
    fase: str,
    confronto: str,
    modello: str,
    semi: str,
    rapporti: Iterable[dict],
    controllo: Callable[[dict], dict],
    variante: Callable[[dict], dict],
    decisione: Callable[[dict], bool | None],
    periodo_determinato_da: str,
) -> list[dict]:
    gruppi = defaultdict(lambda: {"blocchi": 0, "controllo": _cm_vuota(), "variante": _cm_vuota()})
    for rapporto in rapporti:
        righe = rapporto["per_blocco"]
        for indice in range(1, len(righe)):
            precedente = righe[indice - 1]
            corrente = righe[indice]
            esito = decisione(precedente)
            if esito is None:
                continue
            strato = "aggiornamento" if esito else "skip"
            gruppo = gruppi[strato]
            gruppo["blocchi"] += 1
            _accumula(gruppo["controllo"], controllo(corrente))
            _accumula(gruppo["variante"], variante(corrente))

    risultato = []
    for strato in ("aggiornamento", "skip"):
        if strato not in gruppi:
            continue
        g = gruppi[strato]
        c, v = g["controllo"], g["variante"]
        normali = c["vn"] + c["fp"]
        attacchi = c["vp"] + c["fn"]
        fpr_c = c["fp"] / normali if normali else None
        fpr_v = v["fp"] / normali if normali else None
        rec_c = c["vp"] / attacchi if attacchi else None
        rec_v = v["vp"] / attacchi if attacchi else None
        risultato.append(
            {
                "fase": fase,
                "confronto": confronto,
                "modello": modello.upper(),
                "semi": semi,
                "periodo_determinato_da": periodo_determinato_da,
                "strato": strato,
                "blocchi": g["blocchi"],
                "normali": normali,
                "attacchi": attacchi,
                "fp_controllo": c["fp"],
                "fp_variante": v["fp"],
                "delta_fp": v["fp"] - c["fp"],
                "fn_controllo": c["fn"],
                "fn_variante": v["fn"],
                "delta_fn": v["fn"] - c["fn"],
                "fpr_controllo": fpr_c,
                "fpr_variante": fpr_v,
                "delta_fpr_pp": None if fpr_c is None else 100 * (fpr_v - fpr_c),
                "recall_controllo": rec_c,
                "recall_variante": rec_v,
                "delta_recall_pp": None if rec_c is None else 100 * (rec_v - rec_c),
            }
        )
    return risultato


def _tratto_fisso(
    fase: str,
    confronto: str,
    modello: str,
    semi: str,
    rapporto: dict,
    controllo: Callable[[dict], dict],
    variante: Callable[[dict], dict],
    primo: int,
    ultimo: int,
    descrizione: str,
) -> dict:
    selezionate = [
        r for r in rapporto["per_blocco"] if primo <= r["blocco"] <= ultimo
    ]
    attesi = list(range(primo, ultimo + 1))
    if [r["blocco"] for r in selezionate] != attesi:
        raise ValueError(f"intervallo incompleto: {primo}..{ultimo}")
    c, v = _cm_vuota(), _cm_vuota()
    for riga in selezionate:
        _accumula(c, controllo(riga))
        _accumula(v, variante(riga))
    normali = c["vn"] + c["fp"]
    attacchi = c["vp"] + c["fn"]
    fpr_c = c["fp"] / normali if normali else None
    fpr_v = v["fp"] / normali if normali else None
    rec_c = c["vp"] / attacchi if attacchi else None
    rec_v = v["vp"] / attacchi if attacchi else None
    return {
        "fase": fase,
        "confronto": confronto,
        "modello": modello.upper(),
        "semi": semi,
        "periodo_determinato_da": descrizione,
        "strato": f"tratto_{primo}_{ultimo}",
        "blocchi": len(selezionate),
        "normali": normali,
        "attacchi": attacchi,
        "fp_controllo": c["fp"],
        "fp_variante": v["fp"],
        "delta_fp": v["fp"] - c["fp"],
        "fn_controllo": c["fn"],
        "fn_variante": v["fn"],
        "delta_fn": v["fn"] - c["fn"],
        "fpr_controllo": fpr_c,
        "fpr_variante": fpr_v,
        "delta_fpr_pp": None if fpr_c is None else 100 * (fpr_v - fpr_c),
        "recall_controllo": rec_c,
        "recall_variante": rec_v,
        "delta_recall_pp": None if rec_c is None else 100 * (rec_v - rec_c),
    }


def _rapporti_periodi(archivio: Path) -> tuple[list[dict], list[Path]]:
    righe: list[dict] = []
    fonti: list[Path] = []

    a1_paths = [
        archivio / "controlli_appaiati_C" / "rendiconti" / f"replay_C_evidenza_seme{s}.json"
        for s in SEMI
    ]
    a1 = [_carica(p) for p in a1_paths]
    fonti.extend(a1_paths)
    for modello in MODELLI:
        righe.extend(
            _stratifica(
                "ADAPT-01",
                "evidenza meno frozen",
                modello,
                "42-46",
                a1,
                lambda r, m=modello: r[f"congelato_{m}"],
                lambda r, m=modello: r[f"adattivo_{m}"],
                lambda r, m=modello: r.get(f"decisione_{m}") == "procedi",
                "decisione della politica nel blocco precedente",
            )
        )

    for modello in MODELLI:
        fifo_paths = [
            archivio
            / "adapt02_memoria_per_classe"
            / "campagna_C_078a9d8"
            / f"C_seme{s}_fifo_evidenza_inversione.json"
            for s in SEMI
        ]
        pc_paths = [
            archivio
            / "adapt02_memoria_per_classe"
            / "campagna_C_078a9d8"
            / f"C_seme{s}_per_classe_evidenza_inversione.json"
            for s in SEMI
        ]
        fifo = [_carica(p) for p in fifo_paths]
        pc = [_carica(p) for p in pc_paths]
        fonti.extend(fifo_paths + pc_paths)
        coppie = []
        for rapporto_fifo, rapporto_pc in zip(fifo, pc, strict=True):
            if [r["blocco"] for r in rapporto_fifo["per_blocco"]] != [
                r["blocco"] for r in rapporto_pc["per_blocco"]
            ]:
                raise ValueError("ADAPT-02: blocchi FIFO/per-classe non appaiati")
            coppie.append(
                {
                    "per_blocco": [
                        {"fifo": f, "per_classe": p}
                        for f, p in zip(
                            rapporto_fifo["per_blocco"], rapporto_pc["per_blocco"], strict=True
                        )
                    ]
                }
            )
        righe.extend(
            _stratifica(
                "ADAPT-02",
                "per-classe meno FIFO, evidence",
                modello,
                "42-46",
                coppie,
                lambda r, m=modello: r["fifo"][f"adattivo_{m}"],
                lambda r, m=modello: r["per_classe"][f"adattivo_{m}"],
                lambda r, m=modello: r["per_classe"].get(f"decisione_{m}") == "procedi",
                "decisione per-classe nel blocco precedente",
            )
        )

    a3_path = (
        archivio
        / "adapt03_soglia_adattiva"
        / "diagnostico_C_seed42_90597f0"
        / "C_seed42_per_classe_soglia_adattiva.json"
    )
    a3 = _carica(a3_path)
    fonti.append(a3_path)
    for modello in MODELLI:
        righe.extend(
            _stratifica(
                "ADAPT-03",
                "soglia adattiva meno frozen+B",
                modello,
                "42",
                [a3],
                lambda r, m=modello: r[f"congelato_b_{m}"],
                lambda r, m=modello: r[f"soglia_adattiva_{m}"],
                lambda r, m=modello: bool(r["aggiornamento_soglia"][m]["aggiornato"])
                if r.get("aggiornamento_soglia") is not None
                else None,
                "ricalibrazione nel blocco precedente",
            )
        )

    a3p_path = (
        archivio
        / "adapt03_pesi_piu_soglia"
        / "replay_completo_C_seed42_106ee1c"
        / "replay_completo.json"
    )
    a3p = _carica(a3p_path)
    fonti.append(a3p_path)
    righe.extend(
        _stratifica(
            "ADAPT-03",
            "pesi+soglia meno pesi+soglia_B",
            "kan",
            "42",
            [a3p],
            lambda r: r["controllo_soglia_b"],
            lambda r: r["variante_soglia_ricalibrata"],
            lambda r: bool(r["aggiornamento_pesi"]["applicato"])
            if r.get("aggiornamento_pesi") is not None
            else None,
            "aggiornamento dei pesi nel blocco precedente",
        )
    )

    a4_path = (
        archivio
        / "adapt04_calibrazione_recente"
        / "replay_completo_C_seed42_b9fd9fb"
        / "replay_completo.json"
    )
    a4 = _carica(a4_path)
    fonti.append(a4_path)
    righe.extend(
        _stratifica(
            "ADAPT-04",
            "FIT192+CAL64 meno frozen+CAL64",
            "kan",
            "42",
            [a4],
            lambda r: r["frozen_cal64"],
            lambda r: r["fit192_cal64"],
            lambda r: bool(r["aggiornamento"]["applicato"])
            if r.get("aggiornamento") is not None
            else None,
            "decisione 192/64 nel blocco precedente",
        )
    )
    tratto = _tratto_fisso(
        "ADAPT-04",
        "FIT192+CAL64 meno frozen+CAL64",
        "kan",
        "42",
        a4,
        lambda r: r["frozen_cal64"],
        lambda r: r["fit192_cal64"],
        242,
        664,
        "skip decisi nei blocchi 241..663; effetti valutati nei blocchi 242..664",
    )
    if tratto["delta_fp"] != 807 or tratto["delta_fn"] != 219_608:
        raise ValueError("ADAPT-04: il tratto 242..664 non coincide con il riferimento")
    righe.append(tratto)
    return righe, list(dict.fromkeys(fonti))


def _scrivi_csv(percorso: Path, campi: tuple[str, ...], righe: list[dict]) -> None:
    with percorso.open("w", encoding="utf-8", newline="") as f:
        scrittore = csv.DictWriter(f, fieldnames=campi, lineterminator="\n")
        scrittore.writeheader()
        for riga in righe:
            scrittore.writerow(
                {
                    chiave: round(valore, 12) if isinstance(valore, float) else valore
                    for chiave, valore in riga.items()
                }
            )


def _fmt(valore: float | int | None, cifre: int = 3) -> str:
    if valore is None:
        return "n.d."
    if isinstance(valore, int):
        return f"{valore:,}".replace(",", " ")
    return f"{valore:.{cifre}f}"


def _intero_it(valore: int) -> str:
    return f"{valore:,}".replace(",", ".")


def _markdown_periodi(righe: list[dict]) -> str:
    tratto = next(r for r in righe if r["strato"] == "tratto_242_664")
    intestazione = """# Errori dopo aggiornamenti e skip

## Metodo

La decisione presa alla fine del blocco `k` viene associata agli errori del
blocco `k+1`, il primo che puo' usare il nuovo stato. I blocchi senza una
decisione precedente registrata e la decisione finale senza blocco successivo
sono esclusi. Le matrici di confusione sono sommate prima di calcolare recall e
FPR: non si mediano tassi di blocchi di dimensione diversa.

Gli strati descrivono **quando** il controllore ha potuto aggiornare, non
stimano l'effetto causale dell'atto di aggiornare. Stato adattato, composizione
di classe e difficolta' temporale differiscono fra gli strati. I confronti
variante-controllo restano appaiati sullo stesso blocco.

## Risultati

| Fase | Confronto | Modello | Strato | Blocchi | Delta FP | Delta FN | Delta FPR (pp) | Delta recall (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    corpo = []
    for r in righe:
        corpo.append(
            "| {fase} | {confronto} | {modello} | {strato} | {blocchi} | {dfp} | {dfn} | {dfpr} | {drec} |".format(
                fase=r["fase"],
                confronto=r["confronto"],
                modello=r["modello"],
                strato=r["strato"],
                blocchi=r["blocchi"],
                dfp=_fmt(r["delta_fp"], 0),
                dfn=_fmt(r["delta_fn"], 0),
                dfpr=_fmt(r["delta_fpr_pp"]),
                drec=_fmt(r["delta_recall_pp"]),
            )
        )
    separato = f"""

## Tratto ADAPT-04 242--664

Il tratto comprende {tratto['blocchi']} blocchi e viene valutato separatamente
perche' riceve lo stato mantenuto durante lo skip contiguo deciso nei blocchi
241--663. Rispetto a frozen+CAL64, FIT192+CAL64 passa da
{_intero_it(tratto['fp_controllo'])} a {_intero_it(tratto['fp_variante'])} FP
(`+807`) e da {_intero_it(tratto['fn_controllo'])} a
{_intero_it(tratto['fn_variante'])} FN (`+219.608`). La lettura resta descrittiva:
il tratto non e' randomizzato e lo stato incorpora tutti gli aggiornamenti
precedenti.
"""
    chiusura = """

## Lettura

ADAPT-01 mostra che l'aggiornamento selettivo riduce i falsi allarmi rispetto
al congelato sia nei blocchi successivi a un update sia negli skip, ma il
richiamo non si muove sempre nella stessa direzione. ADAPT-02 rende il
compromesso piu' netto: la memoria per classe riduce FP e FPR, mentre aumenta
FN e riduce il recall in entrambi gli strati. In ADAPT-03 la sola soglia
recupera attacchi al prezzo di piu' falsi allarmi; nel replay pesi+soglia non
esiste uno strato skip e il confronto update/skip non e' quindi stimabile.
ADAPT-04 conferma il compromesso sul disegno 192/64: rispetto a frozen+CAL64,
FIT192+CAL64 riduce soprattutto i falsi allarmi ma aumenta gli attacchi mancati.
Nel tratto 242--664, valutato separatamente perche' segue lo skip contiguo
deciso nei blocchi 241--663, la variante produce +807 FP e +219.608 FN.

I conteggi completi e i denominatori sono in `analisi_aggiornamenti_skip.csv` e
`analisi_aggiornamenti_skip.json`.
"""
    return intestazione + "\n".join(corpo) + separato + chiusura


def _sintesi() -> str:
    return """# Sintesi ADAPT-01--04

## Perimetro

Le quattro fasi studiano componenti diverse dell'adattamento temporale su C.
Le righe della tabella sono confronti appaiati entro il gruppo indicato, ma i
gruppi non vanno trattati come un'unica graduatoria: cambiano memoria, soglia,
calendario, disponibilita' dei dati e, in ADAPT-03/04, numero di semi. Tutte le
misure derivano dai rendiconti gia' verificati; D non e' stato usato.

ADAPT-01 confronta congelato, aggiornamento a ogni blocco, politica evidence e
calendario casuale con uguale numero di aggiornamenti sui semi 42--46. Per LR e
MLP l'adattamento riduce il FPR complessivo e aumenta il recall rispetto al
congelato. Per KAN ogni-blocco aumenta il recall ma riduce l'AUROC; evidence
riduce il FPR senza un guadagno sostanziale di recall. Il confronto
evidence-casuale mostra che il momento dell'aggiornamento conta, ma non autorizza
una superiorita' generale della politica.

ADAPT-02 sostituisce FIFO con una memoria per classe a spazio ed etichette
invariati. Su cinque semi la nuova memoria migliora l'AUROC e riduce FP/FPR, ma
aumenta nettamente FN e riduce il recall per tutti i modelli e per entrambe le
politiche. Conservare entrambe le classi non basta dunque a migliorare il
comportamento operativo alla soglia fissata.

ADAPT-03 separa soglia e pesi. Con score congelati, la soglia adattiva cambia
recall e FPR lasciando invariata l'AUROC. Nel caso KAN/per-classe il recall sale
dal 64,89% all'82,61%, mentre il FPR sale dal 13,25% al 14,73%. Quando la soglia
su B viene ricalibrata lungo un'unica traiettoria di pesi, il recall cresce
molto ma anche il FPR: recuperare attacchi non equivale da solo a migliorare il
sistema.

ADAPT-04 usa dati recenti separati, 192 esempi per il fit e 64 per la
calibrazione. A pesi identici, CAL64 aumenta recall e FPR rispetto alla soglia
su B. Nel confronto principale FIT192+CAL64 contro frozen+CAL64, il recall
scende di 2,60 punti percentuali e il FPR di 16,08 punti; la balanced accuracy
globale sale di 6,74 punti, ma l'AUROC media per blocco scende di 0,1093. Anche
questo e' un compromesso, non una prova di superiorita'.

## Evidenza temporale

L'analisi dei periodi associa ogni decisione agli errori del blocco successivo.
La riduzione dei falsi allarmi persiste spesso sia dopo update sia negli skip;
l'aumento degli attacchi mancati della memoria per classe e del disegno 192/64
non e' confinato a una sola delle due condizioni. Gli strati non sono
randomizzati e non misurano l'effetto causale dell'update: servono a localizzare
il compromesso e a evitare che gli aggregati nascondano periodi critici.
In ADAPT-04 il tratto 242--664 viene quindi mostrato separatamente: rispetto a
frozen+CAL64, FIT192+CAL64 aggiunge 807 FP e 219.608 FN sulle stesse righe.

## Conclusione

L'adattamento puo' produrre un beneficio concreto, ma pesi, soglia, memoria e
calendario spostano in modo diverso FP e FN. La conclusione sostenuta dai dati
e' quindi condizionale: occorre valutare insieme recall, FPR, AUROC, balanced
accuracy, andamento temporale, disponibilita' degli update e costo. La tabella
`tabella_evidenze.csv` conserva i gruppi entro cui i confronti sono diretti;
confronti fra gruppi restano descrittivi.
"""


def _protocollo_finale(sovrapposizioni: dict) -> str:
    def numero(n: int) -> str:
        return f"{n:,}".replace(",", ".")

    d = sovrapposizioni["per_intervallo"]["D"]
    visti = sovrapposizioni["D_contro_A_B_C"]
    non_visti = d["righe_valide"] - visti["righe_di_D_con_impronta_gia_vista"]
    normali_non_visti = d["righe_normali"] - visti[
        "righe_normali_di_D_con_impronta_gia_vista"
    ]
    attacchi_non_visti = non_visti - normali_non_visti
    blocchi_d = (d["righe_valide"] + 9_999) // 10_000
    secondi = 5 * (42.928 + 0.160655 * blocchi_d)
    secondi_testo = f"{secondi:.2f}".replace(".", ",")
    minuti_testo = f"{secondi / 60:.1f}".replace(".", ",")
    return f"""# Proposta minima di validazione finale

## Domanda e disegno

Validare su D, una sola volta e dopo il congelamento del protocollo, il
compromesso osservato per KAN fra quattro tracce: `frozen+B`,
`frozen+CAL64`, `FIT192+B` e `FIT192+CAL64`. Il riferimento `frozen+B` separa
il comportamento del modello iniziale con la soglia esterna fissa dagli
effetti della calibrazione recente e dell'aggiornamento dei pesi.

Usare i semi 42--46, un calendario comune per seme, memoria totale 192+64,
quota di etichette 1%, ritardo `j+2`, ruoli immutabili per vettore e la regola
BA con il tie-break gia' fissato. B non viene riletto durante il replay: le
tracce `frozen+B` e `FIT192+B` usano la soglia fissa su B preregistrata prima
dell'avvio. D non interviene in alcuna scelta.

## Inizializzazione esplicita

Per ogni seme, la trasformazione viene stimata soltanto su A. La KAN viene
inizializzata con quel seme e addestrata da zero su A; nessun peso proveniente
da ADAPT-01--04 viene riusato. Le quattro tracce ricevono copie byte-identiche
dei parametri iniziali. `frozen+B` e `frozen+CAL64` mantengono quei pesi per
tutto D; `FIT192+B` e `FIT192+CAL64` condividono una sola traiettoria di pesi
aggiornata su FIT192. Le soglie B sono calcolate prima dell'apertura di D e
restano fisse nelle due tracce `+B`.

## Due aggregazioni sulle stesse predizioni

D intero, {numero(d['righe_valide'])} righe valide, e' l'analisi principale. Il
replay scorre tutto D nell'ordine di `feature_ready_time`; nessuna riga viene
rimossa e D non modifica trasformazione, inizializzazione o criteri.

L'analisi supplementare applica alle stesse predizioni una maschera per i
vettori mai comparsi in A, B o C: {numero(non_visti)} righe, di cui
{numero(normali_non_visti)} normali e {numero(attacchi_non_visti)} attacchi.
Un vettore e' identificato dallo SHA-256 dei 32 byte degli otto `float32`
normalizzati, come in `sovrapposizioni_abcd.json`. La maschera viene applicata
soltanto in fase di aggregazione: non cambia stream, richieste, memoria o
predizioni.

I conteggi del perimetro provengono dall'inventario versionato
`sovrapposizioni_abcd.json`. Questa proposta non apre D e non contiene
risultati su D: D resta riservato fino all'approvazione del protocollo.

## Controlli preregistrati

1. Registrare commit, comando, ambiente e SHA-256 di A, B e D prima dei run.
2. Richiedere blocchi completi e ordinati, digest identici di righe ed etichette
   fra le quattro tracce e digest iniziali uguali entro ciascun seme.
3. Verificare causalita' `j+2`, separazione FIT/CAL, atomicita' degli update e
   identita' di pesi e score entro ciascuna coppia a pesi comuni.
4. Conservare i valori mancanti nei blocchi monoclasse e riportare per ogni
   metrica il denominatore valido; non sostituirli con zero.
5. Conservare raw, confronto per blocco, manifest, hash e tempi effettivi in un
   archivio esterno immutabile.

## Metriche e criteri

Le metriche primarie sono FN, FP, recall e FPR globali, riportate sia su D
intero sia sul sottoinsieme non visto. Balanced accuracy e AUROC per blocco,
con denominatore, sono secondarie. Riportare differenze appaiate per seme,
dispersione, segno in almeno quattro semi su cinque, peggiori periodi
consecutivi, strati update/skip, aggiornamenti disponibili/applicati e costi.

Una variante puo' essere detta superiore solo se non peggiora ne' FN ne' FP e
migliora almeno uno dei due aggregati, con lo stesso segno in almeno quattro
semi su cinque. Segni opposti fra FN e FP definiscono un compromesso. Un aumento
di balanced accuracy o del solo recall non basta; l'AUROC va interpretata
separatamente dalle soglie. I criteri non saranno modificati dopo l'apertura di
D.

## Costo previsto

ADAPT-04 ha richiesto 190,088 s per 916 blocchi: 35,162 s di addestramento
iniziale, 147,160 s di replay e 7,766 s residui. D intero richiede
{blocchi_d} blocchi da 10.000 righe, con l'ultimo parziale. La stima per cinque
semi e' {secondi_testo} s, circa {minuti_testo} minuti seriali, secondo la
formula `5 * (42,928 + 0,160655 * N)`. Il raw atteso e' circa 7,4 MB e lo ZIP
circa 0,9 MB. Il benchmark ha raggiunto circa 1,25 GiB RSS; la somma
conservativa dei massimi osservati del solo stato logico del controllore e'
29.120 byte, non un picco simultaneo e senza overhead Python.
"""


def _leggimi() -> str:
    return """# Materiali per il secondo articolo

Questa cartella organizza le evidenze gia' verificate di ADAPT-01--04. Non
contiene nuovi training o replay e non usa D.

- `tabella_evidenze.csv`: confronti appaiati e gruppi di comparabilita'.
- `sintesi.md`: sintesi scientifica breve (massimo due pagine di testo).
- `analisi_aggiornamenti_skip.*`: errori nel blocco successivo a update o skip.
- `protocollo_validazione_finale.md`: proposta minima prima dell'apertura di D.
- `manifest.json`: SHA-256 di fonti e prodotti.

## Riproduzione

Dalla cartella `adattamento-drift/ton_temporal`:

```powershell
python costruisci_materiali_articolo_02.py
python -m pytest test_materiali_articolo_02.py -q
```

I raw completi devono essere presenti nella cartella documentata
`../archivio_esterno/`. Il generatore non li modifica. I riepiloghi versionati
sono letti dalle cartelle ADAPT-01--04; i conteggi update/skip sono ricalcolati
dai raw. Ogni decisione del blocco `k` e' associata al blocco `k+1`, nel quale
puo' diventare effettiva.
"""


def costruisci(archivio: Path, uscita: Path) -> None:
    uscita.mkdir(parents=True, exist_ok=True)
    evidenze = _tabella_evidenze()
    periodi, fonti_raw = _rapporti_periodi(archivio)
    sovrapposizioni = _carica(RADICE / "sovrapposizioni_abcd.json")
    tratto = next(r for r in periodi if r["strato"] == "tratto_242_664")

    _scrivi_csv(uscita / "tabella_evidenze.csv", CAMPI_TABELLA, evidenze)
    _scrivi_csv(uscita / "analisi_aggiornamenti_skip.csv", CAMPI_PERIODI, periodi)
    (uscita / "analisi_aggiornamenti_skip.json").write_text(
        json.dumps(
            {
                "metodo": "decisione al blocco k, errori osservati al blocco k+1",
                "avvertenza": "stratificazione descrittiva, non stima causale dell'update",
                "tratto_242_664": tratto,
                "righe": periodi,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (uscita / "analisi_aggiornamenti_skip.md").write_text(
        _markdown_periodi(periodi), encoding="utf-8", newline="\n"
    )
    (uscita / "sintesi.md").write_text(_sintesi(), encoding="utf-8", newline="\n")
    (uscita / "protocollo_validazione_finale.md").write_text(
        _protocollo_finale(sovrapposizioni), encoding="utf-8", newline="\n"
    )
    (uscita / "LEGGIMI.md").write_text(_leggimi(), encoding="utf-8", newline="\n")

    fonti_versionate = [
        RADICE / "replay_evidenza" / "confronto_politiche.json",
        RADICE / "adapt02_memoria_per_classe" / "analisi_campagna_C.json",
        RADICE / "adapt03_soglia_adattiva" / "analisi_checkpoint.json",
        RADICE
        / "adapt03_pesi_piu_soglia"
        / "replay_completo"
        / "riepilogo_replay_completo.json",
        RADICE / "adapt04_calibrazione_recente" / "replay_completo" / "riepilogo.json",
        RADICE / "sovrapposizioni_abcd.json",
    ]
    prodotti = [
        uscita / "LEGGIMI.md",
        uscita / "tabella_evidenze.csv",
        uscita / "sintesi.md",
        uscita / "analisi_aggiornamenti_skip.csv",
        uscita / "analisi_aggiornamenti_skip.json",
        uscita / "analisi_aggiornamenti_skip.md",
        uscita / "protocollo_validazione_finale.md",
    ]
    manifest = {
        "schema": 1,
        "nota": "materiali documentali da risultati esistenti; nessun nuovo training o replay",
        "d_riservato": True,
        "fonti_versionate": [
            {
                "percorso": p.relative_to(RADICE).as_posix(),
                "dimensione_byte": p.stat().st_size,
                "sha256": _sha256(p),
            }
            for p in fonti_versionate
        ],
        "fonti_raw_esterne": [
            {
                "percorso": p.relative_to(archivio).as_posix(),
                "dimensione_byte": p.stat().st_size,
                "sha256": _sha256(p),
            }
            for p in fonti_raw
        ],
        "prodotti": [
            {
                "percorso": p.name,
                "dimensione_byte": p.stat().st_size,
                "sha256": _sha256(p),
            }
            for p in prodotti
        ],
    }
    (uscita / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archivio-esterno",
        type=Path,
        default=RADICE.parent / "archivio_esterno",
        help="cartella dei rendiconti completi esterni",
    )
    parser.add_argument(
        "--uscita-dir",
        type=Path,
        default=RADICE / "articolo_02",
        help="cartella dei materiali rigenerati",
    )
    args = parser.parse_args()
    costruisci(args.archivio_esterno.resolve(), args.uscita_dir.resolve())
    print(f"Materiali articolo 02 scritti in {args.uscita_dir.resolve()}")


if __name__ == "__main__":
    main()

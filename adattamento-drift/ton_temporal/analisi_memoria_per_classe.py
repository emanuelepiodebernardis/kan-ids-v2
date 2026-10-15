"""Analisi finale ADAPT-02 dai rendiconti completi della campagna C."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


MODELLI = ("lr", "mlp", "kan")
MEMORIE = ("fifo", "per_classe")
POLITICHE = ("ogni_blocco", "evidenza_inversione")
METRICHE_DIFF = (
    "auroc",
    "recall_attacchi",
    "fpr_complessivo",
    "fpr_medio_blocchi",
    "fn",
    "fp",
)
LINK_ARCHIVIO = (
    "https://drive.google.com/file/d/"
    "1C4uZIcW7rtWk5j1tROcHrCrhdzvzZq5E/view?usp=drive_link"
)


def qui() -> Path:
    return Path(__file__).resolve().parent


def radice_adattamento() -> Path:
    return qui().parent


def archivio_predefinito() -> Path:
    return (radice_adattamento() / "archivio_esterno"
            / "adapt02_memoria_per_classe" / "campagna_C_078a9d8")


def uscita_predefinita() -> Path:
    return qui() / "adapt02_memoria_per_classe"


def carica_rendiconti(cartella: Path) -> list[dict]:
    rendiconti = []
    for p in sorted(cartella.glob("C_seme*_*.json")):
        if p.name == "riepilogo_campagna.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rendiconti.append(d)
    if len(rendiconti) != 20:
        raise SystemExit("attesi 20 rendiconti, trovati %d" % len(rendiconti))
    return rendiconti


def media_blocchi(blocchi: list[dict], chiave: str, metrica: str) -> float:
    valori = [b[chiave][metrica] for b in blocchi
              if b[chiave].get(metrica) is not None]
    if not valori:
        return math.nan
    return mean(valori)


def conta_auroc(blocchi: list[dict], chiave: str) -> tuple[int, int]:
    validi = sum(1 for b in blocchi if b[chiave].get("auroc") is not None)
    return validi, len(blocchi)


def metriche_da_blocchi(rendiconto: dict, modello: str, copia: str) -> dict:
    chiave = "%s_%s" % (copia, modello)
    blocchi = rendiconto["per_blocco"]
    fp = sum(b[chiave]["fp"] for b in blocchi)
    fn = sum(b[chiave]["fn"] for b in blocchi)
    vp = sum(b[chiave]["vp"] for b in blocchi)
    vn = sum(b[chiave]["vn"] for b in blocchi)
    normali = sum(b[chiave]["normali"] for b in blocchi)
    attacchi = sum(b[chiave]["attacchi"] for b in blocchi)
    auroc_validi, blocchi_totali = conta_auroc(blocchi, chiave)
    return {
        "auroc": media_blocchi(blocchi, chiave, "auroc"),
        "auroc_blocchi_validi": auroc_validi,
        "auroc_blocchi_totali": blocchi_totali,
        "recall_attacchi": vp / attacchi,
        "fpr_complessivo": fp / normali,
        "fpr_medio_blocchi": media_blocchi(blocchi, chiave, "falsi_allarmi"),
        "fn": fn,
        "fp": fp,
        "vp": vp,
        "vn": vn,
        "attacchi": attacchi,
        "normali": normali,
    }


def media_memoria(rendiconto: dict) -> dict:
    blocchi = rendiconto["per_blocco"]
    normali = [b.get("memoria_normali") for b in blocchi
               if b.get("memoria_normali") is not None]
    attacchi = [b.get("memoria_attacchi") for b in blocchi
                if b.get("memoria_attacchi") is not None]
    eta_normali = [b.get("memoria_eta_blocchi", {}).get("normali", {}).get("mediana")
                   for b in blocchi]
    eta_attacchi = [b.get("memoria_eta_blocchi", {}).get("attacchi", {}).get("mediana")
                    for b in blocchi]
    eta_normali = [x for x in eta_normali if x is not None]
    eta_attacchi = [x for x in eta_attacchi if x is not None]
    return {
        "normali": mean(normali) if normali else math.nan,
        "attacchi": mean(attacchi) if attacchi else math.nan,
        "eta_normali": mean(eta_normali) if eta_normali else math.nan,
        "eta_attacchi": mean(eta_attacchi) if eta_attacchi else math.nan,
        "secondi": rendiconto["costi"]["secondi_totali"],
    }


def riassunto(valori: list[float]) -> dict:
    valori = list(valori)
    return {
        "media": mean(valori),
        "ds": stdev(valori) if len(valori) > 1 else 0.0,
        "min": min(valori),
        "max": max(valori),
    }


def costruisci(rendiconti: list[dict], link_archivio: str | None) -> dict:
    per_config = {}
    per_chiave = {}
    congelati_per_seme = {}
    auroc_denominatori = set()
    tipi_attacco = set()

    for d in rendiconti:
        for blocco in d["per_blocco"]:
            tipi_attacco.update(
                tipo for tipo in blocco.get("tipi_presenti", [])
                if tipo.lower() != "normal"
            )
        seme = d["parametri"]["seme"]
        memoria = d["parametri"]["memoria_strategia"]
        politica = d["politica"]["nome"]
        voce = {
            "seme": seme,
            "memoria": memoria,
            "politica": politica,
            "memoria_finale": media_memoria(d),
            "modelli": {},
        }
        for modello in MODELLI:
            voce["modelli"][modello] = {
                "congelato": metriche_da_blocchi(d, modello, "congelato"),
                "adattivo": metriche_da_blocchi(d, modello, "adattivo"),
                "aggiornamenti": d["riepilogo"][modello]["aggiornamenti_applicati"],
                "skip": d["riepilogo"][modello]["salti_per_memoria_monoclasse"],
            }
            den = voce["modelli"][modello]["adattivo"]
            auroc_denominatori.add((den["auroc_blocchi_validi"],
                                    den["auroc_blocchi_totali"]))
        if memoria == "fifo" and politica == "ogni_blocco":
            congelati_per_seme[seme] = {
                modello: voce["modelli"][modello]["congelato"]
                for modello in MODELLI
            }
        per_chiave[(seme, memoria, politica)] = voce
        per_config.setdefault(memoria, {}).setdefault(politica, []).append(voce)

    congelato = {}
    for modello in MODELLI:
        righe = [congelati_per_seme[s][modello] for s in sorted(congelati_per_seme)]
        congelato[modello] = {
            metrica: riassunto([r[metrica] for r in righe])
            for metrica in ("auroc", "recall_attacchi", "fpr_complessivo",
                            "fpr_medio_blocchi", "fn", "fp", "vp", "vn")
        }
        tot_fp = sum(r["fp"] for r in righe)
        tot_fn = sum(r["fn"] for r in righe)
        tot_vp = sum(r["vp"] for r in righe)
        tot_vn = sum(r["vn"] for r in righe)
        tot_normali = sum(r["normali"] for r in righe)
        tot_attacchi = sum(r["attacchi"] for r in righe)
        congelato[modello]["complessivo"] = {
            "fp": tot_fp,
            "fn": tot_fn,
            "vp": tot_vp,
            "vn": tot_vn,
            "fpr": tot_fp / tot_normali,
            "recall_attacchi": tot_vp / tot_attacchi,
        }

    aggregati = {}
    for memoria in MEMORIE:
        aggregati[memoria] = {}
        for politica in POLITICHE:
            voci = per_config[memoria][politica]
            aggregati[memoria][politica] = {"modelli": {}, "memoria": {}}
            for modello in MODELLI:
                righe = [v["modelli"][modello]["adattivo"] for v in voci]
                aggregati[memoria][politica]["modelli"][modello] = {
                    metrica: riassunto([r[metrica] for r in righe])
                    for metrica in ("auroc", "recall_attacchi",
                                    "fpr_complessivo", "fpr_medio_blocchi",
                                    "fn", "fp", "vp", "vn")
                }
                tot_fp = sum(r["fp"] for r in righe)
                tot_fn = sum(r["fn"] for r in righe)
                tot_vp = sum(r["vp"] for r in righe)
                tot_vn = sum(r["vn"] for r in righe)
                tot_normali = sum(r["normali"] for r in righe)
                tot_attacchi = sum(r["attacchi"] for r in righe)
                aggregati[memoria][politica]["modelli"][modello]["complessivo"] = {
                    "fp": tot_fp,
                    "fn": tot_fn,
                    "vp": tot_vp,
                    "vn": tot_vn,
                    "fpr": tot_fp / tot_normali,
                    "recall_attacchi": tot_vp / tot_attacchi,
                }
                aggregati[memoria][politica]["modelli"][modello]["aggiornamenti"] = (
                    riassunto([v["modelli"][modello]["aggiornamenti"] for v in voci]))
                aggregati[memoria][politica]["modelli"][modello]["skip"] = (
                    riassunto([v["modelli"][modello]["skip"] for v in voci]))
            for campo in ("normali", "attacchi", "eta_normali",
                          "eta_attacchi", "secondi"):
                aggregati[memoria][politica]["memoria"][campo] = (
                    riassunto([v["memoria_finale"][campo] for v in voci]))

    differenze = {}
    for politica in POLITICHE:
        differenze[politica] = {}
        for modello in MODELLI:
            per_metrica = {}
            for metrica in METRICHE_DIFF:
                valori = []
                for seme in (42, 43, 44, 45, 46):
                    a = per_chiave[(seme, "per_classe", politica)]["modelli"][modello]["adattivo"][metrica]
                    b = per_chiave[(seme, "fifo", politica)]["modelli"][modello]["adattivo"][metrica]
                    valori.append({"seme": seme, "delta": a - b})
                per_metrica[metrica] = {
                    "per_seme": valori,
                    "riassunto": riassunto([v["delta"] for v in valori]),
                }
            differenze[politica][modello] = per_metrica

    return {
        "fonte": "rendiconti completi campagna_C_078a9d8",
        "link_archivio": link_archivio,
        "sha256_archivio": (
            "212B9270EF47D0B997667E83B38CB6C899A5B6ADD00DF61B881AA14DA905AFE0"),
        "dimensione_archivio_byte": 7253160,
        "nota_tassi": (
            "fpr_complessivo usa FP / normali sommati; "
            "fpr_medio_blocchi e' la media dei FPR dei blocchi validi."),
        "tipi_attacco_presenti": sorted(tipi_attacco),
        "nota_tipi_attacco": (
            "I rendiconti registrano la presenza dei tipi per blocco, non "
            "matrici di confusione separate per tipo."),
        "denominatori_auroc_per_blocco": sorted(
            {"%d/%d" % (a, b) for a, b in auroc_denominatori}),
        "congelato": congelato,
        "aggregati": aggregati,
        "differenze_appaiate_per_classe_meno_fifo": differenze,
    }


def fmt_it(x: float, cifre: int = 4) -> str:
    return ("%.*f" % (cifre, x)).replace(".", ",")


def scrivi_differenze_csv(analisi: dict, uscita: Path) -> Path:
    percorso = uscita / "differenze_appaiate_per_seme_C.csv"
    campi = [
        "politica",
        "modello",
        "seme",
        "delta_auroc",
        "delta_recall_attacchi",
        "delta_fpr_complessivo",
        "delta_fpr_medio_blocchi",
        "delta_fn",
        "delta_fp",
    ]
    differenze = analisi["differenze_appaiate_per_classe_meno_fifo"]
    with percorso.open("w", encoding="utf-8", newline="") as f:
        scrittore = csv.DictWriter(f, fieldnames=campi, lineterminator="\n")
        scrittore.writeheader()
        for politica in POLITICHE:
            for modello in MODELLI:
                per_metrica = differenze[politica][modello]
                indicizzati = {
                    metrica: {
                        riga["seme"]: riga["delta"]
                        for riga in per_metrica[metrica]["per_seme"]
                    }
                    for metrica in METRICHE_DIFF
                }
                for seme in (42, 43, 44, 45, 46):
                    scrittore.writerow({
                        "politica": politica,
                        "modello": modello,
                        "seme": seme,
                        **{
                            "delta_" + metrica: indicizzati[metrica][seme]
                            for metrica in METRICHE_DIFF
                        },
                    })
    return percorso


def grafici(analisi: dict, uscita: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cartella = uscita / "figure_campagna_C"
    cartella.mkdir(parents=True, exist_ok=True)
    scritti = []

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = ["ogni blocco", "evidence"]
    x = range(len(labels))
    compl = [
        analisi["aggregati"]["per_classe"]["ogni_blocco"]["modelli"]["kan"]["complessivo"]["fpr"],
        analisi["aggregati"]["per_classe"]["evidenza_inversione"]["modelli"]["kan"]["complessivo"]["fpr"],
    ]
    blocchi = [
        analisi["aggregati"]["per_classe"]["ogni_blocco"]["modelli"]["kan"]["fpr_medio_blocchi"]["media"],
        analisi["aggregati"]["per_classe"]["evidenza_inversione"]["modelli"]["kan"]["fpr_medio_blocchi"]["media"],
    ]
    ax.bar([i - 0.18 for i in x], [v * 100 for v in compl], width=0.36,
           label="complessivo")
    ax.bar([i + 0.18 for i in x], [v * 100 for v in blocchi], width=0.36,
           label="medio per blocco")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("FPR (%)")
    ax.set_title("KAN per classe: FPR complessivo e medio per blocco")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = cartella / "fpr_kan_complessivo_blocchi.png"
    fig.savefig(p, dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    scritti.append(p.name)

    for metrica, titolo, scala, unita in (
        ("auroc", "Delta AUROC per classe - FIFO", 1.0, "delta"),
        ("recall_attacchi", "Delta recall attacchi per classe - FIFO", 1.0,
         "delta"),
        ("fpr_complessivo", "Delta FPR complessivo per classe - FIFO", 100.0,
         "punti percentuali"),
        ("fp", "Delta FP per classe - FIFO", 1.0, "delta"),
        ("fn", "Delta FN per classe - FIFO", 1.0, "delta"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
        colori = plt.get_cmap("tab10").colors[:5]
        for ax, politica, etichetta in zip(
                axes, POLITICHE, ("ogni blocco", "evidence")):
            for i, modello in enumerate(MODELLI):
                righe = analisi["differenze_appaiate_per_classe_meno_fifo"][
                    politica][modello][metrica]["per_seme"]
                valori = [riga["delta"] * scala for riga in righe]
                for j, (riga, valore) in enumerate(zip(righe, valori)):
                    ax.scatter(i + (j - 2) * 0.045, valore, s=32,
                               color=colori[j], alpha=0.85,
                               label=("seme %d" % riga["seme"])
                               if i == 0 and ax is axes[0] else None)
                ax.hlines(mean(valori), i - 0.28, i + 0.28, linewidth=2)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(MODELLI)), [m.upper() for m in MODELLI])
            ax.set_title(etichetta)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(unita)
        maniglie, etichette = axes[0].get_legend_handles_labels()
        fig.legend(maniglie, etichette, title="semi", loc="lower center",
                   ncol=5, fontsize=8, title_fontsize=8)
        fig.suptitle(titolo)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        p = cartella / ("differenze_appaiate_%s.png" % metrica)
        fig.savefig(p, dpi=160, metadata={"Software": "matplotlib"})
        plt.close(fig)
        scritti.append(p.name)

    leggimi = cartella / "LEGGIMI.md"
    leggimi.write_text(
        "# Figure ADAPT-02 campagna C\n\n"
        "Le figure sono rigenerate dai 20 rendiconti completi della campagna C.\n"
        "Non usano i dati grezzi e non includono D.\n\n"
        "- `fpr_kan_complessivo_blocchi.png`: distingue il FPR complessivo "
        "dal FPR medio per blocco per KAN con memoria per classe.\n"
        "- `differenze_appaiate_auroc.png`: differenze per-seme fra memoria "
        "per classe e FIFO sull'AUROC.\n"
        "- `differenze_appaiate_recall_attacchi.png`: differenze per-seme "
        "sul recall degli attacchi.\n"
        "- `differenze_appaiate_fpr_complessivo.png`: differenze per-seme "
        "sul FPR complessivo, in punti percentuali.\n"
        "- `differenze_appaiate_fp.png`: differenze per-seme sui falsi "
        "allarmi.\n"
        "- `differenze_appaiate_fn.png`: differenze per-seme sugli attacchi "
        "mancati.\n",
        encoding="utf-8",
        newline="\n",
    )
    scritti.append(leggimi.name)
    return scritti


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendiconti", type=Path, default=archivio_predefinito())
    ap.add_argument("--uscita", type=Path, default=uscita_predefinita())
    ap.add_argument("--link-archivio", default=LINK_ARCHIVIO)
    args = ap.parse_args()

    rendiconti = carica_rendiconti(args.rendiconti)
    analisi = costruisci(rendiconti, args.link_archivio)
    args.uscita.mkdir(parents=True, exist_ok=True)
    out = args.uscita / "analisi_campagna_C.json"
    out.write_text(json.dumps(analisi, indent=2, ensure_ascii=False),
                   encoding="utf-8", newline="\n")
    csv_differenze = scrivi_differenze_csv(analisi, args.uscita)
    scritti = grafici(analisi, args.uscita)
    print(out)
    print(csv_differenze)
    for nome in scritti:
        print(args.uscita / "figure_campagna_C" / nome)


if __name__ == "__main__":
    main()

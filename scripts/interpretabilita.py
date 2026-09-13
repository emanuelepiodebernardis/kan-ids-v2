#!/usr/bin/env python3
"""Figure XAI della KAN single-layer, senza addestramento.

Le curve e i contributi sono gli addendi interi del kernel fissato. Il
supporto empirico proviene esclusivamente dal train_calibration.npz con
metadata source_split=train. I tre esempi locali provengono dai 200 golden
vectors storici: sono esempi illustrativi selezionati, non un campione per
stimare la qualita' o scegliere il modello. Si mostrano entrambe le etichette.

Non si deducono causalita', controfattuali realizzabili o un ranking univoco
di importanza dagli addendi non centrati. Per porte e codici discreti, i
valori intermedi delle curve non hanno necessariamente significato fisico.
L'equivalenza al kernel va verificata con un confronto C compilato; le figure
non rappresentano una prova di esecuzione o una misura energetica su scheda.

Uso: python scripts/interpretabilita.py [--train-support NPZ --support-meta JSON]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

from kanids import CLIP, RESULTS_DIR                         # noqa: E402
from kanids.config import CATEGORICAL                        # noqa: E402
from kanids.interpretabilita import (contributi, curva, escursione,  # noqa: E402
                                     leggi_modello, leggi_vettori,
                                     logit, tabella_categorica)

INCLUDE = _REPO / "mcu_pio" / "include"
FIGURE = _REPO / "figures"
TRAIN_SUPPORT = _REPO / "artifacts" / "finalization" / "train_calibration.npz"
DISCRETE_CODES = {"src_port", "dst_port", "dns_qtype", "dns_qclass",
                  "dns_rcode", "http_status_code"}
SCALA = 1e6          # i contributi sono interi dell'ordine del milione


# La convenzione del segno, scritta una volta e ripetuta su ogni figura.
# Il relatore l'ha chiesta esplicitamente: un contributo con un segno e basta
# non dice verso quale classe spinge, e chi guarda la figura deve indovinarlo.
SEGNO = ("contributo positivo → spinge verso ATTACCO, "
         "negativo → verso NORMALE; la decisione e' il segno della somma")
BLU, ROSSO = "#1f4e79", "#b03a2e"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leggi_supporto_train(path: Path, metadata: Path, m: dict) -> tuple[dict, dict]:
    """Legge solo un supporto dichiarato train, senza fallback ai golden test.

    L'identita' dello split viene verificata dal replay di preprocessing;
    questo lettore impone la provenienza dichiarata e l'integrita' degli ID.
    """
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if meta.get("source_split") != "train":
        raise ValueError("XAI support must have source_split='train'")
    if not re_sha256(meta.get("source_csv_sha256")):
        raise ValueError("Missing or invalid source_csv_sha256")
    if "calibration_npz_sha256" in meta and meta["calibration_npz_sha256"] != sha256_file(path):
        raise ValueError("Training calibration_npz_sha256 mismatch")
    with np.load(path, allow_pickle=False) as d:
        required = {"Xq", "CAT", "row_ids"}
        if not required.issubset(d.files):
            raise ValueError(f"Training NPZ requires {sorted(required)}")
        x, cat, ids = (np.asarray(d[k]) for k in ("Xq", "CAT", "row_ids"))
    if any(a.dtype.kind not in "iu" for a in (x, cat, ids)):
        raise ValueError("Training inputs and row IDs must be integer arrays")
    n = len(ids)
    if n == 0 or ids.ndim != 1 or x.shape != (n, m["NFEAT"]) or cat.shape != (n, m["NCAT"]):
        raise ValueError("Training support shapes are empty or inconsistent")
    if len(np.unique(ids)) != n or np.any(ids < 0):
        raise ValueError("Training row IDs must be unique and nonnegative")
    row_hash = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    if meta.get("row_ids_sha256") != row_hash:
        raise ValueError("Training row_ids_sha256 mismatch (little-endian int64)")
    if np.any((x < -4096) | (x > 4096)):
        raise ValueError("Training Xq values outside deployed Q12 domain")
    for j in range(m["NCAT"]):
        if np.any((cat[:, j] < 0) | (cat[:, j] >= len(tabella_categorica(m, j)))):
            raise ValueError(f"Training category code out of range: {CATEGORICAL[j]}")
    return {"X": x.astype(np.int64), "CAT": cat.astype(np.int64),
            "row_ids": ids.astype(np.int64)}, meta


def re_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value.lower())


def nomi_numerici() -> list[str]:
    """I nomi nell'ordine in cui il preprocessore li produce, cioe' l'ordine
    delle colonne dell'header. Letti dall'artefatto, non riscritti qui."""
    f = _REPO / "models" / "feature_space.npz"
    if f.exists():
        return [str(x) for x in np.load(f, allow_pickle=True)["feats"]]
    return [f"feature {i}" for i in range(10)]


def nomi_categorie() -> dict[str, list[str]] | None:
    """I nomi veri delle categorie, se sono stati esportati.

    Le tabelle degli header sono indicizzate per posizione e i nomi non ci
    sono: li produce `scripts/export_vocabolari.py`, che ha bisogno del
    dataset. La funzione restituisce None quando manca; il comando di
    generazione delle figure finali richiede il vocabolario semantico."""
    f = _REPO / "models" / "vocabolari_categorici.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["vocabolari"]


def etichette_categoria(voc: dict | None, j: int, n: int) -> list[str]:
    if voc is None:
        return ["UNK"] + [str(i) for i in range(1, n)]
    nomi = voc[CATEGORICAL[j]]
    assert len(nomi) == n, (
        f"{CATEGORICAL[j]}: il vocabolario ha {len(nomi)} categorie, la "
        f"tabella dell'header {n}")
    return list(nomi)


# ─────────────────────────────────────────────────────────────
def figura_funzioni(m: dict, supporto: dict, nomi: list[str], voc: dict | None,
                    dest: Path) -> None:
    """Funzioni del kernel e istogramma/rug degli ingressi di training.

    L'assenza di osservazioni indica un limite del supporto empirico, non
    dimostra da sola se un segmento sia interpolazione o estrapolazione.
    """
    fig, assi = plt.subplots(4, 4, figsize=(15, 11.6))
    assi = assi.ravel()
    for i in range(m["NFEAT"]):
        x, y = curva(m, i, clip=CLIP)
        a = assi[i]
        a.plot(x, y / SCALA, lw=1.8, color=BLU, zorder=3)
        a.axhline(0, lw=0.8, color="0.6")
        titolo = nomi[i] if i < len(nomi) else f"feature {i}"
        if titolo in DISCRETE_CODES:
            titolo += " [codice discreto]"
        a.set_title(titolo, fontsize=10)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.25)

        # Istogramma: tutti i training rows, con la loro frequenza reale.
        # Rug: valori Q12 osservati distinti, per limitare l'overplotting.
        oss = supporto["X"][:, i] / (1 << 12) * CLIP
        basso, alto = a.get_ylim()
        altezza = (alto - basso) * 0.16
        conteggi, bordi = np.histogram(oss, bins=24, range=(-CLIP, CLIP))
        if conteggi.max():
            a.bar(bordi[:-1], conteggi / conteggi.max() * altezza,
                  width=np.diff(bordi), align="edge", bottom=basso,
                  color="0.55", alpha=0.35, linewidth=0, zorder=1)
        rug = np.unique(oss)
        a.plot(rug, np.full(len(rug), basso + altezza * 0.06), "|",
               color="0.25", ms=4, alpha=0.5, zorder=2)
        a.set_ylim(basso, alto)

    for j in range(m["NCAT"]):
        a = assi[m["NFEAT"] + j]
        val = tabella_categorica(m, j) / SCALA
        etichette = etichette_categoria(voc, j, len(val))
        colori = [ROSSO if x < 0 else BLU for x in val]
        a.bar(range(len(val)), val, color=colori)
        a.axhline(0, lw=0.8, color="0.6")
        a.set_title(f"{CATEGORICAL[j]} (categorica)", fontsize=10)
        a.set_xticks(range(len(val)))
        a.set_xticklabels(etichette, fontsize=7, rotation=45, ha="right")
        a.tick_params(labelsize=8)
        a.grid(alpha=0.25, axis="y")
    for k in range(m["NFEAT"] + m["NCAT"], len(assi)):
        assi[k].axis("off")

    nota = ("nomi delle categorie da models/vocabolari_categorici.json"
            if voc else
            "categorie etichettate con l'INDICE: manca "
            "models/vocabolari_categorici.json (python scripts/export_vocabolari.py)")
    fig.suptitle("KAN single-layer: le quattordici funzioni apprese\n"
                 f"{SEGNO}\n"
                 "ordinata = contributo al logit (unita' intere del kernel, x10⁶); "
                 f"in grigio il training (n={len(supporto['X']):,}); rug = valori Q12 distinti",
                 fontsize=12)
    fig.supxlabel("feature dopo trasformazione quantile-normale e clip a "
                  f"±{CLIP:g}  —  {nota}\n"
                  "Porte e codici: gli intermedi della curva non implicano valori validi; "
                  "supporto osservato, non affidabilita' causale.", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"scritto {dest.relative_to(_REPO).as_posix()}")


def scegli_esempi(z: np.ndarray, predizioni: np.ndarray) -> list[tuple[int, str]]:
    """Un attacco netto, un flusso normale netto, e quello piu' vicino alla
    soglia: gli estremi mostrano quali edge decidono, il caso incerto mostra
    che la somma puo' essere il risultato di termini che si oppongono."""
    att = np.flatnonzero(predizioni == 1)
    nor = np.flatnonzero(predizioni == 0)
    scelti = []
    if len(att):
        scelti.append((int(att[np.argmax(z[att])]), "pred. attacco, logit alto"))
    if len(nor):
        scelti.append((int(nor[np.argmin(z[nor])]), "pred. normale, logit basso"))
    scelti.append((int(np.argmin(np.abs(z))), "il piu' vicino alla soglia"))
    return scelti


def figura_contributi(m: dict, v: dict, nomi: list[str], voc: dict | None,
                      scelti, dest: Path):
    """I quattordici addendi di tre flussi reali, con etichetta vera e
    predetta.

    Le due etichette servono a leggere la figura per quello che e': una
    spiegazione della DECISIONE DEL MODELLO, che puo' essere giusta o
    sbagliata. Senza, il terzo esempio — quello vicino alla soglia — si legge
    come se il modello avesse ragione per costruzione."""
    etichette = list(nomi[:m["NFEAT"]]) + [f"{c} (cat)" for c in CATEGORICAL]
    nome_classe = {0: "normale", 1: "attacco"}
    fig, assi = plt.subplots(1, len(scelti), figsize=(5.6 * len(scelti), 6.2),
                             sharey=False)
    assi = np.atleast_1d(assi)
    for a, (k, descr) in zip(assi, scelti):
        num, ctg = contributi(m, v["X"][k], v["CAT"][k])
        val = np.concatenate([num[0], ctg[0]]) / SCALA

        # sull'asse: il valore della feature, cosi' la barra si legge insieme
        # alla curva della prima figura; per le categoriche il nome vero
        etichette_riga = list(etichette)
        for i in range(m["NFEAT"]):
            etichette_riga[i] = f"{etichette[i]} = {v['X'][k][i] / (1 << 12) * CLIP:+.2f}"
        for j in range(m["NCAT"]):
            codice = int(v["CAT"][k][j])
            n = len(tabella_categorica(m, j))
            nome = etichette_categoria(voc, j, n)[codice]
            etichette_riga[m["NFEAT"] + j] = f"{CATEGORICAL[j]} = {nome}"

        ordine = np.argsort(val)
        colori = [ROSSO if x < 0 else BLU for x in val[ordine]]
        a.barh(range(len(val)), val[ordine], color=colori)
        a.set_yticks(range(len(val)))
        a.set_yticklabels([etichette_riga[i] for i in ordine], fontsize=8.5)
        a.axvline(0, lw=0.8, color="0.4")

        tot = val.sum()
        pred = int(num.sum() + ctg.sum() >= 0)
        vero = int(v["LABEL"][k])
        esito = "corretta" if pred == vero else "SBAGLIATA"
        a.set_title(f"vettore #{k} — {descr}\n"
                    f"somma dei 14 addendi = {tot:+.3f} ×10⁶  →  "
                    f"predetta: {nome_classe[pred]}\n"
                    f"etichetta vera: {nome_classe[vero]}  ({esito})",
                    fontsize=10,
                    color="black" if pred == vero else ROSSO)
        a.grid(alpha=0.25, axis="x")
        a.tick_params(labelsize=8)

    fig.suptitle("Contributi locali al logit del kernel single-layer\n"
                 "Somma esatta degli addendi non centrati; esempi illustrativi dai golden vectors, "
                 "senza interpretazione causale\n" + SEGNO, fontsize=11)
    fig.supxlabel("contributo al logit (unita' intere del kernel, x10⁶)  —  "
                  "blu verso ATTACCO, rosso verso NORMALE\n"
                  "Valori numerici nello spazio trasformato; i codici discreti non definiscono "
                  "controfattuali continui realizzabili.", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.895))
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"scritto {dest.relative_to(_REPO).as_posix()}")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-support", type=Path, default=TRAIN_SUPPORT)
    parser.add_argument("--support-meta", type=Path)
    args = parser.parse_args()
    metadata = args.support_meta or args.train_support.with_suffix(".json")
    m = leggi_modello(INCLUDE / "kan14_coeff_int8.h")
    v = leggi_vettori(INCLUDE / "kan14_test_vectors.h")
    supporto, meta = leggi_supporto_train(args.train_support, metadata, m)
    nomi = nomi_numerici()
    if len(nomi) != m["NFEAT"]:
        raise ValueError("Feature names do not match the deployed model width")
    if meta.get("preprocessing", {}).get("numeric_features", nomi) != nomi:
        raise ValueError("Training support numeric feature order mismatch")
    voc = nomi_categorie()
    if voc is None:
        raise SystemExit("Required semantic category vocabulary is missing: "
                         "models/vocabolari_categorici.json")
    for j in range(m["NCAT"]):
        etichette_categoria(voc, j, len(tabella_categorica(m, j)))
    if meta.get("vocabolari", voc) != voc:
        raise ValueError("Training support semantic category vocabulary mismatch")
    if meta.get("model_header_sha256", sha256_file(INCLUDE / "kan14_coeff_int8.h")) != sha256_file(INCLUDE / "kan14_coeff_int8.h"):
        raise ValueError("Training support model header hash mismatch")
    FIGURE.mkdir(exist_ok=True)

    z = logit(m, v["X"], v["CAT"])
    accordo = int(((z >= 0).astype(np.int64) == v["ATTESA"]).sum())
    print(f"scomposizione additiva contro le predizioni attese: "
          f"{accordo}/{len(z)}")
    if accordo != len(z):
        raise SystemExit("la somma degli addendi non riproduce le predizioni "
                         "dell'header: la scomposizione non e' quella del "
                         "kernel")

    figura_funzioni(m, supporto, nomi, voc, FIGURE / "fig_kan_funzioni_apprese.png")

    scelti = scegli_esempi(z, v["ATTESA"])
    figura_contributi(m, v, nomi, voc, scelti,
                      FIGURE / "fig_kan_contributi_locali.png")

    etichette = list(nomi[:m["NFEAT"]]) + [f"{c} (cat)" for c in CATEGORICAL]
    righe = []
    for k, descr in scelti:
        num, ctg = contributi(m, v["X"][k], v["CAT"][k])
        val = np.concatenate([num[0], ctg[0]])
        for e, x in zip(etichette, val):
            righe.append({"vettore": k, "caso": descr, "edge": e,
                          "contributo": int(x)})
        righe.append({"vettore": k, "caso": descr, "edge": "SOMMA = logit",
                      "contributo": int(val.sum())})
        righe.append({"vettore": k, "caso": descr,
                      "edge": "predizione (1 = attacco)",
                      "contributo": int(val.sum() >= 0)})
        # L'etichetta vera accanto alla predizione: la figura mostra una
        # spiegazione della DECISIONE DEL MODELLO, che puo' essere sbagliata,
        # e il CSV deve permettere di verificarlo senza aprire il PNG.
        righe.append({"vettore": k, "caso": descr,
                      "edge": "etichetta vera (1 = attacco)",
                      "contributo": int(v["LABEL"][k])})
    pd.DataFrame(righe).to_csv(
        RESULTS_DIR / "interpretabilita_contributi.csv", index=False, lineterminator="\n")
    print("scritto results/interpretabilita_contributi.csv")

    esc = pd.DataFrame(escursione(m, supporto["X"], supporto["CAT"]))
    esc["edge"] = [etichette[i] if r.tipo == "numerica"
                   else etichette[m["NFEAT"] + r.indice]
                   for i, r in enumerate(esc.itertuples())]
    esc = esc.sort_values("escursione", ascending=False)
    esc["source_split"] = "train"
    esc["n_rows"] = len(supporto["X"])
    esc[["edge", "tipo", "min", "max", "escursione", "media", "source_split", "n_rows"]].to_csv(
        RESULTS_DIR / "interpretabilita_escursione.csv", index=False, lineterminator="\n")
    print("scritto results/interpretabilita_escursione.csv")

    print("\n" + "=" * 74)
    print(f"Escursione degli addendi sul training: {len(supporto['X']):,} righe")
    print("-" * 74)
    for r in esc.head(6).itertuples():
        print(f"  {r.edge:<24}{r.escursione / SCALA:>10.3f} ×10⁶"
              f"   (da {r.min / SCALA:+.3f} a {r.max / SCALA:+.3f})")
    print("-" * 74)
    print("Escursione empirica degli addendi non centrati: non e' un ranking")
    print("causale o un'importanza univoca. Nessuna misura su hardware.")
    print("=" * 74)

    provenance = {
        "schema_version": 1,
        "evidence_type": "independently_generated_host_xai_artifacts",
        "source_split": "train", "n_train_rows": len(supporto["X"]),
        "source_csv_sha256": meta["source_csv_sha256"],
        "row_ids_sha256": meta["row_ids_sha256"],
        "train_npz_sha256": sha256_file(args.train_support),
        "train_metadata_sha256": sha256_file(metadata),
        "model_header_sha256": sha256_file(INCLUDE / "kan14_coeff_int8.h"),
        "golden_header_sha256": sha256_file(INCLUDE / "kan14_test_vectors.h"),
        "vocabulary_sha256": sha256_file(_REPO / "models" / "vocabolari_categorici.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(_REPO / "kanids" / "interpretabilita.py"),
        "histogram_policy": "24 equal-width bins on [-CLIP,+CLIP], frequencies scaled per panel for display",
        "golden_prediction_agreement": {"matches": accordo, "total": len(z)},
        "local_examples_source": "historical_balanced_test_golden_vectors",
        "local_example_rule": "maximum predicted-attack logit; minimum predicted-normal logit; minimum absolute logit",
        "selected_golden_indices": [k for k, _ in scelti],
        "train_support_rug": "unique observed Q12 values; histogram includes every training row",
        "terms_centered": False,
        "limitations": ["computational terms, not causal attribution",
                        "uncentered terms do not define unique feature importance",
                        "discrete-code interpolation need not be semantically feasible",
                        "single-layer original-input decomposition only",
                        "no hardware execution or energy measurement performed by this script"],
        "artifacts": {str(p.relative_to(_REPO)): sha256_file(p) for p in (
            FIGURE / "fig_kan_funzioni_apprese.png", FIGURE / "fig_kan_contributi_locali.png",
            RESULTS_DIR / "interpretabilita_contributi.csv", RESULTS_DIR / "interpretabilita_escursione.csv")},
    }
    (RESULTS_DIR / "interpretabilita_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

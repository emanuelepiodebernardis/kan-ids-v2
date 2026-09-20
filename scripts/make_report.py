#!/usr/bin/env python3
"""Genera un nuovo report di revisione software senza sovrascrivere il PDF RC3.

Le tabelle e i valori del blocco cross-domain sono letti dai CSV prodotti
dagli esperimenti, cosi' non possono divergere dai risultati. Il resto del
testo narrativo cita numeri scritti nel sorgente: e' li' che il report era
rimasto a 3 seed mentre le tabelle erano gia' a 10, e
tests/test_coerenza_artifact.py adesso fallisce se ricompare uno dei valori
ritirati.
"""
from __future__ import annotations

import json
import hashlib
import re
from xml.sax.saxutils import escape
import sys
from pathlib import Path

import pandas as pd
import matplotlib
from reportlab import rl_config
rl_config.invariant = 1  # deterministic PDF identity when checks regenerate it
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
OUT = REPO / "report_KAN_IDS_PAPER1_software_review.pdf"

ACCENT = colors.HexColor("#1F3864")
LIGHT = colors.HexColor("#EDF1F8")
GREY = colors.HexColor("#666666")

# Embed the font so Unicode metrics and glyphs do not depend on PDF viewers.
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, filename in (
    ("KANReport", "DejaVuSans.ttf"),
    ("KANReport-Bold", "DejaVuSans-Bold.ttf"),
    ("KANReport-Oblique", "DejaVuSans-Oblique.ttf"),
    ("KANReport-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
):
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily(
    "KANReport", normal="KANReport", bold="KANReport-Bold",
    italic="KANReport-Oblique", boldItalic="KANReport-BoldOblique")

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontName="KANReport-Bold", fontSize=18,
                            textColor=ACCENT, spaceAfter=2),
    "sub": ParagraphStyle("s", parent=styles["Normal"], fontName="KANReport", fontSize=10,
                          textColor=GREY, alignment=1, spaceAfter=14),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontName="KANReport-Bold", fontSize=13,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontName="KANReport-Bold", fontSize=11,
                         textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=styles["Normal"], fontName="KANReport", fontSize=9.5,
                           leading=13.5, alignment=TA_LEFT, spaceAfter=6),
    "cap": ParagraphStyle("c", parent=styles["Normal"], fontName="KANReport", fontSize=8,
                          textColor=GREY, alignment=1, spaceAfter=10),
    "cell": ParagraphStyle("cl", parent=styles["Normal"], fontName="KANReport", fontSize=8, leading=10),
    "cellright": ParagraphStyle("cr", parent=styles["Normal"], fontName="KANReport", fontSize=8, leading=10, alignment=TA_RIGHT),
    "thead": ParagraphStyle("th", parent=styles["Normal"], fontName="KANReport-Bold", fontSize=8, leading=10, textColor=colors.white),
}


def n(v, dec=4):
    """Numero con la virgola decimale: il resto del report e' in italiano e
    un 0.44 in mezzo agli 0,44 si nota."""
    return f"{v:.{dec}f}".replace(".", ",")


def P(txt, k="body"):
    return Paragraph(txt, S[k])


def table(data, widths, align_right=None, highlight=None):
    prepared = []
    for row_index, row in enumerate(data):
        cells = []
        for col_index, cell in enumerate(row):
            content = cell.text if isinstance(cell, Paragraph) else escape(str(cell))
            style = "thead" if row_index == 0 else (
                "cellright" if col_index in (align_right or []) else "cell")
            cells.append(P(content, style))
        prepared.append(cells)
    t = Table(prepared, colWidths=widths, hAlign="LEFT", repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "KANReport-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4DA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    if align_right:
        for c in align_right:
            st.append(("ALIGN", (c, 1), (c, -1), "RIGHT"))
    for r in (highlight or []):
        st.append(("FONTNAME", (0, r), (-1, r), "KANReport-Bold"))
    t.setStyle(TableStyle(st))
    return t


def fig(name, width=15.5 * cm, caption=None):
    p = FIGURES / name
    if not p.exists():
        return [P(f"[figura mancante: {name}]")]
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    out = [Image(str(p), width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, "cap"))
    return [KeepTogether(out)]


def main():
    story = []
    story.append(P("KAN-IDS — Paper 1: software e misure fisiche", "title"))
    story.append(P("Oleksandr Kuznetsov; Emanuele Pio De Bernardis<br/>"
                   "Aggiornamento delle evidenze — 15 settembre 2026", "sub"))
    story.append(P(
        "<b>Stato: misure fisiche di latenza ed energia disponibili.</b> Le tabelle CV e cross-domain "
        "riproducono gli artefatti salvati dell'autore, non nuovi addestramenti o "
        "repliche indipendenti. Le verifiche software della finalizzazione sono "
        "identificate separatamente. La campagna FNB58 del 15 settembre aggiunge "
        "stime fisiche di energia per la scheda intera, su ingressi preparati; "
        "il peak RAM non e' misurato. Il PDF RC3 originale resta invariato."))

    # ── 1. protocollo ────────────────────────────────────────
    story.append(P("1. Protocollo", "h1"))
    story.append(P(
        "Ogni trasformazione che apprende dai dati — ranking per mutual information, "
        "vocabolari delle categoriche, quantili — è ora fittata <b>esclusivamente sul "
        "training</b> di ciascun fold. Nella versione precedente il ranking era calcolato "
        "su un campione dell'intero dataset prima dello split, e i vocabolari categorici "
        "su train+test. La correzione vive in un solo modulo (<i>kanids/preprocessing.py</i>) "
        "usato da tutti gli script; due test automatici impediscono che il difetto rientri."))
    story.append(P(
        "Gli artefatti dell'autore riportano le stesse dieci feature selezionate "
        "nei quindici fit. Questo e' un controllo descrittivo della selezione, "
        "non dimostra che ogni effetto della storia sperimentale sia nullo. "
        "La selezione sampled-LUT del RC3 usava margini di vettori derivati "
        "dal test: viene corretta separatamente senza riaddestrare la KAN."))
    story.append(P(
        "Validazione: <b>StratifiedKFold(5) ripetuta su 3 seed = 15 fit per modello</b>, "
        "media ± deviazione standard descrittiva. I fold condividono dati di training; "
        "non sono quindici repliche indipendenti. La stratificazione usa sempre l'etichetta a 10 classi, "
        "così i modelli binari e multiclass vedono fold identici e la classe rara (MITM, "
        "0,49% dei flussi) è presente ovunque."))

    story.append(P(
        "La MI della CV nativa usa il target del task: binario per la CV binaria, "
        "multiclasse per quella multiclasse. La stratificazione rimane a dieci classi. "
        "Il fit canonico separato della KAN per export usa invece MI multiclasse. "
        "La rettifica dei metadati v0.12 non cambia feature, fit o metriche salvate."))

    # ── 2. tabella comparativa ───────────────────────────────
    story.append(P("2. Tabella comparativa — TON_IoT in-domain", "h1"))
    b = pd.read_csv(RESULTS / "cv_leakagefree_summary_binary_ALL.csv").sort_values(
        "f1_mean", ascending=False)
    m = pd.read_csv(RESULTS / "cv_leakagefree_summary_multiclass_ALL.csv").set_index("model")

    rows = [["Modello", "F1 binario", "PR-AUC", "FPR",
             "Macro-F1 10 classi", "F1 MITM"]]
    hl = []
    for i, (_, r) in enumerate(b.iterrows(), start=1):
        mm = m.loc[r["model"]] if r["model"] in m.index else None
        rows.append([
            r["model"],
            f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
            f"{r['pr_auc_mean']:.4f}",
            f"{r['fpr_mean']:.4f}",
            f"{mm['macro_f1_mean']:.4f} ± {mm['macro_f1_std']:.4f}" if mm is not None else "—",
            f"{mm['f1_mitm_mean']:.3f}" if mm is not None else "—",
        ])
        if "KAN" in r["model"]:
            hl.append(i)
    story.append(table(rows, [3.4 * cm, 3.3 * cm, 1.9 * cm, 1.7 * cm, 3.4 * cm, 1.8 * cm],
                       align_right=[1, 2, 3, 4, 5], highlight=hl))
    story.append(P("15 fit per modello e per task. Spazio a 14 feature (10 numeriche + 4 "
                   "categoriche); tutti i modelli ricevono esattamente lo stesso input.", "cap"))

    story.append(PageBreak())

    # ── 3. cross-domain ──────────────────────────────────────
    story.append(P("3. Cross-domain TON_IoT ↔ BoT-IoT", "h1"))
    story.append(P(
        "Task binario su uno <b>spazio armonizzato a 13 feature candidate</b>, calcolate con "
        "la stessa formula sui due dataset (durata, byte e pacchetti IP per direzione, totali, "
        "asimmetrie, payload medi, rate) più protocollo e stato mappati su un alfabeto "
        "semantico comune. Escluse porte e indirizzi (identificatori di testbed), gli "
        "aggregati a finestra di BoT-IoT e i metadati DNS/SSL/HTTP di TON_IoT. Nel "
        "cross-domain il target entra <b>solo</b> nella valutazione."))
    story.append(P(
        "<b>Nota sullo spazio di feature.</b> In-domain lo spazio e' <b>10 numeriche + 4 "
        "categoriche</b> (proto, service, conn_state, dns_rejected). Nel cross-domain "
        "diventa <b>10 + 2</b>: delle quattro categoriche solo proto e stato hanno un "
        "corrispettivo nei log Argus di BoT-IoT, e includere le altre violerebbe il "
        "vincolo «solo feature realmente confrontabili». I due blocchi di risultati non "
        "sono quindi sullo stesso spazio, ed e' una conseguenza del vincolo, non una "
        "svista: la colonna in-domain della tabella cross-domain va letta come "
        "riferimento interno a quello spazio, non come confronto con la tabella "
        "precedente."))
    story.append(P(
        "<b>Nota sulle metriche.</b> BoT-IoT è attacco al 99,987%: in quel regime la PR-AUC "
        "sulla classe positiva e' elevata anche per ranking deboli. Si riportano i due "
        "recall e la loro media, la balanced accuracy: 0,50 e' il riferimento di "
        "una predizione indipendente dall'etichetta, non un test statistico del caso."))

    deg = pd.read_csv(RESULTS / "crossdomain_degradation.csv").set_index("model")

    # Il testo del blocco cross-domain legge da qui invece di ripetere le
    # cifre: era rimasto a 3 seed mentre le tabelle erano gia' a 10, e
    # affermava che la KAN multi-layer fosse la peggiore in transfer (0,4026)
    # — cosa che a 10 seed non e' piu' vera, il peggiore e' l'MLP.
    cd = {
        "best_tb": deg["ton->bot"].idxmax(),
        "worst_tb": deg["ton->bot"].idxmin(),
        "tree_tb": deg.loc["DecisionTree(d=5)", "ton->bot"],
        "tree_bt": deg.loc["DecisionTree(d=5)", "bot->ton"],
        "ml_tb": deg.loc["KAN(cat,ML)", "ton->bot"],
        "min_tb": deg["ton->bot"].min(),
        "max_tb": deg["ton->bot"].max(),
        "dmin": deg["delta_ton->bot"].min() * 100,
        "dmax": deg["delta_ton->bot"].max() * 100,
    }
    cd["best_tb_val"] = deg.loc[cd["best_tb"], "ton->bot"]

    # BoT->TON: dispersione per modello e quanti confronti restano separabili
    # dopo Holm. Una versione precedente di questo report dichiarava l'intera
    # direzione "rumore", cioe' nessun ordinamento leggibile; a 10 seed non e'
    # vero, e la differenza fra "alcuni modelli hanno varianza alta" e "nessun
    # ordinamento e' leggibile" e' esattamente cio' che il relatore ha chiesto
    # di non lasciare scritto a mano. Questi numeri vengono dagli artefatti.
    _run_bt = pd.read_csv(RESULTS / "crossdomain_runs_cat.csv")
    _run_bt = _run_bt[_run_bt.exp == "bot->ton"].groupby("model").f1
    _mu, _sd = _run_bt.mean(), _run_bt.std()
    cd["bt_peggiore"] = _sd.index[_mu.argmin()] if len(_mu) else "-"
    cd["bt_peggiore_mu"] = _mu.min() if len(_mu) else float("nan")
    cd["bt_peggiore_sd"] = _sd.loc[cd["bt_peggiore"]] if len(_mu) else float("nan")
    cd["bt_1l_mu"] = _mu.get("KAN(cat,1L)", float("nan"))
    cd["bt_1l_sd"] = _sd.get("KAN(cat,1L)", float("nan"))
    _somm = pd.read_csv(RESULTS / "crossdomain_summary_cat.csv")
    cd["n_in_domain"] = int(_somm[_somm.exp == "ton->ton"].n_runs.iloc[0])
    cd["n_cross"] = int(_somm[_somm.exp == "ton->bot"].n_runs.iloc[0])

    # contributo degli edge categorici: differenza cat - nocat su tutte le
    # celle davvero misurate in entrambe le varianti
    _nc = RESULTS / "crossdomain_summary_nocat.csv"
    if _nc.exists():
        # solo le due direzioni cross: la frase parla di transfer, e sono le
        # stesse undici celle citate dal README (KAN(cat,ML) non ha un run
        # nocat su ton->bot, quindi sono 11 e non 12)
        _cross = ["ton->bot", "bot->ton"]
        a = _somm[_somm.exp.isin(_cross)].set_index(["exp", "model"]).balanced_accuracy_mean
        b = (pd.read_csv(_nc).query("exp in @_cross")
             .set_index(["exp", "model"]).balanced_accuracy_mean)
        d_cat = (a - b).dropna()
        cd["cat_min"], cd["cat_max"] = d_cat.min(), d_cat.max()
        cd["cat_n"], cd["cat_pos"] = len(d_cat), int((d_cat > 0).sum())
    else:
        cd["cat_min"] = cd["cat_max"] = float("nan")
        cd["cat_n"] = cd["cat_pos"] = 0

    deg = deg.reset_index()
    rows = [["Modello", "TON in-dom.", "TON→BoT", "BA loss", "BoT in-dom.", "BoT→TON", "BA loss"]]
    hl = []
    for i, (_, r) in enumerate(deg.iterrows(), start=1):
        def f(v):
            return "—" if pd.isna(v) else f"{v:.4f}"
        rows.append([r["model"], f(r["ton_in_domain"]), f(r["ton->bot"]),
                     f(r["delta_ton->bot"]), f(r["bot_in_domain"]),
                     f(r["bot->ton"]), f(r["delta_bot->ton"])])
        if "KAN" in r["model"]:
            hl.append(i)
    story.append(table(rows, [3.3 * cm, 2.2 * cm, 2.0 * cm, 1.7 * cm, 2.2 * cm, 2.0 * cm, 1.7 * cm],
                       align_right=[1, 2, 3, 4, 5, 6], highlight=hl))
    story.append(P(f"Balanced accuracy. In-domain: {cd['n_in_domain']} fit "
                   f"(5 fold x 10 seed). Direzioni cross: {cd['n_cross']} seed, "
                   f"training sull'intero source, valutazione sull'intero target. "
                   "I seed cross-domain misurano stochasticita' del training sugli stessi "
                   "dataset, non repliche indipendenti di domini.", "cap"))
    story.append(PageBreak())

    # The original RC3 PDF is retained; this is the current software review.
    story.append(P("4. Interpretazione dei risultati salvati", "h1"))
    story.append(P(
        "La KAN single-layer e' additiva e la multilayer permette interazioni. "
        "Il vantaggio medio binario della multilayer e' circa 0,0141 F1, ma "
        "cambiano anche capacita' e ottimizzazione: non e' una misura causale "
        "del beneficio delle interazioni. Gli ordinari p-value su quindici "
        "fold dipendenti non vengono usati per sostenere questa conclusione."))
    story.append(P(
        f"Su TON→BoT il massimo medio e' {escape(str(cd['best_tb']))} "
        f"({n(cd['best_tb_val'])}); il minimo e' {escape(str(cd['worst_tb']))} "
        f"({n(cd['min_tb'])}). La classifica cambia in BoT→TON. Sono "
        "risultati condizionati a questi dataset; non dimostrano superiorita' "
        "generale nel transfer. I confronti fra feature ricche e ridotte "
        "dipendono dal modello e dal dominio. UNSW non fornisce un ceiling "
        "teorico; CIC usa lo spazio ridotto 6+2 e una diversa unita' di "
        "osservazione. Entrambi sono stress test gia' esplorati nella storia "
        "del progetto, non nuovi test storicamente intatti."))
    story.append(P(
        "La scelta operativa <b>h=16, g=8</b> viene mantenuta deliberatamente. "
        "La selezione di validazione salvata sceglie h=32, g=6, con media BA "
        "0,99631 contro 0,99602. Gli array compilati salvati dall'autore "
        "occupano rispettivamente 9.452 e 5.244 B; i valori di stack in "
        "arch_footprint.csv sono risultati della compilazione, non peak RAM "
        "misurata. La preferenza per il modello piccolo non coincide con "
        "l'architettura scelta dalla regola di validazione."))
    story.append(P(
        "La nested CV riseleziona solo k e lo confronta con k=10 fisso. "
        "La differenza fra le medie combina scelta delle feature e procedura "
        "di valutazione; non isola l'ottimismo di selezione e non certifica "
        "l'assenza di effetti della precedente esposizione ai dati. CV float, "
        "fit singolo di export e qualita' integer restano blocchi distinti."))

    review = json.loads((REPO / "evidence/review_v012/saved_results_summary.json").read_text(encoding="utf-8"))
    counts = review["unsw"]["counts_by_space"]
    choice = review["ratio_selection"]
    story.append(P(
        f"La rianalisi salvata v0.12 trova AUROC inferiore a 0,5 in "
        f"{counts['rich']['below_0_5']}/{counts['rich']['n']} run UNSW ricchi e "
        f"{counts['reduced']['below_0_5']}/{counts['reduced']['n']} ridotti. "
        "Con la polarita' originale dello score questo e' ranking inverso; cambiare "
        "solo soglia non corregge il ranking. Non e' stata applicata un'inversione "
        "dello score ottimizzata sul target. Media e SD campionaria per modello "
        "sono in evidence/review_v012/saved_results_unsw_auc.csv. "
        f"Il rapporto 1:5 vince in {choice['ratio5_seed_wins']}/{choice['seeds']} "
        "confronti per seed della BA media sui sei modelli e due domini di validation; "
        "questo descrive stabilita' interna, non nuove repliche di dominio. "
        "CIC usa proxy di finestra/pacchetti: nomi comuni non provano equivalenza "
        "di byte, durata, conteggi o stato con Zeek/Argus."))

    story.append(P("5. Finalizzazione della rappresentazione e delle prove", "h1"))
    selection_path = RESULTS / "lut_selection_protocol.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        story.append(P(
            "<b>Selezione LUT:</b> protocollo presente in "
            "results/lut_selection_protocol.json. Il riepilogo seguente "
            "legge il JSON del run; i campi costituiscono evidenza software "
            "della selezione, non misure fisiche."))
        # Keep all scalar leaf values reviewable without assuming a schema.
        def flatten(value, prefix=""):
            for key, val in value.items():
                label = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(val, dict):
                    yield from flatten(val, label)
                elif isinstance(val, (str, int, float, bool)) or val is None:
                    yield label, str(val)
        selected_fields = [
            (key, val) for key, val in flatten(selection)
            if any(term in key.lower() for term in (
                "selected", "bound", "margin", "uncertified", "coverage",
                "agreement", "rule", "fallback", "n_calibration", "candidate",
                "model_bytes", "minimum_abs_calibration_logit", "certified",
                "calibration_decisions_identical"))
            and not any(term in key.lower() for term in ("sha", "path"))
        ]
        for key, val in selected_fields[:24]:
            story.append(P(escape(f"{key}: {val}"), "cell"))
    else:
        story.append(P("Selezione LUT: evidenza del nuovo run non ancora disponibile."))
    story.append(P(
        "Il vecchio L=257 (5.194 B) e i risultati lut_vs_coeff*.csv del RC3 "
        "sono storici: L era scelto con i margini di duecento vettori dal "
        "test. La correzione usa calibrazione derivata dal training e una "
        "griglia finita fissata; non modifica le funzioni apprese. Il bound "
        "B somma i massimi errori di ciascun edge sull'intera griglia Q12. "
        "Garantisce l'accordo della decisione solo quando |logit coeff| > B, "
        "con termini categorici identici e senza overflow. Non garantisce "
        "la correttezza rispetto all'etichetta o tutte le decisioni future."))
    post_path = RESULTS / "lut_vs_coeff_postfreeze_test.csv"
    if post_path.exists():
        post = pd.read_csv(post_path)
        post_block = [P("Valutazione empirica del test dopo il freeze della LUT: "
                        "results/lut_vs_coeff_postfreeze_test.csv.", "h2")]
        for _, row in post.iterrows():
            for key, val in row.items():
                post_block.append(P(escape(f"{key}: {val}"), "cell"))
        story.append(KeepTogether(post_block))
    else:
        story.append(P("Valutazione post-freeze del test: evidenza non ancora disponibile."))

    story.append(P(
        "La verifica Q15 aggiornata enumera tutti i 32.769 valori t: la somma "
        "delle basi quantizzate varia da 196.607 a 196.609, non e' sempre 196.608. "
        "Il controesempio ammissibile q=-4094 produce t=128 e basi "
        "[32385,131072,33152,0]. Il bound corretto usa ceiling per lo shift "
        "negativo e controlla intermedi e accumulatori sui coefficienti congelati; "
        "scripts/audit_q15_bounds.py riproduce evidence/review_v012/q15_bounds.json. "
        "Questa correzione della prova non modifica l'aritmetica del firmware."))

    overlap = json.loads((REPO / "evidence/review_v012/overlap/overlap_subgroup_results.json").read_text(encoding="utf-8"))
    subgroup = overlap["subgroups"]["non_overlapping"]
    story.append(P(
        "La rianalisi v0.12 trova 5.201 dei 42.209 test con una riga training "
        "identica sulle 44 colonne grezze, incluse etichette e indirizzi. "
        f"Sui 37.008 test senza tale coincidenza, F1 coefficiente/LUT e' "
        f"{subgroup['coefficient']['F1']:.6f}, BA {subgroup['coefficient']['BA']:.6f}; "
        "i 40 disaccordi float/coefficiente ricadono tutti in questo sottogruppo. "
        "Non e' un nuovo split deduplicato o host/time-disjoint. Il replay "
        "float usa i quantili preservati senza fit e verifica tutti gli ingressi "
        "Q12, ma non dimostra identita' bit a bit col trasformatore float originale "
        "non conservato. I risultati separati e gli hash sono nel nuovo artifact map."))
    story.append(P(
        "Il certificato firmato L=1025 ha intervallo [-3426,2324], mentre "
        "L=513 e' una diagnostica con intervallo [-10065,8250]. La guardia "
        "calcolabile dal solo score LUT lascia rispettivamente 3 e 12 casi "
        "test non certificati; entrambe le LUT mantengono empiricamente tutte "
        "le decisioni del coefficiente. Questo non certifica il classificatore "
        "float o la correttezza delle etichette e non cambia L di deployment. "
        "evidence/review_v012/ARTIFACT_MAP.md contiene comandi e input esterni."))

    story.append(P("5.1 Coorte comune e protocollo fisico", "h2"))
    story.append(P(
        "Il nuovo percorso common usa 500 flow ID unici, bilanciati 250+250 "
        "e nello stesso ordine. Ogni modello conserva la propria trasformazione "
        "fissata; vettori numericamente uguali non sono assunti equivalenti "
        "fra preprocessori diversi. La latenza misura ingressi preparati in "
        "RAM → decisione. L'energia usa i primi venti flussi (10+10), ripetuti "
        "in warm cache. Il costo di preprocessing e il carico live di rete "
        "non rientrano in questa misura del kernel. I percorsi E2E e multiclass "
        "rimangono separati."))
    energy_root = REPO / "experiments/hardware_energy_20260915"
    energy_runs = pd.read_csv(energy_root / "results/all_20_acquisitions.csv")
    energy_means = pd.read_csv(energy_root / "results/board_model_means.csv")
    story.append(P(
        "La campagna eseguita il 15 settembre 2026 e' documentata in "
        "experiments/hardware_energy_20260915/README.md. Il FNB58 registra "
        "VBUS e IBUS a 10 campioni/s. Due codici LED ai lati del batch "
        "consentono la sincronizzazione; la potenza media nella porzione "
        "centrale di 60 secondi MCU moltiplicata per T/N stima l'energia "
        "per chiamata. Non si sottrae il consumo idle. Le letture riguardano "
        "la scheda intera e non risolvono singole inferenze. Due acquisizioni "
        "per modello sono ripetibilita' descrittiva, non un'incertezza "
        "strumentale calibrata. I run C3 appaiati condividono lo stesso boot; "
        "i run Mega sono acquisizioni caricate separatamente. "
        "Il protocollo iniziale con marker esterni resta documentazione "
        "storica; i vecchi hook INA219 non forniscono questi risultati."))
    energy_rows = [[P(x, "cell") for x in
                    ("Scheda / modello", "us/chiamata", "Potenza W", "uJ/chiamata")]]
    for row in energy_means.itertuples(index=False):
        energy_rows.append([P(f"{row.board} / {row.model}", "cell"),
                            P(f"{row.mean_call_us:.3f}", "cell"),
                            P(f"{row.mean_power_W:.4f}", "cell"),
                            P(f"{row.energy_uJ_per_call:.4f}", "cell")])
    story.append(KeepTogether([table(energy_rows, [6.5 * cm, 3 * cm, 3 * cm, 3 * cm])]))
    story.append(P(
        f"Fonte: {len(energy_runs)} acquisizioni fisiche, "
        f"{len(energy_means)} medie modello/scheda. SHA-256 e verifica "
        "riproducibile sono nel pacchetto dell'esperimento. Nessun nuovo "
        "training e nessuna misura di peak RAM. I risultati a 500 flussi "
        "e il follow-up fattoriale C3 hanno protocolli diversi e restano separati."))

    story.append(P("5.2 Spiegazioni additive", "h2"))
    story.append(P(
        "I quattordici termini della single-layer sommano esattamente al "
        "logit del kernel, sotto le medesime regole integer. Questa e' una "
        "spiegazione del calcolo, non una causal attribution o un "
        "controfattuale necessariamente realizzabile. Termini non centrati "
        "non definiscono un ranking univoco di importanza. Le distribuzioni "
        "di supporto aggiornate usano il training; le illustrazioni locali "
        "usano i golden vector con etichette vere e predette e vocabolari "
        "semantici. Porte e codici DNS restano discreti. SHAP non e' sempre "
        "un metodo di fitting di surrogati locali; la presenza o assenza di "
        "una dipendenza XAI non dimostra l'interpretabilita'."))
    story.extend(fig("fig_kan_funzioni_apprese.png", 14 * cm,
                     "Funzioni additive e distribuzione di supporto del training; "
                     "provenienza in results/interpretabilita_provenance.json."))
    story.extend(fig("fig_kan_contributi_locali.png", 15 * cm,
                     "Decomposizioni locali: vere etichette e decisioni del modello "
                     "sono mostrate separatamente."))

    story.append(P("6. Stato delle evidenze", "h1"))
    rows = [[P("Blocco", "cell"), P("Provenienza e stato", "cell")]]
    software_status = (
        "Replay host: 42.209 score coefficiente e LUT coincidono con C; "
        "81.930 probe per-coordinate verificano l'errore. Q15: bound corretti "
        "e controlli AVR in tests/test_q15_mul.py. Queste sono verifiche software, "
        "non nuove misure MCU. Ricevute correnti in evidence/review_v012; "
        "gate completo e fresh-apply nella directory integration del rilascio.")
    # Energia disponibile dal 15 settembre; peak RAM rimane non misurato.
    energy_status = (f"Misurata con FNB58: {len(energy_runs)} acquisizioni "
                     f"e {len(energy_means)} medie scheda/modello. Stima USB "
                     "della scheda intera, ingressi preparati, senza sottrazione idle. "
                     "Dati e firmware in experiments/hardware_energy_20260915.")
    entries = [
        ("Risultati ML salvati",
         "CV, transfer, CIC, architettura e linker sizes RC3: salvati; "
         "nessun nuovo training in questa revisione."),
        ("Verifiche software", software_status),
        ("Latenza", "Misurata: 500 flussi comuni su Mega 2560 ed ESP32-C3, "
                    "cinque passate separate da reset, piu' l'esperimento "
                    "fattoriale a otto condizioni su ESP32-C3. Registrata "
                    "negli archivi hardware con i log seriali."),
        ("Energia", energy_status),
        ("Peak RAM", "Non misurata. Le dimensioni statiche del linker non "
         "misurano il picco. Il registro dei venti environment common "
         "rimane distinto dai firmware pilot compilati e misurati il "
         "15 settembre (results/firmware_size_pending.csv)."),
    ]
    for name, state in entries:
        rows.append([P(name, "cell"), P(state, "cell")])
    story.append(table(rows, [5.0 * cm, 10.5 * cm]))
    story.append(P(
        "Fonti: docs/CLAIM_EVIDENCE_MAP.csv. Articolo 2 resta separato. "
        "La campagna fisica conserva la provenienza di sorgenti, coorte e binari. "
        "Pubblicazione e merge restano soggetti alla revisione degli autori."))


    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2.4 * cm, rightMargin=2.4 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                            title="KAN-IDS — Paper 1 software e misure fisiche",
                            author="Oleksandr Kuznetsov; Emanuele Pio De Bernardis")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("KANReport", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2.4 * cm, 1.2 * cm,
                          "KAN-IDS — 15 settembre 2026 — energia disponibile; peak RAM non misurato")
        canvas.drawRightString(A4[0] - 2.4 * cm, 1.2 * cm, str(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    provenance = {
        "version": "0.12.0", "scope": "current report regenerated from saved evidence; no training",
        "output": OUT.name, "output_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "generator": "scripts/make_report.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": [{"path": path.relative_to(REPO).as_posix(),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in [REPO/"evidence/review_v012/saved_results_summary.json",
                                 REPO/"evidence/review_v012/q15_bounds.json",
                                 REPO/"evidence/review_v012/overlap/overlap_subgroup_results.json",
                                 REPO/"results/lut_selection_protocol.json",
                                 REPO/"experiments/hardware_energy_20260915/results/board_model_means.csv"]],
        "historical_receipts": "evidence/finalization/report_visual_qa.json refers to the earlier report only",
    }
    (REPO/"evidence/review_v012/report_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(f"scritto {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

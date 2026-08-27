#!/usr/bin/env python3
"""Genera il report tecnico in PDF a partire dai CSV in results/.

Le tabelle e i valori del blocco cross-domain sono letti dai CSV prodotti
dagli esperimenti, cosi' non possono divergere dai risultati. Il resto del
testo narrativo cita numeri scritti nel sorgente: e' li' che il report era
rimasto a 3 seed mentre le tabelle erano gia' a 10, e
tests/test_coerenza_artifact.py adesso fallisce se ricompare uno dei valori
ritirati.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
OUT = REPO / "report_KAN-IDS_fase2.pdf"

ACCENT = colors.HexColor("#1F3864")
LIGHT = colors.HexColor("#EDF1F8")
GREY = colors.HexColor("#666666")

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontSize=19,
                            textColor=ACCENT, spaceAfter=2),
    "sub": ParagraphStyle("s", parent=styles["Normal"], fontSize=10,
                          textColor=GREY, alignment=1, spaceAfter=14),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13,
                         textColor=ACCENT, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11,
                         textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=styles["Normal"], fontSize=9.5,
                           leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6),
    "cap": ParagraphStyle("c", parent=styles["Normal"], fontSize=8,
                          textColor=GREY, alignment=1, spaceAfter=10),
    "cell": ParagraphStyle("cl", parent=styles["Normal"], fontSize=8, leading=10),
}


def n(v, dec=4):
    """Numero con la virgola decimale: il resto del report e' in italiano e
    un 0.44 in mezzo agli 0,44 si nota."""
    return f"{v:.{dec}f}".replace(".", ",")


def P(txt, k="body"):
    return Paragraph(txt, S[k])


def table(data, widths, align_right=None, highlight=None):
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
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
        st.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
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
    return out


def main():
    story = []
    story.append(P("KAN-IDS — consolidamento sperimentale", "title"))
    story.append(P("Protocollo leakage-free, validazione 5-fold × 3 seed, "
                   "confronto con baseline identiche e cross-domain TON_IoT ↔ BoT-IoT<br/>"
                   "Emanuele Pio De Bernardis — agosto 2026 — "
                   "github.com/emanuelepiodebernardis/kan-ids-v2", "sub"))

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
        "<b>Effetto misurato del difetto: nullo.</b> La selezione per-fold sceglie le stesse "
        "10 feature numeriche in 15 fold su 15, e in-domain non esiste alcuna categoria "
        "assente dal training. I numeri precedentemente riportati restano quindi validi; "
        "il protocollo andava comunque corretto, perché la sua validità non può dipendere "
        "da quanto grande sia risultato l'effetto."))
    story.append(P(
        "Validazione: <b>StratifiedKFold(5) ripetuta su 3 seed = 15 fit per modello</b>, "
        "media ± deviazione standard. La stratificazione usa sempre l'etichetta a 10 classi, "
        "così i modelli binari e multiclass vedono fold identici e la classe rara (MITM, "
        "0,49% dei flussi) è presente ovunque."))

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
    story.extend(fig("fig_indomain_comparison.png", 11.5 * cm))

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
        "sulla classe positiva vale ~1 per costruzione e non discrimina (nei run TON→BoT è "
        "0,9999 mentre i modelli sono al caso). Si riportano i due recall e la loro media, "
        "la balanced accuracy, dove <b>0,50 = caso</b>."))

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
    rows = [["Modello", "TON in-dom.", "TON→BoT", "δ", "BoT in-dom.", "BoT→TON", "δ"]]
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
                   f"training sull'intero source, valutazione sull'intero target.", "cap"))
    story.extend(fig("fig_crossdomain_degradation.png", 14 * cm))
    story.append(PageBreak())
    story.extend(fig("fig_pareto_size_accuracy.png", 16 * cm,
                     "Frontiera di Pareto dimensione/accuratezza, byte contati sugli "
                     "header C effettivamente compilati. A sinistra in-domain: il modello "
                     "KAN da 254 byte è il più piccolo sulla frontiera, l'albero profondo "
                     "5 ne occupa 285 ed è più accurato: un compromesso, non una "
                     "dominanza. A destra cross-domain: la "
                     "classifica si ribalta e il single-layer è il modello che trasferisce "
                     "meglio."))

    story.append(PageBreak())
    story.extend(fig("fig_distribution_overlap.png", 14 * cm,
                     "Sovrapposizione delle distribuzioni marginali fra i due domini. "
                     "Valori sotto 0,2 significano supporti quasi disgiunti."))
    story.extend(fig("fig_confusion_crossdomain.png", 12.5 * cm,
                     "Matrici di confusione TON_IoT → BoT-IoT, normalizzate per riga. "
                     "LightGBM classifica come normale il 71% degli attacchi di BoT-IoT."))

    story.append(PageBreak())

    # ── 4. analisi critica ───────────────────────────────────
    story.append(P("4. Analisi critica", "h1"))

    story.append(P("4.1 Il confronto con le baseline era viziato", "h2"))
    story.append(P(
        "La versione precedente dichiarava che il modello da 254 byte supera gli ensemble ad "
        "alberi «sullo stesso spazio di feature deployabile». Non è così: quel confronto "
        "metteva la KAN sullo spazio grezzo a 14 feature e le baseline sullo spazio derivato "
        "a 10 feature del paper. Con input identici, <b>LightGBM raggiunge 0,9991 contro "
        "0,9835 della KAN single-layer</b>, vincendo 15 fold su 15 "
        "(t-test appaiato p = 1,0·10⁻²⁰). "
        "Il claim è stato corretto nel README."))

    story.append(P("4.2 La profondità colma il divario, e spiega perché esisteva", "h2"))
    story.append(P(
        "La KAN single-layer è un modello additivo generalizzato: somma di funzioni "
        "univariate, incapace per costruzione di rappresentare interazioni fra feature. "
        "La multi-layer, che può, guadagna <b>+0,0141 di F1 vincendo 15 fold su 15</b> e "
        "arriva a 0,9976 ± 0,0002. È la misura diretta che il divario riguardava le "
        "interazioni, non la capacità né l'ottimizzazione — e trasforma un punto debole "
        "in un risultato architetturale."))

    story.append(P("4.3 A parità di conteggio il Pareto è un compromesso, non una dominanza", "h2"))
    story.append(P(
        "Perché il confronto sia una misura e non un artefatto, i byte vanno contati con "
        "una regola sola, e deve essere quella che il codice implementa. Fino a questa "
        "revisione non lo era: <code>footprint.py</code> usava un impacchettamento ideale "
        "(4 byte per nodo interno, 1 per foglia) che l'albero in C non usa. L'header "
        "<code>mcu_pio/include/dt5_model.h</code> memorizza quattro array paralleli su "
        "tutti e 57 i nodi, foglie comprese, e occupa <b>285 byte</b>, non 141. Contati "
        "sugli header compilati (<code>scripts/c_footprint.py</code>, verificabili con "
        "<code>nm</code> sull'oggetto del compilatore), i due modelli più piccoli si "
        "invertono: la KAN single-layer sta in <b>254 byte</b>, l'albero in <b>285</b>."))
    story.append(P(
        "Resta però vero che l'albero è <b>più accurato</b> (F1 0,9944 contro 0,9835) ed è "
        "invariante a trasformazioni monotone, quindi non richiede alcun preprocessing, "
        "mentre la catena integer end-to-end della KAN costa 1.334 byte comprese le "
        "tabelle. Il single-layer non è dominato, ma 31 byte di differenza sono un "
        "arrotondamento su entrambe le schede: l'argomento a favore del single-layer "
        "resta il comportamento cross-domain, non la dimensione. «Accuratezza per byte» "
        "non è un argomento difendibile su TON_IoT, e il repository non lo sostiene."))
    story.append(P(
        "Tre cose invece reggono, e sono quelle su cui costruire. <b>(a)</b> La KAN "
        "multi-layer sta sulla frontiera: 5,12 KB e F1 0,9976 contro l'MLP TensorFlow Lite "
        "Micro del paper originale, 13 KB e 0,9959 con 95 feature — più piccola, più "
        "accurata, 14 feature invece di 95. <b>(b)</b> Nel cross-domain la classifica si "
        f"ribalta: il single-layer ha il valore medio più alto su TON→BoT "
        f"({n(cd['best_tb_val'])}), pur senza essere separabile da XGBoost né "
        f"dall'albero — vedi crossdomain_significativita.csv; l'albero è a "
        f"{n(cd['tree_tb'])} ed è il peggiore di tutti nella direzione BoT→TON "
        f"({n(cd['tree_bt'])}). <b>(c)</b> La KAN offre ciò che un albero non ha: calibrazione conformal "
        "sul modello intero deployato, forma simbolica chiusa e tabelle riscrivibili, che "
        "sono il presupposto della ricalibrazione on-device."))

    story.append(P(
        "Con l'export in C dell'albero (<i>scripts/export_tree_c.py</i>) il confronto si "
        "puo' finalmente fare sul dispositivo, e un primo risultato c'e' gia': "
        "<b>quantizzare le soglie a Q7</b> per il target costa all'albero <b>0,0028 di "
        "F1</b> (0,9944 → 0,9916, agreement 99,55% col modello float). Il divario con la "
        "KAN compilata scende da 0,0109 a <b>0,0081</b>. La KAN paga la quantizzazione "
        "meno dell'albero — un argomento che prima non avevamo, perche' confrontavamo un "
        "albero in virgola mobile con una KAN gia' quantizzata."))

    story.append(P("4.4 Il cross-domain non degrada: collassa", "h2"))
    story.append(P(
        f"TON→BoT lascia ogni modello fra {n(cd['min_tb'], 2)} e {n(cd['max_tb'], 2)} di "
        f"balanced accuracy, cioè al caso o sotto. "
        f"Il δ quantificato nel paper era ≤ 5,95 punti; qui siamo a "
        f"<b>{cd['dmin']:.0f}–{cd['dmax']:.0f} punti</b>. "
        "Due osservazioni non ovvie: il compromesso fra capacità e trasferibilità si "
        "misura <b>dentro la stessa famiglia</b> — aggiungere profondità alla KAN vale "
        "+0,0141 di F1 sul binario e +0,0608 di macro-F1 sulle 10 classi, e le costa quasi "
        f"tutto il transfer ({n(cd['ml_tb'])} di balanced accuracy, <b>sotto il caso</b>), "
        f"mentre la single-layer, ultima in-domain, ha il valore più alto cross-domain. Il "
        f"modello che trasferisce peggio non è però la KAN profonda ma {cd['worst_tb']} "
        f"({n(cd['min_tb'])}): a 3 seed sembrava il contrario, ed è una delle affermazioni "
        f"che i 10 seed hanno ritirato. E la direzione "
        "BoT→TON non è degradata ma <b>indeterminata</b> — con 477 flussi normali in "
        "training la varianza fra seed supera la differenza fra modelli, quindi qualunque "
        "classifica in quella direzione sarebbe rumore."))
    story.append(P(
        "La causa è visibile prima di addestrare qualsiasi cosa: le marginali non si "
        "sovrappongono (byte_rate 0,085, duration 0,106). TON_IoT ha flussi brevi e "
        "bidirezionali, il 5% di BoT-IoT è dominato da flood UDP lunghi e unidirezionali. "
        "Il 21,3% dei flussi di TON_IoT porta uno stato di connessione mai visto "
        "addestrando su BoT-IoT. Gli edge categorici armonizzati aiutano il transfer "
        f"nella maggior parte dei casi ma non sempre: su {cd['cat_n']} celle misurate in "
        f"entrambe le varianti rimuoverli sposta la balanced accuracy fra "
        f"{n(cd['cat_min'])} e {'+' if cd['cat_max'] > 0 else ''}{n(cd['cat_max'])}, e in {cd['cat_n'] - cd['cat_pos']} "
        f"celle la rimozione aiuta. Una versione precedente di questo paragrafo dava "
        f"«0,08–0,16», un intervallo che escludeva proprio le celle che lo contraddicono."))

    story.append(P("4.5 La catena integer end-to-end ora è in C, e ha rivelato tre difetti", "h2"))
    story.append(P(
        "Il percorso dai contatori grezzi alla decisione gira interamente in aritmetica "
        "intera nel firmware: <b>200 golden vector su 200 con logit bit-identico</b> al "
        "riferimento Python, e l'ispezione dell'assembly del percorso di inferenza mostra "
        "<b>zero istruzioni in virgola mobile</b>. Il percorso end-to-end precedente "
        "(<i>mcu_e2e/</i>) interpolava 10.000 knot del QuantileTransformer in doppia "
        "precisione: end-to-end nella struttura, ma non integer-only nel runtime. "
        "Il porting ha fatto emergere tre difetti che si sarebbero manifestati solo sul "
        "dispositivo: il moltiplicatore di scala della feature con scala massima quantizza "
        "a esattamente 32768, che in <i>int16</i> va in overflow silenzioso; la divisione "
        "intera di Python arrotonda verso −∞ mentre quella del C tronca verso zero, "
        "differenza osservabile sulle due feature di asimmetria, che hanno numeratore di "
        "segno qualsiasi; e la stessa saturazione a 2<super>15</super> ricompare nella LUT "
        "della tangente iperbolica del modello a 10 classi."))
    story.append(P(
        "Anche la <b>catena a 10 classi</b> e' ora in C: contatori grezzi → ricerca binaria "
        "sulle soglie per-feature → z in Q12 → primo strato di spline int8 con tabelle "
        "categoriche → LUT tanh → secondo strato → argmax, tutto intero. "
        "<b>200 golden vector su 200 con tutti e dieci gli accumulatori bit-identici</b> al "
        "riferimento, argmax identico, macro-F1 0,9352 contro 0,9378 della pipeline float "
        "(agreement 99,42%), 21,7 KB di tabelle. Le soglie restano a 64 bit perche' su "
        "TON_IoT <i>src_bytes</i> e <i>dst_bytes</i> arrivano a 3,9·10<super>9</super>: "
        "sono i contatori di byte a imporre i 64 bit, non la durata."))

    story.append(P(
        "La catena e' anche <b>deployata</b>, non solo verificata: "
        "<i>mcu_pio/src/main_e2e.cpp</i> parte dai contatori grezzi ed esegue a bordo "
        "l'intera pipeline, feature engineering compreso, sotto il protocollo di "
        "benchmark del paper. Tutte le altre varianti di firmware ricevono vettori gia' "
        "normalizzati fuori dal dispositivo: senza questa, la catena sarebbe dimostrata "
        "corretta ma non sarebbe mai stata <i>la</i> pipeline in esecuzione sull'MCU. "
        "Firmware e harness di verifica includono lo stesso kernel "
        "(<i>include/kan_e2e_infer.h</i>), quindi cio' che e' verificato bit per bit e' "
        "cio' che gira sulla board, non una copia che puo' divergere."))

    story.append(P("4.6 Sul multiclass la profondità pesa quattro volte di più", "h2"))
    story.append(P(
        "Con lo stesso protocollo sulle 10 classi la KAN multi-layer fa "
        "<b>0,9374 ± 0,0036</b> contro 0,8767 della single-layer: <b>+0,0608 di macro-F1, "
        "15 fold su 15</b>, contro il +0,0141 del binario. Separare le famiglie di attacco "
        "richiede interazioni fra feature molto più di quanto ne richieda separare attacco "
        "da normale — stesso argomento strutturale, effetto quattro volte più grande. "
        "E l'albero profondo 5, che sul binario era il modello più piccolo e più accurato, "
        "qui crolla all'ultimo posto, 0,1741 sotto la multi-layer: il suo dominio in-domain "
        "era specifico del problema binario e non sopravvive al task più difficile."))

    story.append(P("4.7 Il soffitto su MITM è informativo, e riguarda tutti", "h2"))
    story.append(P(
        "Sul multiclass la classe MITM (1.043 flussi, 0,49%) è il collo di bottiglia per "
        "<b>ogni</b> modello: LightGBM 0,767, MLP 0,386, KAN single-layer 0,270, albero "
        "profondo 5 0,151. Per ogni modello tranne l'albero profondo 5 tutte le altre "
        "classi stanno sopra 0,88. Nessuno dei sei modelli supera 0,77 su MITM, e né "
        "focal loss né SMOTENC né il class weighting l'hanno spostata: è coerente con un "
        "limite dell'informazione disponibile in questo spazio di feature, ma non lo "
        "dimostra — sei architetture non esauriscono lo spazio delle architetture, e la "
        "dispersione fra loro su questa classe (da 0,151 a 0,767, un fattore cinque) è la "
        "più larga di ogni altra classe, quindi qui l'architettura pesa ancora molto."))

    # ── 5. cosa resta ────────────────────────────────────────
    story.append(P("4.8 La stima riportata non è ottimista: è conservativa", "h2"))
    story.append(P(
        "La cross-validation è corretta per una pipeline fissata, ma la nostra non lo "
        "era: k=10 è stato scelto guardando risultati calcolati sugli stessi dati. "
        "Invece di argomentare che l'effetto fosse piccolo, l'ho misurato con una "
        "<b>cross-validation annidata</b> — la scelta di k avviene dentro ogni fold "
        "esterno, su dati che non partecipano alla valutazione."))
    story.append(P(
        "Risultato: l'ottimismo è <b>negativo</b>. La stima annidata vale 0,9845 ± 0,0006 "
        "contro lo 0,9835 riportato per la KAN single-layer (−0,0009), e 0,9992 contro "
        "0,9991 per LightGBM (−0,0001). I numeri pubblicati sono semmai leggermente "
        "conservativi, perché il protocollo piatto è vincolato a un k=10 ereditato "
        "mentre quello annidato è libero di scegliere meglio."))
    story.append(P(
        "La misura però smentisce un'altra affermazione. La selezione interna <b>non "
        "sceglie mai k=10</b>: prende tutte e 16 le feature candidate in 15 fold su 15 "
        "per la KAN. La curva media è monotona, non ha un picco: 0,9795 a k=5, 0,9836 a "
        "k=10, 0,9845 a k=16. Il claim «l'accuratezza ha il massimo esattamente a 10 "
        "feature» non sopravvive al protocollo corretto. <b>k=10 è una scelta di "
        "deployment</b> — dieci statistiche di flusso da calcolare a bordo invece di "
        "sedici — e ora ha un prezzo misurato: 0,0009 di F1. È un argomento migliore di "
        "un picco che non c'è."))

    story.append(P("4.9 Cosa si puo' effettivamente flashare", "h2"))
    story.append(P(
        "Ogni modello esportato ha un firmware e un environment PlatformIO — "
        "<b>sette su sette</b> — quindi ognuno e' misurabile sulle due board con lo "
        "stesso protocollo: KAN LUT intera, KAN single-layer a coefficienti, KAN "
        "multi-layer, KAN multiclass, catena end-to-end binaria, catena end-to-end a 10 "
        "classi e Decision Tree profondo 5. Prima alcuni esistevano come header ma senza "
        "un <i>main</i> che li usasse: esportati sulla carta, non testabili fisicamente. "
        "Un test lo impedisce ora, e ne compila sette su sette a ogni esecuzione della "
        "suite, senza bisogno di toolchain per MCU."))
    story.append(P(
        "Nello stesso controllo e' emerso che l'export in C del modello multiclass era "
        "rimasto al <b>protocollo v1</b>: tabelle categoriche a 28 righe invece di 32, "
        "cioe' senza lo slot UNK. Il firmware girava su un modello incompatibile con il "
        "preprocessing attuale, e senza errori visibili perche' modello e vettori di test "
        "erano coerenti fra loro. Rigenerato; un test vieta ora qualunque header con "
        "tabelle a 28 righe."))

    story.append(P("5. Stato e passi successivi", "h1"))
    # firmware ed environment contati, non ricordati: erano fermi a 7 e 7
    # mentre gli environment sono diventati 21
    _mcu = REPO / "mcu_pio"
    _n_firmware = len(list((_mcu / "src").glob("main*.cpp")))
    _env = re.findall(r"\[env:([a-z0-9_]+)\]",
                      (_mcu / "platformio.ini").read_text(encoding="utf-8"))
    _n_env = len(_env)
    _n_env_energia = len([e for e in _env if "energy" in e])

    rows = [["Punto", "Stato"],
            ["1. Protocollo leakage-free", "completato, con misura dell'effetto"],
            ["2. Multi-layer con lo stesso protocollo", "completato: 0,9976 ± 0,0002"],
            ["3. BoT-IoT, 4 direzioni", "completato, con analisi del degrado"],
            ["4. Baseline identiche", "completato su binario e multiclass"],
            ["6. Riproducibilità da clone pulito", "completato, verificato"],
            ["5. Integer-only end-to-end (binario)", "completato: 200/200 bit-esatti, 1.334 B"],
            ["5. Integer-only end-to-end (10 classi)", "completato: 200/200 bit-esatti, 21,7 KB"],
            ["5. Firmware che parte dai contatori grezzi",
             "completato: main_e2e.cpp, 200 golden vector verificati, "
             "500 inferenze temporizzate"],
            ["Ogni modello esportato in C e flashabile",
             f"completato: {_n_firmware} firmware, {_n_env} environment PlatformIO"],
            ["Benchmark di energia a batch, senza I/O nella finestra",
             f"completato: main_energy.cpp, {_n_env_energia} environment; "
             f"misure sulle schede a cura del relatore"],
            ["Coerenza degli artefatti deployati", "completato: multiclass rigenerato al protocollo v2"],
            ["Ingombro dei parametri (asse dimensione)", "misurato: results/footprint.csv"],
            ["models/ + manifest (protocollo, seed, metriche)", "completato"],
            ["Conformal e forma simbolica sotto v2", "rigenerati: coperture 99,05/94,90/90,23"],
            ["Benchmark fisici (latenza, energia, code size)", "da fare — richiedono le board"],
            ["Focal loss e SMOTENC (risultati negativi)", "ancora v1; corroborati indipendentemente in v2"],
            ["Multi-layer multiclass in CV 5x3", "completato: 0,9374 +/- 0,0036 su 15 fit"],
            ["Indipendenza della stima (CV annidata)", "misurata: ottimismo -0,0009, conservativo"],
            ["CIC-IoT-2023 come quarto dominio, spazio ridotto 6+2",
             "completato: joint 1:5 valutato anche su CIC, costo della "
             "riduzione misurato sui domini che non ne hanno bisogno"],
            ["MLP piccolo esportato in C intero (baseline hardware)",
             "completato: 760 B misurati sull'header, non stimati"],
            ["Ingombro della configurazione scelta dalla selezione",
             "misurato: 9.452 B contro 5.244 della deployata"],
            ["Statistica dei confronti sull'unita' di analisi giusta",
             "completato: il seed, non le 120 coppie seed x modello x dominio"],
            ["Interpretabilita' diretta della single-layer",
             "completata: i 14 addendi sommano al logit del kernel C, 200/200"]]
    story.append(KeepTogether([table(rows, [8.0 * cm, 7.5 * cm]), Spacer(1, 8)]))
    story.append(P(
        "La priorità che suggerirei è il benchmark fisico: senza gli ingombri misurati, "
        "il confronto con l'albero profondo 5 resta aperto ed è la prima obiezione che "
        "farebbe un revisore. Il cross-domain, dal canto suo, fornisce la motivazione "
        "quantificata al passo successivo: a 40 punti di gap l'adattamento on-device non "
        "è un miglioramento marginale, ma la condizione perché un IDS embedded funzioni "
        "fuori dal laboratorio in cui è stato addestrato."))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2.4 * cm, rightMargin=2.4 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                            title="KAN-IDS — consolidamento sperimentale",
                            author="Emanuele Pio De Bernardis")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2.4 * cm, 1.2 * cm,
                          "KAN-IDS — consolidamento sperimentale, agosto 2026")
        canvas.drawRightString(A4[0] - 2.4 * cm, 1.2 * cm, str(doc_.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"scritto {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

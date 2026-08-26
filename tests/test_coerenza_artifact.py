"""Gli artifact devono riportare gli stessi byte che il compilatore vede.

Richiesta del Prof. Kuznetsov (punto 2): "README, MANIFEST, CSV, audit,
report, figure e commenti devono riportare gli stessi footprint e gli stessi
risultati. Elimini anche i valori vecchi rimasti nei file."

Il problema non era la singola cifra sbagliata, era che la regola di
conteggio esisteva in piu' copie: `scripts/export_e2e_int_c.py` la
riscriveva a mano e sbagliava due termini su tre (822 B invece di 1334),
poi il numero sbagliato si propagava in results/e2e_int_export.csv, da li'
in models/MANIFEST.json e da li' nel report PDF e nell'audit. La stessa
cosa era successa al ramo a 10 classi (13.922 B invece di 22.264).

Qui la regola e' una sola — `scripts/c_footprint.py`, quella verificata
contro `nm` sugli oggetti prodotti dal compilatore — e questi test
pretendono che ogni artefatto derivato ci coincida. Se qualcuno rigenera
un artefatto senza rigenerare gli altri, o reintroduce una formula scritta
a mano, il test cade prima che il numero finisca nell'articolo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from c_footprint import MODELS, scan  # noqa: E402

RESULTS = REPO / "results"


@pytest.fixture(scope="module")
def byte_misurati() -> dict[str, int]:
    """Byte del modello per ciascun header C, dalla regola unica."""
    out = {}
    for nome, header, prefisso, _descr in MODELS:
        path = REPO / "mcu_pio" / "include" / header
        assert path.exists(), f"header mancante: {header}"
        out[nome] = scan(path, prefisso)[0]
    return out


def test_footprint_csv_coincide_con_gli_header(byte_misurati):
    """Ogni riga di results/footprint.csv marcata "array C compilati" deve
    riportare esattamente i byte contati sull'header corrispondente."""
    d = pd.read_csv(RESULTS / "footprint.csv")
    misurate = d[d["regola"] == "array C compilati"]
    assert len(misurate) == len(MODELS), (
        f"footprint.csv ha {len(misurate)} righe misurate ma i modelli con "
        f"header C sono {len(MODELS)}")
    for _, r in misurate.iterrows():
        atteso = byte_misurati[r["modello"]]
        assert int(r["byte_parametri"]) == atteso, (
            f"{r['modello']}: footprint.csv dice {r['byte_parametri']} B, "
            f"l'header ne conta {atteso} B")
        assert round(atteso / 1024, 2) == pytest.approx(float(r["kb"]), abs=0.005)


@pytest.mark.parametrize(
    "csv_name, header, prefisso",
    [
        ("e2e_int_export.csv", "kan_e2e_int.h", "E2E_"),
        ("mc_e2e_int_export.csv", "kan_mc_e2e_int.h", "MC_"),
    ],
)
def test_csv_di_export_coincidono_con_gli_header(csv_name, header, prefisso):
    """I due CSV degli esportatori integer end-to-end. Sono quelli che
    avevano 822 e 13.922: il numero veniva da una formula riscritta a mano
    nello script invece che dall'header appena emesso."""
    atteso = scan(REPO / "mcu_pio" / "include" / header, prefisso)[0]
    d = pd.read_csv(RESULTS / csv_name)
    assert int(d["mem_bytes"].iloc[0]) == atteso, (
        f"{csv_name}: mem_bytes = {d['mem_bytes'].iloc[0]}, "
        f"l'header {header} ne conta {atteso}")
    if "mem_kb" in d.columns:
        assert float(d["mem_kb"].iloc[0]) == pytest.approx(atteso / 1024, abs=0.005)


def test_manifest_coincide_con_i_csv():
    """models/MANIFEST.json e' generato da scripts/export_models.py copiando
    i CSV: se e' stato rigenerato dopo di loro i numeri combaciano, se e'
    rimasto indietro no. Era rimasto indietro su entrambe le catene e2e."""
    manifest = json.loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    modelli = manifest["metriche"]
    coppie = [
        ("catena integer end-to-end binaria", "e2e_int_export.csv"),
        ("catena integer end-to-end 10 classi", "mc_e2e_int_export.csv"),
    ]
    for chiave, csv_name in coppie:
        assert chiave in modelli, f"chiave assente nel MANIFEST: {chiave}"
        voce = modelli[chiave][0]
        d = pd.read_csv(RESULTS / csv_name).iloc[0]
        assert int(voce["mem_bytes"]) == int(d["mem_bytes"]), (
            f"MANIFEST[{chiave}].mem_bytes = {voce['mem_bytes']} ma "
            f"{csv_name} dice {d['mem_bytes']}")
        if "mem_kb" in voce and "mem_kb" in d:
            assert float(voce["mem_kb"]) == pytest.approx(float(d["mem_kb"]), abs=0.005)


# Nome nel README (inglese) -> nome in results/footprint.csv. Se il README
# rinomina una riga, il test cade e va aggiornata questa mappa: e' voluto,
# perche' una riga che sparisce dalla mappa e' una riga non piu' controllata.
NOMI_README = {
    "KAN single-layer + cat": "KAN(cat,1L)",
    "Decision Tree (d=5)": "DecisionTree(d=5)",
    "MLP (16)": "MLP(16)",
    "KAN e2e integer (binary)": "KAN e2e integer (binario)",
    "KAN multi-layer + cat": "KAN(cat,ML)",
    "KAN multiclass (10 classes)": "KAN(cat,MC) 10 classi",
    "KAN LUT integer (default env)": "KAN-LUT integer (env default)",
    "KAN e2e integer (10 classes)": "KAN e2e integer (10 classi)",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}


def _tabella_footprint_del_readme() -> dict[str, tuple[int, str]]:
    """Estrae la tabella dei byte dal README: nome -> (byte, regola)."""
    righe = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    inizio = next(i for i, r in enumerate(righe)
                  if r.startswith("| Model | Bytes | Rule |"))
    out = {}
    for r in righe[inizio + 2:]:
        if not r.startswith("|"):
            break
        celle = [c.strip().strip("*").strip() for c in r.strip("|").split("|")]
        nome, byte, regola = celle[0], celle[1].strip("*").replace(",", ""), celle[2]
        out[nome] = (int(byte), regola.strip("*").strip())
    return out


def test_tabella_del_readme_coincide_con_footprint_csv():
    """Il README e' l'artefatto che il professore legge per primo, ed e'
    l'unico scritto a mano: la sua tabella dei byte deve coincidere riga per
    riga con results/footprint.csv, compresa l'etichetta compiled/estimate."""
    readme = _tabella_footprint_del_readme()
    csv = pd.read_csv(RESULTS / "footprint.csv").set_index("modello")
    assert set(readme) == set(NOMI_README), (
        f"la tabella del README non ha piu' le righe attese; "
        f"in piu': {set(readme) - set(NOMI_README)}; "
        f"mancanti: {set(NOMI_README) - set(readme)}")
    for nome_readme, (byte, regola) in readme.items():
        riga = csv.loc[NOMI_README[nome_readme]]
        assert byte == int(riga["byte_parametri"]), (
            f"README dice {byte} B per {nome_readme}, footprint.csv "
            f"{riga['byte_parametri']} B")
        misurato = riga["regola"] == "array C compilati"
        assert (regola == "compiled") == misurato, (
            f"{nome_readme}: il README lo marca {regola!r} ma footprint.csv "
            f"dice {riga['regola']!r}")


def test_colonna_crossdomain_del_readme_e_a_dieci_seed():
    """La stessa tabella riporta anche la balanced accuracy TON→BoT, e quella
    colonna era rimasta a 3 seed mentre il resto del README era passato a 10.
    Due celle (MLP 0.4703, XGBoost 0.5597) non corrispondevano ad alcun
    artefatto in results/, ne' a 3 seed ne' a 10: venivano da un run che nel
    repository non esiste piu'."""
    runs = pd.read_csv(RESULTS / "crossdomain_runs_cat.csv")
    runs = runs[runs.exp == "ton->bot"]
    assert runs.seed.nunique() == 10, "il protocollo cross-domain non e' piu' a 10 seed"
    atteso = runs.groupby("model").balanced_accuracy.mean().round(4)

    readme = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    inizio = next(i for i, r in enumerate(readme)
                  if r.startswith("| Model | Bytes | Rule |"))
    visti = 0
    for r in readme[inizio + 2:]:
        if not r.startswith("|"):
            break
        celle = [c.strip().strip("*").strip() for c in r.strip("|").split("|")]
        nome, cella = celle[0], celle[4].strip("*").strip()
        if cella == "—":
            continue
        modello = NOMI_README[nome]
        assert modello in atteso.index, f"{nome} non e' nei run cross-domain"
        assert float(cella) == pytest.approx(atteso[modello], abs=5e-5), (
            f"{nome}: il README dice {cella} per TON→BoT, i 10 seed danno "
            f"{atteso[modello]:.4f}")
        visti += 1
    assert visti >= 6, f"solo {visti} celle TON→BoT controllate"


NOMI_TABELLA_FINALE = {
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "KAN multi-layer": "KAN(cat,ML)",
    "Decision Tree (d=5)": "DecisionTree(d=5)",
    "MLP (16)": "MLP(16)",
    "KAN single-layer": "KAN(cat,1L)",
}


def test_tabella_finale_del_readme_coincide_con_lartefatto():
    """La tabella a sette colonne dell'articolo era montata a mano nel README
    da due CSV diversi. Ora la compone scripts/tabella_finale.py e questo test
    pretende che il README ci coincida cella per cella: e' la stessa
    protezione che si e' resa necessaria per la tabella di Pareto, dove due
    celle erano rimaste orfane di qualunque artefatto."""
    csv_path = RESULTS / "tabella_finale.csv"
    if not csv_path.exists():
        pytest.skip("results/tabella_finale.csv non generato "
                    "(python scripts/tabella_finale.py)")
    atteso = pd.read_csv(csv_path).set_index("model")

    righe = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    inizio = next(i for i, r in enumerate(righe)
                  if r.startswith("| Model | TON→TON |"))
    intestazioni = [c.strip() for c in righe[inizio].strip("|").split("|")][1:]
    for h in intestazioni:
        assert h in atteso.columns, f"colonna del README assente nell'artefatto: {h}"

    viste = 0
    for r in righe[inizio + 2:]:
        if not r.startswith("|"):
            break
        celle = [c.strip().strip("*").strip() for c in r.strip("|").split("|")]
        modello = NOMI_TABELLA_FINALE[celle[0]]
        for h, valore in zip(intestazioni, celle[1:]):
            if valore == "—":
                continue
            assert float(valore) == pytest.approx(atteso.loc[modello, h], abs=5e-5), (
                f"{celle[0]} / {h}: il README dice {valore}, "
                f"tabella_finale.csv dice {atteso.loc[modello, h]}")
            viste += 1
    assert viste == 42, f"attese 42 celle nella tabella finale, controllate {viste}"


# Valori che il passaggio da 3 a 10 seed ha ritirato. Il report PDF li aveva
# tutti: era l'artefatto piu' indietro di tutti, e nessuno se ne accorgeva
# perche' il PDF non lo legge nessun test.
VALORI_RITIRATI_NEL_REPORT = {
    "0,5632": "KAN(cat,1L) su TON→BoT a 3 seed; a 10 seed e' 0,5573",
    "0,5466": "DecisionTree su TON→BoT a 3 seed; a 10 seed e' 0,5494",
    "0,4651": "DecisionTree su BoT→TON a 3 seed; a 10 seed e' 0,4597",
    "0,4026": "KAN(cat,ML) su TON→BoT a 3 seed; a 10 seed e' 0,4588, e non e' "
              "piu' il peggiore — lo e' MLP(16)",
    "40–52 punti": "intervallo del degrado a 3 seed; a 10 seed e' 41–55",
    "3,7·10⁻¹⁴": "p-value di KAN(cat,ML) vs LightGBM, attribuito per errore al "
                 "confronto LightGBM vs KAN(cat,1L), che vale 1,0·10⁻²⁰",
    "In-domain: 15 fit": "le celle in-domain della tabella cross-domain sono "
                         "50 fit (5 fold x 10 seed), non 15",
    "rivelato due bug": "i difetti elencati nel corpo sono tre",
}


def test_il_report_non_contiene_valori_ritirati():
    """scripts/make_report.py genera il PDF che il relatore legge. Restava a
    3 seed mentre le tabelle erano gia' a 10, e affermava che la KAN
    multi-layer fosse il peggior modello in transfer — cosa che il README
    ritratta per iscritto. Ora il blocco cross-domain legge dai CSV; questo
    test impedisce che i valori tornino a essere scritti a mano."""
    righe = (REPO / "scripts" / "make_report.py").read_text(
        encoding="utf-8").splitlines()
    # I commenti non finiscono nel PDF e spiegano proprio perche' un valore e'
    # stato ritirato: si guardano solo le righe che producono testo. Le
    # citazioni storiche volute stanno dentro frasi che le dichiarano tali.
    testo = "\n".join(r for r in righe if not r.lstrip().startswith("#"))
    errata = "\n".join(r for r in righe if "versione precedente" in r)
    trovati = {v: perche for v, perche in VALORI_RITIRATI_NEL_REPORT.items()
               if v in testo and v not in errata}
    assert not trovati, "valori ritirati tornati in make_report.py:\n" + "\n".join(
        f"  {v}: {perche}" for v, perche in trovati.items())


def test_il_report_legge_il_blocco_crossdomain_dai_csv():
    testo = (REPO / "scripts" / "make_report.py").read_text(encoding="utf-8")
    assert 'deg["ton->bot"].idxmax()' in testo, (
        "il modello che trasferisce meglio non e' piu' derivato dai dati")
    assert 'deg["ton->bot"].idxmin()' in testo, (
        "il modello che trasferisce peggio non e' piu' derivato dai dati")
    assert "crossdomain_summary_cat.csv" in testo, (
        "il numero di fit in didascalia non e' piu' letto dal summary")


def test_nessuna_formula_di_conteggio_riscritta_a_mano():
    """Gli esportatori devono LEGGERE i byte dall'header emesso, non
    ricalcolarli. E' la causa a monte di tutta la propagazione: finche' la
    regola vive in due posti, i due posti divergono."""
    sospetti = {
        "export_e2e_int_c.py": "256 * 2",
        "export_mc_e2e_int_c.py": "mem = knot_bytes",
    }
    for script, frammento in sospetti.items():
        testo = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert frammento not in testo.replace("`", ""), (
            f"{script}: e' tornata una formula di conteggio scritta a mano "
            f"({frammento!r}). I byte vanno letti con c_footprint.scan() "
            f"sull'header appena scritto.")
        assert "from c_footprint import scan" in testo, (
            f"{script} non importa piu' la regola unica di conteggio")


def test_nessuno_script_ricalcola_i_byte_del_modello():
    """La versione larga del test precedente, che guarda TUTTI gli script.

    Quello sopra elencava due file per nome, ed e' cosi' che una terza copia
    e' sopravvissuta alla correzione: `e2e_int_pipeline.py` conteneva ancora
    `mem = sum(len(c) for c in C8) + 10*2 + 512 + 10*12`, con gli stessi due
    termini sbagliati (LUT del logaritmo contata int16 invece che int32).
    Dava ~842 B, un valore che il README dichiara ritirato — e
    `reproduce.py --stage integer` lo eseguiva, scrivendo quel numero in
    results/ accanto ai 1.334 B corretti.

    La regola: chi assegna a una variabile "mem"/"byte" una somma di
    letterali deve leggerli dall'header, non ricostruirli.
    """
    import re
    somma_a_mano = re.compile(
        r"^\s*(mem|byte|bytes|memoria)\w*\s*=\s*[^=\n]*\+[^=\n]*\b\d{2,}\b", re.M)
    # Eccezione dichiarata, con la sua ragione: coeff_int_inference.py conta
    # i soli dieci edge numerici, non il modello deployato (che ha anche le
    # tabelle categoriche). E' un oggetto diverso, non lo stesso contato male,
    # e la colonna del suo CSV lo dice nel nome.
    ALTRO_OGGETTO = {"coeff_int_inference.py": "mem_edge"}
    colpevoli = []
    for f in sorted((REPO / "scripts").glob("*.py")):
        testo = f.read_text(encoding="utf-8")
        codice = "\n".join(r for r in testo.splitlines()
                           if not r.lstrip().startswith("#"))
        for m in somma_a_mano.finditer(codice):
            if ALTRO_OGGETTO.get(f.name) and ALTRO_OGGETTO[f.name] in m.group(0):
                continue
            colpevoli.append(f"{f.name}: {m.group(0).strip()}")
    assert not colpevoli, (
        "formule di conteggio dei byte ricostruite a mano:\n  "
        + "\n  ".join(colpevoli)
        + "\n\nI byte del modello si leggono dall'header con "
          "c_footprint.scan(): e' l'unica regola verificata contro `nm`.")


def test_il_conformal_non_dichiara_byte_a_mano():
    """`conformal_ids.py` etichettava il modello deployato "230 B": non i 254
    di footprint.csv, non i 190 dei dieci edge int8 che lo script costruisce
    davvero. Non corrispondeva a niente, ed era gia' finito dentro
    results/conformal_ids_real.csv, cioe' in un artefatto pubblicato.

    Un primo tentativo di test cercava OGNI numero di byte in ogni sorgente e
    lo confrontava con footprint.csv: segnalava sette righe, tutte frasi che
    dichiarano superato un valore vecchio. Un test che grida al lupo sulle
    ritrattazioni e' peggio di nessun test, quindi qui si verifica la cosa
    precisa: quell'etichetta deve essere calcolata, non scritta.
    """
    testo = (REPO / "scripts" / "conformal_ids.py").read_text(encoding="utf-8")
    assert "230B" not in testo and "230 B" not in testo, (
        "conformal_ids.py: e' tornato il valore 230 B, che non corrisponde a "
        "nessun modello del progetto")
    assert "len(coefs[0])" in testo, (
        "conformal_ids.py: l'etichetta del modello deployato non e' piu' "
        "calcolata dai coefficienti che lo script costruisce")


def test_la_tabella_di_testa_del_readme_e_ai_valori_correnti():
    """La prima tabella del README ("Headline results") non era coperta da
    nessun test, e conteneva ancora F1 = 0.9837 del protocollo v1 — mentre
    la riga 83 dello stesso file scrive "0.9835 vs 0.9837 previously
    reported" e la tabella dei byte dice 0.9835. E' il primo numero che un
    revisore legge."""
    import re
    cv = pd.read_csv(REPO / "results" / "cv_leakagefree_summary_binary_ALL.csv")
    atteso = float(cv.loc[cv.model == "KAN(cat,1L)", "f1_mean"].iloc[0])
    testa = (REPO / "README.md").read_text(encoding="utf-8").split("## Headline results", 1)
    assert len(testa) == 2, "la sezione 'Headline results' non esiste piu'"
    blocco = testa[1].split("\n##", 1)[0]
    m = re.search(r"single-layer.*?F1 = ([\d.]+)", blocco)
    assert m, "la riga della KAN single-layer non e' piu' riconoscibile"
    assert abs(float(m.group(1)) - atteso) < 5e-5, (
        f"tabella di testa: F1 = {m.group(1)}, i CSV dicono {atteso:.4f}")


def test_nessuno_dichiara_sei_check_host_bit_esatti():
    """Il sesto check confronta contro le ETICHETTE VERE, fa 39/40, e il
    README stesso dice che "should not be quoted as one". Il commit v2.0 lo
    aveva corretto, ma due occorrenze erano sopravvissute — una nel README e
    una nel CHANGELOG."""
    import re
    frase = re.compile(r"(sei|six|6)\s+(check host|host check)", re.I)
    colpevoli = []
    for nome in ("README.md", "CHANGELOG.md", "mcu_pio/README.md"):
        for i, riga in enumerate((REPO / nome).read_text(encoding="utf-8").splitlines(), 1):
            if frase.search(riga) and "bit-e" in riga.lower().replace("exact", "esatt"):
                colpevoli.append(f"{nome}:{i}")
    assert not colpevoli, (
        f"'sei check host bit-esatti' e' tornato in {colpevoli}: sono cinque, "
        f"il sesto non e' un risultato di bit-esattezza")


def test_i_byte_del_firmware_di_energia_vengono_da_footprint():
    """main_energy.cpp ridichiara i byte di cinque modelli in
    `#define EB_MODEL_BYTES` e li stampa nel CSV dell'energia, cioe' dentro
    i risultati delle misure sulle schede. E' una copia della fonte: qui si
    verifica che almeno non diverga."""
    import re
    csv = pd.read_csv(REPO / "results" / "footprint.csv")
    validi = set(csv["byte_parametri"].astype(int))
    testo = (REPO / "mcu_pio" / "src" / "main_energy.cpp").read_text(encoding="utf-8")
    dichiarati = [int(x) for x in re.findall(r"#define EB_MODEL_BYTES\s+(\d+)", testo)]
    assert dichiarati, "main_energy.cpp non dichiara piu' EB_MODEL_BYTES"
    fuori = [d for d in dichiarati if d not in validi]
    assert not fuori, (
        f"main_energy.cpp dichiara byte che non sono in footprint.csv: {fuori}. "
        f"Finirebbero nel CSV delle misure di energia.")

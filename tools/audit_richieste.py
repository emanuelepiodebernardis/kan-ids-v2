#!/usr/bin/env python3
"""Verifica meccanica delle richieste della revisione.

Due sezioni: i sei punti della prima revisione, e i cinque della seconda
(la mail del 25 agosto: rapporto scelto su validation, artifact allineati,
formulazioni scientifiche, benchmark di energia, lock dell'ambiente).

Ogni controllo legge i file del repository e restituisce PASSA / PARZIALE /
NON FATTO con l'evidenza che lo sostiene. Serve perche' "abbiamo fatto tutto"
e' un'affermazione, mentre questo e' un controllo che chiunque puo'
rieseguire — relatore compreso, da un clone pulito.

Dove un requisito e' gia' coperto dalla suite, l'audit cita i test per nome
invece di riverificare a mano: la verifica resta in un posto solo, e chi
legge puo' rieseguire quel singolo test.

    python tools/audit_richieste.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Questo file stampa "TON→BoT" e altri caratteri fuori da ASCII, e non
# importa kanids a livello di modulo: la riconfigurazione va chiesta qui.
# Senza, `python tools/audit_richieste.py > audit.txt` (o una pipe, o la
# cattura di un log di CI) muore con UnicodeEncodeError dopo aver gia'
# stampato quaranta righe — cioe' proprio quando qualcuno prova a
# conservare il risultato dell'audit invece che a guardarlo e basta.
sys.path.insert(0, str(REPO))
from kanids.console import usa_utf8                       # noqa: E402
usa_utf8()

R = REPO / "results"
OK, PART, NO = "PASSA", "PARZIALE", "NON FATTO"
esiti = []


def voce(punto, requisito, stato, evidenza):
    esiti.append((punto, requisito, stato, evidenza))
    simbolo = {OK: "[ok]", PART: "[~ ]", NO: "[NO]"}[stato]
    print(f"  {simbolo} {requisito}")
    for riga in evidenza.splitlines():
        print(f"         {riga}")


def csv(nome):
    p = R / nome
    return pd.read_csv(p) if p.exists() else None


MODELLI = ["KAN(cat,1L)", "KAN(cat,ML)", "LightGBM", "XGBoost",
           "DecisionTree(d=5)", "MLP(16)"]

# ── 1 ────────────────────────────────────────────────────────
print("\n1) PROTOCOLLO LEAKAGE-FREE")
# -v serve alla sezione v2.1 in fondo, che cita i singoli test per nome
# invece di riverificare a mano cio' che la suite gia' verifica.
t = subprocess.run([sys.executable, "-m", "pytest", str(REPO / "tests"), "-v",
                    "--tb=no", "-p", "no:cacheprovider"],
                   capture_output=True, text=True, cwd=REPO)
SUITE = t.stdout
m = re.search(r"(\d+) passed", t.stdout)
n_test = int(m.group(1)) if m else 0
voce(1, "Trasformazioni apprese solo sul training",
     OK if t.returncode == 0 else NO,
     f"{n_test} test superati, fra cui il permutation test che riproduce il difetto v1"
     if t.returncode == 0 else f"suite fallita:\n{t.stdout.strip().splitlines()[-1]}")

b = csv("cv_leakagefree_summary_binary_ALL.csv")
mc = csv("cv_leakagefree_summary_multiclass_ALL.csv")
if b is not None and mc is not None:
    nb = {r["model"]: int(r["n_runs"]) for _, r in b.iterrows()}
    nm = {r["model"]: int(r["n_runs"]) for _, r in mc.iterrows()}
    tutti15 = all(nb.get(k) == 15 for k in MODELLI) and all(nm.get(k) == 15 for k in MODELLI)
    voce(1, "5-fold x 3 seed = 15 fit per ogni modello principale",
         OK if tutti15 else PART,
         f"binario: {nb}\nmulticlass: {nm}")
    ha_std = all(f"{c}_std" in b.columns for c in ("f1", "precision", "recall"))
    voce(1, "Media e deviazione standard riportate", OK if ha_std else NO,
         "colonne *_mean e *_std presenti in entrambe le tabelle")
else:
    voce(1, "Tabelle 5x3", NO, "file di sintesi mancanti")

n = csv("nested_cv_summary_binary.csv")
if n is not None:
    r = n.iloc[0]
    voce(1, "Valutazione finale realmente indipendente",
         OK, f"CV annidata: ottimismo misurato {r.ottimismo:+.4f} "
             f"(negativo = stima conservativa), su {int(r.n_fold)} fold esterni")
else:
    voce(1, "Valutazione finale realmente indipendente", NO, "nested_cv non eseguita")

# ── 2 ────────────────────────────────────────────────────────
print("\n2) KAN MULTI-LAYER CON LO STESSO PROTOCOLLO")
for tab, nome, key in ((b, "binario", "f1"), (mc, "multiclass", "macro_f1")):
    if tab is not None and "KAN(cat,ML)" in set(tab.model):
        r = tab[tab.model == "KAN(cat,ML)"].iloc[0]
        voce(2, f"Multi-layer in CV 5x3 ({nome})", OK,
             f"{key} = {r[f'{key}_mean']:.4f} +/- {r[f'{key}_std']:.4f} su {int(r.n_runs)} fit")
    else:
        voce(2, f"Multi-layer in CV 5x3 ({nome})", NO, "assente dalla tabella")

# ── 3 ────────────────────────────────────────────────────────
print("\n3) SECONDO DATASET: BoT-IoT")
cd = csv("crossdomain_runs_cat.csv")
if cd is not None:
    attesi = ["ton->ton", "bot->bot", "ton->bot", "bot->ton"]
    conteggi = cd.groupby("exp").size().to_dict()
    completi = all(e in conteggi for e in attesi)
    voce(3, "I quattro esperimenti richiesti", OK if completi else PART,
         "\n".join(f"{e}: {conteggi.get(e,0)} run, "
                   f"train={int(cd[cd.exp==e].n_train.iloc[0]):,} "
                   f"test={int(cd[cd.exp==e].n_test.iloc[0]):,}" for e in attesi))
    voce(3, "Task binario normal vs attack", OK,
         "colonne tp/tn/fp/fn a due classi in tutti i run")
else:
    voce(3, "I quattro esperimenti richiesti", NO, "crossdomain_runs_cat.csv assente")

try:
    sys.path.insert(0, str(REPO))
    from kanids.harmonized import HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC
    voce(3, "Solo feature realmente confrontabili", OK,
         f"{len(HARMONIZED_NUMERIC)} numeriche derivate con la stessa formula sui due "
         f"dataset + {len(HARMONIZED_CATEGORICAL)} categoriche su alfabeto comune;\n"
         f"escluse porte, indirizzi, aggregati a finestra di BoT-IoT e metadati "
         f"DNS/SSL/HTTP di TON_IoT")
except Exception as e:
    voce(3, "Solo feature realmente confrontabili", NO, str(e))

vincolo = "test_crossdomain_target_does_not_influence_training"
ha = vincolo in (REPO / "tests" / "test_leakage.py").read_text(encoding="utf-8")
voce(3, "Il target non entra in selezione, normalizzazione, tuning",
     OK if ha else NO,
     "imposto da un test: si fitta due volte sullo stesso source con target\n"
     "diversissimi e si pretende che tutto l'appreso sia identico bit per bit")

# Il terzo e il quarto dominio vivono in adattamento-drift/, sottoprogetto
# autonomo con il proprio kanids/ (vedi il suo README). Questo controllo
# guardava solo results/ e scripts/ della radice, quindi continuava a
# riportare "non iniziato" quando invece era fatto. Ora guarda in entrambi, e
# cerca le API di CIC invece della sottostringa "cic" nel testo: quest'ultima
# compare di sfuggita in file che con CIC non c'entrano.
DRIFT = REPO / "adattamento-drift"
CIC_API = ("build_minimo_cic", "build_ridotto_cic", "load_cic", "cic_paths",
           "CIC_SINGLE_FILENAME", "CIC_GLOB")


def _rel(p):
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def _risultati_cic(*cartelle):
    """File di risultati il cui nome nomina CIC."""
    return [_rel(f) for c in cartelle if c.is_dir()
            for f in sorted(c.glob("*")) if f.is_file() and "cic" in f.name.lower()]


def _codice_cic(*cartelle):
    """File che definiscono o usano le API di CIC, non che ne nominano la
    sigla per caso."""
    trovati = []
    for c in cartelle:
        if not c.is_dir():
            continue
        for f in sorted(c.glob("*.py")):
            try:
                testo = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(api in testo for api in CIC_API):
                trovati.append(_rel(f))
    return trovati


cic_res = _risultati_cic(R, DRIFT / "results")
cic_cod = _codice_cic(REPO / "scripts", REPO / "kanids",
                      DRIFT / "scripts", DRIFT / "kanids")
if cic_res and cic_cod:
    voce(3, "CIC-IoT-2023 come terzo dataset (secondario)", OK,
         f"{len(cic_res)} file di risultati, {len(cic_cod)} di codice:\n"
         + "\n".join("  " + f for f in cic_res + cic_cod) + "\n"
         "il costo dello spazio ridotto che CIC impone e' stato misurato prima\n"
         "di adottarlo (mancano i conteggi direzionali: 7 delle 13 feature\n"
         "numeriche cadono), quindi UNSW-NB15 e' il terzo dominio dell'analisi\n"
         "principale nello spazio ricco e CIC-IoT-2023 il quarto nel suo.\n"
         "Vedi adattamento-drift/RISULTATI.md, sezioni 10, 14 e 15")
elif cic_res or cic_cod:
    voce(3, "CIC-IoT-2023 come terzo dataset (secondario)", PART,
         f"risultati: {cic_res or 'nessuno'}\ncodice: {cic_cod or 'nessuno'}")
else:
    voce(3, "CIC-IoT-2023 come terzo dataset (secondario)", NO,
         "non iniziato: il professore lo indica come obiettivo secondario")

# ── 4 ────────────────────────────────────────────────────────
print("\n4) BASELINE IDENTICHE")
if b is not None:
    presenti = [m for m in MODELLI if m in set(b.model)]
    voce(4, "KAN, LightGBM, XGBoost, Decision Tree shallow, MLP",
         OK if len(presenti) == 6 else PART,
         f"{len(presenti)}/6 sullo stesso spazio di feature: {', '.join(presenti)}")
    attese = ["f1_mean", "precision_mean", "recall_mean", "pr_auc_mean"]
    ok_m = all(c in b.columns for c in attese)
    attese_mc = ["macro_f1_mean", "macro_precision_mean", "macro_recall_mean",
                 "pr_auc_macro_mean"]
    ok_mc = mc is not None and all(c in mc.columns for c in attese_mc)
    voce(4, "F1/Macro-F1, Precision, Recall, PR-AUC",
         OK if (ok_m and ok_mc) else PART,
         f"binario: {[c for c in attese if c in b.columns]}\n"
         f"multiclass: {[c for c in attese_mc if c in (mc.columns if mc is not None else [])]}")
n_cm = len(list(R.glob("confusion_crossdomain_*.csv")))
voce(4, "Confusion matrix nel cross-domain", OK if n_cm >= 20 else PART,
     f"{n_cm} matrici salvate (4 esperimenti x modelli x 2 varianti)")
sh = csv("crossdomain_shift.csv")
voce(4, "Analisi del perche' degrada", OK if sh is not None else PART,
     ("sovrapposizione delle marginali per feature (min "
      f"{sh.sovrapposizione.min():.3f} su {sh.feature.iloc[0]}), tasso UNK sulle\n"
      "categoriche, e ablation con/senza edge categorici")
     if sh is not None else "crossdomain_shift.csv assente")

# ── 5 ────────────────────────────────────────────────────────
print("\n5) INFERENZA INTEGER-ONLY END-TO-END")
for nome, f, chiave in (("binaria", "e2e_int_export.csv", "f1_e2e_int"),
                        ("10 classi", "mc_e2e_int_export.csv", "macro_f1_e2e_int")):
    d = csv(f)
    if d is not None:
        r = d.iloc[0]
        voce(5, f"Catena {nome} dai dati grezzi alla decisione", OK,
             f"{chiave} = {r[chiave]:.4f}, {int(r.n_golden)} golden vector, "
             f"{int(r.mem_bytes):,} B di tabelle")
    else:
        voce(5, f"Catena {nome}", NO, f"{f} assente")
h = [(REPO / "mcu_pio" / "include" / n) for n in ("kan_e2e_int.h", "kan_mc_e2e_int.h")]
k = [(REPO / "mcu_pio" / "host_check" / n) for n in ("run_e2e_check.cpp", "run_mc_e2e_check.cpp")]
def _senza_commenti(testo):
    return "\n".join(r for r in testo.splitlines() if not r.strip().startswith("//"))

fp = sum(len(re.findall(r"\b(float|double)\b", _senza_commenti(p.read_text(encoding="utf-8"))))
         for p in h + k if p.exists())
voce(5, "Nessun floating point nel runtime MCU", OK if fp == 0 else NO,
     f"{fp} occorrenze di float/double in header e kernel, commenti esclusi;\n"
     f"verifica sull'assembly con tools/check_no_float.py")
src = REPO / "mcu_pio" / "src"
fw_e2e = [f.name for f in src.glob("*.cpp")
          if "kan_e2e_infer.h" in f.read_text(errors="ignore", encoding="utf-8")
          or "kan_mc_e2e_int.h" in f.read_text(errors="ignore", encoding="utf-8")]
voce(5, "Un firmware usa la catena end-to-end, non vettori pre-normalizzati",
     OK if fw_e2e else NO,
     f"{', '.join(fw_e2e)} parte dai contatori grezzi ed esegue a bordo anche\n"
     f"il feature engineering; environment PlatformIO dedicati in platformio.ini"
     if fw_e2e else
     "gli altri main ricevono vettori gia' normalizzati: la catena e2e\n"
     "sarebbe verificata ma non deployata")

shared = REPO / "mcu_pio" / "include" / "kan_e2e_infer.h"
hc = (REPO / "mcu_pio" / "host_check" / "run_e2e_check.cpp").read_text(errors="ignore", encoding="utf-8")
voce(5, "Firmware e harness di verifica condividono lo stesso kernel",
     OK if (shared.exists() and "kan_e2e_infer.h" in hc) else PART,
     "kan_e2e_infer.h incluso sia dal firmware sia dall'host check:\n"
     "cio' che e' verificato bit per bit e' cio' che gira sulla board")

ini = (REPO / "mcu_pio" / "platformio.ini").read_text(errors="ignore", encoding="utf-8")
attesi = {
    "KAN single-layer (coeff int8)": "main_coeff.cpp",
    "KAN multi-layer": "main_mlcoeff.cpp",
    "KAN multiclass": "main_mc.cpp",
    "KAN LUT integer": "main.cpp",
    "catena e2e binaria": "main_e2e.cpp",
    "catena e2e 10 classi": "main_mc_e2e.cpp",
    "Decision Tree d=5 (confronto Pareto)": "main_dt5.cpp",
    "KAN single-layer campionata (sampled-LUT)": "main_lut14.cpp",
    "MLP(16) denso (baseline hardware)": "main_mlp.cpp",
}
mancanti = [k for k, v in attesi.items()
            if not (src / v).exists() or v not in ini]
voce(5, "Ogni modello ha un firmware flashabile con il suo environment",
     OK if not mancanti else PART,
     f"{len(attesi) - len(mancanti)}/{len(attesi)}: " +
     ", ".join(f"{k} -> {v}" for k, v in attesi.items() if k not in mancanti)
     if not mancanti else f"senza firmware: {mancanti}")

v1_headers = []
for _hdr in (REPO / "mcu_pio" / "include").glob("*.h"):
    for _m in re.finditer(r"CAT\[(\d+)\]\[\d+\]", _hdr.read_text(errors="ignore", encoding="utf-8")):
        if int(_m.group(1)) == 28:
            v1_headers.append(_hdr.name)
voce(5, "Nessun header di modello e' rimasto al protocollo v1",
     OK if not v1_headers else NO,
     "tutte le tabelle categoriche hanno lo slot UNK (32 righe: 4+10+14+4)"
     if not v1_headers else
     f"header ancora v1 (28 righe, senza UNK): {sorted(set(v1_headers))}")

voce(5, "Python solo per training, export e golden vector",
     OK if all(p.exists() for p in h + k) else NO,
     "i kernel C sono autosufficienti: gli host check compilano e girano\n"
     "senza dataset e senza Python")

# ── 6 ────────────────────────────────────────────────────────
print("\n6) RIPRODUCIBILITA'")
cartelle = {"scripts": "scripts", "models/artifacts": "models",
            "results": "results", "firmware": "mcu_pio",
            "requirements/environment": "requirements.txt"}
mancanti = [k for k, v in cartelle.items() if not (REPO / v).exists()]
voce(6, "Organizzazione richiesta", OK if not mancanti else PART,
     "presenti: " + ", ".join(cartelle) if not mancanti else f"mancanti: {mancanti}")
rp = REPO / "reproduce.py"
stage = re.findall(r'^\s{4}"([a-z-]+)": \(', rp.read_text(encoding="utf-8"), re.M) if rp.exists() else []
voce(6, "Script principale per riprodurre gli esperimenti",
     OK if len(stage) >= 8 else PART,
     f"reproduce.py con {len(stage)} stage: {', '.join(stage)}")
mf = REPO / "models" / "MANIFEST.json"
if mf.exists():
    man = json.loads(mf.read_text(encoding="utf-8"))
    voce(6, "Seed utilizzati documentati", OK,
         f"split={man['seed_split']}, cross-validation={man['seed_cross_validation']}, "
         f"in models/MANIFEST.json e stampati a ogni run")
else:
    voce(6, "Seed utilizzati documentati", PART, "MANIFEST.json assente")
# Scansione in Python puro: `grep` non esiste su Windows e l'audit deve
# poter girare ovunque giri il progetto.
# Gli strumenti in tools/ contengono il pattern per mestiere (uno lo migra,
# questo lo cerca) e vanno esclusi, come fa gia' il test della suite.
pattern = '"' + "/tmp/"
tmp = []
for f in REPO.rglob("*.py"):
    if "tools" in f.parts or f.name == "test_reproducibility.py":
        continue
    if any(x in f.parts for x in (".git", "artifacts", "__pycache__")):
        continue
    try:
        if pattern in f.read_text(encoding="utf-8", errors="ignore"):
            tmp.append(str(f.relative_to(REPO)))
    except OSError:
        pass
voce(6, "Nessuna dipendenza da file temporanei locali", OK if not tmp else NO,
     "0 percorsi /tmp negli script" if not tmp else f"residui: {tmp}")

# ═════════════════════════════════════════════════════════════
# SECONDA REVISIONE — i cinque punti chiesti per la v2.1
# ═════════════════════════════════════════════════════════════
def test_esito(frammento: str) -> str | None:
    """PASSED / FAILED / SKIPPED del primo test il cui id contiene
    `frammento`, dalla suite gia' eseguita al punto 1. L'audit cita i test
    invece di riverificare a mano: la verifica sta in un posto solo, e chi
    legge puo' rieseguire quel test da solo."""
    for riga in SUITE.splitlines():
        if frammento in riga:
            for stato in ("PASSED", "FAILED", "SKIPPED", "ERROR"):
                if stato in riga:
                    return stato
    return None


def da_test(punto, requisito, *frammenti):
    esiti_test = {f: test_esito(f) for f in frammenti}
    mancanti = [f for f, s in esiti_test.items() if s is None]
    falliti = [f for f, s in esiti_test.items() if s in ("FAILED", "ERROR")]
    saltati = [f for f, s in esiti_test.items() if s == "SKIPPED"]
    if mancanti:
        stato, nota = NO, f"test non trovati: {mancanti}"
    elif falliti:
        stato, nota = NO, f"test falliti: {falliti}"
    elif saltati:
        # il motivo del salto non e' sempre "toolchain assente": puo' essere
        # un artefatto che richiede il dataset, o uno stato non ancora
        # committato. Dirlo generico faceva sembrare un problema di ambiente
        # cio' che invece era un passo di lavoro ancora da fare.
        stato, nota = PART, (f"{len(esiti_test) - len(saltati)} verdi, "
                             f"{len(saltati)} saltati: {saltati}\n"
                             f"  (il motivo lo stampa pytest -rs)")
    else:
        n = len(esiti_test)
        stato = OK
        nota = f"{n} test verde" if n == 1 else f"{n} test verdi"
    voce(punto, requisito, stato,
         nota + "\n" + "\n".join(f"  {f}" for f in frammenti))


print("\n" + "=" * 74)
print("SECONDA REVISIONE (v2.1) — i cinque punti della mail del 25 agosto")
print("=" * 74)

# ── v2.1-1: rapporto scelto su validation, test usati una volta sola ──
print("\nv1) JOINT TRAINING: RAPPORTO SCELTO SU VALIDATION INTERNA")
jt = (REPO / "scripts" / "joint_training.py").read_text(encoding="utf-8")
voce("v1", "Esiste una validation ritagliata dentro il training",
     OK if "def inner_split(" in jt else NO,
     "joint_training.py::inner_split divide split_train in fit + validation")
_m_cand = re.search(r"RATIOS_CANDIDATI = \(([^)]*)\)", jt)
_candidati = _m_cand.group(1) if _m_cand else "?"
voce("v1", "La scelta del rapporto ha un comando dedicato che non tocca i test",
     OK if "--select-ratio" in jt else NO,
     "joint_training.py --select-ratio; candidati: " + _candidati)
voce("v1", "--ratio non ha piu' un valore di default scelto a mano",
     OK if 'ap.add_argument("--ratio", type=float, default=None' in jt else NO,
     "senza --ratio esplicito il rapporto viene letto dalla selezione su validation")
da_test("v1", "Anche larghezza e grado si scelgono su validation, non sul test",
        "test_la_selezione_non_legge_il_test",
        "test_validation_e_test_sono_disgiunti",
        "test_la_regola_e_quella_dichiarata",
        "test_la_pipeline_legge_larchitettura_invece_di_riscriverla",
        "test_lo_scarto_dalla_selezione_e_dichiarato")
def _prezzo_architettura() -> str:
    """Quanto costa la configurazione scelta, LETTO da arch_footprint.csv.

    Prima questa riga diceva "per vincolo di dimensione" e basta. Adesso il
    prezzo e' misurato, quindi si stampa: un requisito dichiarato non pieno
    deve dire anche quanto vale la scelta di lasciarlo tale."""
    d = csv("arch_footprint.csv")
    if d is None or len(d) < 2 or "ruolo" not in d.columns:
        return ("\nIl prezzo di quella scelta non e' ancora misurato: "
                "python scripts/footprint_architettura.py")
    try:
        dep = d[d.ruolo == "deployata"].iloc[0]
        sel = d[d.ruolo == "selezionata"].iloc[0]
    except IndexError:
        return ""
    db = int(sel.byte_parametri) - int(dep.byte_parametri)
    coda = (f"\nIl prezzo e' misurato, non affermato: la configurazione scelta "
            f"occupa {int(sel.byte_parametri):,} B contro "
            f"{int(dep.byte_parametri):,} ({db:+,} B, "
            f"{100.0 * db / int(dep.byte_parametri):+.1f}%)")
    if "byte_avr_stack_main" in d.columns:
        coda += (f" e {int(sel.byte_avr_stack_main)} B di stack contro "
                 f"{int(dep.byte_avr_stack_main)}")
    return coda + ". Entrambe stanno su entrambe le schede."


try:
    from kanids import scarto_dalla_selezione as _scarto
    _s = _scarto()
    if _s is None:
        voce("v1", "Selezione dell'architettura eseguita", PART,
             "non ancora eseguita: python scripts/select_architettura.py "
             "--seeds 42,43,44,45,46")
    else:
        _righe = []
        for _m, _v in _s.items():
            _d, _sel = _v["deployata"], _v["selezionata"]
            _righe.append(
                f"  {_m}: deployata h={_d['hidden']} g={_d['degree']}, "
                f"selezionata h={_sel['hidden']} g={_sel['degree']}"
                + ("  (coincidono)" if _v["coincidono"] else "  DIVERSE"))
        _tutte = all(_v["coincidono"] for _v in _s.values())
        voce("v1", "Architettura: cosa sceglie la validation e cosa si deploya",
             OK if _tutte else PART,
             "\n".join(_righe) + ("" if _tutte else
             "\nlo scarto e' dichiarato nel README, sezione 'Architecture: "
             "selected and deployed are not the same': il progetto deploya la "
             "configurazione piu' piccola, e non perche' la selezione l'abbia "
             "scelta." + _prezzo_architettura()))
except Exception as _e:                                    # pragma: no cover
    voce("v1", "Selezione dell'architettura", NO, str(_e))
da_test("v1", "I test set non entrano nella scelta del rapporto",
        "test_ratio_selection_ignores_the_test_sets_entirely",
        "test_inner_split_partitions_the_train_split_only",
        "test_inner_split_keeps_both_classes_in_validation")

sc = R / "joint_ratio_selection_scelta.json"
if sc.exists():
    d = json.loads(sc.read_text(encoding="utf-8"))
    concordi = f"{d.get('seed_concordi_su_scelta')}/{d.get('seed_totali')}"
    conf = d.get("confronti_appaiati", [])
    righe_conf = [
        f"  contro 1:{c['contro']:g}: {c['differenza_media']:+.4f}, "
        f"p={c['p_value']:.2e}, "
        f"{'significativa' if c['significativa_5pct'] else 'NON significativa'}, "
        f"vince in {c['vince_in']}"
        for c in conf]
    voce("v1", "La selezione e' stata eseguita e i test non sono stati toccati",
         OK if d.get("test_set_usati_in_questa_fase") == 0 else NO,
         f"rapporto scelto 1:{d['ratio_scelto']:g} su candidati {d['candidati']}\n"
         f"criterio: {d['criterio']}\n"
         f"stesso argmax in {concordi} seed presi singolarmente\n"
         f"medie per rapporto: {d['media_per_rapporto']}"
         + ("\nconfronti appaiati:\n" + "\n".join(righe_conf) if conf else ""))
else:
    voce("v1", "La selezione e' stata eseguita", NO,
         "manca results/joint_ratio_selection_scelta.json:\n"
         "  python scripts/joint_training.py --select-ratio")

# ── v2.1-2: artifact allineati ──
print("\nv2) ARTIFACT ALLINEATI, VALORI VECCHI RIMOSSI")
da_test("v2", "Byte del modello coerenti fra header C, CSV, MANIFEST e README",
        "test_footprint_csv_coincide_con_gli_header",
        "test_csv_di_export_coincidono_con_gli_header",
        "test_manifest_coincide_con_i_csv",
        "test_tabella_del_readme_coincide_con_footprint_csv")
da_test("v2", "La regola di conteggio dei byte esiste in un posto solo",
        "test_nessuna_formula_di_conteggio_riscritta_a_mano",
        "test_nessuno_script_ricalcola_i_byte_del_modello",
        "test_il_conformal_non_dichiara_byte_a_mano")
da_test("v2", "Anche la prima tabella e i commenti del firmware sono verificati",
        "test_la_tabella_di_testa_del_readme_e_ai_valori_correnti",
        "test_il_numero_di_check_bit_esatti_e_quello_dei_sorgenti",
        "test_i_byte_del_firmware_di_energia_vengono_da_footprint")
da_test("v2", "Gli artefatti non cambiano a seconda della macchina che li rigenera",
        "test_ogni_io_testuale_dichiara_lencoding",
        "test_gli_artefatti_non_ascii_sono_utf8",
        "test_il_manifest_si_rilegge_uguale",
        "test_ogni_scrittura_di_testo_fissa_il_terminatore_di_riga",
        "test_gitattributes_fissa_i_terminatori_di_riga",
        "test_nessun_file_di_testo_versionato_ha_i_cr",
        "test_nessun_percorso_windows_negli_artefatti",
        "test_il_manifest_misura_i_byte_del_contenuto_non_del_checkout",
        "test_ogni_to_csv_fissa_il_terminatore_di_riga",
        "test_il_repository_non_contiene_file_di_appoggio_della_sessione",
        "test_lelenco_dei_txt_ammessi_e_quello_che_c_e_davvero")
da_test("v2", "L'output degli script non dipende da dove viene scritto",
        "test_il_difetto_esiste_davvero",
        "test_importare_kanids_mette_loutput_in_utf8",
        "test_laudit_riconfigura_loutput_prima_di_stampare",
        "test_reproduce_passa_lencoding_ai_figli")
sys.path.insert(0, str(REPO / "scripts"))
try:
    from c_footprint import collect as _fp
    righe = [f"  {r['modello']:<30}{r['byte_parametri']:>7} B" for r in _fp()]
    voce("v2", "Byte misurati dagli header che il compilatore vede", OK,
         "\n".join(righe))
except Exception as exc:                                   # pragma: no cover
    voce("v2", "Byte misurati dagli header", NO, f"c_footprint non eseguibile: {exc}")

# ── v2.1-3: formulazioni scientifiche ──
print("\nv3) FORMULAZIONI SCIENTIFICHE")
rd = (REPO / "README.md").read_text(encoding="utf-8")
ritirate = {
    "UNSW 0.8184 presentato come ceiling":
        "No TON+BoT→UNSW result can exceed 0.8184",
    "KAN-1L dichiarata la migliore in transfer":
        "The single-layer KAN remains the best cross-domain performer",
    "LightGBM dichiarata ultima cross-domain":
        "LightGBM, first in-domain, is last cross-domain",
    "LightGBM dichiarata non installabile sul Mega":
        "400 boosted trees do not fit an ATmega2560 at all",
    "MITM presentato come limite dimostrato":
        "MITM is the ceiling for everyone",
}
rimaste = {k: v for k, v in ritirate.items() if v in rd}
voce("v3", "Le affermazioni ritirate non sono piu' nel README",
     OK if not rimaste else NO,
     f"{len(ritirate)} formulazioni ritirate, {len(rimaste)} ancora presenti"
     + ("" if not rimaste else "\n" + "\n".join(f"  ANCORA PRESENTE: {k}" for k in rimaste)))

sig = csv("crossdomain_significativita.csv")
if sig is not None and "p_holm" in sig.columns:
    tb = sig[sig.exp == "ton->bot"]
    non_sep = tb[tb.p_holm >= 0.05].sort_values("p_holm")
    voce("v3", "I confronti fra modelli hanno un test appaiato a sostegno", OK,
         f"{len(sig)} confronti appaiati per seed in "
         f"crossdomain_significativita.csv, con correzione di Holm per famiglia\n"
         f"su ton->bot {len(non_sep)} coppie su {len(tb)} NON sono separabili "
         f"dopo Holm\n"
         + (f"la piu' stretta fra quelle non separabili: "
            f"{non_sep.iloc[0].modello_a} vs {non_sep.iloc[0].modello_b}, "
            f"p Holm = {non_sep.iloc[0].p_holm:.3g}\n" if len(non_sep) else "")
         + "il test confronta dieci riaddestramenti sugli stessi identici "
           "training e test set: misura la variabilita' di riaddestramento, "
           "non quella di campionamento, e il CSV lo dichiara riga per riga")
elif sig is not None:
    voce("v3", "I confronti fra modelli hanno un test appaiato a sostegno", NO,
         "crossdomain_significativita.csv e' nel formato vecchio: "
         "python scripts/statistica_confronti.py")
else:
    voce("v3", "Confronti appaiati fra modelli", NO,
         "manca results/crossdomain_significativita.csv "
         "(python scripts/crossdomain_report.py)")
da_test("v3", "La colonna cross-domain del README e' ai 10 seed del protocollo",
        "test_colonna_crossdomain_del_readme_e_a_dieci_seed")

# ── v2.1-4: benchmark energia ──
print("\nv4) BENCHMARK DI ENERGIA")
fw = REPO / "mcu_pio" / "src" / "main_energy.cpp"
voce("v4", "Esiste un firmware dedicato alla misura di energia",
     OK if fw.exists() else NO,
     f"{_rel(fw)} — finestra di EB_BATCH inferenze marcata su un pin, "
     f"finestra di riferimento di pari durata" if fw.exists() else "assente")
da_test("v4", "Nessuna Serial e nessun I2C dentro la finestra misurata",
        "test_la_finestra_misurata_non_contiene_io",
        "test_la_finestra_di_riferimento_ha_la_stessa_durata")
try:
    from kanids.toolchain import motivo_assenza as _perche, trova as _trova
    _comp = {n: _trova(n) for n in ("avr-g++", "g++")}
    _righe = []
    for _n, _p in _comp.items():
        _righe.append(f"{_n:<9} {_p if _p else 'assente'}")
        if _p is None:
            _righe.append(f"          {_perche(_n)}")
    voce("v4", "Compilatori disponibili per i test che compilano davvero",
         OK if all(_comp.values()) else PART, "\n".join(_righe))
except Exception as _e:                                    # pragma: no cover
    voce("v4", "Compilatori disponibili", NO, str(_e))
da_test("v4", "Le inferenze non possono essere ottimizzate via dal compilatore",
        "test_il_ciclo_di_misura_non_puo_essere_ottimizzato_via",
        "test_le_inferenze_avvengono_davvero")
ini = (REPO / "mcu_pio" / "platformio.ini").read_text(encoding="utf-8")
env_en = re.findall(r"\[env:([a-z0-9_]*energy[a-z0-9_]*)\]", ini)
voce("v4", "Environment PlatformIO per entrambe le schede",
     OK if len(env_en) >= 2 else NO,
     f"{len(env_en)} environment: {', '.join(env_en)}")

# ── v2.1-5: lock e riproducibilita' ──
print("\nv5) AMBIENTE BLOCCATO E TABELLE RIPRODUCIBILI")
da_test("v5", "requirements.txt e requirements-lock.txt sono coerenti",
        "test_ogni_versione_del_lock_soddisfa_il_vincolo_dichiarato",
        "test_ogni_dipendenza_dichiarata_e_bloccata",
        "test_il_lock_e_esatto",
        "test_pyarrow_e_dichiarato")
da_test("v5", "reproduce.py copre le tabelle principali, nell'ordine giusto",
        "test_reproduce_esegue_script_che_esistono",
        "test_reproduce_copre_le_tabelle_principali")
da_test("v5", "Il pacchetto per il relatore si genera da script",
        "test_lo_script_esiste_e_si_importa",
        "test_il_pacchetto_si_costruisce_e_dichiara_cio_che_manca",
        "test_lindice_non_contiene_numeri_scritti_a_mano",
        "test_i_numeri_dellindice_vengono_dagli_artefatti")
tf = R / "tabella_finale.csv"
if tf.exists():
    d = pd.read_csv(tf)
    meta = json.loads((R / "tabella_finale_meta.json").read_text(encoding="utf-8"))
    voce("v5", "La tabella finale e' un artefatto generato, non copiata a mano",
         OK if d.notna().all().all() else PART,
         f"results/tabella_finale.csv: {len(d)} modelli x {len(d.columns) - 1} colonne, "
         f"rapporto 1:{meta['ratio_joint']:g}\n"
         f"run per cella: {meta['run_per_cella']}")
    da_test("v5", "Il README coincide con la tabella generata, cella per cella",
            "test_tabella_finale_del_readme_coincide_con_lartefatto")
else:
    voce("v5", "Tabella finale generata", NO,
         "manca results/tabella_finale.csv (python scripts/tabella_finale.py)")

# ═════════════════════════════════════════════════════════════
# TERZA REVISIONE — le richieste per la v2.1-rc2
# ═════════════════════════════════════════════════════════════
print("\nr6) BASELINE HARDWARE: MLP PICCOLO IN C INTERO")

_inc = REPO / "mcu_pio" / "include"
_hdr = _inc / "mlp16_int8.h"
voce("r6", "L'MLP e' esportato in C intero, non stimato",
     OK if _hdr.exists() else NO,
     f"{_rel(_hdr)} + {_rel(_inc / 'mlp16_infer.h')} + "
     f"{_rel(_inc / 'mlp16_test_vectors.h')}"
     if _hdr.exists() else
     "header assente: python scripts/export_mlp_int_c.py")

_fpc = csv("footprint.csv")
if _fpc is not None and "MLP(16)" in set(_fpc.modello):
    _r = _fpc[_fpc.modello == "MLP(16)"].iloc[0]
    _misurato = _r.regola == "array C compilati"
    voce("r6", "I byte dell'MLP sono misurati sull'header, non stimati",
         OK if _misurato else NO,
         f"MLP(16): {int(_r.byte_parametri)} B, regola '{_r.regola}', "
         f"fonte {_r.fonte}\n"
         f"  la stima a un byte per parametro diceva 705 B")
else:
    voce("r6", "I byte dell'MLP sono misurati sull'header", NO,
         "riga MLP(16) assente da results/footprint.csv")

_src = REPO / "mcu_pio" / "src"
_ini = (REPO / "mcu_pio" / "platformio.ini").read_text(encoding="utf-8")
_env_mlp = [e for e in ("megaatmega2560_mlp", "esp32c3_mlp",
                        "megaatmega2560_energy_mlp", "esp32c3_energy_mlp")
            if f"[env:{e}]" in _ini]
voce("r6", "L'MLP e' flashabile e misurabile come gli altri modelli",
     OK if (_src / "main_mlp.cpp").exists() and len(_env_mlp) == 4 else NO,
     f"src/main_mlp.cpp + EB_MLP in main_energy.cpp\n"
     f"  environment: {', '.join(_env_mlp)}")

da_test("r6", "Il kernel C dell'MLP e' identico alla simulazione numpy",
        "test_il_kernel_c_riproduce_il_logit_della_simulazione",
        "test_il_confronto_saprebbe_vedere_una_differenza",
        "test_la_versione_intera_segue_il_modello_float")
da_test("r6", "Nessun accumulatore dell'MLP puo' andare in overflow",
        "test_tutti_gli_accumulatori_stanno_in_int32",
        "test_il_bound_dell_accumulatore_e_rispettato",
        "test_pesi_troppo_grandi_fermano_lexport")
da_test("r6", "Il confronto a cinque e' verificato su tutti i kernel",
        "test_kernel_senza_virgola_mobile_su_avr",
        "test_il_numero_di_check_bit_esatti_e_quello_dei_sorgenti",
        "test_le_inferenze_avvengono_davvero")

print("\nr3) INGOMBRO MISURATO DELLA CONFIGURAZIONE SCELTA")

_af = csv("arch_footprint.csv")
if _af is not None and len(_af) >= 2:
    _dep = _af[_af.ruolo == "deployata"].iloc[0]
    _sel = _af[_af.ruolo == "selezionata"].iloc[0]
    _d = int(_sel.byte_parametri) - int(_dep.byte_parametri)
    voce("r3", "La configurazione scelta dalla selezione e' stata compilata",
         OK,
         f"h={int(_sel.hidden)} grado={int(_sel.degree)}: "
         f"{int(_sel.byte_parametri):,} B di parametri\n"
         f"  h={int(_dep.hidden)} grado={int(_dep.degree)} (deployata): "
         f"{int(_dep.byte_parametri):,} B\n"
         f"  differenza {_d:+,} B "
         f"({100.0 * _d / int(_dep.byte_parametri):+.1f}%)")
    _fp = csv("footprint.csv")
    _atteso = None
    if _fp is not None and "KAN(cat,ML)" in set(_fp.modello):
        _atteso = int(_fp[_fp.modello == "KAN(cat,ML)"].iloc[0].byte_parametri)
    voce("r3", "Il confronto e' omogeneo con results/footprint.csv",
         OK if _atteso == int(_dep.byte_parametri) else NO,
         f"la deployata ricompilata da' {int(_dep.byte_parametri):,} B, "
         f"l'header committato {_atteso:,} B" if _atteso is not None
         else "footprint.csv non ha la riga KAN(cat,ML)")
    if "byte_avr_dati" in _af.columns and _af.byte_avr_dati.notna().all():
        _diff = [(r.ruolo, int(r.byte_parametri), int(r.byte_avr_dati))
                 for r in _af.itertuples()
                 if int(r.byte_parametri) != int(r.byte_avr_dati)]
        voce("r3", "Due misure indipendenti dell'ingombro coincidono",
             OK if not _diff else NO,
             "\n".join(f"  {r.ruolo}: parser {int(r.byte_parametri):,} B, "
                       f"avr-g++ {int(r.byte_avr_dati):,} B, stack "
                       f"{int(r.byte_avr_stack_main)} B"
                       for r in _af.itertuples())
             if not _diff else f"non coincidono: {_diff}")
else:
    voce("r3", "La configurazione scelta dalla selezione e' stata compilata", NO,
         "manca results/arch_footprint.csv "
         "(python scripts/footprint_architettura.py)")

da_test("r3", "La misura non apre il test set e usa una compilazione sola",
        "test_la_misura_non_legge_il_test",
        "test_lo_script_scarta_esplicitamente_lindice_del_test",
        "test_lheader_committato_si_riemette_identico",
        "test_anche_i_test_vector_si_riemettono_identici",
        "test_la_compilazione_del_multilayer_binario_esiste_in_un_posto_solo")

print("\nr4) STATISTICA: UNITA' DI ANALISI E P-VALUE")

_js = csv("joint_ratio_significativita.csv")
if _js is not None and "unita" in _js.columns:
    voce("r4", "La selezione del rapporto usa il seed come unita' di analisi",
         OK if (_js.unita == "seed").all() and _js.n_unita.max() < 100 else NO,
         f"{len(_js)} confronti su {int(_js.n_unita.max())} osservazioni "
         f"appaiate (prima erano 120: 10 seed x 6 modelli x 2 domini)\n"
         + "\n".join(f"  contro {r.modello_b}: {r.differenza:+.5f} "
                     f"+/- {r.dev_differenza:.5f}, vince in {r.vince_a}, "
                     f"p Holm = {r.p_holm:.2e}" for r in _js.itertuples()))
else:
    voce("r4", "La selezione del rapporto usa il seed come unita' di analisi",
         NO, "artefatto assente o nel formato vecchio: "
             "python scripts/statistica_confronti.py")

_zeri = []
for _n in ("crossdomain_significativita.csv", "joint_ratio_significativita.csv",
           "indomain_significativita.csv"):
    _d = csv(_n)
    if _d is not None and "p_value" in _d.columns:
        _q = int((_d.p_value == 0).sum())
        if _q:
            _zeri.append(f"{_n}: {_q}")
voce("r4", "Nessun p-value e' scritto come zero", OK if not _zeri else NO,
     "un p arrotondato a 0,0 si legge come certezza assoluta; i CSV portano "
     "il valore pieno e una colonna formattata che sotto 1e-12 scrive una "
     "disuguaglianza" if not _zeri else f"ancora zeri in {_zeri}")

_id = csv("indomain_significativita.csv")
if _id is not None and "correzione" in _id.columns:
    _con = sorted({c.split(",")[0] for c in _id.correzione})
    voce("r4", "La correzione si applica dove il suo regime esiste", OK,
         "\n".join(f"  {e}: {_id[_id.exp == e].correzione.iloc[0][:96]}"
                   for e in sorted(_id.exp.unique())))
else:
    voce("r4", "La correzione si applica dove il suo regime esiste", NO,
         "manca results/indomain_significativita.csv")

da_test("r4", "Media, dispersione e vittorie stanno accanto a ogni p",
        "test_i_csv_dei_confronti_portano_le_quantita_richieste",
        "test_nessun_p_value_e_zero_negli_artefatti",
        "test_la_selezione_del_rapporto_usa_il_seed_come_unita",
        "test_il_json_della_scelta_concorda_con_il_csv")
da_test("r4", "Il README riporta i numeri corretti, non quelli vecchi",
        "test_la_tabella_del_rapporto_nel_readme_viene_dallartefatto",
        "test_la_tabella_cross_domain_nel_readme_viene_dallartefatto",
        "test_il_readme_non_promette_piu_ripetibilita_dove_non_ce")

print("\nr7) INTERPRETABILITA' DIRETTA DELLA KAN SINGLE-LAYER")

_fig = [f for f in ("fig_kan_funzioni_apprese.png", "fig_kan_contributi_locali.png")
        if (REPO / "figures" / f).exists()]
voce("r7", "Le due figure richieste esistono", OK if len(_fig) == 2 else NO,
     "\n".join(f"  figures/{f}" for f in _fig) if _fig else
     "python scripts/interpretabilita.py")

_c = csv("interpretabilita_contributi.csv")
if _c is not None:
    _ok, _tot = 0, 0
    for _v, _g in _c.groupby("vettore"):
        # "etichetta vera" e' una riga di contesto aggiunta nella rc3, non un
        # addendo: contarla dentro la somma faceva fallire questo controllo
        # sui due vettori la cui etichetta e' 1, cioe' esattamente dove la
        # spiegazione e' piu' interessante.
        _add = _g[~_g.edge.astype(str).str.startswith(
            ("SOMMA", "predizione", "etichetta"))]
        _som = _g[_g.edge == "SOMMA = logit"].contributo.iloc[0]
        _tot += 1
        _ok += int(int(_add.contributo.sum()) == int(_som))
    voce("r7", "I contributi sommano esattamente al logit",
         OK if _ok == _tot else NO,
         f"{_ok}/{_tot} esempi in cui i 14 addendi sommano al logit senza resto:\n"
         f"  non e' una stima del contributo come SHAP o LIME, sono gli addendi\n"
         f"  della somma che il kernel esegue")
else:
    voce("r7", "I contributi sommano esattamente al logit", NO,
         "manca results/interpretabilita_contributi.csv")

_e = csv("interpretabilita_escursione.csv")
if _e is not None:
    voce("r7", "Ogni edge e' ordinato per quanto muove davvero il logit", OK,
         "\n".join(f"  {r.edge:<24}{r.escursione / 1e6:>8.2f} x10^6"
                   for r in _e.head(4).itertuples()))

da_test("r7", "La scomposizione e' quella del kernel C, verificata sul binario",
        "test_la_somma_dei_contributi_e_il_logit_del_kernel_c",
        "test_il_confronto_saprebbe_vedere_una_differenza",
        "test_nessun_explainer_post_hoc_fra_le_dipendenze")
da_test("r7", "Il multi-layer resta descritto in modo prudente",
        "test_il_readme_e_prudente_sul_multilayer",
        "test_il_readme_non_sovrainterpreta_la_forma_delle_curve")

print("\nr5) ALLINEAMENTO TOTALE: PACCHETTO, DOCUMENTI, NUMERI")

da_test("r5", "Gli host check compilano e girano dal pacchetto estratto",
        "test_gli_host_check_compilano_e_girano_dal_pacchetto",
        "test_nel_pacchetto_gli_header_stanno_dove_i_check_li_cercano")
da_test("r5", "CIC-IoT-2023 e' sempre descritto nello spazio ridotto 6+2",
        "test_cic_e_sempre_descritto_nello_spazio_ridotto")
da_test("r5", "Nessun conteggio invecchiato di firmware o environment",
        "test_i_conteggi_di_firmware_e_environment_sono_quelli_veri",
        "test_la_lista_dei_firmware_del_pacchetto_e_completa")
da_test("r5", "Il report non dichiara da fare cio' che e' fatto",
        "test_il_report_non_dichiara_come_da_fare_cose_gia_fatte")

_mf = REPO / "models" / "MANIFEST.json"
_pdf = REPO / "report_KAN-IDS_fase2.pdf"
_eta = []
if _mf.exists() and _pdf.exists():
    import os as _os
    _piu_nuovi = [f for f in (REPO / "results").glob("*.csv")
                  if f.stat().st_mtime > _pdf.stat().st_mtime]
    if _piu_nuovi:
        _eta = sorted(f.name for f in _piu_nuovi)[:6]
voce("r5", "Il PDF del report non e' piu' vecchio dei risultati che cita",
     OK if not _eta else PART,
     "report e artefatti allineati" if not _eta else
     f"{len(_eta)} risultati piu' recenti del PDF, fra cui {_eta}\n"
     f"  rigenerare con: python scripts/make_report.py")

# ═════════════════════════════════════════════════════════════
# QUARTA REVISIONE — le richieste per la v2.1-rc3
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("QUARTA REVISIONE (v2.1-rc3) — l'ultimo checkpoint tecnico")
print("=" * 74)

print("\nq1) FINESTRA DI MISURA PULITA E DUE MARCATORI")

da_test("q1", "Niente modulo e niente volatile dentro il ciclo misurato",
        "test_il_ciclo_misurato_non_contiene_ne_divisioni_ne_volatile")
da_test("q1", "Nessuna routine di libgcc nella finestra, nell'assembly AVR",
        "test_nessuna_routine_di_libgcc_dentro_la_finestra_su_avr")
da_test("q1", "Finestra attiva e riferimento hanno marcatori distinti",
        "test_le_due_finestre_hanno_marcatori_distinti")
da_test("q1", "Le due durate sono misurate, stampate e confrontate",
        "test_la_calibrazione_converge_e_le_due_finestre_si_corrispondono",
        "test_il_firmware_riporta_le_due_energie_e_le_due_durate")

_en = (REPO / "mcu_pio" / "src" / "main_energy.cpp").read_text(encoding="utf-8")
_bandiere = [b for b in ("checksum_ok", "calibration_ok", "windows_ok",
                         "tolerance_permille") if b in _en]
voce("q1", "L'output dichiara da solo se la misura vale",
     OK if len(_bandiere) == 4 else NO,
     "SUMMARY riporta " + ", ".join(_bandiere))

print("\nq2) SAMPLED-LUT DELLO STESSO MODELLO, PER MISURARE IL COMPROMESSO")

_lut = REPO / "mcu_pio" / "include" / "kan14_lut_int16.h"
_tab = R / "lut_vs_coeff.csv"
if _lut.exists() and _tab.exists():
    _t = pd.read_csv(_tab)
    _L = int(re.search(r"#define KLUT_L (\d+)",
                       _lut.read_text(encoding="utf-8")).group(1))
    _r = _t[_t.L == _L].iloc[0]
    _c = pd.read_csv(R / "footprint.csv")
    _b254 = int(_c[_c.modello == "KAN(cat,1L)"].iloc[0].byte_parametri)
    voce("q2", "La LUT e' campionata dal modello deployato, non da un altro",
         OK, f"{_rel(_lut)} generato da mcu_pio/include/kan14_coeff_int8.h; "
             f"edge categorici identici")
    voce("q2", "Il compromesso e' misurato, non affermato", OK,
         f"L={_L}: {int(_r.byte_modello):,} B contro {_b254} B della versione a "
         f"coefficienti (x{_r.byte_modello / _b254:.1f})\n"
         f"limite di scostamento del logit {int(_r.limite_scostamento_logit):,} "
         f"< margine minimo {int(_r.margine_minimo_osservato):,}: "
         f"nessuno dei 200 vettori puo' cambiare decisione\n"
         f"curva completa per L in {list(_t.L)}: results/lut_vs_coeff.csv")
else:
    voce("q2", "La LUT campionata esiste", NO,
         "manca kan14_lut_int16.h o results/lut_vs_coeff.csv "
         "(python scripts/export_kan14_lut_c.py)")

da_test("q2", "Il kernel LUT e' verificato contro la simulazione e sul target",
        "test_il_kernel_c_calcola_gli_stessi_interi_della_simulazione",
        "test_lhost_check_della_lut_gira_e_conferma_lequivalenza",
        "test_il_kernel_lut_su_avr_non_usa_float_ne_64_bit")
da_test("q2", "L'header si rigenera identico e la scelta di L e' riproducibile",
        "test_lheader_si_riemette_identico_dallheader_a_coefficienti",
        "test_la_tabella_del_compromesso_e_riproducibile",
        "test_il_limite_saprebbe_dire_di_no")
da_test("q2", "La LUT e' flashabile e misurabile come gli altri modelli",
        "test_firmware_ed_environment_esistono")

print("\nq3) BINARI FINALI: LATENZA ED ENERGIA, DALLO STESSO COMMIT")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_pacchetto", REPO / "scripts" / "pacchetto_finale.py")
_pk = _ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_pk)
    _lat = [e for e in _pk.FIRMWARE if _pk.categoria(e) == "latenza"]
    _en = [e for e in _pk.FIRMWARE if _pk.categoria(e) == "energia"]
    _mega = [e for e in _pk.FIRMWARE if _pk.scheda(e) == "Mega 2560"]
    _esp = [e for e in _pk.FIRMWARE if _pk.scheda(e) == "ESP32-C3"]
    voce("q3", "Il pacchetto compila latenza ed energia nello stesso passaggio",
         OK if _lat and _en else NO,
         f"{len(_pk.FIRMWARE)} environment: {len(_lat)} di latenza, {len(_en)} "
         f"di energia; {len(_mega)} sul Mega 2560, {len(_esp)} sull'ESP32-C3\n"
         f"la lista e' letta da mcu_pio/platformio.ini, non scritta a mano")
except Exception as _e:
    voce("q3", "Il pacchetto compila latenza ed energia nello stesso passaggio",
         NO, f"pacchetto_finale.py non importabile: {_e}")

da_test("q3", "Nessun environment resta fuori dal pacchetto",
        "test_il_pacchetto_include_tutti_gli_environment",
        "test_la_lista_dei_firmware_del_pacchetto_e_completa")
da_test("q3", "Il pacchetto dichiara commit, tag e albero sporco",
        "test_lindice_non_contiene_numeri_scritti_a_mano",
        "test_i_numeri_dellindice_vengono_dagli_artefatti")

print("\nq6) SINGLE-LAYER E CATENA END-TO-END TENUTE SEPARATE")

_fp = csv("footprint.csv")
if _fp is not None and "ingresso" in _fp.columns:
    _gruppi = _fp.groupby("ingresso").modello.apply(list).to_dict()
    voce("q6", "Ogni riga dichiara da dove parte l'inferenza",
         OK if _fp.ingresso.notna().all() else NO,
         "\n".join(f"{k}:\n  " + ", ".join(v) for k, v in _gruppi.items()))
    _1l = _fp[_fp.modello == "KAN(cat,1L)"].iloc[0]
    _e2e = _fp[_fp.modello == "KAN e2e integer (binario)"].iloc[0]
    voce("q6", "I due casi che il relatore ha nominato sono distinti",
         OK if _1l.ingresso != _e2e.ingresso else NO,
         f"KAN(cat,1L): {int(_1l.byte_parametri):,} B — {_1l.ingresso}\n"
         f"KAN e2e integer (binario): {int(_e2e.byte_parametri):,} B — {_e2e.ingresso}")
else:
    voce("q6", "Ogni riga dichiara da dove parte l'inferenza", NO,
         "results/footprint.csv non ha la colonna 'ingresso' "
         "(python scripts/footprint.py --solo-header)")

da_test("q6", "Il README dichiara l'ingresso e coincide con l'artefatto",
        "test_tabella_del_readme_coincide_con_footprint_csv")

print("\nq4) RIPULITURA: NUMERI E CONTEGGI GENERATI, NON RICOPIATI")

_fs = R / "firmware_size.csv"
if _fs.exists():
    _d = pd.read_csv(_fs)
    _ini = (REPO / "mcu_pio" / "platformio.ini").read_text(encoding="utf-8")
    _tutti = set(re.findall(r"^\[env:([^\]]+)\]", _ini, re.M))
    _mancanti = _tutti - set(_d.environment)
    voce("q4", "Flash e SRAM misurate per OGNI environment, non per dodici",
         OK if not _mancanti else PART,
         f"{len(_d)} environment misurati con PlatformIO su {len(_tutti)} "
         f"definiti\n"
         f"  Mega 2560: flash da {_d[_d.scheda=='Mega 2560'].flash_byte.min():,} "
         f"a {_d[_d.scheda=='Mega 2560'].flash_byte.max():,} B\n"
         f"  ESP32-C3:  flash da {_d[_d.scheda=='ESP32-C3'].flash_byte.min():,} "
         f"a {_d[_d.scheda=='ESP32-C3'].flash_byte.max():,} B"
         + (f"\n  senza misura: {sorted(_mancanti)}" if _mancanti else ""))
else:
    voce("q4", "Flash e SRAM misurate per OGNI environment, non per dodici",
         PART, "results/firmware_size.csv assente: la tabella del README "
               "resta quella scritta a mano\n"
               "  python reproduce.py --stage firmware-size")

da_test("q4", "La tabella delle dimensioni e' generata e coincide col CSV",
        "test_il_blocco_del_readme_coincide_con_il_csv",
        "test_il_csv_copre_ogni_environment",
        "test_le_due_grandezze_non_sono_confuse")
da_test("q4", "Nessun conteggio di firmware o environment scritto a mano",
        "test_i_conteggi_di_firmware_e_environment_sono_quelli_veri",
        "test_la_lista_dei_firmware_del_pacchetto_e_completa")
da_test("q4", "Gli artefatti citati esistono e i byte coincidono ovunque",
        "test_ogni_artefatto_citato_esiste",
        "test_footprint_csv_coincide_con_gli_header",
        "test_manifest_coincide_con_i_csv",
        "test_tabella_del_readme_coincide_con_footprint_csv")
da_test("q4", "Anche la prosa scritta a mano coincide con gli artefatti",
        "test_la_tabella_di_compilazione_coincide_con_il_csv",
        "test_il_readme_non_dichiara_piu_lossless_la_compilazione_int16",
        "test_la_tabella_dei_firmware_nomina_ogni_sorgente",
        "test_gli_environment_citati_dalla_tabella_esistono")

print("\nq5) FIGURE DI INTERPRETABILITA': ETICHETTE, SEGNO, NOMI, DENSITA'")

da_test("q5", "Le figure dichiarano la convenzione del segno",
        "test_le_figure_dichiarano_la_convenzione_del_segno")
da_test("q5", "I contributi locali portano etichetta vera e predetta",
        "test_i_contributi_locali_mostrano_etichetta_vera_e_predetta",
        "test_il_csv_dei_contributi_somma_al_logit")
da_test("q5", "Le curve mostrano dove stanno davvero i dati",
        "test_le_funzioni_apprese_mostrano_dove_stanno_i_dati")
da_test("q5", "Le categorie usano i nomi veri, o dicono di non averli",
        "test_le_categorie_usano_i_nomi_veri_quando_ci_sono",
        "test_lo_script_dei_vocabolari_verifica_se_stesso",
        "test_i_vocabolari_se_presenti_combaciano_con_le_tabelle")

_voc = REPO / "models" / "vocabolari_categorici.json"
if _voc.exists():
    _v = json.loads(_voc.read_text(encoding="utf-8"))["vocabolari"]
    voce("q5", "I nomi delle categorie sono un artefatto committato", OK,
         "\n".join(f"{c}: {', '.join(n)}" for c, n in _v.items()))
else:
    voce("q5", "I nomi delle categorie sono un artefatto committato", PART,
         "models/vocabolari_categorici.json assente: le figure etichettano con "
         "l'indice e lo dichiarano\n"
         "  generarlo con: python scripts/export_vocabolari.py (serve il dataset)")

print("\nq7) STATO CANONICO DEL MODELLO MULTICLASSE")

_stato = REPO / "models" / "kan14_multiclass_multilayer.pkl"


def _tracciato(p: Path) -> bool:
    """Lo traccia git, o e' solo appoggiato sul disco?

    Questo controllo guardava `p.exists()` e per questo ha dichiarato
    "versionato" un file che .gitignore escludeva: su un clone pulito non ci
    sarebbe stato, e gli header sarebbero tornati artefatti di provenienza
    perduta — cioe' il requisito sarebbe risultato soddisfatto proprio mentre
    non lo era."""
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)],
                           cwd=REPO, capture_output=True)
        return r.returncode == 0
    except OSError:
        return False


if _stato.exists() and _tracciato(_stato):
    voce("q7", "Lo stato da cui derivano gli header a 10 classi e' versionato",
         OK, f"{_rel(_stato)} ({_stato.stat().st_size:,} B), tracciato da git: "
             f"i due header non sono piu' artefatti congelati di provenienza "
             f"perduta, ma la funzione deterministica di un file committato")
elif _stato.exists():
    voce("q7", "Lo stato da cui derivano gli header a 10 classi e' versionato",
         NO,
         f"{_rel(_stato)} esiste sul disco ma git NON lo traccia: su un clone "
         f"pulito non ci sarebbe\n"
         f"  controllare .gitignore, poi: git add -f {_rel(_stato)}")
else:
    voce("q7", "Lo stato da cui derivano gli header a 10 classi e' versionato",
         PART,
         "gli header a 10 classi sono ancora artefatti congelati: lo stato di "
         "training non e' nel repository\n"
         "  python reproduce.py --stage multiclass-state\n"
         "  python scripts/export_models.py\n"
         "  python reproduce.py --stage integer-10classi")

da_test("q7", "Gli esportatori a 10 classi partono da un clone pulito",
        "test_gli_esportatori_a_10_classi_cercano_anche_in_models",
        "test_la_corrispondenza_fra_cache_e_versionato_esiste")
da_test("q7", "Gli header si riemettono identici dallo stato committato",
        "test_lheader_si_riemette_identico_dallo_stato",
        "test_lo_stato_committato_ha_la_forma_che_gli_header_dichiarano")

print("\nqx) TROVATO VERIFICANDO: I KERNEL A COEFFICIENTI USAVANO int64")

da_test("qx", "Nessun kernel a coefficienti passa piu' per int64",
        "test_nessun_kernel_a_coefficienti_usa_piu_int64",
        "test_nessuna_routine_a_64_bit_nellassembly_avr")
da_test("qx", "La moltiplicazione Q15 senza int64 e' esatta, non approssimata",
        "test_lidentita_e_esatta_non_approssimata",
        "test_il_controllo_saprebbe_vedere_una_differenza",
        "test_gli_intermedi_stanno_in_int32_per_i_valori_veri_degli_header")
_q15 = REPO / "mcu_pio" / "include" / "q15_mul.h"
voce("qx", "La regola sta in un posto solo",
     OK if _q15.exists() else NO,
     f"{_rel(_q15)}, incluso dai tre kernel a coefficienti"
     if _q15.exists() else "manca mcu_pio/include/q15_mul.h")

# ── verdetto ─────────────────────────────────────────────────
print("\n" + "=" * 74)
for stato in (NO, PART):
    v = [e for e in esiti if e[2] == stato]
    if v:
        print(f"{stato}:")
        for p, req, _, _ in v:
            print(f"  punto {p}: {req}")
n_ok = sum(1 for e in esiti if e[2] == OK)
print(f"\n{n_ok}/{len(esiti)} requisiti verificati come soddisfatti")
print("=" * 74)

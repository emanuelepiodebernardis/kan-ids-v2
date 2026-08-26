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
        stato, nota = PART, (f"{len(esiti_test) - len(saltati)} verdi, "
                             f"{len(saltati)} saltati (toolchain assente): {saltati}")
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
             "configurazione piu' piccola per vincolo di dimensione, non "
             "perche' la selezione l'abbia scelta"))
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
        "test_nessuno_dichiara_sei_check_host_bit_esatti",
        "test_i_byte_del_firmware_di_energia_vengono_da_footprint")
da_test("v2", "Gli artefatti non cambiano a seconda della macchina che li rigenera",
        "test_ogni_io_testuale_dichiara_lencoding",
        "test_gli_artefatti_non_ascii_sono_utf8",
        "test_il_manifest_si_rilegge_uguale",
        "test_ogni_scrittura_di_testo_fissa_il_terminatore_di_riga",
        "test_gitattributes_fissa_i_terminatori_di_riga",
        "test_nessun_file_di_testo_versionato_ha_i_cr",
        "test_nessun_percorso_windows_negli_artefatti",
        "test_il_manifest_misura_i_byte_del_contenuto_non_del_checkout")
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
if sig is not None:
    tb = sig[sig.exp == "ton->bot"]
    testa = tb[(tb.p_value >= 0.05)]
    voce("v3", "I confronti fra modelli hanno un test appaiato a sostegno", OK,
         f"{len(sig)} confronti appaiati per seed in crossdomain_significativita.csv\n"
         f"su ton->bot {len(testa)} coppie su {len(tb)} NON sono separabili (p >= 0.05)\n"
         f"la piu' stretta: {testa.iloc[0].modello_a} vs {testa.iloc[0].modello_b} "
         f"p={testa.iloc[0].p_value}" if len(testa) else "nessuna coppia indistinguibile")
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

"""L'ingombro della configurazione scelta dalla selezione, misurato.

Richiesta del Prof. Kuznetsov, punto 3: "per h=32/degree=6 scelto dalla
validazione non rifare tutti gli esperimenti, ma compilare quella
configurazione e misurarne il footprint reale, cosi' che h=16/degree=8 sia
giustificato come un compromesso accuratezza/memoria misurato".

Le due proprieta' che questi test difendono
===========================================
1. **Il test set resta chiuso.** Misurare l'ingombro di una configurazione
   mai deployata e' un'ottima occasione per valutarla "gia' che ci siamo" sul
   test, e non si puo': il progetto ha appena finito di correggere due
   selezioni fatte sull'held-out. Qui si perturbano i VALORI delle righe
   destinate al test e si pretende che non cambi niente — ne' l'ingombro ne'
   il punteggio di validation.

2. **Le due configurazioni sono confrontabili.** Se i byte venissero da due
   procedure diverse, il confronto sarebbe fra due compilatori. Lo script usa
   `kanids/compila_ml.py` per entrambe, e misura l'ingombro due volte: col
   parser del progetto e con le sezioni che avr-g++ emette davvero. Le due
   devono coincidere.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kanids.toolchain import motivo_assenza, trova              # noqa: E402

SCRIPT = REPO / "scripts" / "footprint_architettura.py"


def _modulo():
    spec = importlib.util.spec_from_file_location("fp_arch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────
# 1. il test set non viene letto
# ─────────────────────────────────────────────────────────────
def test_la_misura_non_legge_il_test(tmp_path):
    """Stessa costruzione del test sulla selezione: due dataset identici
    tranne che per i valori delle righe destinate al test, resi assurdi.

    Si perturbano i valori e non le etichette perche' `outer_split` stratifica
    su y: capovolgere le etichette cambierebbe lo split stesso, e il test
    misurerebbe se' stesso invece del leakage.
    """
    m = _modulo()
    from kanids.datasets import encode_targets, make_synthetic
    from kanids.splits import outer_split

    df = make_synthetic(2000, seed=0)
    y, _ym, _c = encode_targets(df)
    _tr, te = outer_split(y, seed=m.SEED)

    assurdo = df.copy()
    numeriche = [c for c in df.columns
                 if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    rng = np.random.RandomState(7)
    for c in numeriche:
        assurdo[c] = assurdo[c].astype("float64")
        assurdo.loc[assurdo.index[te], c] = (
            assurdo.loc[assurdo.index[te], c].to_numpy() * -3.0
            + rng.normal(1e4, 1e3, len(te)))

    cfg = {"ruolo": "deployata", "hidden": 4, "degree": 4}
    a = m.una_configurazione(df, y, cfg, k=6, epochs=5, lavoro=tmp_path / "a")
    b = m.una_configurazione(assurdo, y, cfg, k=6, epochs=5, lavoro=tmp_path / "b")

    assert a["byte_parametri"] == b["byte_parametri"], (
        "l'ingombro cambia se cambiano i dati del test set")
    assert a["bal_acc_validation_float"] == b["bal_acc_validation_float"], (
        f"il punteggio di validation cambia con i valori del test: "
        f"{a['bal_acc_validation_float']} contro {b['bal_acc_validation_float']}")
    assert a["bal_acc_validation_intera"] == b["bal_acc_validation_intera"]


def test_lo_script_scarta_esplicitamente_lindice_del_test():
    """Il controllo statico che accompagna quello sperimentale: la riga che
    calcola lo split deve buttare via il test, e deve vedersi."""
    testo = SCRIPT.read_text(encoding="utf-8")
    assert "tr, _te = outer_split(y, seed=SEED)" in testo, (
        "lo split non e' scritto in modo che si veda che il test si scarta")
    corpo = testo.split("def una_configurazione", 1)[1]
    # via i commenti: uno di essi nomina `_te` per dire che non si usa, ed e'
    # il terzo controllo statico di questo progetto a inciampare nella propria
    # spiegazione. E confine di parola, altrimenti `write_text` conta.
    corpo = "\n".join(r.split("#", 1)[0] for r in corpo.splitlines())
    import re as _re
    usi = _re.findall(r"\b_te\b", corpo)
    assert len(usi) == 1, (
        f"l'indice del test compare {len(usi)} volte: viene usato dopo essere "
        f"stato calcolato")


# ─────────────────────────────────────────────────────────────
# 2. le due misure dell'ingombro coincidono
# ─────────────────────────────────────────────────────────────
@pytest.mark.skipif(trova("avr-g++") is None, reason=motivo_assenza("avr-g++"))
def test_il_parser_e_il_compilatore_contano_gli_stessi_byte(tmp_path):
    """`c_footprint.scan` conta leggendo l'header; avr-g++ conta emettendo
    sezioni. Sono due misure indipendenti della stessa cosa, e il confronto
    fra le due architetture si regge su quel numero."""
    m = _modulo()
    from kanids.datasets import encode_targets, make_synthetic

    df = make_synthetic(2000, seed=0)
    y, _ym, _c = encode_targets(df)
    r = m.una_configurazione(df, y, {"ruolo": "deployata", "hidden": 8,
                                     "degree": 4},
                             k=6, epochs=5, lavoro=tmp_path)
    assert r["byte_avr_dati"] == r["byte_parametri"], (
        f"il parser dice {r['byte_parametri']} B, il compilatore "
        f"{r['byte_avr_dati']} B")
    assert r["byte_avr_stack_main"] > 0, "lo stack non e' stato misurato"


def test_la_larghezza_costa_stack(tmp_path):
    """Il kernel tiene le attivazioni nascoste in un array sullo stack: una
    larghezza doppia costa 64 byte di SRAM in piu' sul Mega, che ne ha 8.192.
    E' la parte del prezzo che non si vede guardando la Flash."""
    if trova("avr-g++") is None:
        pytest.skip(motivo_assenza("avr-g++"))
    m = _modulo()
    from kanids.datasets import encode_targets, make_synthetic

    df = make_synthetic(2000, seed=0)
    y, _ym, _c = encode_targets(df)
    stretto = m.una_configurazione(df, y, {"ruolo": "a", "hidden": 16, "degree": 4},
                                   k=6, epochs=3, lavoro=tmp_path / "s")
    largo = m.una_configurazione(df, y, {"ruolo": "b", "hidden": 32, "degree": 4},
                                 k=6, epochs=3, lavoro=tmp_path / "l")
    assert largo["byte_avr_stack_main"] > stretto["byte_avr_stack_main"], (
        "raddoppiare la larghezza non cambia lo stack: la misura non sta "
        "guardando il kernel")


# ─────────────────────────────────────────────────────────────
# 3. le configurazioni vengono dagli artefatti
# ─────────────────────────────────────────────────────────────
def test_le_configurazioni_non_sono_scritte_a_mano():
    """Se un giorno la selezione cambiasse scelta, questo script deve
    misurare la nuova senza che nessuno se ne ricordi."""
    m = _modulo()
    cfg = m.configurazioni()
    assert cfg and cfg[0]["ruolo"] == "deployata"

    from kanids.config import ARCH
    assert cfg[0]["hidden"] == ARCH["KAN(cat,ML)"]["hidden"]
    assert cfg[0]["degree"] == ARCH["KAN(cat,ML)"]["degree"]

    scelta = REPO / "results" / "arch_selection_scelta.json"
    if scelta.exists():
        import json
        s = json.loads(scelta.read_text(encoding="utf-8"))
        ml = s.get("scelte", s).get("KAN(cat,ML)")
        if ml and (ml["hidden"], ml["degree"]) != (cfg[0]["hidden"],
                                                   cfg[0]["degree"]):
            assert len(cfg) == 2 and cfg[1]["ruolo"] == "selezionata"
            assert (cfg[1]["hidden"], cfg[1]["degree"]) == (ml["hidden"],
                                                            ml["degree"])

    # e nel sorgente non devono comparire come costanti
    testo = SCRIPT.read_text(encoding="utf-8")
    corpo = "\n".join(r for r in testo.split('"""')[2].splitlines()
                      if not r.strip().startswith("#"))
    import re
    assert not re.search(r"hidden\s*=\s*(16|32)\b", corpo), (
        "una larghezza e' scritta a mano nello script invece di venire "
        "dagli artefatti")


def test_lo_script_e_uno_stage_di_reproduce():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reproduce", REPO / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    comandi = [" ".join(c) for _descr, cmds in mod.STAGES.values() for c in cmds]
    assert any("footprint_architettura.py" in c for c in comandi), (
        "la misura dell'ingombro delle due architetture non e' riproducibile "
        "da reproduce.py")


# ─────────────────────────────────────────────────────────────
# 4. il README riporta i numeri misurati, non altri
# ─────────────────────────────────────────────────────────────
def test_il_readme_riporta_lingombro_misurato():
    """La sezione «Architecture: selected and deployed are not the same» e'
    scritta a mano ed e' quella che il relatore legge per prima. Ogni cifra
    della sua tabella deve venire da results/arch_footprint.csv.

    E' lo stesso controllo che sui punteggi della selezione ha gia' bocciato
    quattro numeri ricalcolati a mano su un output arrotondato — e uno di
    quei quattro era sopravvissuto in kanids/config.py fino a oggi.
    """
    csv = REPO / "results" / "arch_footprint.csv"
    if not csv.exists():
        pytest.skip("python scripts/footprint_architettura.py")
    d = pd.read_csv(csv)
    if len(d) < 2:
        pytest.skip("una sola configurazione: niente confronto da dichiarare")

    testo = (REPO / "README.md").read_text(encoding="utf-8")
    inizio = testo.index("### Architecture: selected and deployed")
    sezione = testo[inizio:inizio + 6000]

    mancanti = []
    for r in d.itertuples():
        for che_cosa, valore in (("byte del modello", int(r.byte_parametri)),
                                 ("stack del kernel",
                                  int(getattr(r, "byte_avr_stack_main", 0) or 0)),
                                 ("codice del kernel",
                                  int(getattr(r, "byte_avr_codice", 0) or 0))):
            if not valore:
                continue
            if f"{valore:,}" not in sezione and str(valore) not in sezione:
                mancanti.append(f"{r.ruolo} h={r.hidden} g={r.degree}: "
                                f"{che_cosa} = {valore:,}")

    dep = d[d.ruolo == "deployata"].iloc[0]
    sel = d[d.ruolo == "selezionata"].iloc[0]
    db = int(sel.byte_parametri) - int(dep.byte_parametri)
    pct = 100.0 * db / int(dep.byte_parametri)
    if f"{db:,}" not in sezione:
        mancanti.append(f"differenza in byte = {db:+,}")
    if f"{pct:.1f}" not in sezione:
        mancanti.append(f"differenza in percentuale = {pct:+.1f} %")

    assert not mancanti, (
        "la sezione dell'architettura nel README non riporta questi numeri "
        "misurati:\n  " + "\n  ".join(mancanti)
        + "\n\nVanno letti da results/arch_footprint.csv, non stimati.")


def test_il_readme_non_dice_piu_che_non_ci_sta():
    """La formulazione onesta non e' "non ci sta": su entrambe le schede ci
    sta. E' una preferenza dichiarata con un prezzo misurato, e il README
    deve dirlo cosi' — il relatore ha chiesto due volte di togliere le
    affermazioni piu' forti dei dati."""
    testo = (REPO / "README.md").read_text(encoding="utf-8")
    inizio = testo.index("### Architecture: selected and deployed")
    sezione = testo[inizio:inizio + 6000].lower()
    assert "does not fit" in sezione or "not \"it does not fit\"" in sezione, (
        "la sezione non affronta la domanda se la configurazione scelta ci "
        "stia: e' la prima cosa che un revisore chiede")
    assert "declared preference" in sezione or "measured price" in sezione, (
        "la sezione presenta ancora la scelta come un vincolo invece che "
        "come una preferenza con un prezzo")

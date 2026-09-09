"""Il protocollo di valutazione, imposto invece che dichiarato.

Il primo lavoro ha stabilito la regola con un test, non con una frase: si
fitta due volte sullo stesso source con target diversi e si pretende che
tutto l'appreso sia identico bit per bit, cosi' che introducendo la
violazione classica il test fallisca. Qui vale lo stesso per la scelta
degli iperparametri: `iters` e `ridge` non devono poter essere scelti
guardando le righe che finiscono nelle tabelle.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "scripts", _ROOT / "src"):
    sys.path.insert(0, str(p))

from kanids.valutazione import (  # noqa: E402
    MODO_FINALE, MODO_SELEZIONE, AccessoAlTestVietato, dividi_target,
    imposta_modo, modo_corrente)


@pytest.fixture(autouse=True)
def _modo_pulito():
    yield
    imposta_modo(MODO_FINALE)


def _target(n=20000, p_rara=0.0013, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.random(n) > p_rara).astype(int)


# ── la partizione ────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [42, 43, 51])
@pytest.mark.parametrize("p_rara", [0.0013, 0.24])
def test_partizioni_disgiunte_ed_esaustive(seed, p_rara):
    y = _target(p_rara=p_rara, seed=seed)
    sel = np.random.RandomState(seed).choice(len(y), 32, replace=False)
    sp = dividi_target(y, sel, seed)
    tutte = np.concatenate([sp.idx_adattamento, sp.idx_validation, sp._idx_test])
    assert len(np.unique(tutte)) == len(tutte), "le partizioni si sovrappongono"
    assert len(tutte) == len(y), "le partizioni non coprono tutto il target"


@pytest.mark.parametrize("seed", [42, 47, 51])
def test_stratificazione_tiene_la_classe_rara_su_entrambi_i_lati(seed):
    """Con BoT-IoT come target i normali sono lo 0,013 %: una divisione non
    stratificata puo' lasciarne zero da una parte, e la balanced accuracy
    diventa indefinita per campionamento invece che per metodo."""
    y = _target(p_rara=0.0013, seed=seed)
    sel = np.random.RandomState(seed).choice(len(y), 32, replace=False)
    sp = dividi_target(y, sel, seed)
    assert (y[sp.idx_validation] == 0).sum() > 0
    assert (y[sp._idx_test] == 0).sum() > 0


@pytest.mark.parametrize("seed", [42, 43])
def test_la_partizione_dipende_solo_dal_seed(seed):
    """Metodi diversi dentro lo stesso seed devono vedere lo stesso test,
    altrimenti i confronti appaiati per seed non sono appaiati."""
    y = _target(seed=0)
    sel = np.random.RandomState(seed).choice(len(y), 32, replace=False)
    a = dividi_target(y, sel, seed)
    b = dividi_target(y, sel, seed)
    assert np.array_equal(a._idx_test, b._idx_test)
    c = dividi_target(y, sel, seed + 1)
    assert not np.array_equal(a._idx_test, c._idx_test)


@pytest.mark.parametrize("seed", [42, 47])
def test_la_partizione_non_dipende_dalla_selezione(seed):
    """La proprieta' che rende confrontabili budget diversi: cambiando le
    righe selezionate, il test resta lo stesso insieme meno quelle righe --
    non un insieme ridisegnato."""
    y = _target(seed=1)
    rng = np.random.RandomState(seed)
    sel_a = rng.choice(len(y), 8, replace=False)
    sel_b = rng.choice(len(y), 512, replace=False)
    a, b = dividi_target(y, sel_a, seed), dividi_target(y, sel_b, seed)
    base = dividi_target(y, np.array([], int), seed)
    assert np.array_equal(a._idx_test, np.setdiff1d(base._idx_test, sel_a))
    assert np.array_equal(b._idx_test, np.setdiff1d(base._idx_test, sel_b))
    # e nessuna riga etichettata finisce nella valutazione
    assert not np.intersect1d(a._idx_test, sel_a).size
    assert not np.intersect1d(b.idx_validation, sel_b).size


def test_senza_etichette_non_valuta_sulle_righe_etichettate():
    """Il difetto trovato in drift_senza_etichette.py: i metodi
    supervisionati di confronto venivano valutati anche sulle righe le cui
    etichette avevano usato."""
    testo = (_ROOT / "scripts" / "drift_senza_etichette.py").read_text(encoding="utf-8")
    assert "dividi_target(y_tgt, idx_usati" in testo
    assert "balanced_accuracy_score(y_tgt[ev], pred[ev])" in testo


# ── il guardiano ─────────────────────────────────────────────────────
def test_il_test_set_e_inaccessibile_in_modo_selezione():
    y = _target()
    sel = np.arange(32)
    sp = dividi_target(y, sel, 42)
    imposta_modo(MODO_SELEZIONE)
    assert modo_corrente() == MODO_SELEZIONE
    with pytest.raises(AccessoAlTestVietato):
        sp.test
    sp.validation          # la validation resta leggibile
    imposta_modo(MODO_FINALE)
    assert len(sp.test) > 0


def test_il_repr_non_rivela_la_dimensione_del_test():
    sp = dividi_target(_target(), np.arange(32), 42)
    assert "nascosto" in repr(sp) and str(len(sp._idx_test)) not in repr(sp)


# ── regressione: il difetto non deve poter rientrare ─────────────────
# Tutti gli script che valutano su un complemento delle righe selezionate.
# Devono passare dalla partizione unica, cosi' che ogni numero del documento
# stia sullo stesso test set e le sezioni siano confrontabili fra loro.
SCRIPT_CON_SELEZIONE = [
    "drift_int_adapt.py", "sweep_iperparametri.py", "drift_adapt.py",
    "drift_baselines.py", "drift_sampling.py", "drift_trasferimenti.py",
    "spazio_ridotto.py", "tre_domini.py",
]


@pytest.mark.parametrize("nome", SCRIPT_CON_SELEZIONE)
def test_nessuna_maschera_complemento_unica(nome):
    """Il difetto originale, nella forma in cui era scritto: una maschera
    sola sul complemento delle righe selezionate, usata sia per calibrare
    sia per riportare. Se questo schema rientra, il test fallisce."""
    testo = (_ROOT / "scripts" / nome).read_text(encoding="utf-8")
    assert "mask = np.ones(len(y_tgt), bool)" not in testo, (
        f"{nome} e' tornato a usare un complemento unico come insieme di "
        "valutazione: validation e test devono restare disgiunti "
        "(kanids/valutazione.py)")


def test_lo_sweep_gira_in_modo_selezione():
    """Verifica statica: `imposta_modo(MODO_SELEZIONE)` deve comparire in
    main() dello sweep, prima di qualunque chiamata agli sweep."""
    testo = (_ROOT / "scripts" / "sweep_iperparametri.py").read_text(encoding="utf-8")
    albero = ast.parse(testo)
    main = next(n for n in albero.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    chiamate = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    nomi = [n.func.id for n in chiamate if isinstance(n.func, ast.Name)]
    assert "imposta_modo" in nomi, "lo sweep non entra in modo selezione"
    assert nomi.index("imposta_modo") < min(
        (nomi.index(s) for s in ("sweep_iters", "sweep_ridge") if s in nomi),
        default=len(nomi)), "il modo selezione va impostato prima degli sweep"


def test_lo_sweep_rifiuta_i_seed_di_riporto():
    """Lo sweep del ridge gira su seed di calibrazione. Se qualcuno gli passa
    i seed su cui si riportano i risultati, deve fermarsi."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "sweep_iperparametri.py"),
         "--seeds", "42,43", "--solo", "iters"],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=300)
    assert r.returncode != 0, "lo sweep ha accettato seed di riporto"
    assert "seed di riporto" in (r.stdout + r.stderr)


def test_header_c_solo_dalla_configurazione_canonica():
    """Una prova esplorativa non deve poter sovrascrivere `kan_int_adapt.h`,
    che e' un artefatto committato e verificato bit per bit su 200 golden
    vector. Prima bastava essere il primo seed di un rilancio qualsiasi."""
    testo = (_ROOT / "scripts" / "drift_int_adapt.py").read_text(encoding="utf-8")
    assert "canonica = (first" in testo
    for condizione in ("exp == EXP_CANONICO", "seed == SEEDS[0]",
                       "args.iters == 6000", "args.ratio == 50.0"):
        assert condizione in testo, (
            f"la guardia sulla scrittura dell'header ha perso «{condizione}»: "
            "un rilancio esplorativo puo' di nuovo sovrascrivere l'header C")


def test_header_c_scritto_con_terminatori_lf():
    """`Path.write_text` su Windows traduce \\n in \\r\\n: l'header rigenerato
    differirebbe da quello committato per i soli terminatori, contro
    .gitattributes (eol=lf). Difetto gia' corretto una volta nel primo
    lavoro, rientrato qui."""
    testo = (_ROOT / "scripts" / "drift_int_adapt.py").read_text(encoding="utf-8")
    assert 'newline="\\n"' in testo, (
        "il generatore dell'header C non forza i terminatori LF: su Windows "
        "produrra' CRLF e l'artefatto risultera' modificato senza esserlo")
    for h in ("mcu/kan_int_adapt.h", "mcu_pio/include/kan_int_adapt.h"):
        assert b"\r\n" not in (_ROOT / h).read_bytes(), f"{h} contiene CRLF"

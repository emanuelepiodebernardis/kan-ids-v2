"""La spiegazione della KAN single-layer e' il calcolo, non una sua stima.

Richiesta del Prof. Kuznetsov, punto 7: "sfruttare l'interpretabilita' della
KAN single-layer: una figura semplice con le funzioni apprese per feature e
2-3 esempi di contributi locali al logit — una spiegazione diretta, non
post-hoc. Mantenere una formulazione piu' prudente per il multi-layer."

Il test che conta
=================
"Diretta, non post-hoc" e' un'affermazione verificabile, e qui viene
verificata nel modo piu' duro disponibile: i quattordici addendi vengono
sommati e confrontati con il **logit che il kernel C compilato restituisce**,
sui 200 vettori reali dell'header. Non "quasi uguale": uguale. Se lo fossero
solo quasi, la scomposizione sarebbe un'approssimazione — cioe' esattamente
la cosa da cui il punto 7 vuole distinguersi.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
sys.path.insert(0, str(REPO))

from kanids.interpretabilita import (contributi, curva, escursione,  # noqa: E402
                                     leggi_modello, leggi_vettori, logit,
                                     tabella_categorica)
from kanids.toolchain import ambiente, motivo_assenza, trova         # noqa: E402

GPP = trova("g++")
MODELLO = INCLUDE / "kan14_coeff_int8.h"
VETTORI = INCLUDE / "kan14_test_vectors.h"


@pytest.fixture(scope="module")
def m():
    return leggi_modello(MODELLO)


@pytest.fixture(scope="module")
def v():
    return leggi_vettori(VETTORI)


# ─────────────────────────────────────────────────────────────
# 1. la somma degli addendi E' il logit del kernel compilato
# ─────────────────────────────────────────────────────────────
@pytest.mark.skipif(GPP is None, reason=motivo_assenza("g++"))
def test_la_somma_dei_contributi_e_il_logit_del_kernel_c(tmp_path, m, v):
    """Il confronto con il C, non con un'altra funzione Python: e' l'unico
    modo per dire che la spiegazione riguarda il modello che gira davvero."""
    righe = []
    for k in range(len(v["X"])):
        righe.append("  {{" + ", ".join(str(int(x)) for x in v["X"][k]) + "}, {"
                     + ", ".join(str(int(c)) for c in v["CAT"][k]) + "}},")
    (tmp_path / "drv.cpp").write_text(
        '#include <cstdio>\n#include <cstdint>\n'
        '#include "kan14_coeff_int8.h"\n#include "kan14_coeff_infer.h"\n'
        "struct caso { int16_t x[10]; uint8_t c[4]; };\n"
        f"static const caso CASI[{len(v['X'])}] = {{\n" + "\n".join(righe) + "\n};\n"
        "int main(){ for (int k = 0; k < (int)(sizeof(CASI)/sizeof(CASI[0])); k++)\n"
        "    printf(\"%ld\\n\", (long)kan14_coeff_logit(CASI[k].x, CASI[k].c));\n"
        "  return 0; }\n", encoding="utf-8", newline="\n")
    exe = tmp_path / "drv"
    r = subprocess.run([GPP, "-O2", "-std=c++11", "-I", str(INCLUDE),
                        str(tmp_path / "drv.cpp"), "-o", str(exe)],
                       capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-2000:]
    dal_c = np.array([int(x) for x in subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=120).stdout.split()],
        dtype=np.int64)

    num, ctg = contributi(m, v["X"], v["CAT"])
    somma = num.sum(axis=1) + ctg.sum(axis=1)
    diversi = np.flatnonzero(somma != dal_c)
    assert diversi.size == 0, (
        f"{diversi.size} logit su {len(dal_c)} non coincidono con la somma "
        f"degli addendi; primo: vettore {diversi[0]}, C={dal_c[diversi[0]]}, "
        f"somma={somma[diversi[0]]}")


@pytest.mark.skipif(GPP is None, reason=motivo_assenza("g++"))
def test_il_confronto_saprebbe_vedere_una_differenza(m, v):
    """Controllo del controllo: spostando un coefficiente di uno, la somma
    deve cambiare. Senza, il test precedente potrebbe confrontare due volte
    lo stesso numero."""
    rotto = {k: (x.copy() if hasattr(x, "copy") else x) for k, x in m.items()}
    rotto["COEF"][0][5] += 1
    a = logit(m, v["X"], v["CAT"])
    b = logit(rotto, v["X"], v["CAT"])
    assert not np.array_equal(a, b)


def test_la_scomposizione_riproduce_le_predizioni_attese(m, v):
    z = logit(m, v["X"], v["CAT"])
    assert ((z >= 0).astype(np.int64) == v["ATTESA"]).all()


def test_gli_addendi_sono_quattordici(m):
    assert m["NFEAT"] == 10 and m["NCAT"] == 4, (
        "la figura e il testo parlano di quattordici funzioni apprese")


# ─────────────────────────────────────────────────────────────
# 2. le funzioni apprese
# ─────────────────────────────────────────────────────────────
def test_ogni_edge_ha_una_curva_non_costante(m):
    """Un edge che non muove mai il logit sarebbe un parametro inutile in
    Flash: qui si verifica che non ce ne siano."""
    piatti = []
    for i in range(m["NFEAT"]):
        _x, y = curva(m, i)
        if int(y.max() - y.min()) == 0:
            piatti.append(i)
    assert not piatti, f"edge numerici costanti: {piatti}"
    for j in range(m["NCAT"]):
        assert len(tabella_categorica(m, j)) > 1


def test_lescursione_ordina_gli_edge_e_non_e_una_stima(m, v):
    e = escursione(m, v["X"], v["CAT"])
    assert len(e) == m["NFEAT"] + m["NCAT"]
    for r in e:
        assert r["max"] >= r["min"]
        assert r["escursione"] == r["max"] - r["min"]


# ─────────────────────────────────────────────────────────────
# 3. "non post-hoc" e' un'affermazione con conseguenze
# ─────────────────────────────────────────────────────────────
def test_nessun_explainer_post_hoc_fra_le_dipendenze():
    """Se il progetto dichiara una spiegazione diretta, non deve dipendere da
    uno strumento che ne stima una."""
    for nome in ("requirements.txt", "requirements-lock.txt"):
        testo = (REPO / nome).read_text(encoding="utf-8").lower()
        for pacchetto in ("shap", "lime", "captum", "eli5"):
            righe = [r for r in testo.splitlines()
                     if r.strip().startswith(pacchetto)]
            assert not righe, f"{nome} dipende da {pacchetto}: {righe}"


def test_il_readme_e_prudente_sul_multilayer():
    """Il punto 7 chiede esplicitamente una formulazione piu' prudente per il
    multi-layer: li' il secondo strato mescola le unita' nascoste e una
    scomposizione additiva esatta non esiste. Il README deve dirlo, non
    lasciarlo intuire."""
    testo = (REPO / "README.md").read_text(encoding="utf-8")
    assert "fig_kan_funzioni_apprese" in testo, (
        "il README non mostra la figura delle funzioni apprese")
    i = testo.index("fig_kan_funzioni_apprese")
    sezione = testo[max(0, i - 4000):i + 6000].lower()
    assert "multi-layer" in sezione, (
        "la sezione sull'interpretabilita' non dice niente del multi-layer")
    for frase in ("does not decompose", "no exact additive", "not additive",
                  "cannot be decomposed"):
        if frase in sezione:
            break
    else:
        pytest.fail("il README non dichiara che per il multi-layer la "
                    "scomposizione additiva esatta non esiste")


def test_il_readme_non_sovrainterpreta_la_forma_delle_curve():
    """Le funzioni apprese oscillano: grado 8 senza penalizzazione di
    smoothness. La scomposizione resta esatta, ma leggere le ondulazioni come
    una legge del dominio sarebbe la stessa sovra-interpretazione che il
    relatore ha gia' fatto togliere due volte."""
    testo = (REPO / "README.md").read_text(encoding="utf-8")
    i = testo.index("fig_kan_funzioni_apprese")
    sezione = testo[max(0, i - 4000):i + 6000].lower()
    assert "oscillat" in sezione or "wiggl" in sezione or "not monotone" in sezione, (
        "il README mostra le curve senza avvertire che oscillano e che la "
        "loro forma non va letta come una legge del dominio")


# ─────────────────────────────────────────────────────────────
# 4. gli artefatti prodotti
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("nome", ["fig_kan_funzioni_apprese.png",
                                  "fig_kan_contributi_locali.png"])
def test_le_figure_esistono(nome):
    f = REPO / "figures" / nome
    if not f.exists():
        pytest.skip("python scripts/interpretabilita.py")
    assert f.stat().st_size > 20_000, f"{nome} sembra vuota"


def test_il_csv_dei_contributi_somma_al_logit():
    import pandas as pd
    f = REPO / "results" / "interpretabilita_contributi.csv"
    if not f.exists():
        pytest.skip("python scripts/interpretabilita.py")
    d = pd.read_csv(f)
    for vet, g in d.groupby("vettore"):
        addendi = g[~g.edge.str.startswith(("SOMMA", "predizione", "etichetta"))]
        somma = g[g.edge == "SOMMA = logit"].contributo.iloc[0]
        assert int(addendi.contributo.sum()) == int(somma), (
            f"vettore {vet}: gli addendi del CSV non sommano al logit")
        assert len(addendi) == 14

        # predizione ed etichetta vera stanno entrambe nel CSV: la figura
        # mostra la decisione del modello, che puo' essere sbagliata, e chi
        # legge deve poterlo verificare senza aprire il PNG
        pred = int(g[g.edge == "predizione (1 = attacco)"].contributo.iloc[0])
        vera = g[g.edge == "etichetta vera (1 = attacco)"]
        assert len(vera) == 1, f"vettore {vet}: manca l'etichetta vera"
        assert pred == int(somma >= 0)


# ─────────────────────────────────────────────────────────────
# rc3 punto 5: cosa devono dire le figure
# ─────────────────────────────────────────────────────────────
def _script():
    return (REPO / "scripts" / "interpretabilita.py").read_text(encoding="utf-8")


def test_le_figure_dichiarano_la_convenzione_del_segno():
    """"Contributo +0,7" non dice verso quale classe spinge. Il relatore ha
    chiesto che la convenzione sia scritta sulla figura, non lasciata da
    dedurre."""
    s = _script()
    assert "SEGNO = " in s, "la convenzione del segno non e' definita"
    assert s.count("SEGNO") >= 3, (
        "la convenzione e' definita ma non finisce su entrambe le figure")
    for parola in ("ATTACCO", "NORMALE"):
        assert parola in s, f"la convenzione non nomina {parola}"


def test_i_contributi_locali_mostrano_etichetta_vera_e_predetta():
    s = _script()
    for atteso in ("etichetta vera", "predetta", "SBAGLIATA"):
        assert atteso in s, (
            f"la figura dei contributi non riporta {atteso!r}: una "
            f"spiegazione senza etichetta vera si legge come se il modello "
            f"avesse ragione per costruzione")


def test_le_funzioni_apprese_mostrano_dove_stanno_i_dati():
    """Densita' e rug dei 200 vettori sotto ogni curva: dove non ci sono
    osservazioni la spline e' estrapolazione, e la figura deve farlo vedere."""
    s = _script()
    assert "np.histogram" in s and '"|"' in s, (
        "mancano istogramma o rug sotto le curve")


def test_le_categorie_usano_i_nomi_veri_quando_ci_sono(tmp_path, monkeypatch):
    """Con il vocabolario esportato le barre portano il nome della categoria;
    senza, portano l'indice e la figura lo DICHIARA invece di far passare un
    numero per un nome."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "interp_script", REPO / "scripts" / "interpretabilita.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    senza = mod.etichette_categoria(None, 0, 4)
    assert senza == ["UNK", "1", "2", "3"], senza

    voc = {"proto": ["UNK", "tcp", "udp", "icmp"]}
    assert mod.etichette_categoria(voc, 0, 4) == ["UNK", "tcp", "udp", "icmp"]

    # un vocabolario di lunghezza sbagliata deve fermare la figura: le
    # etichette scivolerebbero di una posizione rispetto ai contributi
    with pytest.raises(AssertionError):
        mod.etichette_categoria({"proto": ["UNK", "tcp"]}, 0, 4)


def test_lo_script_dei_vocabolari_verifica_se_stesso():
    """Non basta esportare quattro liste: l'ordine deve essere quello che il
    preprocessore ha imparato, e un ordine sbagliato non produce errori — solo
    una figura in cui due protocolli sono scambiati. Lo script ricodifica il
    test set col vocabolario esportato e pretende gli stessi indici."""
    s = (REPO / "scripts" / "export_vocabolari.py").read_text(encoding="utf-8")
    assert "ricodificato != atteso" in s, (
        "l'export dei vocabolari non si verifica contro la codifica del "
        "preprocessore")
    assert "cardinalita_dell_header" in s, (
        "l'export non confronta le cardinalita' con l'header deployato")


def test_i_vocabolari_se_presenti_combaciano_con_le_tabelle():
    """Quando il file c'e' (macchina col dataset), la sua forma deve
    corrispondere alle tabelle categoriche dell'header deployato."""
    import json
    f = REPO / "models" / "vocabolari_categorici.json"
    if not f.exists():
        pytest.skip("models/vocabolari_categorici.json assente: "
                    "python scripts/export_vocabolari.py (serve il dataset)")
    voc = json.loads(f.read_text(encoding="utf-8"))["vocabolari"]
    from kanids.config import CATEGORICAL
    from kanids.interpretabilita import leggi_modello
    m = leggi_modello(REPO / "mcu_pio" / "include" / "kan14_coeff_int8.h")
    off = list(m["CAT_OFF"]) + [len(m["CAT"])]
    for j, c in enumerate(CATEGORICAL):
        assert c in voc, f"manca il vocabolario di {c}"
        assert len(voc[c]) == off[j + 1] - off[j], (
            f"{c}: {len(voc[c])} categorie contro {off[j+1] - off[j]} righe "
            f"nella tabella dell'header")
        assert voc[c][0] == "UNK", f"{c}: l'indice 0 non e' UNK"

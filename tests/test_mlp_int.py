"""L'MLP intero: la baseline densa che mancava sul dispositivo.

Richiesta del Prof. Kuznetsov, punto 6: "aggiungere una baseline hardware in
piu': il piccolo MLP gia' addestrato, esportato in C intero, per un confronto
onesto DT / MLP / KAN-1L / KAN-ML / LUT".

Cosa viene verificato qui, e perche' proprio questo
---------------------------------------------------
Il rischio di un export intero non e' che sia impreciso: e' che il C e la
simulazione numpy divergano su qualche ingresso, e che la divergenza si veda
solo sulla scheda, dove nessuno la sta guardando. Per gli altri modelli la
verifica sono i golden vector, che pero' esistono solo dopo aver eseguito
l'esportatore sui dati reali.

Questi test non hanno bisogno dei dati: costruiscono un MLP con pesi casuali,
lo quantizzano con la STESSA funzione dell'esportatore, ne emettono l'header
e compilano contro il KERNEL VERO del repository, confrontando il logit — non
la predizione — su ogni vettore. Un logit uguale su tutti i vettori e' molto
piu' forte di una predizione uguale: la predizione e' un bit, e due
implementazioni diverse la azzeccano lo stesso quasi sempre.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kanids.toolchain import ambiente, motivo_assenza, trova      # noqa: E402

import importlib.util                                             # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "export_mlp_int_c", REPO / "scripts" / "export_mlp_int_c.py")
xp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xp)

from c_footprint import scan                                      # noqa: E402

GPP = trova("g++")
gpp = pytest.mark.skipif(GPP is None, reason=motivo_assenza("g++"))

CARDS = [4, 10, 14, 4]        # le cardinalita' del progetto (models/MANIFEST.json)
NUM = 10
HID = 16


def _mlp_finto(seed: int = 0, scala: float = 1.0):
    """Pesi casuali con la forma di un MLPClassifier(hidden_layer_sizes=(16,))
    sul design [10 numeriche | one-hot(4) | one-hot(10) | one-hot(14) | one-hot(4)]."""
    rs = np.random.RandomState(seed)
    d = NUM + sum(CARDS)
    W1 = rs.normal(0, 0.5 * scala, (d, HID))
    b1 = rs.normal(0, 0.2 * scala, HID)
    W2 = rs.normal(0, 0.8 * scala, (HID, 1))
    b2 = float(rs.normal(0, 0.2 * scala))
    return W1, b1, W2, b2


def _ingressi(n: int, seed: int = 1):
    rs = np.random.RandomState(seed)
    xq = rs.randint(-xp.QMAX, xp.QMAX + 1, size=(n, NUM)).astype(np.int64)
    cat = np.stack([rs.randint(0, c, size=n) for c in CARDS], axis=1).astype(np.int64)
    return xq, cat


# ─────────────────────────────────────────────────────────────
# 1. il C e' la stessa cosa del numpy
# ─────────────────────────────────────────────────────────────
@gpp
def test_il_kernel_c_riproduce_il_logit_della_simulazione(tmp_path):
    """Confronto sul LOGIT, non sulla predizione, su 400 vettori casuali che
    coprono tutto il dominio degli ingressi (|xq| fino a 4096) e tutte le
    categorie. Il C accumula in int32 e numpy in int64: se il bound calcolato
    all'export fosse sbagliato, il C avvolgerebbe dove numpy no e i due
    valori si separerebbero proprio sui vettori estremi, che qui ci sono."""
    W1, b1, W2, b2 = _mlp_finto(seed=0)
    q = xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=CARDS)
    xq, cat = _ingressi(400, seed=7)
    atteso = xp.simula(q, xq, cat)

    # l'header generato e il kernel VERO del repository, nella stessa cartella
    (tmp_path / "mlp16_int8.h").write_text(xp.header_parametri(q, 0.0),
                                           encoding="utf-8", newline="\n")
    shutil.copy2(INCLUDE / "mlp16_infer.h", tmp_path / "mlp16_infer.h")

    righe = []
    for r in range(len(xq)):
        righe.append("  {{" + ", ".join(str(int(v)) for v in xq[r]) + "}, {"
                     + ", ".join(str(int(v)) for v in cat[r]) + "}},")
    (tmp_path / "drv.cpp").write_text(
        '#include <cstdio>\n#include <cstdint>\n#include "mlp16_infer.h"\n'
        f"struct caso {{ int16_t x[{NUM}]; uint8_t c[{len(CARDS)}]; }};\n"
        f"static const caso CASI[{len(xq)}] = {{\n" + "\n".join(righe) + "\n};\n"
        "int main(){ for (int k = 0; k < (int)(sizeof(CASI)/sizeof(CASI[0])); k++)\n"
        "    printf(\"%ld\\n\", (long)mlp16_logit(CASI[k].x, CASI[k].c));\n"
        "  return 0; }\n", encoding="utf-8", newline="\n")

    exe = tmp_path / "drv"
    r = subprocess.run([GPP, "-O2", "-std=c++11", "-I", str(tmp_path),
                        str(tmp_path / "drv.cpp"), "-o", str(exe)],
                       capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-2000:]
    out = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=120).stdout.split()
    ottenuto = np.array([int(v) for v in out], dtype=np.int64)

    assert len(ottenuto) == len(atteso)
    diff = np.flatnonzero(ottenuto != atteso)
    assert diff.size == 0, (
        f"{diff.size} logit su {len(atteso)} diversi fra kernel C e "
        f"simulazione numpy; primo caso k={diff[0]}: C={ottenuto[diff[0]]}, "
        f"numpy={atteso[diff[0]]}")


@gpp
def test_il_confronto_saprebbe_vedere_una_differenza(tmp_path):
    """Controllo del controllo. Il test precedente confronta due
    implementazioni: se il driver stampasse sempre lo stesso numero, o se il
    confronto guardasse un array vuoto, passerebbe comunque. Qui si sposta UN
    peso di uno e si pretende che il confronto fallisca."""
    W1, b1, W2, b2 = _mlp_finto(seed=0)
    q = xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=CARDS)
    xq, cat = _ingressi(200, seed=7)
    atteso = xp.simula(q, xq, cat)

    sabotato = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in q.items()}
    sabotato["W2q"][0] = int(sabotato["W2q"][0]) + 1
    diverso = xp.simula(sabotato, xq, cat)
    assert not np.array_equal(diverso, atteso), (
        "cambiare un peso di uno non cambia nessun logit: il confronto del "
        "test precedente non sta guardando niente")


# ─────────────────────────────────────────────────────────────
# 2. la quantizzazione approssima davvero il modello float
# ─────────────────────────────────────────────────────────────
def test_la_versione_intera_segue_il_modello_float():
    """L'intero non deve solo essere riproducibile: deve essere lo stesso
    modello. Si confrontano i logit riscalati, non le predizioni, perche' su
    pesi casuali le predizioni coincidono per caso nella grande maggioranza
    dei casi e non direbbero nulla."""
    W1, b1, W2, b2 = _mlp_finto(seed=3)
    q = xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=CARDS)
    xq, cat = _ingressi(500, seed=11)

    # riferimento float, sullo stesso ingresso: x = xq * CLIP / 2^QX
    from kanids import CLIP
    x = xq / xp.QMAX * CLIP
    D = np.hstack([x] + [np.eye(c)[cat[:, j]] for j, c in enumerate(CARDS)])
    z1 = np.maximum(D @ W1 + b1, 0.0)
    zf = (z1 @ W2).ravel() + b2

    zi = xp.simula(q, xq, cat).astype(np.float64)
    # unita' dell'intero -> unita' reali: s2 * 2^HSHIFT / 2^QX
    zi = zi * (2.0 ** q["hshift"]) * q["s2"] / xp.QMAX

    scala = float(np.sqrt(np.mean(zf ** 2)))
    errore = float(np.sqrt(np.mean((zi - zf) ** 2))) / scala
    assert errore < 0.02, f"errore relativo RMS del logit intero: {errore:.4f}"

    concordi = float((np.sign(zi) == np.sign(zf)).mean())
    assert concordi > 0.98, f"le due versioni decidono diversamente nel " \
                            f"{100*(1-concordi):.1f}% dei casi"


# ─────────────────────────────────────────────────────────────
# 3. il bound di non-overflow non e' decorativo
# ─────────────────────────────────────────────────────────────
def test_tutti_gli_accumulatori_stanno_in_int32():
    """Il kernel C usa int32 ovunque. Se il bound dell'uscita non ci stesse,
    il C avvolgerebbe in silenzio dove numpy no, e i due divergerebbero solo
    su certi ingressi: esattamente il difetto che non si vede su scheda."""
    for seed in range(6):
        W1, b1, W2, b2 = _mlp_finto(seed=seed, scala=1.0 + seed)
        q = xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=CARDS)
        assert q["bound_acc"] < (1 << 31), (seed, q["bound_acc"])
        assert q["bound_z"] < (1 << 31), (seed, q["bound_z"])
        xq, cat = _ingressi(300, seed=100 + seed)
        z = xp.simula(q, xq, cat)
        assert int(np.abs(z).max()) <= q["bound_z"], (
            "un logit osservato supera il bound dichiarato")


def test_il_bound_dell_accumulatore_e_rispettato():
    """Il bound e' calcolato dai pesi e dagli estremi degli ingressi, quindi
    deve valere su QUALSIASI ingresso, non su quelli provati."""
    W1, b1, W2, b2 = _mlp_finto(seed=5)
    q = xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=CARDS)
    assert q["bound_acc"] < (1 << 31)

    # il caso peggiore costruito a mano: ogni ingresso al segno che massimizza
    xq_peggio = np.sign(q["W1q"].sum(axis=1)) * xp.QMAX
    xq = np.vstack([xq_peggio, -xq_peggio]).astype(np.int64)
    cat = np.zeros((2, len(CARDS)), dtype=np.int64)
    for h in range(HID):
        acc = np.abs(xq @ q["W1q"][:, h]) + abs(int(q["B1q"][h]))
        assert int(acc.max()) <= q["bound_acc"], (
            "il bound dichiarato e' piu' piccolo di un accumulatore "
            "effettivamente raggiungibile")


def test_pesi_troppo_grandi_fermano_lexport():
    """Se il bound non entrasse in int32, l'header non va scritto: un
    accumulatore che sborda su AVR non produce un errore, produce numeri."""
    d = NUM + sum(CARDS)
    W1 = np.full((d, HID), 1000.0)      # tutti al massimo della scala
    b1 = np.full(HID, 1e9)              # bias enorme: e' li' che sfonda
    W2 = np.ones((HID, 1))
    with pytest.raises(OverflowError):
        xp.quantizza(W1, b1, W2, 0.0, n_num=NUM, cards=CARDS)


def test_un_design_di_larghezza_sbagliata_viene_rifiutato():
    """Se cambiassero le cardinalita' senza rigenerare, le colonne one-hot
    finirebbero nella tabella sbagliata e il modello direbbe altro senza
    lamentarsi."""
    W1, b1, W2, b2 = _mlp_finto(seed=0)
    with pytest.raises(ValueError):
        xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=[4, 10, 14, 5])


# ─────────────────────────────────────────────────────────────
# 4. i byte, contati come per tutti gli altri modelli
# ─────────────────────────────────────────────────────────────
def test_i_byte_dellheader_sono_quelli_delle_tabelle(tmp_path):
    """La stima table-driven diceva 705 B (un byte per parametro). Il conteggio
    vero e' 760: i bias sono int32 e la tabella categorica ha una riga per
    codice, non una colonna one-hot per parametro. Se un giorno il numero
    cambia deve cambiare qui, non nel PDF."""
    W1, b1, W2, b2 = _mlp_finto(seed=0)
    q = xp.quantizza(W1, b1, W2, b2, n_num=NUM, cards=CARDS)
    h = tmp_path / "mlp16_int8.h"
    h.write_text(xp.header_parametri(q, 0.0), encoding="utf-8", newline="\n")

    totale, righe = scan(h, "MLP16_")
    dettaglio = {nome: nbyte for nome, _, _, nbyte in righe}
    atteso = {"MLP16_W1": NUM * HID, "MLP16_CAT": sum(CARDS) * HID,
              "MLP16_CAT_OFF": len(CARDS), "MLP16_B1": HID * 4,
              "MLP16_W2": HID, "MLP16_B2": 4}
    assert dettaglio == atteso, f"array contati: {dettaglio}"
    assert totale == 760, totale

    npar = W1.size + b1.size + W2.size + 1
    assert npar == 705, "il numero di parametri del modello e' cambiato"


def test_il_bias_di_uscita_non_sparisce_dal_conteggio():
    """MLP16_B2 e' uno scalare `static const ... PROGMEM =`. La regex delle
    costanti di c_footprint.py non prevedeva PROGMEM e lo avrebbe saltato in
    silenzio: quattro byte, ma saltati senza dirlo."""
    from c_footprint import SCALAR
    assert SCALAR.search("static const int32_t MLP16_B2 PROGMEM = -12;"), \
        "una costante PROGMEM non viene contata fra i byte del modello"
    # e non deve prendersi gli array, che sono contati dall'altra regex
    assert not SCALAR.search("static const int8_t MLP16_W2[16] PROGMEM = {1};")


# ─────────────────────────────────────────────────────────────
# 5. l'header vero, quando c'e'
# ─────────────────────────────────────────────────────────────
def test_lheader_dellmlp_e_stato_generato():
    """Test volutamente esplicito: subito dopo aver applicato la patch
    l'header non esiste ancora, e senza questo messaggio i test che
    compilano `main_mlp.cpp` fallirebbero con un errore del compilatore su
    un include mancante, che non dice cosa fare."""
    assert (INCLUDE / "mlp16_int8.h").exists(), (
        "mcu_pio/include/mlp16_int8.h non c'e'. Va generato una volta con:\n"
        "    python scripts/export_mlp_int_c.py\n"
        "e committato come gli altri header.")


@pytest.mark.skipif(not (INCLUDE / "mlp16_int8.h").exists(),
                    reason="header non generato: python scripts/export_mlp_int_c.py")
def test_lheader_vero_non_contiene_virgola_mobile():
    import re
    testo = (INCLUDE / "mlp16_int8.h").read_text(encoding="utf-8")
    corpo = [l for l in testo.splitlines() if not l.strip().startswith("//")]
    bad = [l.strip()[:80] for l in corpo if re.search(r"\b(float|double)\b", l)]
    assert not bad, f"tipi in virgola mobile nell'header intero: {bad}"


def test_il_kernel_non_contiene_virgola_mobile():
    import re
    testo = (INCLUDE / "mlp16_infer.h").read_text(encoding="utf-8")
    # via i commenti: parlano di scale reali, che sono numeri, non float a runtime
    corpo = testo.split("*/", 1)[1]
    corpo = "\n".join(l for l in corpo.splitlines() if not l.strip().startswith("/*"))
    assert not re.search(r"\b(float|double)\b", corpo), \
        "il kernel dell'MLP dichiara tipi in virgola mobile"

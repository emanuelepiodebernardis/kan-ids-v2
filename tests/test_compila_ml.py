"""La compilazione della KAN multi-layer sta in un posto solo, e non e' cambiata.

Perche' questi test
===================
La procedura Chebyshev -> B-spline -> int8 stava dentro
`scripts/export_kan14_ml_coeff_c.py`, scritta di seguito, e serviva solo a
produrre l'header del modello deployato. La richiesta del Prof. Kuznetsov
(punto 3) chiede di compilare **un'altra** configurazione e misurarne
l'ingombro: farlo con una seconda copia del codice avrebbe trasformato il
confronto fra due architetture in un confronto fra due compilatori.

Spostare codice numerico e' pero' il modo classico di cambiarlo senza
accorgersene. Il primo test qui sotto rimette la questione su un piano
verificabile: legge l'header **committato**, ne estrae tutti i numeri, li da'
in pasto all'emettitore nuovo e pretende il file identico byte per byte. Se
una parentesi, uno spazio o un a capo fossero cambiati, o se un array venisse
emesso in un ordine diverso, il confronto fallirebbe.

Non copre la parte di quantizzazione, che richiede il dataset: quella si
verifica rilanciando l'esportatore e guardando `git diff` (istruzione nel
docstring dello script). Copre pero' tutto cio' che riguarda la FORMA delle
tabelle, che e' esattamente cio' che il punto 3 chiede di misurare.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kanids.compila_ml import N_INT, TL, header_parametri   # noqa: E402
from c_footprint import scan                                # noqa: E402

HEADER = INCLUDE / "kan14_ml_coeff_int8.h"


# ─────────────────────────────────────────────────────────────
# lettura dell'header committato
# ─────────────────────────────────────────────────────────────
def _graffe(testo: str):
    """Parentesi graffe annidate -> liste annidate di interi."""
    pila, corrente = [], None
    numero = ""
    for ch in testo:
        if ch == "{":
            nuova = []
            if corrente is not None:
                pila.append(corrente)
            corrente = nuova
        elif ch in ",}":
            if numero.strip():
                corrente.append(int(numero.strip()))
                numero = ""
            if ch == "}":
                chiusa = corrente
                if pila:
                    corrente = pila.pop()
                    corrente.append(chiusa)
                else:
                    return chiusa
        elif ch in "-0123456789":
            numero += ch
    raise ValueError("graffe non bilanciate")


def _array(testo: str, nome: str):
    inizio = testo.index(f" {nome}[")
    inizio = testo.index("= {", inizio) + 2
    livello, i = 0, inizio
    while True:
        if testo[i] == "{":
            livello += 1
        elif testo[i] == "}":
            livello -= 1
            if livello == 0:
                break
        i += 1
    return _graffe(testo[inizio:i + 1])


@pytest.fixture(scope="module")
def q_dallheader() -> dict:
    """Ricostruisce il dizionario di `compila()` leggendo l'header committato."""
    testo = HEADER.read_text(encoding="utf-8")
    hid = int(re.search(r"#define KML_HID (\d+)", testo).group(1))
    idx_mult = int(re.search(r"#define KML_IDX_MULT (-?\d+)L", testo).group(1))
    off = _array(testo, "KML_CAT_OFF")
    cat = np.array(_array(testo, "KML_CAT"), dtype=np.int64)
    cards = [off[i + 1] - off[i] for i in range(len(off) - 1)] + \
            [len(cat) - off[-1]]

    c1 = np.array(_array(testo, "KML_C1"), dtype=np.int64)      # (10, HID, 19)
    c2 = np.array(_array(testo, "KML_C2"), dtype=np.int64)      # (HID, 19)
    m1 = np.array(_array(testo, "KML_M1"), dtype=np.int64)      # (10, HID)
    m2 = np.array(_array(testo, "KML_M2"), dtype=np.int64)      # (HID,)
    tm = _array(testo, "KML_CAT_MULT")
    tanh_q15 = np.array(_array(testo, "KML_TANH"), dtype=np.int64)

    t8 = []
    for j, c in enumerate(cards):
        t8.append((cat[off[j]:off[j] + c, :], 0.0))

    return {"C1q": [c1[i].T for i in range(c1.shape[0])],
            "C2q": [c2[h][:, None] for h in range(c2.shape[0])],
            "m1": m1, "m2": m2[:, None], "t8": t8, "tm": tm,
            "tanh_q15": tanh_q15, "idx_mult": idx_mult,
            "K": c1.shape[0], "HID": hid, "DEG": 8, "J": len(cards),
            "cards": cards}


# ─────────────────────────────────────────────────────────────
# 1. l'emissione non e' cambiata
# ─────────────────────────────────────────────────────────────
def test_lheader_committato_si_riemette_identico(q_dallheader):
    """Il vincolo vero dello spostamento: stessi numeri dentro, stesso file
    fuori. Byte per byte, commento di testa compreso."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_ml", REPO / "scripts" / "export_kan14_ml_coeff_c.py")
    # il modulo importa il dataset legacy: qui serve solo la costante di testa,
    # che si legge dal sorgente senza eseguirlo
    sorgente = (REPO / "scripts" / "export_kan14_ml_coeff_c.py").read_text(
        encoding="utf-8")
    m = re.search(r'INTESTAZIONE = \((.*?)\)\n', sorgente, re.S)
    assert m, "INTESTAZIONE non trovata nell'esportatore"
    intestazione = eval("(" + m.group(1) + ")")            # solo stringhe letterali
    assert spec is not None

    atteso = HEADER.read_text(encoding="utf-8")
    ottenuto = header_parametri(q_dallheader, intestazione)
    assert ottenuto == atteso, (
        "l'header riemesso non coincide con quello committato: lo spostamento "
        "della compilazione in kanids/compila_ml.py ha cambiato l'output.\n"
        f"lunghezze: {len(ottenuto)} vs {len(atteso)}")


def test_il_confronto_saprebbe_vedere_una_differenza(q_dallheader):
    """Controllo del controllo: cambiando un coefficiente di uno, il file
    riemesso deve smettere di coincidere. Senza, un emettitore che leggesse
    l'header e lo ricopiasse passerebbe il test precedente."""
    rotto = dict(q_dallheader)
    rotto["C1q"] = [c.copy() for c in q_dallheader["C1q"]]
    rotto["C1q"][0][0, 0] += 1
    assert header_parametri(rotto, "/* x */") != \
        header_parametri(q_dallheader, "/* x */")


def test_anche_i_test_vector_si_riemettono_identici():
    """Stesso vincolo sull'altro header prodotto dall'esportatore. Sono i
    vettori contro cui gli host check verificano la bit-esattezza: un a capo
    di differenza non romperebbe niente, ma vorrebbe dire che lo spostamento
    ha toccato l'emissione, e allora non si saprebbe piu' cos'altro ha
    toccato."""
    from kanids.compila_ml import header_test_vectors
    f = INCLUDE / "kan14_ml_test_vectors.h"
    testo = f.read_text(encoding="utf-8")
    X = np.array(_numeri(_grezzo(testo, "KMLTV_X")), dtype=np.int64).reshape(-1, 10)
    C = np.array(_numeri(_grezzo(testo, "KMLTV_CAT")), dtype=np.int64).reshape(-1, 4)
    E = np.array(_numeri(_grezzo(testo, "KMLTV_EXPECTED")), dtype=np.int64)
    L = np.array(_numeri(_grezzo(testo, "KMLTV_LABEL")), dtype=np.int64)
    sel = np.arange(len(E))
    assert header_test_vectors(X, C, E, L, sel) == testo


def _grezzo(testo: str, nome: str) -> str:
    inizio = testo.index(f" {nome}[")
    inizio = testo.index("= {", inizio) + 2
    livello, i = 0, inizio
    while True:
        if testo[i] == "{":
            livello += 1
        elif testo[i] == "}":
            livello -= 1
            if livello == 0:
                break
        i += 1
    return testo[inizio:i + 1]


def _numeri(s: str):
    return [int(x) for x in re.findall(r"-?\d+", s)]


# ─────────────────────────────────────────────────────────────
# 2. l'ingombro dipende dalla larghezza, non dal grado
# ─────────────────────────────────────────────────────────────
def _q_finto(hid: int, cards=(4, 10, 14, 4), k: int = 10) -> dict:
    rs = np.random.RandomState(hid)
    return {"C1q": [rs.randint(-127, 128, (N_INT + 3, hid)) for _ in range(k)],
            "C2q": [rs.randint(-127, 128, (N_INT + 3, 1)) for _ in range(hid)],
            "m1": rs.randint(0, 32767, (k, hid)),
            "m2": rs.randint(0, 32767, (hid, 1)),
            "t8": [(rs.randint(-127, 128, (c, hid)), 0.0) for c in cards],
            "tm": list(rs.randint(0, 32767, len(cards))),
            "tanh_q15": rs.randint(-32767, 32768, TL),
            "idx_mult": 12345, "K": k, "HID": hid, "DEG": 8,
            "J": len(cards), "cards": list(cards)}


@pytest.mark.parametrize("hid, atteso", [(16, 5244), (32, 9452)])
def test_i_byte_di_una_configurazione_si_contano_dalle_forme(tmp_path, hid, atteso):
    """L'ingombro compilato e' una proprieta' delle FORME delle tabelle, e le
    forme dipendono dalla larghezza nascosta, dal numero di segmenti e dalle
    cardinalita'. Il grado di Chebyshev non compare: dopo la compilazione a
    B-spline ogni funzione ha NSEG+3 coefficienti, qualunque fosse il grado
    del polinomio da cui viene.

    Qui il numero e' contato dallo stesso `c_footprint.scan()` che produce
    results/footprint.csv, su un header emesso davvero. I 5.244 B della
    configurazione deployata devono uscire da questo conto: se non uscissero,
    il confronto del punto 3 non sarebbe fra grandezze omogenee.
    """
    h = tmp_path / f"kml{hid}.h"
    h.write_text(header_parametri(_q_finto(hid), "/* prova */"),
                 encoding="utf-8", newline="\n")
    totale, righe = scan(h, "KML_")
    assert totale == atteso, {n: b for n, _, _, b in righe}


def test_il_grado_non_cambia_lingombro():
    """Detto in modo diretto: due `q` con lo stesso HID e gradi diversi
    producono header della stessa dimensione in byte di parametri. E' la
    ragione per cui il punto 3 si puo' chiudere compilando, senza rifare gli
    esperimenti."""
    a, b = _q_finto(16), _q_finto(16)
    a["DEG"], b["DEG"] = 8, 6
    ta = header_parametri(a, "/* x */")
    tb = header_parametri(b, "/* x */")
    forme = re.compile(r"static const \w+ (KML_\w+)((?:\[[^\]]*\])*)")
    assert forme.findall(ta) == forme.findall(tb), (
        "il grado di Chebyshev cambia la forma delle tabelle emesse")


# ─────────────────────────────────────────────────────────────
# 3. la procedura non e' stata duplicata
# ─────────────────────────────────────────────────────────────
# Script che contengono la firma della compilazione senza importarla, e
# perche' sono un'altra cosa. Un elenco DICHIARATO, non un'eccezione muta:
# se ne comparisse un terzo, il test lo direbbe.
ALTRE_COMPILAZIONI = {
    "export_kan14_mc_coeff_c.py":
        "modello a 10 classi: C2 ha 10 uscite e la LUT e' indicizzata "
        "diversamente. E' inoltre un artefatto congelato, fuori da --stage all",
    "kan14_ml_compile.py":
        "studio sulla quantizzazione (int16 contro int8), non il generatore "
        "dell'header deployato: produce results/kan14_ml_compile_real.csv",
}


def test_la_compilazione_del_multilayer_binario_esiste_in_un_posto_solo():
    """La formula dei byte era stata riscritta a mano in tre script, e
    sbagliava due termini su tre. Qui si impedisce lo stesso all'algoritmo di
    quantizzazione: chi compila il multi-layer binario per il deployment deve
    importarlo, non riscriverlo.

    Le due eccezioni sono dichiarate sopra con la ragione. Il test confronta
    l'insieme trovato con quello dichiarato nei due sensi: un file nuovo che
    duplica la procedura fa fallire, e una voce dell'elenco che non serve piu'
    fa fallire ugualmente, invece di restare li' a coprire qualcosa.
    """
    firme = ("lstsq", "np.tanh(np.linspace(-8, 8")
    trovati = set()
    for f in sorted((REPO / "scripts").glob("*.py")):
        testo = f.read_text(encoding="utf-8", errors="replace")
        if "from kanids.compila_ml import" in testo:
            continue
        if all(s in testo for s in firme):
            trovati.add(f.name)
    assert trovati == set(ALTRE_COMPILAZIONI), (
        f"copie non dichiarate: {sorted(trovati - set(ALTRE_COMPILAZIONI))}; "
        f"dichiarate ma non piu' presenti: "
        f"{sorted(set(ALTRE_COMPILAZIONI) - trovati)}")


def test_lesportatore_usa_il_modulo_condiviso():
    testo = (REPO / "scripts" / "export_kan14_ml_coeff_c.py").read_text(
        encoding="utf-8")
    assert "from kanids.compila_ml import" in testo
    for nome in ("compila", "header_parametri", "header_test_vectors", "simula"):
        assert nome in testo, f"l'esportatore non usa {nome}()"


def test_il_grado_viene_letto_dal_modello_non_assunto():
    """L'esportatore chiamava `cheb_T(x)` col grado 8 di default. Su un modello
    di grado diverso — cioe' proprio quello che il punto 3 chiede di
    compilare — sarebbe morto dentro un einsum, con un messaggio che non
    c'entrava niente."""
    testo = (REPO / "kanids" / "compila_ml.py").read_text(encoding="utf-8")
    assert "DEG = C1.shape[2] - 1" in testo, (
        "il grado non viene ricavato dalla forma di C1")
    assert not re.search(r"cheb_T\([^,)]+\)\s*[^,]", testo.replace("def cheb_T", "")), \
        "c'e' ancora una chiamata a cheb_T senza grado esplicito"

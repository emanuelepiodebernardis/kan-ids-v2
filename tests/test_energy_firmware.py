"""Il firmware di benchmark energetico e i suoi due vincoli.

Richiesta del Prof. Kuznetsov (punto 4): "un benchmark energia corretto, a
batch di molte inferenze e senza Serial/I2C dentro la finestra misurata".

I due vincoli sono verificabili e qui vengono verificati:

1. **Niente I/O nella finestra.** Fra l'alzata e l'abbassata del pin di
   marcatura non deve comparire nessuna Serial, nessun Wire, nessun delay.
   E' un controllo sul sorgente e non sul binario perche' e' li' che
   l'errore si reintroduce: nei sette firmware di latenza la finestra e'
   pulita, ma fra una misura e l'altra ci sono da cinque a nove
   Serial.print, ed e' proprio su quegli intervalli che il vecchio
   integratore INA219 accumulava.

2. **Le inferenze avvengono davvero.** Tolta la Serial dalla finestra,
   l'unica cosa che consumava il risultato sparisce, e i kernel sono
   `static inline` con ingressi costanti: `-O2` potrebbe cancellare
   l'intero ciclo, e una finestra vuota sembrerebbe solo un modello molto
   efficiente. Il firmware confronta la somma delle predizioni con quella
   attesa dai golden vector; qui lo si compila e lo si ESEGUE sull'host per
   ogni variante, pretendendo checksum_ok=1.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MCU = REPO / "mcu_pio"
SORGENTE = MCU / "src" / "main_energy.cpp"

VARIANTI = ["EB_COEFF", "EB_MLCOEFF", "EB_MC", "EB_E2E", "EB_DT5", "EB_MLP"]
ARCHI = ["__AVR__", "ARDUINO_ARCH_ESP32"]

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kanids.toolchain import ambiente, motivo_assenza, trova      # noqa: E402
from artefatti import include_mancanti, motivo                    # noqa: E402


def _serve_lheader(variante: str) -> None:
    """La variante compila il firmware con l'header del suo modello: se e'
    un header generato non ancora prodotto, si salta dicendolo."""
    manca = include_mancanti(SORGENTE)
    if not manca:
        return
    # il ramo della catena `#if defined(EB_X) ... #elif defined(EB_Y)` che
    # sceglie gli include di QUESTA variante, non il guard dei default
    testo = SORGENTE.read_text(encoding="utf-8")
    catena = testo[testo.index("#if defined(EB_COEFF)"):]
    rami = re.split(r"#(?:if|elif) defined\(", catena)
    blocco = next((r for r in rami if r.startswith(variante + ")")), "")
    for h in manca:
        if h in blocco:
            pytest.skip(motivo(h))

# Questo test NON usa avr-g++: compila il firmware per l'HOST e lo ESEGUE,
# per confrontare il checksum delle predizioni con i golden vector. Serve un
# compilatore host (g++, clang++ o c++), e per mesi il messaggio di skip ha
# detto genericamente "toolchain assente", tanto che la nota del tag v2.1-rc
# attribuiva il salto ad avr-g++ e descriveva un test diverso.
GPP = trova("g++")
gpp = pytest.mark.skipif(GPP is None, reason=motivo_assenza("g++"))


def senza_commenti(sorgente: str) -> str:
    """Sorgente C senza commenti.

    Serve perche' i controlli statici di questo progetto hanno gia' sbagliato
    tre volte contando la propria spiegazione: un test che cercava
    `with_suffix` lo trovava nel commento che ne vietava l'uso, uno che
    cercava "selected and deployed" lo trovava nella frase che ci rimandava,
    e questo contava la chiamata a micros() nominata in un commento. Un
    commento che spiega un difetto non e' il difetto.
    """
    fuori, i, n = [], 0, len(sorgente)
    while i < n:
        if sorgente.startswith("/*", i):
            j = sorgente.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif sorgente.startswith("//", i):
            j = sorgente.find("\n", i)
            i = n if j < 0 else j
        else:
            fuori.append(sorgente[i])
            i += 1
    return "".join(fuori)


def test_la_finestra_misurata_non_contiene_io():
    """Fra `digitalWrite(EB_PIN, HIGH)` e `digitalWrite(EB_PIN, LOW)` non
    deve esserci I/O di alcun tipo."""
    testo = SORGENTE.read_text(encoding="utf-8")
    # solo il corpo, non l'intestazione di commento che parla di Serial
    corpo = testo.split("void setup()", 1)[1]
    apri = corpo.index("digitalWrite(EB_PIN, HIGH)")
    chiudi = corpo.index("digitalWrite(EB_PIN, LOW)", apri)
    finestra = corpo[apri:chiudi]

    vietati = ["Serial", "Wire.", "delay(", "delayMicroseconds(",
               "pgm_read", "memcpy_P", "digitalRead", "analogRead"]
    trovati = [v for v in vietati if v in finestra]
    assert not trovati, (
        f"dentro la finestra misurata compare {trovati}: la misura di "
        f"energia includerebbe UART, I2C o accessi a Flash invece della sola "
        f"inferenza.\nFinestra:\n{finestra}")

    assert "eb_one(" in finestra, "la finestra non contiene nessuna inferenza"
    assert "EB_BATCH" in finestra, "la finestra non e' un batch"


def test_la_finestra_di_riferimento_ha_la_stessa_durata():
    """La linea di base va misurata sulla stessa durata della finestra
    attiva, altrimenti la sottrazione delle due potenze non e' un'energia."""
    testo = SORGENTE.read_text(encoding="utf-8")
    assert re.search(r"eb_riferimento\(\s*durata\[rep\]\s*\)", testo), (
        "la finestra di riferimento non usa la durata misurata di quella attiva")

    # La calibrazione deve cronometrare LO STESSO ciclo che poi esegue, con
    # micros() fuori. La prima versione ne cronometrava uno diverso — con una
    # chiamata a micros() per giro — e su AVR il conto usciva 5.900 giri in
    # 20 ms invece di ~320.000: la divisione intera dava 0, il guard lo
    # portava a 1, e la finestra di riferimento durava un sedicesimo di
    # quella attiva. La formula (P_alta - P_bassa)*T/N presuppone che le due
    # durate coincidano, quindi il difetto falsava la misura, non la rifiniva.
    assert "eb_nop_loop" in testo, "non esiste un ciclo di riferimento unico"
    codice = senza_commenti(testo)
    corpo = codice[codice.index("static void eb_calibra"):]
    corpo = corpo[:corpo.index("static uint32_t eb_riferimento")]
    assert corpo.count("micros()") == 2, (
        f"eb_calibra chiama micros() {corpo.count('micros()')} volte: deve "
        f"chiamarlo due volte in tutto, fuori dal ciclo cronometrato")
    assert "eb_nop_loop(giri)" in corpo, (
        "la calibrazione non cronometra la stessa funzione della finestra")


def test_il_firmware_riporta_le_due_energie_e_le_due_durate():
    """Il relatore ha chiesto di distinguere energia totale per inferenza ed
    energia dinamica rispetto al baseline. La prima si ottiene dalla sola
    finestra attiva, la seconda dalla differenza: servono entrambe le durate
    misurate, non una misurata e una promessa."""
    testo = (MCU / "src" / "main_energy.cpp").read_text(encoding="utf-8")
    assert "E_totale per inferenza" in testo and "E_dinamica per inferenza" in testo, (
        "il firmware non documenta le due energie separatamente")
    assert "ref_us" in testo, "la durata della finestra di riferimento non e' stampata"
    assert "ref_vs_active_permille" in testo, (
        "il firmware non dichiara di quanto le due finestre si discostano: "
        "'stessa durata' resta una promessa non verificabile")
    assert "calibration_ok" in testo, (
        "il firmware non dice se la calibrazione e' riuscita")


def test_il_ciclo_di_misura_non_puo_essere_ottimizzato_via():
    testo = SORGENTE.read_text(encoding="utf-8")
    assert "volatile uint32_t eb_acc" in testo, (
        "l'accumulatore non e' volatile: il compilatore puo' eliminare le "
        "inferenze e la finestra risulterebbe vuota")
    assert "attesa_per_batch" in testo, "manca il confronto con la somma attesa"


@gpp
@pytest.mark.parametrize("variante", VARIANTI)
@pytest.mark.parametrize("arch", ARCHI)
def test_compila_per_entrambe_le_architetture(tmp_path, variante, arch):
    _serve_lheader(variante)
    r = subprocess.run(
        [GPP, "-fsyntax-only", "-std=c++11", "-Iinclude", "-Ihost_check",
         "-DHOST_CHECK", f"-D{arch}", f"-D{variante}",
         "-include", "host_check/arduino_stub.h", "src/main_energy.cpp"],
        cwd=MCU, capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-2000:]


@gpp
@pytest.mark.parametrize("variante", VARIANTI)
def test_le_inferenze_avvengono_davvero(tmp_path, variante):
    """Compila ed ESEGUE il firmware sull'host: il checksum delle predizioni
    deve coincidere con quello atteso dai golden vector."""
    _serve_lheader(variante)
    main = tmp_path / "main.cpp"
    main.write_text("int main() { setup(); return 0; }\n", encoding="utf-8", newline="\n")
    exe = tmp_path / "eb"
    r = subprocess.run(
        [GPP, "-O2", "-std=c++11", "-Iinclude", "-Ihost_check",
         "-DHOST_CHECK", "-DARDUINO_ARCH_ESP32", f"-D{variante}",
         "-DEB_BATCH=500", "-DEB_REPS=3",
         "-include", "host_check/arduino_stub.h",
         "-include", "src/main_energy.cpp", str(main), "-o", str(exe)],
        cwd=MCU, capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-2000:]

    out = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=120).stdout
    assert "checksum_ok=1" in out, (
        f"{variante}: il checksum non torna, le inferenze potrebbero essere "
        f"state eliminate dal compilatore.\n{out}")
    righe = [r for r in out.splitlines()
             if r.startswith(("coeff", "ml_", "mc_", "e2e", "dt5", "mlp"))]
    assert len(righe) == 3, f"attese 3 righe di misura, trovate {len(righe)}:\n{out}"
    for riga in righe:
        campi = riga.split(",")
        assert campi[-1] == "1", f"ripetizione con checksum errato: {riga}"
        assert int(campi[2]) == 500, f"batch sbagliato: {riga}"


def test_platformio_definisce_gli_environment_di_energia():
    ini = (MCU / "platformio.ini").read_text(encoding="utf-8")
    attesi = ["megaatmega2560_energy", "esp32c3_energy",
              "megaatmega2560_energy_dt5", "esp32c3_energy_mc",
              "megaatmega2560_energy_mlp", "esp32c3_energy_mlp"]
    for env in attesi:
        assert f"[env:{env}]" in ini, f"manca l'environment {env}"
    # Il pin di marcatura non deve essere quello del LED di bordo: il LED
    # assorbe corrente e finirebbe dentro la misura. Si guarda il codice, non
    # l'intestazione di commento, che LED_BUILTIN lo nomina per spiegare
    # perche' non si usa.
    codice = SORGENTE.read_text(encoding="utf-8").split("*/", 1)[1]
    assert "LED_BUILTIN" not in codice, (
        "il pin di marcatura e' quello del LED: il LED assorbe corrente e "
        "finirebbe dentro la misura")

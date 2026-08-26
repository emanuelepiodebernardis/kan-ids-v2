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

VARIANTI = ["EB_COEFF", "EB_MLCOEFF", "EB_MC", "EB_E2E", "EB_DT5"]
ARCHI = ["__AVR__", "ARDUINO_ARCH_ESP32"]

sys.path.insert(0, str(REPO))
from kanids.toolchain import ambiente, motivo_assenza, trova      # noqa: E402

# Questo test NON usa avr-g++: compila il firmware per l'HOST e lo ESEGUE,
# per confrontare il checksum delle predizioni con i golden vector. Serve un
# compilatore host (g++, clang++ o c++), e per mesi il messaggio di skip ha
# detto genericamente "toolchain assente", tanto che la nota del tag v2.1-rc
# attribuiva il salto ad avr-g++ e descriveva un test diverso.
GPP = trova("g++")
gpp = pytest.mark.skipif(GPP is None, reason=motivo_assenza("g++"))


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
    righe = [r for r in out.splitlines() if r.startswith(("coeff", "ml_", "mc_",
                                                          "e2e", "dt5"))]
    assert len(righe) == 3, f"attese 3 righe di misura, trovate {len(righe)}:\n{out}"
    for riga in righe:
        campi = riga.split(",")
        assert campi[-1] == "1", f"ripetizione con checksum errato: {riga}"
        assert int(campi[2]) == 500, f"batch sbagliato: {riga}"


def test_platformio_definisce_gli_environment_di_energia():
    ini = (MCU / "platformio.ini").read_text(encoding="utf-8")
    attesi = ["megaatmega2560_energy", "esp32c3_energy",
              "megaatmega2560_energy_dt5", "esp32c3_energy_mc"]
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

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

VARIANTI = ["EB_COEFF", "EB_MLCOEFF", "EB_MC", "EB_E2E", "EB_DT5", "EB_MLP",
            "EB_LUT14"]
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


def corpo_funzione(testo: str, firma: str) -> str:
    """Il corpo di una funzione C, dalla graffa aperta alla sua chiusa.

    Serve perche' il ciclo misurato non sta piu' in linea dentro setup(): e'
    una funzione, e un controllo che guardasse solo fra i due digitalWrite
    vedrebbe una chiamata e dichiarerebbe pulito tutto quello che c'e'
    dentro."""
    i = testo.index(firma)
    apri = testo.index("{", i)
    livello, j = 0, apri
    while j < len(testo):
        if testo[j] == "{":
            livello += 1
        elif testo[j] == "}":
            livello -= 1
            if livello == 0:
                return testo[apri:j + 1]
        j += 1
    raise AssertionError(f"corpo non terminato per {firma}")


def test_la_finestra_misurata_non_contiene_io():
    """Fra `digitalWrite(EB_PIN, HIGH)` e `digitalWrite(EB_PIN, LOW)` non
    deve esserci I/O di alcun tipo — ne' in linea ne' dentro la funzione che
    la finestra chiama."""
    testo = SORGENTE.read_text(encoding="utf-8")
    # solo il corpo, non l'intestazione di commento che parla di Serial
    corpo = testo.split("void setup()", 1)[1]
    apri = corpo.index("digitalWrite(EB_PIN, HIGH)")
    chiudi = corpo.index("digitalWrite(EB_PIN, LOW)", apri)
    finestra = corpo[apri:chiudi]

    codice = senza_commenti(testo)
    misurato = finestra + corpo_funzione(codice, "eb_finestra_attiva(uint32_t")

    vietati = ["Serial", "Wire.", "delay(", "delayMicroseconds(",
               "pgm_read", "memcpy_P", "digitalRead", "analogRead"]
    trovati = [v for v in vietati if v in misurato]
    assert not trovati, (
        f"dentro la finestra misurata compare {trovati}: la misura di "
        f"energia includerebbe UART, I2C o accessi a Flash invece della sola "
        f"inferenza.\nFinestra:\n{misurato}")

    assert "eb_finestra_attiva(" in finestra, (
        "la finestra non contiene nessuna inferenza")
    assert "EB_BATCH" in finestra, "la finestra non e' un batch"


def test_il_ciclo_misurato_non_contiene_ne_divisioni_ne_volatile():
    """I due costi estranei che il relatore ha chiesto di togliere.

    Il ciclo scriveva `eb_acc += eb_one(k % EB_CACHE)` con eb_acc volatile:
    per ogni inferenza una divisione a 32 bit (EB_CACHE non e' una potenza di
    due, quindi su AVR e' una chiamata a __udivmodsi4) e una lettura-somma-
    riscrittura di quattro byte in RAM. Nessuno dei due appartiene al modello
    che si vuole misurare. Il controllo statico e' qui, quello sull'assembly
    davvero emesso per ATmega2560 e' piu' sotto."""
    codice = senza_commenti(SORGENTE.read_text(encoding="utf-8"))
    ciclo = corpo_funzione(codice, "eb_finestra_attiva(uint32_t")

    assert "%" not in ciclo, (
        f"il ciclo misurato contiene un'operazione di modulo:\n{ciclo}")
    assert ciclo.count("eb_acc") == 1, (
        "l'accumulatore volatile va scritto una volta sola, alla fine del "
        f"batch, non a ogni inferenza:\n{ciclo}")
    assert "eb_acc = acc" in ciclo, (
        "il risultato del batch non finisce nel volatile: senza, il "
        "compilatore puo' eliminare le inferenze")


def test_le_due_finestre_hanno_marcatori_distinti():
    """Con un pin solo, il livello basso significava sia "finestra di
    riferimento" sia "tutto il resto": stampe, calibrazione, intervalli fra
    le ripetizioni. Chi integra la corrente doveva fidarsi dell'ordine
    invece di leggerlo dalla traccia."""
    testo = SORGENTE.read_text(encoding="utf-8")
    codice = senza_commenti(testo)
    assert "EB_PIN_REF" in codice, "non esiste un marcatore per il riferimento"

    rif = corpo_funzione(codice, "eb_riferimento(uint32_t")
    assert "digitalWrite(EB_PIN_REF, HIGH)" in rif and \
           "digitalWrite(EB_PIN_REF, LOW)" in rif, (
        "la finestra di riferimento non alza il proprio marcatore")
    assert "digitalWrite(EB_PIN," not in rif, (
        "la finestra di riferimento tocca il marcatore della finestra attiva")

    for pin in ("EB_PIN", "EB_PIN_REF"):
        assert f"pinMode({pin}, OUTPUT)" in codice, (
            f"{pin} non viene configurato come uscita")
    assert "marker_pin_ref" in testo, (
        "l'intestazione dell'output non dice su quale pin esce la finestra "
        "di riferimento")


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
    intestazione = next(r for r in out.splitlines() if r.startswith("variant,"))
    colonne = intestazione.split(",")
    for attesa in ("window_us", "ref_us", "ref_vs_active_permille",
                   "windows_match"):
        assert attesa in colonne, f"manca la colonna {attesa}: {intestazione}"

    righe = [r for r in out.splitlines()
             if r.startswith(("coeff", "ml_", "mc_", "e2e", "dt5", "mlp", "lut"))]
    assert len(righe) == 3, f"attese 3 righe di misura, trovate {len(righe)}:\n{out}"
    for riga in righe:
        campi = riga.split(",")
        assert len(campi) == len(colonne), f"riga e intestazione non combaciano: {riga}"
        assert campi[-1] == "1", f"ripetizione con checksum errato: {riga}"
        assert int(campi[2]) == 500, f"batch sbagliato: {riga}"


@gpp
def test_la_calibrazione_converge_e_le_due_finestre_si_corrispondono(tmp_path):
    """La finestra di riferimento deve durare quanto quella attiva, e questo
    va verificato eseguendo, non leggendo il sorgente.

    Due cose che erano rotte e che qui fallirebbero. (1) La calibrazione si
    fermava dopo dodici raddoppi: su un core veloce non arrivava ai 50 ms
    richiesti, ripiegava su un valore arbitrario e dichiarava
    calibration_ok=0 — cioe' non calibrava proprio dove la suite gira. (2) Il
    difetto originale, la finestra di riferimento lunga un sedicesimo di
    quella attiva, qui darebbe -937 parti per mille.

    La soglia e' larga (200 permille) perche' l'host non e' un
    microcontrollore: e' un sistema multitasking, e su finestre di pochi
    millisecondi il rumore di scheduling e' reale. Serve a intercettare un
    difetto di un ordine di grandezza, non a certificare la scheda: quella
    verifica la fa il firmware a bordo, riga per riga, con
    tolerance_permille=50.
    """
    main = tmp_path / "main.cpp"
    main.write_text("int main() { setup(); return 0; }\n", encoding="utf-8",
                    newline="\n")
    exe = tmp_path / "eb"
    r = subprocess.run(
        [GPP, "-O2", "-std=c++11", "-Iinclude", "-Ihost_check",
         "-DHOST_CHECK", "-DARDUINO_ARCH_ESP32", "-DEB_COEFF",
         "-DEB_BATCH=200000", "-DEB_REPS=3",
         "-include", "host_check/arduino_stub.h",
         "-include", "src/main_energy.cpp", str(main), "-o", str(exe)],
        cwd=MCU, capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-2000:]

    # Cinque esecuzioni, e conta la MIGLIORE. L'host e' un sistema
    # multitasking: una finestra da qualche millisecondo puo' prendersi lo
    # scheduler in faccia (visto: 374 parti per mille su una macchina carica,
    # 26 su quella scarica un secondo dopo), e il rumore puo' solo allungare
    # una delle due finestre, mai accorciarla. L'esecuzione meno disturbata e'
    # quindi la stima giusta di cio' che farebbe un microcontrollore, dove
    # nessuno interrompe. Il difetto storico — finestra di riferimento lunga
    # un sedicesimo, cioe' -937 — non e' rumore: comparirebbe in tutte e
    # cinque, e anche la migliore lo vedrebbe.
    scarti, sommari = [], []
    for _ in range(5):
        out = subprocess.run([str(exe)], capture_output=True, text=True,
                             timeout=300).stdout
        sommario = next(r for r in out.splitlines() if r.startswith("SUMMARY"))
        campi = dict(p.split("=", 1) for p in sommario.split() if "=" in p)
        assert campi["calibration_ok"] == "1", (
            f"la calibrazione non converge: la finestra di riferimento sarebbe "
            f"arbitraria\n{sommario}")
        assert campi["checksum_ok"] == "1", sommario
        assert int(campi["mean_ref_us"]) > 0, sommario
        scarti.append(abs(int(campi["ref_vs_active_permille"])))
        sommari.append(sommario)

    migliore = min(scarti)
    assert migliore <= 200, (
        f"nemmeno l'esecuzione meno disturbata avvicina le due finestre: "
        f"{migliore} parti per mille (su {scarti}). La differenza fra le due "
        f"potenze non sarebbe un'energia\n" + "\n".join(sommari))


AVR = trova("avr-g++")
avr = pytest.mark.skipif(AVR is None, reason=motivo_assenza("avr-g++"))


def assembly_avr(tmp_path: Path, variante: str) -> str:
    """Il firmware compilato PER ATmega2560, non per l'host. E' l'unico posto
    dove si vede che cosa esegue il processore: sull'host un modulo fra
    interi e' un'istruzione, su AVR e' una chiamata a libgcc."""
    asm = tmp_path / f"{variante}.s"
    r = subprocess.run(
        [AVR, "-mmcu=atmega2560", "-Os", "-std=c++11", "-S",
         "-Iinclude", "-Ihost_check", "-DHOST_CHECK", f"-D{variante}",
         "-include", "host_check/arduino_stub.h",
         "src/main_energy.cpp", "-o", str(asm)],
        cwd=MCU, capture_output=True, text=True, env=ambiente("avr-g++"))
    assert r.returncode == 0, f"compilazione AVR fallita:\n{r.stderr[-1500:]}"
    return asm.read_text(encoding="utf-8", errors="replace")


def corpo_assembly(asm: str, funzione: str) -> str:
    """Le istruzioni di UNA funzione, fra la sua etichetta e la direttiva
    .size che ne dichiara la fine. Il nome e' decorato (`_ZL18...`) e puo'
    portare un suffisso di ottimizzazione (`.constprop.14`)."""
    m = re.search(rf"^(\S*{funzione}\S*):\n(.*?)\n\t\.size", asm,
                  re.S | re.M)
    assert m, (f"{funzione} non compare come funzione nell'assembly: se e' "
               f"stata incorporata in setup() il ciclo misurato non ha piu' "
               f"confini ispezionabili")
    return m.group(2)


@avr
@pytest.mark.parametrize("variante", VARIANTI)
def test_nessuna_routine_di_libgcc_dentro_la_finestra_su_avr(tmp_path, variante):
    """Il controllo che conta, sul codice davvero emesso per la scheda.

    Dentro la finestra misurata devono comparire le chiamate al kernel del
    modello e nient'altro: nessuna routine di libgcc — divisione a 32 bit
    (__udivmodsi4), helper a 64 bit (__adddi3, __ashrdi3), soft-float
    (__mulsf3). Il costo del kernel e' cio' che si vuole misurare; tutto il
    resto e' il costo del banco di misura, e finirebbe nei microjoule
    attribuiti al modello.

    Controllo del controllo: rimettendo `k % EB_CACHE` nel ciclo, qui compare
    __udivmodsi4 e il corpo passa da 69 a 99 istruzioni.
    """
    _serve_lheader(variante)
    asm = assembly_avr(tmp_path, variante)
    corpo = corpo_assembly(asm, "eb_finestra_attiva")
    chiamate = set(re.findall(r"\b(?:r?call|r?jmp)\s+([^\s,]+)", corpo))
    libgcc = sorted(c for c in chiamate if c.startswith("__"))
    assert not libgcc, (
        f"{variante}: dentro la finestra misurata il compilatore chiama "
        f"{libgcc}. Non e' codice del modello: e' overhead del banco di "
        f"misura, e l'energia per inferenza lo conterebbe come se lo fosse.")


@avr
def test_il_firmware_di_energia_non_usa_virgola_mobile_su_avr(tmp_path):
    """Il firmware intero, non solo i kernel: stampare un rapporto o una
    media in virgola mobile tirerebbe dentro le routine soft-float in un
    programma che esiste per misurare un modello integer-only."""
    asm = assembly_avr(tmp_path, "EB_COEFF")
    sorgente = tmp_path / "energy.s"
    sorgente.write_text(asm, encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, str(REPO / "tools" / "check_no_float.py"),
                        str(sorgente)], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"virgola mobile nel firmware di energia su AVR\n{r.stdout}\n{r.stderr}")


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

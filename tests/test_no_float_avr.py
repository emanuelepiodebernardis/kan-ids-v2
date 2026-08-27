"""Il percorso di inferenza non deve contenere virgola mobile SUL TARGET.

Il README dichiara "zero floating-point instructions". La verifica che
c'era dietro girava su assembly x86 dell'host e cercava solo mnemonici x86.
Su AVR e su RISC-V senza estensione F pero' una FPU non esiste: il floating
point non compare come istruzione ma come chiamata alle routine soft-float
di libgcc (`__addsf3`, `__mulsf3`, `__floatsisf`, `__gesf2`, ...). Quella
regex non ne intercettava nemmeno una, quindi l'affermazione piu' importante
del progetto — che l'inferenza gira su un microcontrollore senza FPU — non
era coperta dal suo stesso controllo.

Qui i kernel vengono compilati davvero per ATmega2560 e l'assembly emesso
viene passato a tools/check_no_float.py, che adesso cerca anche i simboli
soft-float. I test che richiedono avr-g++ si saltano se il toolchain non
c'e': sono una verifica in piu' dove esiste, non un requisito per far
girare la suite.

Il controllo si invoca con l'interprete corrente e non con bash: su Windows
`bash` puo' risolvere a una WSL non installata, e il test falliva li' per
ragioni che con la virgola mobile non c'entrano nulla.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
CHECK = REPO / "tools" / "check_no_float.py"

# (nome, header, corpo del main). Il kernel a 10 classi end-to-end non c'e':
# il suo array di golden vector sfora i limiti di AVR e infatti platformio.ini
# lo compila solo per ESP32-C3.
KERNEL = [
    ("KAN(cat,1L)", ["kan14_coeff_int8.h", "kan14_coeff_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = kan14_coeff_predict(x, c);"),
    ("KAN(cat,ML)", ["kan14_ml_coeff_int8.h", "kan14_ml_coeff_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = kan14_ml_predict(x, c);"),
    ("KAN(cat,MC)", ["kan14_mc_coeff_int8.h", "kan14_mc_coeff_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = kan14_mc_predict(x, c);"),
    ("KAN e2e integer", ["kan_e2e_int.h", "kan_e2e_infer.h"],
     "sink = e2e_predict(1, 1, 1, 1, 1000);"),
    ("DecisionTree(d=5)", ["dt5_model.h"],
     "int16_t x[10]={0}; sink = dt5_predict(x);"),
    # La baseline densa: e' quella con l'accumulatore int64, cioe' quella dove
    # e' piu' facile che il compilatore tiri dentro una routine di libgcc.
    ("MLP(16)", ["mlp16_int8.h", "mlp16_infer.h"],
     "int16_t x[10]={0}; uint8_t c[4]={0}; sink = mlp16_predict(x, c);"),
]

# Non basta shutil.which: su Windows avr-g++ sta quasi sempre dentro i
# pacchetti di PlatformIO e non nel PATH, e questi test si saltavano proprio
# sulla macchina dove il codice viene scritto — cioe' l'affermazione "zero
# virgola mobile su AVR" non era controllata dove serviva.
sys.path.insert(0, str(REPO))
from kanids.toolchain import ambiente, motivo_assenza, trova      # noqa: E402

AVR = trova("avr-g++")
avr = pytest.mark.skipif(AVR is None, reason=motivo_assenza("avr-g++"))


def _assembly(tmp_path: Path, headers: list[str], corpo: str) -> Path:
    src = tmp_path / "probe.cpp"
    src.write_text(
        "#include <stdint.h>\n"
        + "".join(f'#include "{h}"\n' for h in headers)
        + "volatile int32_t sink;\n"
        f"int main(void) {{ {corpo} return 0; }}\n", encoding="utf-8", newline="\n")
    asm = tmp_path / "probe.s"
    r = subprocess.run(
        [AVR, "-mmcu=atmega2560", "-Os", "-I", str(INCLUDE),
         "-S", str(src), "-o", str(asm)],
        capture_output=True, text=True, env=ambiente("avr-g++"))
    assert r.returncode == 0, f"compilazione per AVR fallita:\n{r.stderr[-1500:]}"
    return asm


@avr
@pytest.mark.parametrize("nome, headers, corpo", KERNEL,
                         ids=[k[0] for k in KERNEL])
def test_kernel_senza_virgola_mobile_su_avr(tmp_path, nome, headers, corpo):
    for h in headers:
        if not (INCLUDE / h).exists():
            pytest.skip(f"{h} non generato: vedi scripts/export_*.py")
    asm = _assembly(tmp_path, headers, corpo)
    r = subprocess.run([sys.executable, str(CHECK), str(asm)],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{nome}: virgola mobile nel percorso di inferenza su AVR\n"
        f"{r.stdout}\n{r.stderr}")
    assert "0 istruzioni, 0 chiamate soft-float" in r.stdout, r.stdout


@avr
def test_il_controllo_riconosce_davvero_la_virgola_mobile(tmp_path):
    """Controllo del controllo: un sorgente che USA float deve far fallire
    lo script. Senza questo, un check che non trova mai niente perche' la
    sua regex e' sbagliata sarebbe indistinguibile da un check che passa.
    La vecchia regex, solo x86, su questo assembly AVR trovava zero."""
    src = tmp_path / "conFloat.cpp"
    src.write_text("float f(float a, float b) { return a * b + a / b; }\n"
                   "int g(int x) { return (int)((float)x * 1.5f); }\n", encoding="utf-8", newline="\n")
    asm = tmp_path / "conFloat.s"
    r = subprocess.run([AVR, "-mmcu=atmega2560", "-Os",
                        "-S", str(src), "-o", str(asm)],
                       capture_output=True, text=True, env=ambiente("avr-g++"))
    assert r.returncode == 0, r.stderr

    r = subprocess.run([sys.executable, str(CHECK), str(asm)],
                       capture_output=True, text=True)
    assert r.returncode == 1, (
        "lo script NON ha visto la virgola mobile in un sorgente che la usa "
        f"apertamente:\n{r.stdout}")
    assert "__mulsf3" in r.stdout or "__addsf3" in r.stdout, r.stdout


def test_lo_script_segnala_un_file_assente(tmp_path):
    """Un file assembly assente deve dare un errore riconoscibile. La
    versione in bash stampava conteggi vuoti e usciva con un errore di
    sintassi della shell, che a occhio somiglia a un successo."""
    r = subprocess.run([sys.executable, str(CHECK), str(tmp_path / "non_esiste.s")],
                       capture_output=True, text=True)
    assert r.returncode == 2, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"

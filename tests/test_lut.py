"""La versione sampled-LUT della KAN single-layer, e cosa deve restare vero.

Richiesta del Prof. Kuznetsov (rc3, punto 2): misurare il trade-off LUT
contro coefficienti in memoria, latenza ed energia. Perche' il confronto sia
attribuibile alla sola rappresentazione, la LUT e' campionata DAL MODELLO
DEPLOYATO — stesse funzioni apprese, stessi edge categorici, stessi vettori —
e questi test difendono esattamente quella proprieta':

* l'header si rigenera identico byte per byte dall'header a coefficienti;
* il kernel C calcola gli stessi interi della simulazione numpy;
* le decisioni coincidono con quelle della versione a coefficienti su tutti e
  200 i vettori, e non per caso: il limite di deviazione calcolato su TUTTI
  gli 8.193 ingressi possibili sta sotto il margine minimo osservato;
* i byte dichiarati sono quelli che il compilatore mette in Flash;
* su AVR non compare virgola mobile ne' aritmetica a 64 bit.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
INCLUDE = REPO / "mcu_pio" / "include"
HEADER = INCLUDE / "kan14_lut_int16.h"
SORGENTE = INCLUDE / "kan14_coeff_int8.h"
VETTORI = INCLUDE / "kan14_test_vectors.h"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from kanids import lut as klut                                   # noqa: E402
from kanids.interpretabilita import (leggi_modello, leggi_vettori,  # noqa: E402
                                     logit as logit_coeff)
from kanids.toolchain import ambiente, motivo_assenza, trova     # noqa: E402

GPP = trova("g++")
AVR = trova("avr-g++")
gpp = pytest.mark.skipif(GPP is None, reason=motivo_assenza("g++"))
avr = pytest.mark.skipif(AVR is None, reason=motivo_assenza("avr-g++"))
serve_header = pytest.mark.skipif(
    not HEADER.exists(),
    reason="kan14_lut_int16.h non generato: python scripts/export_kan14_lut_c.py")


def _L_dell_header() -> int:
    return int(re.search(r"#define KLUT_L (\d+)",
                         HEADER.read_text(encoding="utf-8")).group(1))


@serve_header
def test_lheader_si_riemette_identico_dallheader_a_coefficienti():
    """L'header della LUT non contiene informazione propria: e' una funzione
    dell'header a coefficienti. Se rigenerarlo dai suoi stessi ingressi
    producesse un file diverso, qualcuno l'avrebbe modificato a mano — ed e'
    proprio il caso in cui il confronto smetterebbe di misurare la sola
    rappresentazione."""
    import scripts.export_kan14_lut_c as exp                     # noqa: WPS433

    m = leggi_modello(SORGENTE)
    lu = klut.campiona(m, _L_dell_header())
    testo = HEADER.read_text(encoding="utf-8")
    intestazione = testo[:testo.index("#pragma once")].rstrip("\n")
    riemesso = klut.header(lu, m, intestazione)
    assert riemesso == testo, (
        "l'header committato non coincide con quello riemesso dai suoi stessi "
        "numeri: rigenerarlo con python scripts/export_kan14_lut_c.py")
    assert exp.USCITA == HEADER


@serve_header
def test_le_decisioni_coincidono_e_la_garanzia_e_un_limite_non_un_campione():
    """200/200 di accordo non basta come prova: con L=9 l'accordo e' ancora
    200/200 pur sbagliando il logit di oltre un milione. Cio' che rende la
    coincidenza una garanzia e' che la somma delle deviazioni massime — su
    tutti gli ingressi possibili, non su un campione — stia sotto il margine
    piu' piccolo osservato."""
    m = leggi_modello(SORGENTE)
    v = leggi_vettori(VETTORI)
    lu = klut.campiona(m, _L_dell_header())

    z = logit_coeff(m, v["X"], v["CAT"])
    zl = klut.logit(lu, m, v["X"], v["CAT"])
    assert ((z >= 0) == (zl >= 0)).all(), "la LUT cambia qualche decisione"

    limite = int(klut.deviazione_esaustiva(lu, m).sum())
    margine = int(np.abs(z).min())
    assert limite < margine, (
        f"il limite di deviazione {limite} non sta sotto il margine minimo "
        f"{margine}: le decisioni coincidono ma niente lo garantisce")


@serve_header
def test_il_limite_saprebbe_dire_di_no():
    """Controllo del controllo: con pochi campioni il limite deve superare il
    margine, altrimenti la disuguaglianza sopra passerebbe sempre e non
    starebbe verificando nulla."""
    m = leggi_modello(SORGENTE)
    v = leggi_vettori(VETTORI)
    margine = int(np.abs(logit_coeff(m, v["X"], v["CAT"])).min())
    grossolana = klut.campiona(m, 9)
    limite = int(klut.deviazione_esaustiva(grossolana, m).sum())
    assert limite > margine, (
        "nemmeno con nove campioni per edge il limite supera il margine: la "
        "disuguaglianza non discrimina")


@serve_header
def test_i_byte_dichiarati_sono_quelli_che_il_compilatore_mette_in_flash():
    from c_footprint import scan                                 # noqa: WPS433

    m = leggi_modello(SORGENTE)
    lu = klut.campiona(m, _L_dell_header())
    dal_parser, _ = scan(HEADER, "KLUT_")
    assert dal_parser == klut.byte_modello(lu, m), (
        "il conteggio del modulo e quello del parser non coincidono")

    testo = HEADER.read_text(encoding="utf-8")
    dichiarati = int(re.search(r"\((\d[\d.]*) B di modello", testo)
                     .group(1).replace(".", ""))
    assert dichiarati == dal_parser, (
        f"l'intestazione dichiara {dichiarati} B, gli array ne occupano "
        f"{dal_parser}")

    csv = pd.read_csv(REPO / "results" / "footprint.csv")
    riga = csv[csv.fonte.astype(str).str.endswith("kan14_lut_int16.h")]
    assert len(riga) == 1, "la LUT non compare in results/footprint.csv"
    assert int(riga.iloc[0].byte_parametri) == dal_parser


@serve_header
def test_la_tabella_del_compromesso_e_riproducibile():
    """results/lut_vs_coeff.csv e' la risposta alla domanda del relatore: si
    ricalcola e si pretende identica, cosi' la curva byte/errore non puo'
    invecchiare rispetto all'header."""
    f = REPO / "results" / "lut_vs_coeff.csv"
    assert f.exists(), "manca results/lut_vs_coeff.csv"
    import scripts.export_kan14_lut_c as exp                     # noqa: WPS433

    atteso = exp.tabella(leggi_modello(SORGENTE), leggi_vettori(VETTORI))
    trovato = pd.read_csv(f)
    pd.testing.assert_frame_equal(trovato, atteso, check_dtype=False)

    scelto = trovato[trovato.L == _L_dell_header()].iloc[0]
    assert bool(scelto.decisioni_garantite), (
        "l'header e' stato generato con un L che non garantisce le decisioni")
    piu_piccoli = trovato[(trovato.L < _L_dell_header())
                          & trovato.decisioni_garantite]
    assert piu_piccoli.empty, (
        f"esiste un L piu' piccolo con la stessa garanzia: {list(piu_piccoli.L)}")


@gpp
@serve_header
def test_il_kernel_c_calcola_gli_stessi_interi_della_simulazione(tmp_path):
    """Non le stesse predizioni: gli stessi logit. Confrontare le decisioni
    lascerebbe passare un kernel che sbaglia di poco ovunque."""
    src = tmp_path / "dump.cpp"
    src.write_text(
        '#include <cstdio>\n#include <cstdint>\n'
        '#include "kan14_lut_infer.h"\n#include "kan14_test_vectors.h"\n'
        "int main(){ for (int k=0;k<KTV_N;k++){ int16_t x[10]; uint8_t c[4];\n"
        "  for(int i=0;i<10;i++) x[i]=KTV_X[k][i];\n"
        "  for(int j=0;j<4;j++)  c[j]=KTV_CAT[k][j];\n"
        "  printf(\"%ld\\n\",(long)kan14_lut_logit(x,c)); } return 0; }\n",
        encoding="utf-8", newline="\n")
    exe = tmp_path / "dump"
    r = subprocess.run([GPP, "-O2", "-I", str(INCLUDE), str(src), "-o", str(exe)],
                       capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-1500:]
    dal_c = np.array([int(x) for x in
                      subprocess.run([str(exe)], capture_output=True, text=True,
                                     timeout=120).stdout.split()])

    m = leggi_modello(SORGENTE)
    v = leggi_vettori(VETTORI)
    dal_python = klut.logit(klut.campiona(m, _L_dell_header()), m,
                            v["X"], v["CAT"])
    assert dal_c.tolist() == dal_python.tolist(), (
        "kernel C e simulazione numpy non danno lo stesso logit intero")


@gpp
@serve_header
def test_lhost_check_della_lut_gira_e_conferma_lequivalenza(tmp_path):
    exe = tmp_path / "chk"
    r = subprocess.run([GPP, "-O2", "-o", str(exe), "run_lut_check.cpp"],
                       cwd=REPO / "mcu_pio" / "host_check",
                       capture_output=True, text=True, env=ambiente("g++"))
    assert r.returncode == 0, r.stderr[-1500:]
    out = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stdout
    assert "200/200" in out.stdout, out.stdout
    assert "decisioni identiche alla versione a coefficienti: 200/200" in out.stdout, \
        out.stdout


@avr
@serve_header
def test_il_kernel_lut_su_avr_non_usa_float_ne_64_bit(tmp_path):
    src = tmp_path / "probe.cpp"
    src.write_text('#include <stdint.h>\n#include "kan14_lut_infer.h"\n'
                   "volatile int32_t sink;\n"
                   "int main(void){ int16_t x[10]={0}; uint8_t c[4]={0};\n"
                   "  sink = kan14_lut_logit(x,c); return 0; }\n",
                   encoding="utf-8", newline="\n")
    asm = tmp_path / "probe.s"
    r = subprocess.run([AVR, "-mmcu=atmega2560", "-Os", "-I", str(INCLUDE),
                        "-S", str(src), "-o", str(asm)],
                       capture_output=True, text=True, env=ambiente("avr-g++"))
    assert r.returncode == 0, r.stderr[-1500:]

    r = subprocess.run([sys.executable, str(REPO / "tools" / "check_no_float.py"),
                        str(asm)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    testo = asm.read_text(encoding="utf-8", errors="replace")
    a_64 = sorted(set(re.findall(
        r"__(?:mulsidi3|muldi3|adddi3|subdi3|ashrdi3|ashldi3|lshrdi3)", testo)))
    assert not a_64, f"il kernel LUT chiama routine a 64 bit: {a_64}"


def test_firmware_ed_environment_esistono():
    """La LUT serve a essere MISURATA: senza firmware e senza environment, i
    numeri di latenza ed energia non arriverebbero mai."""
    assert (REPO / "mcu_pio" / "src" / "main_lut14.cpp").exists()
    ini = (REPO / "mcu_pio" / "platformio.ini").read_text(encoding="utf-8")
    for env in ("megaatmega2560_lut14", "esp32c3_lut14",
                "megaatmega2560_energy_lut14", "esp32c3_energy_lut14"):
        assert f"[env:{env}]" in ini, f"manca l'environment {env}"
    energia = (REPO / "mcu_pio" / "src" / "main_energy.cpp").read_text(encoding="utf-8")
    assert "EB_LUT14" in energia, "la variante di energia non esiste"


def test_i_numeri_del_readme_sul_test_set_vengono_dallartefatto():
    """Il paragrafo del README che dichiara l'accordo sull'intero test set
    cita sei numeri. Vengono da `results/lut_vs_coeff_test.csv`, prodotto da
    `export_kan14_lut_c.py --su-test` sulla macchina che ha il dataset: qui si
    pretende che coincidano, invece di fidarsi di chi li ha ricopiati.

    Il CSV non c'e' in un clone senza dataset, e allora il test si salta
    dicendolo: e' una verifica in piu' dove il file esiste, non un requisito
    per far girare la suite."""
    f = REPO / "results" / "lut_vs_coeff_test.csv"
    if not f.exists():
        pytest.skip("results/lut_vs_coeff_test.csv assente: serve il dataset "
                    "(python scripts/export_kan14_lut_c.py --su-test)")
    r = pd.read_csv(f).iloc[0]
    testo = (REPO / "README.md").read_text(encoding="utf-8")
    blocco = testo[testo.index("Measured on the **whole test set**"):]
    blocco = blocco[:blocco.index("\n\n")]

    attesi = {
        "flussi di test": f"{int(r.n_test):,}",
        "decisioni diverse": str(int(r.decisioni_diverse)),
        "F1": f"{float(r.f1_coefficienti):.6f}",
        "flussi entro il limite": str(int(r.flussi_entro_il_limite)),
        "scostamento massimo": f"{int(r.scostamento_max_osservato):,}",
        "limite": f"{int(r.limite_scostamento_logit):,}",
    }
    mancanti = {k: v for k, v in attesi.items() if v not in blocco}
    assert not mancanti, (
        f"il README non riporta i valori dell'artefatto: {mancanti}\n{blocco}")
    assert float(r.f1_coefficienti) == float(r.f1_lut), (
        "il README dice che gli F1 coincidono, il CSV dice di no")
    assert int(r.decisioni_diverse) == 0, (
        "il README dichiara decisioni identiche su tutto il test set")

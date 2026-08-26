"""Trova i compilatori, anche quando non stanno nel PATH.

Perche' esiste. Sei test del progetto compilano davvero: cinque kernel per
ATmega2560, per verificare che nel percorso di inferenza non ci sia una sola
istruzione in virgola mobile ne' una chiamata soft-float; e il firmware di
energia per l'host, per eseguirlo e confrontare il checksum delle predizioni
con i golden vector. Sono i test che sostengono due affermazioni centrali del
lavoro, e finche' si saltano quelle affermazioni sono verificate altrove ma
non qui.

Cercarli con `shutil.which` soltanto li fa saltare su quasi ogni macchina
Windows, compresa quella dove il progetto viene sviluppato — e li' PlatformIO
**ha gia' scaricato** la toolchain AVR per costruire i firmware:

    ~/.platformio/packages/toolchain-atmelavr/bin/avr-g++.exe

Chiunque possa compilare il firmware ha quel compilatore. Pretendere che lo
metta anche nel PATH e' una barriera senza scopo, e il prezzo lo paga la
verifica: `test_no_float_avr.py` si saltava in silenzio, quindi la frase
"zero virgola mobile su AVR" non era controllata dove il codice viene
scritto.

Qui si cerca prima nel PATH, poi nei pacchetti PlatformIO. Non si installa e
non si scarica niente: se il compilatore non c'e', i test si saltano come
prima, ma dicendo QUALE manca e dove lo si e' cercato.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# compilatore -> pacchetti PlatformIO che potrebbero contenerlo
PACCHETTI = {
    "avr-g++": ("toolchain-atmelavr",),
    "avr-gcc": ("toolchain-atmelavr",),
    "avr-objdump": ("toolchain-atmelavr",),
}

# per l'host non c'e' un pacchetto PlatformIO: si accettano gli equivalenti
# che capiscono le stesse opzioni da riga di comando (-O2, -std=, -include)
EQUIVALENTI = {
    "g++": ("g++", "clang++", "c++"),
}


def _radice_platformio() -> Path:
    return Path(os.environ.get("PLATFORMIO_CORE_DIR",
                               Path.home() / ".platformio")) / "packages"


# Variabili d'ambiente convenzionali: CXX e' quella che ogni build system
# rispetta da trent'anni, ed e' il modo pulito di indicare un compilatore che
# sta in una cartella qualunque senza metterlo nel PATH. Su Windows conta piu'
# del solito: la cartella bin di w64devkit contiene anche find.exe, sort.exe e
# tar.exe di busybox, che messi davanti al PATH oscurano gli omonimi di
# Windows e rompono script che con questo progetto non c'entrano niente.
VARIABILI = {"g++": ("CXX",), "avr-g++": ("AVR_CXX",)}


def trova(nome: str) -> str | None:
    """Percorso dell'eseguibile, o None.

    Nell'ordine: variabile d'ambiente convenzionale, PATH (con gli
    equivalenti dichiarati), pacchetti PlatformIO.
    """
    for var in VARIABILI.get(nome, ()):
        valore = os.environ.get(var)
        if valore:
            p = Path(valore)
            if p.is_file():
                return str(p)
            trovato = shutil.which(valore)
            if trovato:
                return trovato
    for candidato in EQUIVALENTI.get(nome, (nome,)):
        trovato = shutil.which(candidato)
        if trovato:
            return trovato
    radice = _radice_platformio()
    for pacchetto in PACCHETTI.get(nome, ()):
        for ext in ("", ".exe"):
            p = radice / pacchetto / "bin" / (nome + ext)
            if p.is_file():
                return str(p)
    return None


def motivo_assenza(nome: str) -> str:
    """Il messaggio di skip: dice cosa manca e dove si e' guardato.

    "toolchain assente" non aiuta nessuno a installarla."""
    dove = []
    if VARIABILI.get(nome):
        dove.append("variabile " + " o ".join("$" + v for v in VARIABILI[nome]))
    dove.append(f"PATH ({', '.join(EQUIVALENTI.get(nome, (nome,)))})")
    for pacchetto in PACCHETTI.get(nome, ()):
        dove.append(f"{_radice_platformio() / pacchetto / 'bin'}")
    suggerimento = {
        "avr-g++": " — si ottiene installando un environment AVR di PlatformIO "
                   "(pio pkg install -t toolchain-atmelavr)",
        "g++": " — su Windows: w64devkit (zip, senza installer) oppure MSYS2",
    }.get(nome, "")
    return f"{nome} non trovato in: " + "; ".join(dove) + suggerimento


def ambiente(nome: str, base: dict | None = None) -> dict:
    """Ambiente per invocare `nome`, con la sua cartella davanti al PATH.

    gcc non e' un solo eseguibile: chiama `as` per assemblare e `ld` per
    linkare, e li cerca anche nel PATH. Invocare il driver con un percorso
    assoluto mentre la sua cartella NON e' nel PATH da':

        g++.exe: fatal error: cannot execute 'as': CreateProcess: No such
        file or directory

    che sembra un compilatore rotto e invece e' un compilatore monco. La
    soluzione non e' rimettere quella cartella nel PATH globale — su Windows
    w64devkit ci tiene anche find.exe e sort.exe, che oscurerebbero quelli di
    sistema — ma metterla nel PATH del SOLO sottoprocesso, dove non da'
    fastidio a nessun altro programma.
    """
    env = dict(os.environ if base is None else base)
    exe = trova(nome)
    if exe:
        cartella = str(Path(exe).parent)
        env["PATH"] = cartella + os.pathsep + env.get("PATH", "")
    return env

"""Dove sta uno stato di training, quando `artifacts/` e' vuoto.

Il difetto che questo modulo chiude
===================================
`artifacts/` e' cache rigenerabile e non e' versionata; `models/` e'
versionata. `scripts/export_models.py` copia i checkpoint dalla prima alla
seconda proprio per questo. Gli esportatori pero' leggevano solo dalla cache:

    with open(artifact_path("kan14_mlbin.pkl"), "rb") as fh:   # FileNotFoundError

Su un clone pulito — o semplicemente su una macchina dove la cache e' stata
svuotata — `scripts/export_kan14_ml_coeff_c.py` moriva con un
FileNotFoundError su un file che **e' nel repository**, sotto un altro nome,
committato e verificato. L'header piu' importante del progetto dopo quello
single-layer risultava cosi' non rigenerabile, e non per una ragione vera.

E' la stessa forma del difetto che `kanids/legacy.py::prepare14_dict` aveva
gia' corretto per i dati: uno script che dipende da un effetto collaterale di
un altro, senza che niente dica quale sia quell'altro.

Qui la corrispondenza sta in un posto solo, `export_models.py` la usa per
sapere cosa copiare, e gli esportatori la usano per sapere dove cercare. Se
manca davvero, il messaggio dice quale script lo produce invece di dire solo
che un file non c'e'.
"""
from __future__ import annotations

from pathlib import Path

from kanids.config import MODELS_DIR, artifact_path

# nome nella cache -> (nome versionato in models/, script che lo produce)
VERSIONATI = {
    "kan14_bin_model.npz": ("kan14_binary_singlelayer.npz",
                            "scripts/kan14_binary.py"),
    "kan14_mlbin.pkl": ("kan14_binary_multilayer.pkl",
                        "scripts/kan14_ml_binary.py"),
    "mlcat_state.pkl": ("kan14_multiclass_multilayer.pkl",
                        "scripts/kan_ml_cat_mc.py"),
}


def trova(nome: str) -> Path | None:
    """La cache se c'e', altrimenti la copia versionata, altrimenti None."""
    cache = Path(artifact_path(nome))
    if cache.exists():
        return cache
    dst = VERSIONATI.get(nome)
    if dst:
        versionato = MODELS_DIR / dst[0]
        if versionato.exists():
            return versionato
    return None


def motivo(nome: str) -> str:
    """Perche' non si trova, e cosa lanciare. Un messaggio che dice solo
    'file non trovato' costringe chi legge a fare l'archeologia del
    repository per capire quale script lo produce."""
    dst = VERSIONATI.get(nome)
    dove = [str(artifact_path(nome))]
    if dst:
        dove.append(str(MODELS_DIR / dst[0]))
    coda = f"; si rigenera con `python {dst[1]}`" if dst else ""
    return f"checkpoint {nome} non trovato in: {', '.join(dove)}{coda}"

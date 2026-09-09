"""Il driver di rigenerazione deve accorgersi dei checkpoint vecchi.

Il guardiano esiste perche' l'errore era gia' successo: un checkpoint
completo scritto sotto il protocollo precedente veniva riusato in silenzio
-- gli script saltano cio' che risulta gia' fatto -- e i numeri vecchi
rientravano mescolati ai nuovi senza che nulla lo segnalasse.

La prima versione del guardiano cercava i checkpoint con una glob e saltava
ogni file con "ratio" nel nome, cioe' esattamente gli otto file della
sezione 18. Ora ogni stage dichiara i propri checkpoint per nome, e questo
test blocca il ritorno della scorciatoia.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rigenera", _ROOT / "rigenera.py")
rigenera = importlib.util.module_from_spec(_spec)
sys.modules["rigenera"] = rigenera
_spec.loader.exec_module(rigenera)


def test_ogni_stage_dichiara_i_suoi_checkpoint():
    nomi = set()
    for nome, argomenti, gruppo, desc, ckpt in rigenera.STAGE:
        assert gruppo in {"paper", "secondario", "sezione18", "guardia"}, nome
        assert nome not in nomi, f"stage duplicato: {nome}"
        nomi.add(nome)
        assert ckpt and all(c.endswith(".jsonl") for c in ckpt), nome
        assert Path(argomenti[0]).suffix == ".py", nome


def test_un_checkpoint_piu_vecchio_del_protocollo_viene_segnalato(tmp_path):
    """Anche -- soprattutto -- se ha 'ratio' nel nome."""
    finto = _ROOT / "artifacts" / "_prova_guardiano_ratio7.jsonl"
    finto.write_text("{}\n", encoding="utf-8")
    try:
        os.utime(finto, (0, 0))
        segnalato = rigenera.checkpoint_obsoleto([finto.name])
        assert segnalato == finto
        os.utime(finto, None)   # ora e' piu' recente del protocollo
        assert rigenera.checkpoint_obsoleto([finto.name]) is None
    finally:
        finto.unlink()


def test_un_checkpoint_inesistente_non_e_obsoleto():
    """Uno stage mai eseguito deve poter partire, non essere bloccato."""
    assert rigenera.checkpoint_obsoleto(["_non_esiste_affatto.jsonl"]) is None


@pytest.mark.parametrize("nome,argomenti,gruppo,desc,ckpt",
                         [s for s in rigenera.STAGE if s[2] == "sezione18"])
def test_il_suffisso_del_checkpoint_coincide_col_rapporto(
        nome, argomenti, gruppo, desc, ckpt):
    """Il suffisso `_ratio{n}` nel nome del checkpoint e' l'unica cosa che
    impedisce a un run a rapporto diverso di sovrascrivere quello a 50."""
    rapporto = argomenti[argomenti.index("--ratio") + 1]
    assert rapporto != "50", f"{nome}: a ratio 50 il suffisso e' vuoto"
    for c in ckpt:
        assert c.endswith(f"_ratio{rapporto}.jsonl"), (nome, c)


def test_la_sezione_18_non_rigenera_gli_script_prequenziali():
    """`drift_graduale.py` e `drift_graduale_int.py` non usano la partizione
    validation/test: valutano ogni batch prima di addestrarci sopra. Non
    vanno rilanciati, e metterli qui vorrebbe dire buttare ore di calcolo."""
    for nome, argomenti, gruppo, _desc, _ck in rigenera.STAGE:
        if gruppo == "sezione18":
            assert "graduale" not in argomenti[0], nome
    for script in ("scripts/drift_graduale.py", "scripts/drift_graduale_int.py"):
        testo = (_ROOT / script).read_text(encoding="utf-8")
        assert "dividi_target" not in testo, (
            f"{script} ora usa la partizione: la sezione 18 va rilanciata "
            "anche per lui")

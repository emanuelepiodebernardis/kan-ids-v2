"""Separazione fra i dati su cui si SCEGLIE e quelli su cui si RIPORTA.

Il difetto che questo modulo esiste per rendere impossibile: `iters=6000`
(sezione 16.1) e `ridge=0,1` (sezione 16.2) sono stati scelti massimizzando
la balanced accuracy calcolata sulle stesse righe target che poi il
documento riporta come risultato. E' lo stesso difetto del rapporto 1:5 nel
joint training del primo lavoro, dove la regola era: *i test devono essere
usati una volta sola, alla fine*.

Lo schema qui e' il minimo che lo corregge senza alterare la statistica
della selezione delle etichette -- che e' il collo di bottiglia centrale
del lavoro e non va toccata:

    target ricampionato
      |-- righe SELEZIONATE dalla regola  -> adattamento (budget di n etichette)
      `-- resto
            |-- VALIDATION (30 %)  -> qui, e solo qui, si scelgono le costanti
            `-- TEST       (70 %)  -> letto una volta, alla fine

La regola di selezione continua a vedere tutto il target, esattamente come
prima: un dispositivo in campo sceglie dallo stream che gli passa davanti.
A cambiare e' solo che il complemento non e' piu' un unico insieme usato sia
per calibrare sia per riportare.

Il partizionamento dipende dal solo `seed`, non dal metodo: dentro uno
stesso seed tutti i metodi vedono la stessa validation e lo stesso test,
quindi i confronti appaiati per seed restano validi.

## Il guardiano

`SplitValutazione.test` solleva `AccessoAlTestVietato` quando il processo
gira in modo selezione (`KANIDS_MODO=selezione`, oppure
`imposta_modo(MODO_SELEZIONE)`). Gli sweep di iperparametri girano in quel
modo, quindi non possono leggere il test nemmeno per sbaglio: e' un errore
di esecuzione, non una convenzione che qualcuno deve ricordare.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

MODO_SELEZIONE = "selezione"
MODO_FINALE = "finale"

FRAZIONE_VALIDATION = 0.30

_modo = None


class AccessoAlTestVietato(RuntimeError):
    """Sollevata quando codice in modo selezione tenta di leggere il test."""


def imposta_modo(modo: str) -> None:
    global _modo
    if modo not in (MODO_SELEZIONE, MODO_FINALE):
        raise ValueError(f"modo sconosciuto: {modo!r}")
    _modo = modo


def modo_corrente() -> str:
    if _modo is not None:
        return _modo
    return os.environ.get("KANIDS_MODO", MODO_FINALE)


@dataclass(frozen=True)
class SplitValutazione:
    """Le tre partizioni del target, come indici nell'array del target."""

    idx_adattamento: np.ndarray
    idx_validation: np.ndarray
    _idx_test: np.ndarray

    @property
    def test(self) -> np.ndarray:
        if modo_corrente() == MODO_SELEZIONE:
            raise AccessoAlTestVietato(
                "questo processo gira in modo selezione: il test set non e' "
                "leggibile. Scegli le costanti su .idx_validation. Se stai "
                "producendo i risultati finali, esegui senza "
                "KANIDS_MODO=selezione."
            )
        return self._idx_test

    @property
    def validation(self) -> np.ndarray:
        return self.idx_validation

    def __repr__(self) -> str:  # niente dimensioni del test nel repr
        return (f"SplitValutazione(adattamento={len(self.idx_adattamento)}, "
                f"validation={len(self.idx_validation)}, test=nascosto)")


def dividi_target(y_target: np.ndarray, idx_selezionati: np.ndarray, seed: int,
                  frazione_validation: float = FRAZIONE_VALIDATION
                  ) -> SplitValutazione:
    """Divide il complemento delle righe selezionate in validation e test.

    Stratificato sull'etichetta: con BoT-IoT come target i normali sono lo
    0,013 %, quindi una divisione non stratificata puo' lasciarne zero da
    una delle due parti e produrre una balanced accuracy indefinita per
    ragioni di campionamento invece che di metodo.
    """
    n = len(y_target)
    resto = np.ones(n, bool)
    resto[idx_selezionati] = False
    idx_resto = np.flatnonzero(resto)

    rng = np.random.RandomState(1000 + int(seed))
    val_parts = []
    for classe in np.unique(y_target[idx_resto]):
        della_classe = idx_resto[y_target[idx_resto] == classe]
        k = int(round(len(della_classe) * frazione_validation))
        k = min(max(k, 1), len(della_classe) - 1) if len(della_classe) > 1 else 0
        if k:
            val_parts.append(rng.choice(della_classe, k, replace=False))
    idx_val = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], int)

    in_val = np.zeros(n, bool)
    in_val[idx_val] = True
    idx_test = idx_resto[~in_val[idx_resto]]

    return SplitValutazione(np.asarray(idx_selezionati), idx_val, idx_test)

"""Proprieta' delle regole di selezione di `scripts/drift_sampling.py`.

Il primo test documenta un'identita' che i risultati mostravano ma il testo
non dichiarava: `conformal_top` e `margine` selezionano **le stesse righe**,
e non per caso.

    anomalia(z) = 1 - max(p, 1-p)   con p = sigma(z)
                = 1 - sigma(|z|)

e' strettamente decrescente in |z|. Ordinare per anomalia decrescente e'
quindi ordinare per |z| crescente, cioe' esattamente il criterio del
margine, ristretto al `pool` conformal. E il pool -- le righe il cui
insieme di predizione non ha esattamente una classe -- contiene sempre le
righe piu' vicine al confine, perche' li' entrambe le classi sono
plausibili. Le due regole coincidono ogni volta che il pool copre le n
righe di margine minimo, che e' il caso in tutti i run pubblicati
(80 celle su 80 di `results/drift_sampling_runs.csv`, tutte le colonne).

Non cambia nessun numero. Cambia cosa si puo' scrivere: sono una regola
sola, e "selezionare i flussi conformalmente piu' anomali" e' il criterio
del margine sotto un'altra veste, non un secondo metodo indipendente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "scripts", _ROOT / "src"):
    sys.path.insert(0, str(p))

from drift_sampling import selectors, set_sizes  # noqa: E402


def _finto_target(n=4000, seed=0):
    """Punteggi plausibili: due gaussiane sovrapposte, classe rara all'1%."""
    rng = np.random.RandomState(seed)
    y = (rng.random(n) > 0.01).astype(int)
    z = np.where(y == 1, rng.normal(1.5, 1.2, n), rng.normal(-1.0, 1.2, n))
    return z, y


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("budget", [8, 32, 128])
def test_conformal_top_e_margine_sono_la_stessa_regola(seed, budget):
    z, y = _finto_target(seed=seed)
    q = 0.6
    pool = np.flatnonzero(set_sizes(z, q) != 1)
    ordine = np.argsort(np.abs(z))[:budget]
    if not np.isin(ordine, pool).all():
        pytest.skip("pool non copre le righe di margine minimo per questo q")

    sel = selectors(z, y, q, budget, seed)
    assert np.array_equal(sel["margine"], sel["conformal_top"]), (
        "margine e conformal_top devono coincidere quando il pool conformal "
        "copre le righe di margine minimo: e' la stessa regola. Se questo "
        "test fallisce, una delle due definizioni e' cambiata e il documento "
        "va aggiornato di conseguenza."
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_anomalia_conformal_e_monotona_nel_margine(seed):
    """La ragione dell'identita', isolata: l'ordinamento per anomalia e'
    l'inverso esatto dell'ordinamento per |z|."""
    z, _ = _finto_target(seed=seed)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    anomalia = 1.0 - np.maximum(p, 1.0 - p)
    assert np.array_equal(
        np.argsort(-anomalia, kind="stable"),
        np.argsort(np.abs(z), kind="stable"),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solo_bilanciato_e_adattiva_guardano_le_etichette(seed):
    """Le regole dichiarate applicabili sul dispositivo non devono dipendere
    da y, tranne `adattiva` (che spende etichette del budget per il sondaggio)
    e `bilanciato` (dichiarato non applicabile)."""
    z, y = _finto_target(seed=seed)
    sel_a = selectors(z, y, 0.6, 32, seed)
    sel_b = selectors(z, 1 - y, 0.6, 32, seed)
    for regola in ("casuale", "margine", "strat_z", "conformal", "conformal_top", "misto"):
        assert np.array_equal(sel_a[regola], sel_b[regola]), (
            f"la regola '{regola}' cambia selezione se si invertono le "
            "etichette: sta guardando y, quindi non e' applicabile a bordo"
        )

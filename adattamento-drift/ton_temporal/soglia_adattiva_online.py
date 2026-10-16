"""Controllore causale della sola soglia per ADAPT-03."""

from __future__ import annotations

import time

import numpy as np

import soglia_bilanciata


class ControlloreSoglia:
    """Mantiene una soglia senza modificare score o parametri del modello."""

    def __init__(self, soglia_b, indice_score, minimo_per_classe=5,
                 eta_massima=50):
        if minimo_per_classe <= 0:
            raise ValueError('minimo_per_classe deve essere positivo')
        if eta_massima < 0:
            raise ValueError('eta_massima deve essere non negativa')
        self.soglia_b = float(soglia_b)
        self.soglia = float(soglia_b)
        self.indice_score = int(indice_score)
        self.minimo_per_classe = int(minimo_per_classe)
        self.eta_massima = int(eta_massima)
        self.ultimo_aggiornamento = None
        self.tentativi = 0
        self.aggiornamenti = 0
        self.senza_evidenza = 0
        self.tempo_secondi = []

    def tenta_aggiornamento(self, memoria, blocco_corrente):
        """Aggiorna dai soli esempi ammissibili; altrimenti conserva lo stato."""
        self.tentativi += 1
        X, y, eta = memoria.array_ammissibile(
            blocco_corrente, self.eta_massima)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        base = {
            'tentativo': self.tentativi,
            'blocco': int(blocco_corrente),
            'soglia_prima': self.soglia,
            'soglia_b': self.soglia_b,
            'esempi_ammissibili': int(len(y)),
            'normali_ammissibili': n0,
            'attacchi_ammissibili': n1,
            'eta_massima_ammessa': self.eta_massima,
            'eta_osservata': _riassumi_eta(eta),
            'minimo_per_classe': self.minimo_per_classe,
        }
        if min(n0, n1) < self.minimo_per_classe:
            self.senza_evidenza += 1
            base.update({
                'aggiornato': False,
                'motivo': 'minimo_per_classe_non_raggiunto',
                'soglia_dopo': self.soglia,
                'fallback': 'mantieni_ultima_soglia_ammessa',
                'ms_aggiornamento': 0.0,
            })
            return base

        t0 = time.perf_counter()
        nuova, dettaglio = soglia_bilanciata.scegli_soglia(
            X[:, self.indice_score], y,
            origine='gli esempi ammissibili della memoria')
        durata = time.perf_counter() - t0
        precedente = self.soglia
        self.soglia = float(nuova)
        self.ultimo_aggiornamento = int(blocco_corrente)
        self.aggiornamenti += 1
        self.tempo_secondi.append(durata)
        base.update({
            'aggiornato': True,
            'motivo': 'evidenza_ammissibile',
            'soglia_prima': precedente,
            'soglia_dopo': self.soglia,
            'fallback': None,
            'ms_aggiornamento': round(1000 * durata, 3),
            'scelta': dettaglio,
        })
        return base

    def riepilogo(self):
        return {
            'soglia_b': self.soglia_b,
            'soglia_finale': self.soglia,
            'minimo_per_classe': self.minimo_per_classe,
            'eta_massima': self.eta_massima,
            'tentativi': self.tentativi,
            'aggiornamenti': self.aggiornamenti,
            'blocchi_senza_evidenza': self.senza_evidenza,
            'ultimo_aggiornamento': self.ultimo_aggiornamento,
        }

    @staticmethod
    def byte_stato_logico():
        """Due float64, un indice int64 e quattro contatori int64."""
        return 2 * 8 + 5 * 8


def _riassumi_eta(eta):
    if len(eta) == 0:
        return {'n': 0, 'min': None, 'mediana': None, 'max': None}
    return {
        'n': int(len(eta)),
        'min': int(np.min(eta)),
        'mediana': float(np.median(eta)),
        'max': int(np.max(eta)),
    }

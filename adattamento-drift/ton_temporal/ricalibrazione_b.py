"""Ricalibrazione esatta della soglia KAN sull'archivio B completo."""

from __future__ import annotations

import time

import numpy as np

import replay
import soglia_bilanciata


class ArchivioCalibrazioneB:
    """Cache della rappresentazione KAN di B, separata dalla memoria online."""

    def __init__(self, modello, trasformazione, X_b, y_b, blocco=200_000):
        X_b = np.asarray(X_b)
        y_b = np.asarray(y_b, dtype=np.uint8)
        if len(X_b) == 0:
            raise ValueError('B non puo essere vuoto')
        if len(X_b) != len(y_b):
            raise ValueError('feature ed etichette di B hanno lunghezza diversa')
        classi = np.unique(y_b)
        if not np.array_equal(classi, np.array([0, 1], dtype=np.uint8)):
            raise ValueError('B deve contenere entrambe le classi binarie 0/1')
        if blocco <= 0:
            raise ValueError('il blocco della cache deve essere positivo')

        self.y = y_b
        self.righe = int(len(y_b))
        self.blocco = int(blocco)
        self.byte_x = int(X_b.nbytes)
        self.byte_y = int(y_b.nbytes)
        self.tempi = []

        t0 = time.perf_counter()
        prima = None
        for i in range(0, self.righe, self.blocco):
            R = modello.rappresentazione(trasformazione(X_b[i:i + self.blocco]))
            if prima is None:
                self.rappresentazione = np.empty(
                    (self.righe, R.shape[1]), dtype=np.float64)
                prima = R.shape[1]
            self.rappresentazione[i:i + len(R)] = R
        self.secondi_costruzione = time.perf_counter() - t0
        self.byte_cache = int(self.rappresentazione.nbytes)

    @classmethod
    def da_rappresentazione(cls, rappresentazione, y_b):
        """Costruttore leggero per le controprove, senza modello o flusso C."""
        R = np.asarray(rappresentazione, dtype=np.float64)
        y = np.asarray(y_b, dtype=np.uint8)
        if len(R) == 0:
            raise ValueError('B non puo essere vuoto')
        if len(R) != len(y):
            raise ValueError('rappresentazione ed etichette di B non allineate')
        if not np.array_equal(np.unique(y), np.array([0, 1], dtype=np.uint8)):
            raise ValueError('B deve contenere entrambe le classi binarie 0/1')
        oggetto = cls.__new__(cls)
        oggetto.y = y
        oggetto.righe = int(len(y))
        oggetto.blocco = int(len(y))
        oggetto.byte_x = None
        oggetto.byte_y = int(y.nbytes)
        oggetto.rappresentazione = np.array(R, copy=True)
        oggetto.byte_cache = int(oggetto.rappresentazione.nbytes)
        oggetto.secondi_costruzione = 0.0
        oggetto.tempi = []
        return oggetto

    def ricalibra(self, parametri):
        """Ricalcola score e soglia su tutto B con la regola BA preregistrata."""
        p = np.asarray(parametri, dtype=np.float64)
        if p.ndim != 1 or len(p) != self.rappresentazione.shape[1] + 1:
            raise ValueError('parametri KAN incompatibili con la cache di B')

        t0 = time.perf_counter()
        score = self.rappresentazione @ p[:-1] + p[-1]
        t_score = time.perf_counter()
        soglia, dettaglio = soglia_bilanciata.scegli_soglia(
            score, self.y, origine='intero B, archivio separato dalla memoria online')
        t_fine = time.perf_counter()
        rss, metodo = replay.picco_rss()
        costo = {
            'secondi_score_b': t_score - t0,
            'secondi_scelta_ba': t_fine - t_score,
            'secondi_totali': t_fine - t0,
            'byte_score_b': int(score.nbytes),
            'picco_rss_processo_mib': rss,
            'picco_rss_metodo': metodo,
        }
        dettaglio['soglia'] = float(soglia)
        self.tempi.append(costo)
        return float(soglia), dettaglio, costo

    def risorse(self):
        return {
            'righe': self.righe,
            'byte_x_b': self.byte_x,
            'byte_y_b': self.byte_y,
            'byte_cache_rappresentazione_kan': self.byte_cache,
            'colonne_cache': int(self.rappresentazione.shape[1]),
            'dtype_cache': str(self.rappresentazione.dtype),
            'blocco_costruzione_cache': self.blocco,
            'secondi_costruzione_cache': self.secondi_costruzione,
        }

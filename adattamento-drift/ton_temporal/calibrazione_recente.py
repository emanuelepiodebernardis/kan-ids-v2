"""Primitive causali di ADAPT-04 per fit192 e calibration64 separati."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

import replay
import soglia_bilanciata


SALT = b'ADAPT04-C-seed42-v1'
N_FEATURE = 8
FIT_CAPACITA = 192
CAL_CAPACITA = 64
ETA_MASSIMA = 50
MINIMO_CAL_CLASSE = 5


def vettore_canonico(x):
    """Restituisce gli otto input come bit IEEE-754 little-endian canonici."""
    valori = np.asarray(x, dtype=np.float64)
    if valori.shape != (N_FEATURE,):
        raise ValueError('il ruolo richiede esattamente otto feature')
    valori = np.ascontiguousarray(valori, dtype='<f8').copy()
    valori[valori == 0.0] = 0.0
    bit = valori.view('<u8')
    bit[np.isnan(valori)] = np.uint64(0x7FF8000000000000)
    return bit.tobytes(order='C')


def digest_ruolo(x):
    h = hashlib.sha256()
    h.update(SALT)
    h.update(b'\0')
    h.update(vettore_canonico(x))
    return h.digest()


def assegna_ruolo(x):
    digest = digest_ruolo(x)
    ruolo = 'calibration' if int.from_bytes(digest, 'big') % 4 == 0 else 'fit'
    return ruolo, digest.hex()


def assegna_ruoli(X):
    ruoli, digest = [], []
    for x in np.asarray(X):
        ruolo, impronta = assegna_ruolo(x)
        ruoli.append(ruolo)
        digest.append(impronta)
    return np.asarray(ruoli, dtype='U11'), digest


def primo_blocco_influenzabile(blocco_osservazione, ritardo=1):
    if blocco_osservazione < 0 or ritardo < 0:
        raise ValueError('blocco e ritardo devono essere non negativi')
    return int(blocco_osservazione + ritardo + 1)


class BufferRuolo(replay.MemoriaReplay):
    """Memoria per classe con digest del vettore, senza archivio laterale."""

    def __init__(self, capacita):
        super().__init__(capacita, 'per_classe')

    def aggiungi(self, X, y, blocco_osservazione, digest):
        X = np.asarray(X)
        y = np.asarray(y)
        if not (len(X) == len(y) == len(digest)):
            raise ValueError('feature, etichette e digest hanno lunghezze diverse')
        for x, classe, impronta in zip(X, y, digest):
            c = int(classe)
            if c not in (0, 1):
                raise ValueError('classe attesa binaria 0/1, trovata ' + str(c))
            if len(impronta) != 64:
                raise ValueError('digest SHA-256 non valido')
            self.voci.append({
                'x': np.asarray(x).copy(),
                'y': c,
                'blocco_arrivo': int(blocco_osservazione),
                'ordine': self._ordine,
                'digest': impronta,
            })
            self._ordine += 1
        self._riduci()

    def digest(self):
        return [v['digest'] for v in self.voci]

    def vettori_unici(self):
        return len(set(self.digest()))

    def payload(self):
        X, y = self.array()
        metadati = len(self.voci) * (8 + 8 + 32)
        return {
            'feature_byte': int(X.nbytes),
            'etichette_byte': int(y.nbytes),
            'metadati_byte': int(metadati),
            'totale_byte': int(X.nbytes + y.nbytes + metadati),
            'nota': (
                'metadati logici: blocco osservazione, ordine e digest SHA-256; '
                'escluso overhead degli oggetti Python'),
        }


class MemorieSeparate:
    def __init__(self):
        self.fit = BufferRuolo(FIT_CAPACITA)
        self.calibration = BufferRuolo(CAL_CAPACITA)

    def aggiungi(self, X, y, ruoli, blocco_osservazione, digest):
        X = np.asarray(X)
        y = np.asarray(y)
        ruoli = np.asarray(ruoli)
        if not (len(X) == len(y) == len(ruoli) == len(digest)):
            raise ValueError('lotto etichettato incoerente')
        noti = set(ruoli.tolist())
        if not noti.issubset({'fit', 'calibration'}):
            raise ValueError('ruolo non riconosciuto')
        for nome, buffer in (('fit', self.fit), ('calibration', self.calibration)):
            maschera = ruoli == nome
            buffer.aggiungi(
                X[maschera], y[maschera], blocco_osservazione,
                [d for d, usa in zip(digest, maschera) if usa])
        sovrapposti = set(self.fit.digest()) & set(self.calibration.digest())
        if sovrapposti:
            raise RuntimeError('un vettore compare sia nel fit sia nella calibrazione')

    def descrizione(self, blocco_corrente):
        Xf, yf = self.fit.array()
        Xc, yc = self.calibration.array()
        Xr, yr, eta = self.calibration.array_ammissibile(
            blocco_corrente, ETA_MASSIMA)
        return {
            'fit192': {
                'normali': int((yf == 0).sum()),
                'attacchi': int((yf == 1).sum()),
                'vettori_unici': self.fit.vettori_unici(),
                'eta_blocchi': self.fit.eta(blocco_corrente),
                'payload': self.fit.payload(),
            },
            'calibration64': {
                'normali': int((yc == 0).sum()),
                'attacchi': int((yc == 1).sum()),
                'vettori_unici': self.calibration.vettori_unici(),
                'eta_blocchi': self.calibration.eta(blocco_corrente),
                'recenti': {
                    'totale': int(len(yr)),
                    'normali': int((yr == 0).sum()),
                    'attacchi': int((yr == 1).sum()),
                    'eta_min': int(eta.min()) if len(eta) else None,
                    'eta_massima': int(eta.max()) if len(eta) else None,
                },
                'payload': self.calibration.payload(),
            },
            'totale_esempi': int(len(Xf) + len(Xc)),
        }


@dataclass
class StatoPolitiche:
    parametri_adattati: np.ndarray
    soglia_frozen_cal64: float
    soglia_fit192_cal64: float
    soglia_b: float

    def copia(self):
        return StatoPolitiche(
            np.array(self.parametri_adattati, copy=True),
            float(self.soglia_frozen_cal64),
            float(self.soglia_fit192_cal64),
            float(self.soglia_b),
        )


def _parametri_validi(parametri, n):
    p = np.asarray(parametri, dtype=np.float64)
    return p.shape == (n,) and bool(np.isfinite(p).all())


def proponi_transazione(modello, parametri_iniziali, stato, memorie,
                        blocco_corrente):
    """Calcola una transazione senza mutare modello, memorie o stato."""
    Xf, yf = memorie.fit.array()
    conteggi_fit = memorie.fit.conteggi()
    if not (conteggi_fit[0] and conteggi_fit[1]):
        return {'applicabile': False, 'motivo': 'fit_classe_mancante'}

    Xc, yc, eta = memorie.calibration.array_ammissibile(
        blocco_corrente, ETA_MASSIMA)
    normali = int((yc == 0).sum())
    attacchi = int((yc == 1).sum())
    if normali < MINIMO_CAL_CLASSE:
        return {
            'applicabile': False,
            'motivo': 'cal_normali_recenti_insufficienti',
            'cal_normali_recenti': normali,
            'cal_attacchi_recenti': attacchi,
        }
    if attacchi < MINIMO_CAL_CLASSE:
        return {
            'applicabile': False,
            'motivo': 'cal_attacchi_recenti_insufficienti',
            'cal_normali_recenti': normali,
            'cal_attacchi_recenti': attacchi,
        }

    t_fit = time.perf_counter()
    try:
        Rf = modello.rappresentazione(Xf)
        nuovo = replay.aggiorna(Rf, yf)
    except (ValueError, FloatingPointError):
        return {'applicabile': False, 'motivo': 'fit_invalido'}
    secondi_fit = time.perf_counter() - t_fit
    if nuovo is None:
        return {'applicabile': False, 'motivo': 'fit_invalido'}
    candidato = np.concatenate([nuovo[0], [nuovo[1]]])
    if not _parametri_validi(candidato, len(stato.parametri_adattati)):
        return {'applicabile': False, 'motivo': 'fit_invalido'}

    try:
        Rc = modello.rappresentazione(Xc)
        t_cal_frozen = time.perf_counter()
        score_frozen = Rc @ parametri_iniziali[:-1] + parametri_iniziali[-1]
        soglia_frozen, dettaglio_frozen = soglia_bilanciata.scegli_soglia(
            score_frozen, yc, origine='calibration64 recente')
        secondi_cal_frozen = time.perf_counter() - t_cal_frozen

        t_cal_adattata = time.perf_counter()
        score_adattato = Rc @ candidato[:-1] + candidato[-1]
        soglia_adattata, dettaglio_adattato = soglia_bilanciata.scegli_soglia(
            score_adattato, yc, origine='calibration64 recente')
        secondi_cal_adattata = time.perf_counter() - t_cal_adattata
    except (ValueError, FloatingPointError):
        return {'applicabile': False, 'motivo': 'fit_invalido'}
    if not (np.isfinite(soglia_frozen) and np.isfinite(soglia_adattata)):
        return {'applicabile': False, 'motivo': 'fit_invalido'}

    return {
        'applicabile': True,
        'motivo': None,
        'parametri_candidati': candidato,
        'soglia_frozen_candidata': float(soglia_frozen),
        'soglia_adattata_candidata': float(soglia_adattata),
        'dettaglio_frozen': dettaglio_frozen,
        'dettaglio_adattato': dettaglio_adattato,
        'cal_normali_recenti': normali,
        'cal_attacchi_recenti': attacchi,
        'cal_eta_min': int(eta.min()),
        'cal_eta_massima': int(eta.max()),
        'score_cal_frozen_sha256': _sha_float64(score_frozen),
        'score_cal_adattato_sha256': _sha_float64(score_adattato),
        'costi_secondi': {
            'fit': secondi_fit,
            'calibrazione_frozen': secondi_cal_frozen,
            'calibrazione_adattata': secondi_cal_adattata,
        },
    }


def applica_transazione(stato, proposta):
    if not proposta['applicabile']:
        return stato.copia()
    return StatoPolitiche(
        np.array(proposta['parametri_candidati'], copy=True),
        proposta['soglia_frozen_candidata'],
        proposta['soglia_adattata_candidata'],
        stato.soglia_b,
    )


def _sha_float64(valori):
    a = np.ascontiguousarray(valori, dtype='<f8')
    h = hashlib.sha256(str(len(a)).encode('ascii') + b':')
    h.update(a.tobytes(order='C'))
    return h.hexdigest()

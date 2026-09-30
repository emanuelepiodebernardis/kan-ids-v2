"""Replay a blocchi con etichette scarse e ritardate, contro copia congelata.

Il protocollo, attuato alla lettera
-----------------------------------
Per ogni blocco di `--blocco` righe in ordine di feature_ready_time:

  1. PREVEDI      il modello adattivo e la sua copia congelata emettono le
                  predizioni su tutte le righe del blocco. Le predizioni
                  emesse non vengono mai ricalcolate.
  2. RICHIEDI     si chiedono le etichette per l'1% delle righe: 100 su un
                  blocco completo, floor(0,01*n) sull'ultimo. Gli indici sono
                  estratti con un generatore inizializzato a `seme + k`, quindi
                  sono GLI STESSI per tutti i metodi, come il confronto richiede.
  3. ATTENDI      le etichette del blocco k arrivano alla FINE del blocco k+1.
                  Entrano allora nella memoria, e l'aggiornamento che ne segue
                  incide per la prima volta sul blocco k+2.
  4. AGGIORNA     memoria FIFO di `--memoria` esempi, inizializzata da A. Se la
                  memoria contiene una sola classe l'aggiornamento e' SALTATO e
                  il salto e' registrato.

Le tre regole di aggiornamento sono quelle chieste, e hanno la stessa forma:
una regressione logistica su una rappresentazione piccola, calcolata dal
modello congelato.

  regressione logistica   gli 8 coefficienti e l'intercetta
  MLP                     il solo ultimo strato, sulle attivazioni nascoste
  KAN additiva            un guadagno per feature e l'intercetta, sui contributi
                          additivi della base B-spline congelata

Misure per blocco, con NA dove non sono definite (per esempio l'AUROC quando il
blocco contiene una sola classe). Nessun riequilibrio: la prevalenza e' quella
osservata.

Trasformazione dichiarata: log1p sulle otto feature, poi standardizzazione con
media e deviazione stimate SU A. Le otto feature sono conteggi di byte e
pacchetti con code lunghe; la trasformazione e' fissata una volta su A e non
viene piu' toccata, quindi non usa informazione dei flussi successivi.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

NA = None


def base_bspline(u, nodi, grado=3):
    """Base B-spline valutata in u, con nodi aperti; restituisce (n, K)."""
    t = np.asarray(nodi, dtype=float)
    n_base = len(t) - grado - 1
    B = np.zeros((len(u), n_base))
    for i in range(n_base):
        B[:, i] = _bspline_uno(u, t, i, grado)
    return B


def _bspline_uno(u, t, i, k):
    if k == 0:
        return ((u >= t[i]) & (u < t[i + 1])).astype(float)
    d1 = t[i + k] - t[i]
    d2 = t[i + k + 1] - t[i + 1]
    a = np.zeros_like(u) if d1 == 0 else (u - t[i]) / d1 * _bspline_uno(u, t, i, k - 1)
    b = np.zeros_like(u) if d2 == 0 else (t[i + k + 1] - u) / d2 * _bspline_uno(u, t, i + 1, k - 1)
    return a + b


class Trasformazione:
    """log1p e standardizzazione, stimate una volta su A."""

    def __init__(self, X):
        L = np.log1p(np.maximum(X, 0.0))
        self.media = L.mean(axis=0)
        self.scala = L.std(axis=0)
        self.scala[self.scala == 0] = 1.0

    def __call__(self, X):
        return (np.log1p(np.maximum(X, 0.0)) - self.media) / self.scala


class ModelloLR:
    """Regressione logistica; l'aggiornamento tocca gli 8 coefficienti."""
    nome = 'lr'

    def __init__(self, Z, y, seme):
        self.m = LogisticRegression(max_iter=1000, C=1.0)
        self.m.fit(Z, y)

    def rappresentazione(self, Z):
        return Z

    def punteggio(self, Z):
        return self.m.decision_function(Z)

    def parametri(self):
        return np.concatenate([self.m.coef_.ravel(), self.m.intercept_])

    def applica(self, w, b):
        self.m.coef_ = w.reshape(1, -1).copy()
        self.m.intercept_ = np.array([b])


class ModelloMLP:
    """MLP a uno strato nascosto; l'aggiornamento tocca il solo ultimo strato."""
    nome = 'mlp'

    def __init__(self, Z, y, seme, nascosti=16, campione=400000):
        rng = np.random.default_rng(seme)
        idx = (rng.choice(len(y), campione, replace=False)
               if len(y) > campione else np.arange(len(y)))
        self.campione_usato = int(len(idx))
        self.m = MLPClassifier(hidden_layer_sizes=(nascosti,), max_iter=60,
                               random_state=seme, early_stopping=False)
        self.m.fit(Z[idx], y[idx])
        self.W1, self.b1 = self.m.coefs_[0], self.m.intercepts_[0]
        self.w2 = self.m.coefs_[1].ravel().copy()
        self.b2 = float(self.m.intercepts_[1][0])

    def rappresentazione(self, Z):
        return np.maximum(Z @ self.W1 + self.b1, 0.0)      # ReLU

    def punteggio(self, Z):
        return self.rappresentazione(Z) @ self.w2 + self.b2

    def parametri(self):
        return np.concatenate([self.w2, [self.b2]])

    def applica(self, w, b):
        self.w2 = w.copy()
        self.b2 = float(b)


class ModelloKAN:
    """KAN additiva su base B-spline; l'aggiornamento tocca guadagni e intercetta."""
    nome = 'kan'

    def __init__(self, Z, y, seme, n_base=8, campione=400000):
        rng = np.random.default_rng(seme)
        idx = (rng.choice(len(y), campione, replace=False)
               if len(y) > campione else np.arange(len(y)))
        self.campione_usato = int(len(idx))
        self.lo = Z.min(axis=0)
        self.hi = Z.max(axis=0)
        self.n_base = n_base
        self.nodi = []
        for j in range(Z.shape[1]):
            interni = np.linspace(0, 1, n_base - 2)
            self.nodi.append(np.concatenate([[0, 0, 0], interni, [1, 1, 1]]))
        B = self._basi(Z[idx])
        m = LogisticRegression(max_iter=1000, C=1.0)
        m.fit(B, y[idx])
        self.theta = m.coef_.ravel().copy()
        self.guadagni = np.ones(Z.shape[1])
        self.b = float(m.intercept_[0])

    def _u(self, Z):
        span = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)
        return np.clip((Z - self.lo) / span, 0.0, 1.0 - 1e-9)

    def _basi(self, Z):
        U = self._u(Z)
        return np.concatenate([base_bspline(U[:, j], self.nodi[j])
                               for j in range(Z.shape[1])], axis=1)

    def rappresentazione(self, Z):
        """Un contributo additivo per feature: e' su questi che agiscono i guadagni."""
        B = self._basi(Z)
        k = self.n_base - 1 + 3 - 2   # base per feature effettive
        per_feature = B.shape[1] // Z.shape[1]
        out = np.zeros((len(Z), Z.shape[1]))
        for j in range(Z.shape[1]):
            fetta = slice(j * per_feature, (j + 1) * per_feature)
            out[:, j] = B[:, fetta] @ self.theta[fetta]
        return out

    def punteggio(self, Z):
        return self.rappresentazione(Z) @ self.guadagni + self.b

    def parametri(self):
        return np.concatenate([self.guadagni, [self.b]])

    def applica(self, w, b):
        self.guadagni = w.copy()
        self.b = float(b)


def misure(y, punteggio):
    """Misure per blocco; None dove non sono definite."""
    pred = (punteggio > 0).astype(np.uint8)
    n = len(y)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    vp = int(((pred == 1) & (y == 1)).sum()); vn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out = {'righe': n, 'normali': n0, 'attacchi': n1,
           'accuratezza': (vp + vn) / n if n else NA,
           'richiamo_attacchi': vp / n1 if n1 else NA,
           'richiamo_normali': vn / n0 if n0 else NA,
           'falsi_allarmi': fp / n0 if n0 else NA,
           'vp': vp, 'vn': vn, 'fp': fp, 'fn': fn}
    if n0 and n1:
        ordine = np.argsort(punteggio)
        rango = np.empty(n, dtype=float)
        rango[ordine] = np.arange(1, n + 1)
        # ranghi medi sui pari merito
        _, inizio, conteggi = np.unique(punteggio[ordine], return_index=True, return_counts=True)
        for i0, c in zip(inizio, conteggi):
            if c > 1:
                rango[ordine[i0:i0 + c]] = rango[ordine[i0:i0 + c]].mean()
        somma = rango[y == 1].sum()
        out['auroc'] = (somma - n1 * (n1 + 1) / 2) / (n0 * n1)
    else:
        out['auroc'] = NA
    return out


def campiona(modo, punteggio_guida, quanti, rng):
    """Indici da etichettare dentro il blocco. Nessun uso delle etichette."""
    n = len(punteggio_guida)
    if quanti <= 0:
        return np.array([], dtype=int)
    if modo == 'uniforme' or quanti >= n:
        return rng.choice(n, min(quanti, n), replace=False)
    # dieci fasce di uguale ampiezza sul rango del punteggio
    fasce = 10
    ordine = np.argsort(punteggio_guida, kind='stable')
    pezzi = np.array_split(ordine, fasce)
    per_fascia = quanti // fasce
    resto = quanti - per_fascia * fasce
    scelti = []
    for i, pezzo in enumerate(pezzi):
        k = per_fascia + (1 if i < resto else 0)
        k = min(k, len(pezzo))
        if k:
            scelti.append(rng.choice(pezzo, k, replace=False))
    scelti = np.concatenate(scelti) if scelti else np.array([], dtype=int)
    if len(scelti) < quanti:      # fasce piccole: completa a caso fra i restanti
        restanti = np.setdiff1d(np.arange(n), scelti, assume_unique=False)
        if len(restanti):
            extra = rng.choice(restanti, min(quanti - len(scelti), len(restanti)),
                               replace=False)
            scelti = np.concatenate([scelti, extra])
    return np.sort(scelti)


def aggiorna(rappr, y, ridge=1.0):
    """Rifitta i pochi parametri sulla memoria. None se la memoria e monoclasse."""
    if len(np.unique(y)) < 2:
        return None
    m = LogisticRegression(max_iter=1000, C=1.0 / ridge)
    m.fit(rappr, y)
    return m.coef_.ravel().copy(), float(m.intercept_[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--iniziale', required=True, help='flusso A .npz, per l addestramento iniziale')
    p.add_argument('--flusso', required=True, help='flusso su cui fare il replay, .npz')
    p.add_argument('--uscita', required=True)
    p.add_argument('--blocco', type=int, default=10000)
    p.add_argument('--budget', type=float, default=0.01)
    p.add_argument('--memoria', type=int, default=256)
    p.add_argument('--ritardo', type=int, default=1,
                   help='blocchi di attesa: le etichette di k arrivano a fine k+ritardo')
    p.add_argument('--seme', type=int, default=42)
    p.add_argument('--max-blocchi', type=int, default=None, help='per uno smoke breve')
    p.add_argument('--campionamento', choices=('uniforme', 'strati_punteggio'),
                   default='uniforme',
                   help=("uniforme: l'1%% estratto a caso, come prescrive la scheda. "
                         "strati_punteggio: lo stesso budget, ma distribuito su dieci "
                         "fasce del punteggio della LR CONGELATA. Usa solo informazione "
                         "disponibile al momento della predizione, non le etichette, e "
                         "poiche' il punteggio di riferimento e' uno solo gli indici "
                         "restano gli STESSI per tutti i metodi."))
    a = p.parse_args(argv)

    t0 = time.monotonic()
    A = np.load(a.iniziale, allow_pickle=False)
    S = np.load(a.flusso, allow_pickle=False)
    XA, yA = A['X'], A['y']
    X, y, rid, frt = S['X'], S['y'], S['row_id'], S['frt']
    tipi = [str(t) for t in S['tipi']]
    tp = S['tipo']
    if not np.all(np.diff(frt) >= 0):
        raise SystemExit('il flusso non e ordinato per feature_ready_time')

    tr = Trasformazione(XA)
    ZA = tr(XA)
    print('addestramento iniziale su %d righe di A (normali %d, attacchi %d)'
          % (len(yA), int((yA == 0).sum()), int((yA == 1).sum())), flush=True)

    modelli = {}
    for cls in (ModelloLR, ModelloMLP, ModelloKAN):
        t1 = time.monotonic()
        modelli[cls.nome] = cls(ZA, yA, a.seme)
        print('  %-4s addestrato in %.1fs' % (cls.nome, time.monotonic() - t1), flush=True)

    # copia congelata: gli stessi parametri iniziali, mai aggiornati
    congelati = {n: np.array(m.parametri(), copy=True) for n, m in modelli.items()}
    adattivi = {n: np.array(m.parametri(), copy=True) for n, m in modelli.items()}

    # memoria FIFO inizializzata da A
    rng0 = np.random.default_rng(a.seme)
    sem = rng0.choice(len(yA), min(a.memoria, len(yA)), replace=False)
    memoria_X = [ZA[i] for i in sem]
    memoria_y = [int(yA[i]) for i in sem]
    print('memoria iniziale: %d esempi da A, normali %d'
          % (len(memoria_y), sum(1 for v in memoria_y if v == 0)), flush=True)

    n = len(y)
    n_blocchi = (n + a.blocco - 1) // a.blocco
    if a.max_blocchi:
        n_blocchi = min(n_blocchi, a.max_blocchi)

    in_attesa = {}          # blocco -> (indici campionati)
    righe = []
    salti = {nm: 0 for nm in modelli}
    etichette_spese = 0

    for k in range(n_blocchi):
        i0, i1 = k * a.blocco, min((k + 1) * a.blocco, n)
        Zb = tr(X[i0:i1]); yb = y[i0:i1]

        # 1. PREVEDI, con i parametri correnti; nulla viene ricalcolato dopo
        voce = {'blocco': k, 'righe': int(i1 - i0),
                'frt_inizio_utc': float(frt[i0]), 'frt_fine_utc': float(frt[i1 - 1]),
                'tipi_presenti': sorted({tipi[t] for t in tp[i0:i1]})}
        for nm, m in modelli.items():
            R = m.rappresentazione(Zb)
            pc = R @ congelati[nm][:-1] + congelati[nm][-1]
            pa = R @ adattivi[nm][:-1] + adattivi[nm][-1]
            voce['congelato_' + nm] = misure(yb, pc)
            voce['adattivo_' + nm] = misure(yb, pa)

        # 2. RICHIEDI l'1%, stessi indici per tutti i metodi
        quanti = int(a.budget * (i1 - i0))
        rng = np.random.default_rng(a.seme + k)
        # punteggio guida: quello della LR CONGELATA, unico e uguale per tutti i
        # metodi, cosi' gli indici campionati restano gli stessi. Non dipende
        # dallo stato di adattamento, quindi non introduce retroazione.
        guida = (modelli['lr'].rappresentazione(Zb) @ congelati['lr'][:-1]
                 + congelati['lr'][-1])
        scelti = campiona(a.campionamento, guida, quanti, rng)
        in_attesa[k] = (i0 + scelti)
        etichette_spese += quanti
        voce['etichette_richieste'] = quanti
        voce['row_id_campionati_sha'] = int(np.sum(rid[i0 + scelti]) % 10**9) if quanti else 0

        # 3. ATTENDI: a fine blocco k arrivano le etichette del blocco k-ritardo
        pronte = k - a.ritardo
        voce['etichette_arrivate_dal_blocco'] = pronte if pronte >= 0 else NA
        if pronte >= 0 and pronte in in_attesa:
            idx = in_attesa.pop(pronte)
            for i in idx:
                memoria_X.append(tr(X[i:i + 1])[0]); memoria_y.append(int(y[i]))
            del memoria_X[:max(0, len(memoria_X) - a.memoria)]
            del memoria_y[:max(0, len(memoria_y) - a.memoria)]
            MX = np.array(memoria_X); My = np.array(memoria_y)
            voce['memoria_normali'] = int((My == 0).sum())
            voce['memoria_attacchi'] = int((My == 1).sum())
            # 4. AGGIORNA: incide dal blocco k+1, cioe' dal (pronte+ritardo+1)
            for nm, m in modelli.items():
                m.applica(adattivi[nm][:-1], adattivi[nm][-1])
                R = m.rappresentazione(MX)
                nuovo = aggiorna(R, My)
                if nuovo is None:
                    salti[nm] += 1
                    voce['salto_' + nm] = True
                else:
                    adattivi[nm] = np.concatenate([nuovo[0], [nuovo[1]]])
                    voce['salto_' + nm] = False
        righe.append(voce)
        if k % 50 == 0:
            print('  blocco %4d/%d  etichette spese %d' % (k, n_blocchi, etichette_spese), flush=True)

    # ------- riepilogo -------
    def media(nm, variante, campo):
        v = [r[variante + '_' + nm][campo] for r in righe
             if r[variante + '_' + nm][campo] is not None]
        return float(np.mean(v)) if v else NA

    riepilogo = {}
    for nm in modelli:
        riepilogo[nm] = {
            'salti_per_memoria_monoclasse': salti[nm],
            'congelato': {c: media(nm, 'congelato', c) for c in
                          ('accuratezza', 'richiamo_attacchi', 'richiamo_normali',
                           'falsi_allarmi', 'auroc')},
            'adattivo': {c: media(nm, 'adattivo', c) for c in
                         ('accuratezza', 'richiamo_attacchi', 'richiamo_normali',
                          'falsi_allarmi', 'auroc')},
        }
    fuori = {
        'parametri': {'blocco': a.blocco, 'budget': a.budget, 'memoria': a.memoria,
                      'ritardo_in_blocchi': a.ritardo, 'seme': a.seme,
                      'max_blocchi': a.max_blocchi},
        'flusso': {'file': Path(a.flusso).name, 'righe': int(n),
                   'blocchi_eseguiti': n_blocchi,
                   'normali': int((y == 0).sum()), 'attacchi': int((y == 1).sum())},
        'iniziale': {'file': Path(a.iniziale).name, 'righe': int(len(yA)),
                     'normali': int((yA == 0).sum())},
        'trasformazione': 'log1p poi standardizzazione, stimata su A',
        'campionamento': a.campionamento,
        'etichette_spese': etichette_spese,
        'quota_etichette_effettiva': etichette_spese / (n_blocchi * a.blocco),
        'riepilogo': riepilogo,
        'secondi': round(time.monotonic() - t0, 1),
        'per_blocco': righe,
    }
    u = Path(a.uscita)
    u.parent.mkdir(parents=True, exist_ok=True)
    u.write_text(json.dumps(fuori, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                 encoding='utf-8')

    print()
    print('blocchi %d   etichette spese %d (%.4f delle righe)'
          % (n_blocchi, etichette_spese, fuori['quota_etichette_effettiva']))
    print('%-5s %-10s %10s %10s %10s %10s' % ('', '', 'accur.', 'ric.att', 'ric.norm', 'auroc'))
    def fmt(x):
        return 'NA' if x is None else '%.4f' % x

    for nm in modelli:
        for v in ('congelato', 'adattivo'):
            r = riepilogo[nm][v]
            print('%-5s %-10s %10s %10s %10s %10s'
                  % (nm, v, fmt(r['accuratezza']), fmt(r['richiamo_attacchi']),
                     fmt(r['richiamo_normali']), fmt(r['auroc'])))
        print('%-5s salti per memoria monoclasse: %d' % (nm, salti[nm]))
    print('scritto ' + str(u))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

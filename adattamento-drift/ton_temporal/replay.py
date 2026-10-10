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
import hashlib
import json
import os
import platform
import sys
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


def descrizione_hardware():
    """Hardware su cui la corsa e' avvenuta, letto dal sistema, non dichiarato a mano.

    Deve funzionare su Linux e su Windows, perche' le corse si fanno su una
    macchina e le prove si eseguono anche sull'altra. Ogni voce porta accanto il
    **metodo** con cui e' stata ottenuta: quando una lettura non e' possibile il
    rendiconto dice perche', invece di lasciare un campo vuoto che sembra una
    misura mancata.
    """
    d = {'cpu': 'non leggibile', 'cpu_thread_visibili': NA,
         'ram_totale_mib': NA, 'metodo_cpu': 'nessuno', 'metodo_ram': 'nessuno'}
    # --- CPU
    try:
        testo = Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='replace')
        nomi = [r.split(':', 1)[1].strip() for r in testo.splitlines()
                if r.startswith('model name')]
        if nomi:
            d['cpu'] = nomi[0]
            d['cpu_thread_visibili'] = len(nomi)
            d['metodo_cpu'] = '/proc/cpuinfo'
    except OSError:
        pass
    if d['metodo_cpu'] == 'nessuno':
        nome = (os.environ.get('PROCESSOR_IDENTIFIER')
                or platform.processor() or platform.machine())
        if nome:
            d['cpu'] = nome.strip()
            d['metodo_cpu'] = ('PROCESSOR_IDENTIFIER'
                               if os.environ.get('PROCESSOR_IDENTIFIER')
                               else 'platform.processor')
        if os.cpu_count():
            d['cpu_thread_visibili'] = os.cpu_count()
    # --- RAM totale
    try:
        for r in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if r.startswith('MemTotal'):
                d['ram_totale_mib'] = round(int(r.split()[1]) / 1024, 1)
                d['metodo_ram'] = '/proc/meminfo'
                break
    except OSError:
        pass
    if d['metodo_ram'] == 'nessuno' and sys.platform == 'win32':
        try:
            import ctypes

            class _Memoria(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

            m = _Memoria()
            m.dwLength = ctypes.sizeof(_Memoria)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                d['ram_totale_mib'] = round(m.ullTotalPhys / 1024 ** 2, 1)
                d['metodo_ram'] = 'GlobalMemoryStatusEx'
        except Exception:
            pass
    if d['metodo_ram'] == 'nessuno':
        try:
            n = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
            d['ram_totale_mib'] = round(n / 1024 ** 2, 1)
            d['metodo_ram'] = 'os.sysconf'
        except (AttributeError, ValueError, OSError):
            pass
    d['cpu_utilizzabili_dal_processo'] = (
        len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
        else (os.cpu_count() or NA))
    d['thread_blas'] = {v: os.environ.get(v) for v in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    return d


def picco_rss(forza_senza_resource=False):
    """Picco di memoria residente del processo, in MiB, col metodo usato.

    Ritorna (valore, metodo). Il valore e' None solo se nessuna via e'
    disponibile, e in quel caso il metodo lo dichiara: un campo vuoto senza
    spiegazione si leggerebbe come una misura andata male.

    `forza_senza_resource` serve alle prove per esercitare la via di riserva
    anche dove `resource` esiste.
    """
    if not forza_senza_resource:
        try:
            import resource
        except ImportError:
            pass
        else:
            v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux conta in kibibyte, macOS in byte
            fattore = 1024 if sys.platform.startswith('linux') else 1024 ** 2
            return round(v / fattore, 1), 'resource.getrusage'
    if sys.platform == 'win32':
        try:
            import ctypes

            class _Contatori(ctypes.Structure):
                _fields_ = [('cb', ctypes.c_ulong),
                            ('PageFaultCount', ctypes.c_ulong),
                            ('PeakWorkingSetSize', ctypes.c_size_t),
                            ('WorkingSetSize', ctypes.c_size_t),
                            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                            ('PagefileUsage', ctypes.c_size_t),
                            ('PeakPagefileUsage', ctypes.c_size_t)]

            c = _Contatori()
            c.cb = ctypes.sizeof(_Contatori)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            if ok:
                return round(c.PeakWorkingSetSize / 1024 ** 2, 1), 'GetProcessMemoryInfo'
        except Exception:
            pass
    return NA, 'non disponibile su questa piattaforma'


def picco_rss_mib():
    """Il solo valore, per compatibilita' con chi legge un numero."""
    return picco_rss()[0]


def riassumi_tempi(valori):
    """n, totale e quantili dei tempi di un singolo aggiornamento, in millisecondi."""
    if not valori:
        return {'n': 0, 'secondi_totali': 0.0, 'ms_medio': NA, 'ms_mediano': NA,
                'ms_minimo': NA, 'ms_massimo': NA}
    v = sorted(valori)
    return {'n': len(v),
            'secondi_totali': round(sum(v), 3),
            'ms_medio': round(1000 * sum(v) / len(v), 3),
            'ms_mediano': round(1000 * v[(len(v) - 1) // 2], 3),
            'ms_minimo': round(1000 * v[0], 3),
            'ms_massimo': round(1000 * v[-1], 3)}


def impronta_row_id(row_id):
    """SHA-256 della sequenza dei row_id campionati, su una serializzazione dichiarata.

    Serializzazione: i row_id **nell'ordine in cui il campionatore li ha
    restituiti**, convertiti in interi decimali senza zeri iniziali, uniti da una
    virgola, codificati in UTF-8. L'ordine e' parte dell'impronta di proposito:
    due politiche devono ricevere gli stessi record **nello stesso ordine**, e un
    digest insensibile all'ordine non lo verificherebbe.

    Sostituisce un campo precedente che conteneva `sum(row_id) % 10**9`. Quella
    somma era insensibile all'ordine e alle permutazioni, e due insiemi diversi
    con la stessa somma si costruiscono scambiando +1 e -1 fra due id: usarla
    come prova che gli indici campionati coincidono era un controllo molto piu'
    debole di quanto il nome `_sha` suggerisse.
    """
    if len(row_id) == 0:
        return hashlib.sha256(b'').hexdigest()
    testo = ','.join(str(int(x)) for x in row_id)
    return hashlib.sha256(testo.encode('utf-8')).hexdigest()


def punteggio_a_blocchi(modello, trasformazione, X, parametri, blocco=200_000):
    """Punteggio del modello su un flusso intero, senza tenerne in memoria la
    rappresentazione: quella della base B-spline su milioni di righe pesa
    gigabyte, e il processo viene ucciso senza messaggio."""
    fuori = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), blocco):
        R = modello.rappresentazione(trasformazione(X[i:i + blocco]))
        fuori[i:i + blocco] = R @ parametri[:-1] + parametri[-1]
    return fuori


def _carica_soglia():
    """soglia_bilanciata.py accanto a questo file, senza dipendere da sys.path."""
    import importlib.util
    percorso = Path(__file__).resolve().parent / 'soglia_bilanciata.py'
    if not percorso.is_file():
        raise SystemExit(f'manca {percorso}, necessario per --calibrazione')
    spec = importlib.util.spec_from_file_location('soglia_bilanciata', percorso)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules['soglia_bilanciata'] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def misure(y, punteggio, soglia=0.0):
    """Misure per blocco; None dove non sono definite.

    `soglia` e' il punto di decisione: la predizione e' `punteggio > soglia`.
    Con il valore predefinito 0.0 si ottiene il riferimento SENZA calibrazione.
    Con --calibrazione la soglia viene scelta su B e resta identica e fissa
    nella coppia congelato / adattivo di ciascun modello; l'AUROC non dipende
    dalla soglia, quindi cambiano solo le misure di decisione.
    """
    pred = (punteggio > soglia).astype(np.uint8)
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
    out['auroc'] = auroc(y, punteggio)
    return out


def auroc(y, punteggio):
    """AUROC con ranghi medi sui pari merito; None se manca una delle due classi.

    Estratta da misure() perche' la usa anche la stima del verso sulle etichette
    gia' arrivate: le due devono essere LA STESSA funzione, altrimenti il
    confronto fra la stima e il valore vero non misurerebbe la stima.
    """
    y = np.asarray(y)
    punteggio = np.asarray(punteggio, dtype=float)
    n = len(y)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    if not (n0 and n1):
        return NA
    ordine = np.argsort(punteggio)
    rango = np.empty(n, dtype=float)
    rango[ordine] = np.arange(1, n + 1)
    _, inizio, conteggi = np.unique(punteggio[ordine], return_index=True, return_counts=True)
    for i0, c in zip(inizio, conteggi):
        if c > 1:
            rango[ordine[i0:i0 + c]] = rango[ordine[i0:i0 + c]].mean()
    somma = rango[y == 1].sum()
    return (somma - n1 * (n1 + 1) / 2) / (n0 * n1)


def scrivi_compatto(documento):
    """JSON con i metadati indentati e UN BLOCCO PER RIGA.

    Con l'indentazione piena un rendiconto di 916 blocchi occupa 93.000 righe, e
    un commit che ne porta una decina rende il diff del PR illeggibile a occhio
    pur pesando pochi megabyte compressi. Qui i metadati restano indentati,
    perche' sono quelli che si leggono, e ogni blocco sta su una riga sola: il
    file resta JSON valido — `json.load` lo apre senza sapere nulla di questo —
    diventa circa 930 righe, e un diff mostra esattamente quali blocchi sono
    cambiati invece di spalmare la differenza su migliaia di righe.
    """
    blocchi = documento.get('per_blocco')
    testa = {k: v for k, v in documento.items() if k != 'per_blocco'}
    pezzi = [json.dumps(testa, ensure_ascii=False, indent=1, allow_nan=False)[:-2]]
    if blocchi is None:
        return pezzi[0] + '\n}\n'
    pezzi.append(',\n "per_blocco": [\n')
    righe = ['  ' + json.dumps(b, ensure_ascii=False, allow_nan=False) for b in blocchi]
    pezzi.append(',\n'.join(righe))
    pezzi.append('\n ]\n}\n')
    return ''.join(pezzi)


def _blocchi_ammissibili(percorso, n_blocchi, ritardo):
    """I blocchi in cui un aggiornamento e' davvero possibile.

    Il controllo negativo deve pareggiare il numero di aggiornamenti
    APPLICATI. Estraendo a sorte fra tutti i blocchi, circa il 43% delle
    estrazioni finisce su un blocco in cui la memoria contiene una sola classe,
    l'aggiornamento viene saltato dalla guardia, e il controllo ne applica quasi
    la meta' di quelli che dovrebbe: misurerebbe la differenza fra i conteggi
    invece della differenza fra i criteri.

    La memoria e' riempita dalle etichette che arrivano, non dalla politica,
    quindi l'insieme dei blocchi ammissibili e' lo stesso per tutte le politiche
    e per tutti i modelli dello stesso seme. Lo si legge dal rendiconto della
    corsa `ogni_blocco`, che per ogni blocco registra se l'aggiornamento e'
    stato applicato o saltato.
    """
    if percorso is None:
        return np.arange(max(0, n_blocchi - ritardo))
    with open(percorso, encoding='utf-8') as f:
        righe = json.load(f)['per_blocco']
    ammissibili = []
    for voce in righe:
        salti_modelli = {voce.get('salto_' + nm) for nm in ('lr', 'mlp', 'kan')}
        if salti_modelli == {False}:
            # il blocco le cui etichette erano arrivate, non quello corrente
            pronte = voce.get('etichette_arrivate_dal_blocco')
            if pronte is not None:
                ammissibili.append(int(pronte))
        elif len(salti_modelli - {None}) > 1:
            raise SystemExit(
                f"blocco {voce.get('blocco')}: la guardia sulla memoria non e' "
                "uguale per i tre modelli, l'insieme ammissibile non e' unico")
    return np.array(sorted(set(ammissibili)))


def _quanti_per_modello(testo, modelli):
    """Un numero per tutti i modelli, oppure "lr=54,mlp=20,kan=23"."""
    testo = str(testo)
    if '=' not in testo:
        return {nm: int(testo) for nm in modelli}
    fuori = {}
    for pezzo in testo.split(','):
        nome, _, valore = pezzo.partition('=')
        nome = nome.strip()
        if nome not in modelli:
            raise SystemExit(f'modello non riconosciuto in --quanti-aggiornamenti: {nome}')
        fuori[nome] = int(valore)
    mancanti = set(modelli) - set(fuori)
    if mancanti:
        raise SystemExit(f'--quanti-aggiornamenti non copre: {sorted(mancanti)}')
    return fuori


def _seme_politica_effettivo(seme, seme_politica):
    """Seme del calendario della politica, con default storico."""
    return seme if seme_politica is None else seme_politica


def scegli_blocchi_casuali(modelli, candidati, quanti_per_modello, seme, seme_politica=None):
    """Calendario del controllo casuale, separato dagli altri usi di `--seme`.

    Se `seme_politica` non e' passato, il seme effettivo coincide con `seme`:
    e' il comportamento dei rendiconti gia' pubblicati. Passarlo cambia solo i
    blocchi scelti dal controllo negativo, non inizializzazione, campionamento
    delle etichette, soglie o conteggi richiesti.
    """
    seme_pol = _seme_politica_effettivo(seme, seme_politica)
    fuori = {}
    for nm in modelli:
        # un generatore per modello, indipendente dal campionamento: il
        # controllo non deve dipendere da quali righe sono etichettate
        rng_pol = np.random.default_rng(
            10_000 + seme_pol * 97 + sum(ord(c) for c in nm))
        quanti = min(quanti_per_modello[nm], len(candidati))
        fuori[nm] = set(int(x) for x in
                        rng_pol.choice(candidati, quanti, replace=False))
    return fuori


def decidi(politica, modello, blocco_pronte, righe, blocchi_scelti):
    """Se aggiornare, con la sola informazione disponibile al momento.

    Tre politiche, piu' un controllo negativo:

      ogni_blocco           aggiorna sempre che si possa. E' quella del §10.
      evidenza_inversione   aggiorna solo se la stima FUORI CAMPIONE del blocco
                            le cui etichette sono appena arrivate dice che il
                            verso e' rovesciato. Se la stima non esiste — fra le
                            righe campionate manca una classe — la decisione e'
                            `non_disponibile`, e NON viene confusa con
                            «nessuna evidenza»: la prima dice che non si sa, la
                            seconda che si sa e non c'e'.
      casuale               aggiorna su un insieme di blocchi estratto a sorte e
                            fissato in anticipo. E' il controllo negativo: se
                            non viene battuto dalla politica su evidenza, allora
                            quello che conta e' il NUMERO di aggiornamenti e non
                            l'inversione.

    Non legge nulla del blocco corrente: solo il rendiconto del blocco
    `blocco_pronte`, che e' gia' stato predetto e le cui etichette sono arrivate.
    """
    if politica == 'ogni_blocco':
        return 'procedi'
    if politica == 'casuale':
        return 'procedi' if blocco_pronte in blocchi_scelti[modello] else 'non_scelto'
    if politica != 'evidenza_inversione':
        raise SystemExit(f'politica non riconosciuta: {politica}')
    if blocco_pronte >= len(righe):
        return 'non_disponibile'
    stima = righe[blocco_pronte].get('stima_campione_' + modello)
    if stima is None or stima['auroc_sul_campione'] is None:
        return 'non_disponibile'
    return 'procedi' if stima['auroc_sul_campione'] < 0.5 else 'nessuna_evidenza'


def stima_verso(modello, parametri, memoria_X, memoria_y):
    """Il verso stimato dalle SOLE etichette gia' arrivate, cioe' dalla memoria.

    Risponde alla domanda che il referente pone come primo passo: il verso e'
    stimabile in esercizio? L'AUROC del blocco corrente richiede le etichette di
    tutto il blocco e non e' disponibile; questa usa i soli esempi in memoria,
    che sono quelli le cui etichette sono arrivate.

    Quando in memoria manca una delle due classi la decisione e' **non
    disponibile**, e non va confusa con «non invertito»: sono due cose diverse e
    il rendiconto le tiene separate.
    """
    y = np.asarray(memoria_y)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    if not (n0 and n1):
        return {'disponibile': False, 'auroc_in_memoria': NA,
                'normali_in_memoria': n0, 'attacchi_in_memoria': n1,
                'verso_stimato_invertito': NA}
    Z = np.asarray(memoria_X)
    p = modello.rappresentazione(Z) @ parametri[:-1] + parametri[-1]
    a = auroc(y, p)
    return {'disponibile': True, 'auroc_in_memoria': float(a),
            'normali_in_memoria': n0, 'attacchi_in_memoria': n1,
            'verso_stimato_invertito': bool(a < 0.5)}


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
    p.add_argument('--seme-politica', type=int, default=None,
                   help='solo per --politica casuale: seme del calendario dei '
                        'blocchi scelti. Se omesso coincide con --seme, per '
                        'riprodurre i rendiconti storici')
    p.add_argument('--max-blocchi', type=int, default=None, help='per uno smoke breve')
    p.add_argument('--politica', default='ogni_blocco',
                   choices=('ogni_blocco', 'evidenza_inversione', 'casuale'),
                   help='quando applicare l aggiornamento')
    p.add_argument('--ammissibili', default=None,
                   help='solo per --politica casuale: il rendiconto della corsa '
                        'ogni_blocco dello stesso seme, da cui leggere i blocchi '
                        'in cui un aggiornamento e davvero possibile')
    p.add_argument('--quanti-aggiornamenti', default=None,
                   help='solo per --politica casuale: quanti blocchi scegliere, '
                        'un numero per tutti o "lr=54,mlp=20,kan=23"')
    p.add_argument('--calibrazione', default=None,
                   help='flusso B .npz: la soglia di decisione viene scelta su B '
                        'col massimo della balanced accuracy, invece di restare a zero')
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
    costo_addestramento = {}
    for cls in (ModelloLR, ModelloMLP, ModelloKAN):
        t1 = time.monotonic()
        modelli[cls.nome] = cls(ZA, yA, a.seme)
        costo_addestramento[cls.nome] = round(time.monotonic() - t1, 2)
        print('  %-4s addestrato in %.1fs' % (cls.nome, costo_addestramento[cls.nome]), flush=True)
    secondi_iniziale = round(time.monotonic() - t0, 2)
    congelati_iniziali = {n: np.array(m.parametri(), copy=True) for n, m in modelli.items()}

    # soglia di decisione: zero, oppure scelta su B col massimo della balanced
    # accuracy. In entrambi i casi e' UNA per modello e resta fissa, identica
    # per la copia congelata e per quella adattiva.
    soglie = {nm: 0.0 for nm in modelli}
    rendiconto_soglie = {'modo': 'zero', 'per_modello': {}}
    if a.calibrazione is not None:
        sb = _carica_soglia()
        B = np.load(a.calibrazione, allow_pickle=True)
        XB, yB = B['X'], B['y']
        rendiconto_soglie['modo'] = 'balanced_accuracy_su_B'
        rendiconto_soglie['file'] = Path(a.calibrazione).name
        rendiconto_soglie['righe_di_B'] = int(len(yB))
        for nm, m in modelli.items():
            # punteggi del modello CONGELATO su B: il modello adattivo su B non
            # esiste, perche' B precede C e non viene replayato. A blocchi:
            # la rappresentazione B-spline di milioni di righe non sta in memoria.
            pB = punteggio_a_blocchi(m, tr, XB, congelati_iniziali[nm])
            soglie[nm], dettaglio = sb.scegli_soglia(pB, yB)
            dettaglio['soglia'] = soglie[nm]
            rendiconto_soglie['per_modello'][nm] = dettaglio
            print('  %-4s soglia su B: %+.4f (balanced accuracy %.4f, '
                  '%d candidate a pari merito%s)'
                  % (nm, soglie[nm], dettaglio['balanced_accuracy'],
                     dettaglio['candidate_a_pari_merito'],
                     '' if dettaglio['candidate_contigue'] else ', NON contigue'),
                  flush=True)

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
    decisioni = {nm: {} for nm in modelli}
    blocchi_scelti = {nm: set() for nm in modelli}
    if a.politica == 'casuale':
        if a.quanti_aggiornamenti is None:
            raise SystemExit('--politica casuale richiede --quanti-aggiornamenti')
        # Il controllo negativo deve pareggiare il numero di aggiornamenti
        # MODELLO PER MODELLO: la politica su evidenza ne applica un numero
        # diverso per ciascuno, e confrontarla con un numero unico misurerebbe
        # la differenza fra i conteggi invece della differenza fra i criteri.
        quanti_per_modello = _quanti_per_modello(a.quanti_aggiornamenti, modelli)
        candidati = _blocchi_ammissibili(a.ammissibili, n_blocchi, a.ritardo)
        blocchi_scelti = scegli_blocchi_casuali(
            modelli, candidati, quanti_per_modello, a.seme, a.seme_politica)
    salti = {nm: 0 for nm in modelli}
    aggiornamenti = {nm: 0 for nm in modelli}
    etichette_spese = 0
    righe_scorse = 0
    # residuo 3: il tempo del SINGOLO aggiornamento, per modello, misurato sul
    # tratto che un sistema reale pagherebbe (rappresentazione della memoria +
    # rifitting dei parametri), separato dal costo del replay.
    tempi_aggiornamento = {nm: [] for nm in modelli}
    tempi_aggiornamento_saltati = {nm: [] for nm in modelli}
    byte_stato_adattamento = {nm: 0 for nm in modelli}
    byte_memoria_fifo = 0
    t_flusso = time.monotonic()

    for k in range(n_blocchi):
        i0, i1 = k * a.blocco, min((k + 1) * a.blocco, n)
        Zb = tr(X[i0:i1]); yb = y[i0:i1]
        righe_scorse += int(i1 - i0)

        # 1. PREVEDI, con i parametri correnti; nulla viene ricalcolato dopo
        voce = {'blocco': k, 'righe': int(i1 - i0),
                'frt_inizio_utc': float(frt[i0]), 'frt_fine_utc': float(frt[i1 - 1]),
                'tipi_presenti': sorted({tipi[t] for t in tp[i0:i1]})}
        punteggi_emessi = {}
        for nm, m in modelli.items():
            R = m.rappresentazione(Zb)
            pc = R @ congelati[nm][:-1] + congelati[nm][-1]
            pa = R @ adattivi[nm][:-1] + adattivi[nm][-1]
            s = soglie[nm]
            voce['congelato_' + nm] = misure(yb, pc, s)
            voce['adattivo_' + nm] = misure(yb, pa, s)
            # il verso stimato dalle sole etichette gia' arrivate, prima che
            # arrivino quelle di questo blocco: e' cio' che una politica
            # utilizzabile in esercizio avrebbe a disposizione adesso
            voce['stima_verso_' + nm] = stima_verso(
                m, adattivi[nm], memoria_X, memoria_y)
            punteggi_emessi[nm] = pa

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

        # stima FUORI CAMPIONE del verso: l'AUROC sulle sole righe campionate di
        # questo blocco, con i punteggi GIA' EMESSI. Il modello che li ha emessi
        # non aveva visto queste etichette, quindi la stima non e' in-campione
        # come quella sulla memoria. Diventera' disponibile alla fine del blocco
        # k+1, quando le etichette arrivano: e' quindi una stima utilizzabile per
        # decidere sui blocchi successivi, non su questo.
        for nm in modelli:
            voce['stima_campione_' + nm] = {
                'righe_campionate': int(len(scelti)),
                'normali_campionati': int((yb[scelti] == 0).sum()),
                'attacchi_campionati': int((yb[scelti] == 1).sum()),
                'auroc_sul_campione': auroc(yb[scelti], punteggi_emessi[nm][scelti]),
            }
        etichette_spese += quanti
        voce['etichette_richieste'] = quanti
        # Il campo storico e' una SOMMA modulo 10^9, non un digest: insensibile
        # all'ordine e banale da far collidere. Resta per continuita' con i
        # rendiconti gia' pubblicati, col nome che dice cosa e'; il confronto fra
        # politiche usa il digest vero qui sotto.
        voce['row_id_campionati_somma_storica'] = (
            int(np.sum(rid[i0 + scelti]) % 10**9) if quanti else 0)
        voce['row_id_campionati_sha256'] = impronta_row_id(rid[i0 + scelti])

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
                # la politica decide SE aggiornare, usando solo informazione
                # disponibile adesso: la stima fuori campione del blocco
                # `pronte`, le cui etichette sono appena arrivate.
                esito = decidi(a.politica, nm, pronte, righe, blocchi_scelti)
                voce['decisione_' + nm] = esito
                if esito != 'procedi':
                    decisioni[nm][esito] = decisioni[nm].get(esito, 0) + 1
                    voce['salto_' + nm] = False
                    continue
                t_agg = time.perf_counter()
                m.applica(adattivi[nm][:-1], adattivi[nm][-1])
                R = m.rappresentazione(MX)
                nuovo = aggiorna(R, My)
                d_agg = time.perf_counter() - t_agg
                voce['ms_aggiornamento_' + nm] = round(1000 * d_agg, 3)
                byte_memoria_fifo = max(byte_memoria_fifo, MX.nbytes + My.nbytes)
                byte_stato_adattamento[nm] = max(
                    byte_stato_adattamento[nm], R.nbytes + adattivi[nm].nbytes)
                if nuovo is None:
                    tempi_aggiornamento_saltati[nm].append(d_agg)
                    salti[nm] += 1
                    voce['salto_' + nm] = True
                    decisioni[nm]['saltato_memoria_monoclasse'] = \
                        decisioni[nm].get('saltato_memoria_monoclasse', 0) + 1
                else:
                    tempi_aggiornamento[nm].append(d_agg)
                    adattivi[nm] = np.concatenate([nuovo[0], [nuovo[1]]])
                    aggiornamenti[nm] += 1
                    voce['salto_' + nm] = False
                    decisioni[nm]['applicato'] = decisioni[nm].get('applicato', 0) + 1
        righe.append(voce)
        if k % 50 == 0:
            print('  blocco %4d/%d  etichette spese %d' % (k, n_blocchi, etichette_spese), flush=True)

    # ------- riepilogo -------
    def media(nm, variante, campo):
        v = [r[variante + '_' + nm][campo] for r in righe
             if r[variante + '_' + nm][campo] is not None]
        return float(np.mean(v)) if v else NA

    secondi_flusso = round(time.monotonic() - t_flusso, 2)
    riepilogo = {}
    for nm in modelli:
        riepilogo[nm] = {
            'salti_per_memoria_monoclasse': salti[nm],
            'aggiornamenti_applicati': aggiornamenti[nm],
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
                      'seme_politica': _seme_politica_effettivo(
                          a.seme, a.seme_politica),
                      'seme_politica_esplicito': a.seme_politica is not None,
                      'max_blocchi': a.max_blocchi},
        'flusso': {'file': Path(a.flusso).name, 'righe': int(n),
                   'blocchi_eseguiti': n_blocchi,
                   'normali': int((y == 0).sum()), 'attacchi': int((y == 1).sum())},
        'iniziale': {'file': Path(a.iniziale).name, 'righe': int(len(yA)),
                     'normali': int((yA == 0).sum())},
        'trasformazione': 'log1p poi standardizzazione, stimata su A',
        'campionamento': a.campionamento,
        'politica': {'nome': a.politica,
                     'seme': _seme_politica_effettivo(a.seme, a.seme_politica)
                     if a.politica == 'casuale' else None,
                     'quanti_aggiornamenti_richiesti': a.quanti_aggiornamenti,
                     'blocchi_scelti': {nm: sorted(v) for nm, v in blocchi_scelti.items()
                                        if v} or None},
        'soglia_di_decisione': rendiconto_soglie,
        'etichette_spese': etichette_spese,
        # righe EFFETTIVAMENTE scorse, non blocchi x dimensione: l'ultimo blocco
        # e' incompleto e il prodotto sovrastima il denominatore
        'righe_scorse': int(righe_scorse),
        'quota_etichette_effettiva': etichette_spese / righe_scorse,
        'quota_etichette_denominatore': 'righe effettivamente scorse nel replay',
        'riepilogo': riepilogo,
        'ambiente': {
            'hardware_misurato': descrizione_hardware(),
            'python': platform.python_version(),
            'implementazione': platform.python_implementation(),
            'sistema': platform.system() + ' ' + platform.release(),
            'macchina': platform.machine(),
            'numpy': np.__version__,
            'scikit_learn': __import__('sklearn').__version__,
            'hardware': 'solo CPU: nessuna libreria di accelerazione usata',
        },
        'comando': ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]),
        'costi': {
            'secondi_totali': round(time.monotonic() - t0, 2),
            'secondi_addestramento_iniziale': secondi_iniziale,
            'secondi_per_modello_iniziale': costo_addestramento,
            'secondi_replay': secondi_flusso,
            'secondi_per_blocco': round(secondi_flusso / max(1, n_blocchi), 4),
            'etichette_richieste': etichette_spese,
            'etichette_per_blocco': int(a.budget * a.blocco),
            'aggiornamenti_applicati': aggiornamenti,
            'aggiornamenti_saltati': salti,
            'decisioni_della_politica': decisioni,
            'tempo_del_singolo_aggiornamento': {
                nm: riassumi_tempi(tempi_aggiornamento[nm]) for nm in modelli},
            'tempo_degli_aggiornamenti_saltati': {
                nm: riassumi_tempi(tempi_aggiornamento_saltati[nm]) for nm in modelli},
            'tempo_aggiornamento_cosa_comprende':
                'applica + rappresentazione(memoria) + rifitting dei parametri, '
                'cronometrato con time.perf_counter attorno a quel solo tratto; '
                'escluso il punteggio del blocco, che si paga in ogni politica',
            'memoria': {
                'picco_rss_processo_mib': picco_rss()[0],
                'picco_rss_metodo': picco_rss()[1],
                'byte_memoria_fifo': int(byte_memoria_fifo),
                'byte_stato_per_modello': {nm: int(v) for nm, v
                                           in byte_stato_adattamento.items()},
                'nota': 'il picco RSS e dominato dagli array dello stream tenuti '
                        'in memoria dal banco di prova, non dall\'adattamento: non '
                        'e la memoria che un sistema in linea userebbe',
            },
            'avvertenza': 'il numero di aggiornamenti non dimostra un guadagno di '
                          'latenza ne di memoria: il costo per aggiornamento e '
                          'misurato qui e va moltiplicato per il numero, mentre il '
                          'punteggio di ogni blocco si paga in tutte le politiche',
        },
        'per_blocco': righe,
    }
    u = Path(a.uscita)
    u.parent.mkdir(parents=True, exist_ok=True)
    u.write_text(scrivi_compatto(fuori), encoding='utf-8')

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

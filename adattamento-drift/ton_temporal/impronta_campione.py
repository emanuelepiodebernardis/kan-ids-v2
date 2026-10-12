"""Identita' delle righe campionate fra due esecuzioni.

Perche' esiste questo modulo
----------------------------
Il confronto fra due esecuzioni (soglia a zero contro soglia calibrata, oppure
politiche diverse sullo stesso flusso) si regge su un presupposto: le righe
etichettate sono le stesse, nello stesso ordine, in tutte le esecuzioni
confrontate. Se non lo fossero, le differenze misurate potrebbero venire dal
budget speso su righe diverse e non da cio' che si vuole confrontare.

Il campo che serviva a verificarlo si chiamava `row_id_campionati_sha` ma
conteneva `sum(row_id) % 10**9`. Una somma non e' un digest:

  * e' insensibile all'ordine: [1, 2, 3] e [3, 1, 2] danno lo stesso valore,
    mentre per il presupposto l'ordine conta;
  * le collisioni si costruiscono a mano scambiando +1 e -1 fra due indici:
    {10, 20, 30} e {11, 19, 30} hanno la stessa somma;
  * e' ridotta modulo 10**9, quindi lo spazio dei valori e' piccolo.

Il campo resta nei rendiconti gia' pubblicati e li' va letto per quello che e'.
Dalle esecuzioni successive c'e' anche `row_id_campionati_sha256`, che e' lo
SHA-256 di una serializzazione dichiarata: i `row_id` nell'ordine restituito dal
campionatore, interi decimali, separati da virgola, codificati in UTF-8.

Le funzioni qui sotto confrontano due voci di blocco usando il digest quando
c'e' in entrambe, e dicono sempre su quale dei due campi si sono basate: un
confronto passato sulla somma storica non vale quanto uno passato sul digest, e
lo strumento che lo usa deve poterlo scrivere nel proprio rendiconto.
"""

from __future__ import annotations

import hashlib

CAMPO_DIGEST = 'row_id_campionati_sha256'
CAMPO_SOMMA = 'row_id_campionati_somma_storica'
CAMPO_SOMMA_NOME_VECCHIO = 'row_id_campionati_sha'

# forza del confronto, dalla piu' debole alla piu' forte
ASSENTE = 'assente'
SOMMA_STORICA = 'somma_storica'
DIGEST = 'digest'


def serializza(row_id) -> str:
    """La serializzazione dichiarata: ordine di campionamento, decimali, virgole."""
    return ','.join(str(int(x)) for x in row_id)


def impronta(row_id) -> str:
    """SHA-256 della serializzazione dichiarata. Stringa vuota -> digest del vuoto."""
    return hashlib.sha256(serializza(row_id).encode('utf-8')).hexdigest()


def somma_storica(row_id) -> int:
    """Il vecchio valore, riprodotto qui per poterne mostrare i limiti nei test."""
    return int(sum(int(x) for x in row_id) % 10 ** 9)


def _valori(voce):
    """(digest, somma) della voce, leggendo anche il vecchio nome del campo."""
    d = voce.get(CAMPO_DIGEST)
    s = voce.get(CAMPO_SOMMA)
    if s is None:
        s = voce.get(CAMPO_SOMMA_NOME_VECCHIO)
    return d, s


def confronta(x, y):
    """Confronta due voci di blocco.

    Ritorna (uguali, forza). `forza` dice su che cosa si e' basato il confronto:
    DIGEST se entrambe le voci hanno lo SHA-256, SOMMA_STORICA se si e' potuto
    usare soltanto il campo storico, ASSENTE se nessuno dei due campi c'e' in
    entrambe. Con ASSENTE, `uguali` e' None: non e' stato verificato niente.
    """
    dx, sx = _valori(x)
    dy, sy = _valori(y)
    if dx is not None and dy is not None:
        return dx == dy, DIGEST
    if sx is not None and sy is not None:
        return sx == sy, SOMMA_STORICA
    return None, ASSENTE


class Riepilogo:
    """Accumula l'esito dei confronti e la forza piu' debole incontrata."""

    def __init__(self):
        self.confronti = 0
        self.per_forza = {DIGEST: 0, SOMMA_STORICA: 0, ASSENTE: 0}

    def aggiungi(self, forza):
        self.confronti += 1
        self.per_forza[forza] = self.per_forza.get(forza, 0) + 1

    @property
    def forza_minima(self):
        for f in (ASSENTE, SOMMA_STORICA, DIGEST):
            if self.per_forza.get(f):
                return f
        return ASSENTE

    def descrizione(self):
        if not self.confronti:
            return 'nessun confronto sugli indici campionati'
        f = self.forza_minima
        testo = {
            DIGEST: 'indici campionati identici, verificati sullo SHA-256 della '
                    'serializzazione dichiarata',
            SOMMA_STORICA: 'indici campionati identici, ma verificati solo sulla '
                           'somma storica dei row_id: controllo piu\' debole di un '
                           'digest, insensibile all\'ordine',
            ASSENTE: 'indici campionati NON verificati: il campo manca',
        }[f]
        return f'{testo} ({self.confronti:,} blocchi confrontati)'

    def rendiconto(self):
        return {'blocchi_confrontati': self.confronti,
                'forza_minima': self.forza_minima,
                'per_forza': dict(self.per_forza),
                'descrizione': self.descrizione()}

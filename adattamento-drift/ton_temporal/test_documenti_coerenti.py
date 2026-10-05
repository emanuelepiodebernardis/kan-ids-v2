"""I documenti dicono quello che l'evidenza dice: ogni cifra ricalcolata e cercata nel testo.

Perche' questa suite esiste
---------------------------
Due errori sono arrivati fino al referente perche' una cifra scritta in un
documento non era piu' quella che l'evidenza produceva: la tabella del richiamo
sui normali nel protocollo riportava valori non riproducibili, e il conteggio dei
blocchi `ddos` era 580 contro i 586 effettivi. Nessuno dei due si vedeva
rileggendo, perche' il testo era internamente coerente: erano sbagliati rispetto
ai dati, non rispetto a se stessi.

Questa suite chiude quella falla. Ogni valore riassuntivo viene **ricalcolato
dai rendiconti per blocco**, formattato nella convenzione italiana usata nei
documenti, e cercato come stringa nei file. Se un documento viene modificato a
mano e la cifra non torna, la prova diventa rossa.

Verifica anche che le grafie **sbagliate** non ricompaiano: un errore corretto
una volta puo' rientrare da un copia e incolla.

Non dipende da evidenza esterna: i rendiconti sono quelli pubblicati nel ramo, e
se non si trovano la suite viene saltata invece di passare in silenzio.
"""

from __future__ import annotations

import importlib.util
import re
import json
import statistics as st
import sys
from pathlib import Path

import pytest

QUI = Path(__file__).resolve().parent

SEMI = (42, 43, 44, 45, 46)
MODELLI = ('lr', 'mlp', 'kan')


def _cartella_evidenza():
    for c in (QUI / 'replay_evidenza', QUI.parent / 'replay_evidenza',
              QUI.parent / 'ton_temporal' / 'replay_evidenza'):
        if (c / 'replay_C_seme42.json').is_file():
            return c
    return None


def _cartella_documenti():
    for c in (QUI, QUI.parent, QUI.parent / 'ton_temporal'):
        if (c / 'protocol.md').is_file() and (c / 'sintesi_pilota.md').is_file():
            return c
    return None


CARTELLA_EV = _cartella_evidenza()
CARTELLA_DOC = _cartella_documenti()

if CARTELLA_EV is None:
    pytest.skip('rendiconti per blocco non trovati', allow_module_level=True)
if CARTELLA_DOC is None:
    pytest.skip('documenti non trovati', allow_module_level=True)

MANCANTI = [s for s in SEMI if not (CARTELLA_EV / f'replay_C_seme{s}.json').is_file()]
if MANCANTI:
    pytest.skip(f'rendiconti mancanti per i semi {MANCANTI}', allow_module_level=True)


@pytest.fixture(scope='module')
def rendiconti():
    fuori = {}
    for s in SEMI:
        with open(CARTELLA_EV / f'replay_C_seme{s}.json', encoding='utf-8') as f:
            d = json.load(f)
        assert d['parametri']['seme'] == s, f'il file del seme {s} dichiara {d["parametri"]["seme"]}'
        fuori[s] = d['per_blocco']
    return fuori


@pytest.fixture(scope='module')
def testi():
    """I documenti con gli spazi normalizzati.

    I file sono a righe di 79 colonne, quindi una frase come «246 blocchi su
    916» e' spezzata da un ritorno a capo. Cercarla cosi' com'e' darebbe falsi
    allarmi, e una prova che allarma a vuoto viene disattivata: qui gli spazi
    bianchi consecutivi diventano uno spazio singolo prima del confronto.
    """
    return {nome: ' '.join((CARTELLA_DOC / nome).read_text(encoding='utf-8').split())
            for nome in ('protocol.md', 'sintesi_pilota.md', 'nota_confini_componenti.md')
            if (CARTELLA_DOC / nome).is_file()}


# --- formattazione nella convenzione dei documenti ------------------------

def dec(x, cifre=3, segno=False):
    """0.2107 -> '0,211'; con segno -> '+0,211'. Il meno e' U+2212, come nei testi."""
    s = f'{abs(x):.{cifre}f}'.replace('.', ',')
    if segno:
        return ('+' if x >= 0 else '−') + s
    return ('−' if x < 0 else '') + s


def mille(n, segno=False):
    """12412 -> '12.412'; con segno -> '−12.412'."""
    s = f'{abs(int(n)):,}'.replace(',', '.')
    if segno:
        return ('+' if n >= 0 else '−') + s
    return s


def presente(testi, stringa, dove=None):
    """La stringa compare in almeno uno dei documenti indicati."""
    nomi = dove or list(testi)
    return any(stringa in testi[n] for n in nomi if n in testi)


# --- ricalcolo ------------------------------------------------------------

def fpr_medio(blocchi, chiave):
    v = [b[chiave]['fp'] / b[chiave]['normali'] for b in blocchi if b[chiave]['normali'] > 0]
    return st.mean(v)


def fpr_compl(blocchi, chiave):
    fp = sum(b[chiave]['fp'] for b in blocchi)
    vn = sum(b[chiave]['vn'] for b in blocchi)
    return fp / (fp + vn), fp


def coppie_auroc(blocchi, m):
    return [(b[f'congelato_{m}']['auroc'], b[f'adattivo_{m}']['auroc']) for b in blocchi
            if b[f'congelato_{m}']['auroc'] is not None
            and b[f'adattivo_{m}']['auroc'] is not None]


# --- le prove -------------------------------------------------------------

def test_i_rendiconti_sono_cinque_e_di_semi_distinti(rendiconti):
    assert sorted(rendiconti) == list(SEMI)
    assert len({len(v) for v in rendiconti.values()}) == 1, 'numeri di blocchi diversi'
    assert len(rendiconti[42]) == 916


@pytest.mark.parametrize('m', MODELLI)
def test_le_due_aggregazioni_del_fpr_sono_quelle_scritte(rendiconti, testi, m):
    """§3 della sintesi e §10 del protocollo: delta medio per blocco e complessivo."""
    dm = [fpr_medio(b, f'adattivo_{m}') - fpr_medio(b, f'congelato_{m}')
          for b in rendiconti.values()]
    dc, dfp = [], []
    for b in rendiconti.values():
        ca, fa = fpr_compl(b, f'adattivo_{m}')
        cc, fc = fpr_compl(b, f'congelato_{m}')
        dc.append(ca - cc)
        dfp.append(fa - fc)

    s_medio = dec(st.mean(dm), 3, segno=True)
    s_compl = dec(st.mean(dc), 3, segno=True)
    s_fp = mille(round(st.mean(dfp)), segno=True)
    assert presente(testi, s_medio), f'{m}: FPR medio per blocco {s_medio} non compare'
    assert presente(testi, s_compl), f'{m}: FPR complessivo {s_compl} non compare'
    assert presente(testi, s_fp), f'{m}: falsi positivi {s_fp} non compare'

    # e il verso dichiarato e' quello misurato
    assert len({x > 0 for x in dm}) == 1, f'{m}: il verso della media per blocco non e\' unanime'
    assert len({x > 0 for x in dc}) == 1, f'{m}: il verso del complessivo non e\' unanime'


@pytest.mark.parametrize('m', MODELLI)
def test_il_richiamo_sui_normali_medio_per_blocco_e_quello_scritto(rendiconti, testi, m):
    c = st.mean(1 - fpr_medio(b, f'congelato_{m}') for b in rendiconti.values())
    a = st.mean(1 - fpr_medio(b, f'adattivo_{m}') for b in rendiconti.values())
    assert presente(testi, f'{dec(c, 3)} → {dec(a, 3)}') or \
           presente(testi, f'da {dec(c, 3)} a {dec(a, 3)}'), \
        f'{m}: richiamo {dec(c, 3)} -> {dec(a, 3)} non compare in nessuna forma'


@pytest.mark.parametrize('m,atteso_42', [('lr', 174), ('mlp', 154), ('kan', 10)])
def test_i_conteggi_dell_inversione_del_seme_42_sono_quelli_scritti(rendiconti, testi, m, atteso_42):
    v = coppie_auroc(rendiconti[42], m)
    inv = [(c, a) for c, a in v if c < 0.5]
    assert len(inv) == atteso_42, f'{m}: seme 42 da {len(inv)} invertiti, non {atteso_42}'
    assert len(v) == 875, f'blocchi misurabili {len(v)}, non 875'
    assert presente(testi, f'{atteso_42}'), f'{m}: il conteggio {atteso_42} non compare'


@pytest.mark.parametrize('m', MODELLI)
def test_i_livelli_di_auroc_dei_due_regimi_sono_quelli_scritti(rendiconti, testi, m):
    """Le quattro medie 'da X a Y' del §1, sul seme 42."""
    v = coppie_auroc(rendiconti[42], m)
    inv = [(c, a) for c, a in v if c < 0.5]
    alt = [(c, a) for c, a in v if c >= 0.5]
    for gruppo in (inv, alt):
        da = dec(st.mean(c for c, _ in gruppo), 3)
        a_ = dec(st.mean(a for _, a in gruppo), 3)
        assert presente(testi, f'da {da} a {a_}'), f'{m}: "da {da} a {a_}" non compare'


@pytest.mark.parametrize('m', MODELLI)
def test_i_guadagni_sui_cinque_semi_sono_quelli_scritti(rendiconti, testi, m):
    gi, ga, conteggi = [], [], []
    for b in rendiconti.values():
        v = coppie_auroc(b, m)
        inv = [(c, a) for c, a in v if c < 0.5]
        alt = [(c, a) for c, a in v if c >= 0.5]
        gi.append(st.mean(a - c for c, a in inv))
        ga.append(st.mean(a - c for c, a in alt))
        conteggi.append(len(inv))
    assert presente(testi, dec(st.mean(gi), 3, segno=True)), \
        f'{m}: guadagno dove invertito {dec(st.mean(gi), 3, True)} non compare'
    assert presente(testi, dec(st.mean(ga), 3, segno=True)), \
        f'{m}: guadagno altrove {dec(st.mean(ga), 3, True)} non compare'
    lo, hi = min(conteggi), max(conteggi)
    atteso = (f'{lo} in tutti e cinque' if lo == hi else
              (f'{lo} o {hi}' if hi - lo == 1 else f'da {lo} a {hi}'))
    assert presente(testi, atteso), f'{m}: intervallo "{atteso}" non compare'


def test_la_concentrazione_dei_normali_e_quella_scritta(rendiconti, testi):
    bl = rendiconti[42]
    ricchi = [b for b in bl if b['congelato_lr']['normali'] >= 200]
    vuoti = [b for b in bl if b['congelato_lr']['normali'] == 0]
    tot = sum(b['congelato_lr']['normali'] for b in bl)
    nric = sum(b['congelato_lr']['normali'] for b in ricchi)
    assert presente(testi, f'{len(ricchi)} blocchi su {len(bl)}')
    assert presente(testi, f'{mille(nric)} dei {mille(tot)} normali')
    assert presente(testi, f'{len(vuoti)} blocchi non ne hanno')


@pytest.mark.parametrize('m', MODELLI)
def test_il_trasferimento_fra_bande_si_chiude(rendiconti, testi, m):
    """La tabella 'poveri / ricchi / netto': i tre numeri e la loro somma."""
    bl = rendiconti[42]
    cong, adat = f'congelato_{m}', f'adattivo_{m}'
    poveri = sum(b[adat]['fp'] - b[cong]['fp'] for b in bl
                 if 1 <= b[cong]['normali'] < 200)
    ricchi = sum(b[adat]['fp'] - b[cong]['fp'] for b in bl
                 if b[cong]['normali'] >= 200)
    netto = sum(b[adat]['fp'] for b in bl) - sum(b[cong]['fp'] for b in bl)
    assert poveri + ricchi == netto, 'le due bande non sommano al netto'
    for valore in (poveri, ricchi, netto):
        assert presente(testi, mille(valore, segno=True)), \
            f'{m}: {mille(valore, True)} non compare'


def test_i_tipi_dei_blocchi_invertiti_sono_quelli_scritti(rendiconti, testi):
    """Definizione dichiarata: blocchi in cui il tipo compare, fra i misurabili."""
    bl = [b for b in rendiconti[42] if b['congelato_lr']['auroc'] is not None]
    atteso = {'dos': (122, 123), 'injection': (46, 49), 'ddos': (6, 586), 'password': (0, 120)}
    for tipo, (inv_att, tot_att) in atteso.items():
        presenti = [b for b in bl if tipo in b['tipi_presenti']]
        invertiti = [b for b in presenti if b['congelato_lr']['auroc'] < 0.5]
        assert (len(invertiti), len(presenti)) == (inv_att, tot_att), \
            f'{tipo}: {len(invertiti)} su {len(presenti)}, non {inv_att} su {tot_att}'
        assert presente(testi, f'{inv_att}'), f'{tipo}: {inv_att} non compare'


def test_il_conteggio_ddos_a_tipo_unico_e_quello_scritto(rendiconti, testi):
    """Il documento dichiara anche l'altra definizione: va verificata."""
    bl = [b for b in rendiconti[42] if b['congelato_lr']['auroc'] is not None]
    unici = [b for b in bl
             if [x for x in b['tipi_presenti'] if x != 'normal'] == ['ddos']]
    assert len(unici) == 584, f'ddos a tipo unico: {len(unici)}, non 584'
    assert presente(testi, '584')


# --- le grafie sbagliate non devono rientrare -----------------------------

@pytest.mark.parametrize('sbagliato,motivo', [
    ('(580)', 'il conteggio dei blocchi ddos era 580, il valore e\' 586'),
    ('| 0,734 |', 'richiamo normali MLP: era 0,734, il valore e\' 0,732'),
    ('| 0,462 |', 'richiamo normali additivo: era 0,462, il valore e\' 0,483'),
    ('| 0,276 |', 'richiamo normali additivo adattivo: era 0,276, il valore e\' 0,297'),
    ('| KAN |', 'il terzo modello non e\' una KAN: si chiama "additivo"'),
    ('spendendo meno etichette', 'il risparmio e\' di aggiornamenti, non di etichette'),
    ('non invertiti −0,073', 'i poveri non invertiti danno −0,136, non −0,073'),
])
def test_le_grafie_corrette_non_rientrano(testi, sbagliato, motivo):
    for nome, t in testi.items():
        assert sbagliato not in t, f'{nome}: "{sbagliato}" — {motivo}'


def test_la_dicitura_KAN_additiva_compare_solo_dove_viene_ritirata(testi):
    """Era il nome troppo generoso del terzo modello.

    Non si proibisce la stringa, perche' i documenti la citano per ritirarla: si
    proibisce di usarla come nome del modello. L'occorrenza e' ammessa solo se
    nei 200 caratteri che precedono si dichiara che e' una formulazione
    precedente o imprecisa.
    """
    for nome, t in testi.items():
        i = 0
        while True:
            i = t.find('KAN additiva', i)
            if i == -1:
                break
            prima = t[max(0, i - 200):i]
            assert any(x in prima for x in ('versione precedente', 'chiamava',
                                            'precedenti', 'troppo generosa')), (
                f'{nome}: «KAN additiva» usata come nome, non ritirata, '
                f'nel contesto: ...{t[max(0, i - 90):i + 40]}...')
            i += 1


def test_i_falsi_allarmi_non_aumentano_mai_senza_dire_secondo_quale_aggregazione(testi):
    """La correzione chiesta: la frase e' ammessa solo se l'aggregazione e' nominata.

    Non si proibisce «aumentano sempre», che sulla media per blocco e' vero in
    cinque semi su cinque: si proibisce di scriverlo senza dire rispetto a che
    cosa. L'occorrenza e' ammessa in due casi soli: l'aggregazione e' nominata
    nei 160 caratteri che precedono, oppure la frase sta dentro il richiamo alla
    versione precedente, cioe' e' citata per essere ritirata.
    """
    for nome, t in testi.items():
        da = 0
        while True:
            i = t.find('aumentano sempre', da)
            if i == -1:
                break
            prima = t[max(0, i - 160):i]
            qualificata = 'medio per blocco' in prima
            ritirata = 'versione precedente' in prima
            assert qualificata or ritirata, (
                f'{nome}: «aumentano sempre» senza aggregazione dichiarata e '
                f'senza ritiro, nel contesto: ...{t[max(0, i - 90):i + 40]}...')
            da = i + 1


def test_le_due_aggregazioni_sono_definite_prima_di_essere_usate(testi):
    """Il referente ha chiesto che l'aggregazione sia dichiarata, non sottintesa."""
    for nome in ('protocol.md', 'sintesi_pilota.md'):
        if nome not in testi:
            continue
        t = testi[nome]
        assert 'FPR medio per blocco' in t, f'{nome}: manca il nome della prima aggregazione'
        assert 'FPR complessivo' in t, f'{nome}: manca il nome della seconda'
        assert 'somma delle matrici di confusione' in t or \
               'sommando le matrici di confusione' in t, \
            f'{nome}: la seconda aggregazione non e\' definita'
        i_def = min(t.index('FPR medio per blocco'), t.index('FPR complessivo'))
        i_tab = t.find('−0,072')
        if i_tab != -1:
            assert i_def < i_tab, f'{nome}: i valori compaiono prima della definizione'


def test_la_concentrazione_e_i_costi_del_collo_di_bottiglia_sono_quelli_scritti(rendiconti, testi):
    """§4: mediana, media, quota del 5% piu' ricco, normali attesi nel blocco mediano."""
    nm = [b['congelato_lr']['normali'] for b in rendiconti[42]]
    mediana, media = round(st.median(nm)), round(st.mean(nm))
    ordinati = sorted(nm, reverse=True)
    k = round(len(ordinati) * 0.05)
    quota = 100 * sum(ordinati[:k]) / sum(nm)
    # normali attesi fra i cento campionati nel blocco mediano: mediana * budget
    with open(CARTELLA_EV / 'replay_C_seme42.json', encoding='utf-8') as f:
        par = json.load(f)['parametri']
    attesi = st.median(nm) * par['budget']
    assert presente(testi, f'mediana {mediana}'), f'mediana {mediana} non compare'
    assert presente(testi, f'media di {media}'), f'media {media} non compare'
    assert presente(testi, f'{dec(quota, 1)}%'), f'quota {dec(quota, 1)}% non compare'
    assert presente(testi, dec(attesi, 2)), f'normali attesi {dec(attesi, 2)} non compare'


def test_la_ricchezza_di_normali_e_dichiarata_per_seme(rendiconti, testi):
    """§5: il valore sui blocchi poveri varia fra i semi, quindi va attribuito.

    Sul seme 42 vale −0,132, ma sui cinque semi arriva a −0,217: scriverlo senza
    dire di quale seme e' la cifra e' l'errore che il referente ha chiesto di non
    ripetere. La prova impone che l'intervallo dei cinque semi sia dichiarato.
    """
    estremi = {'ricchi': [], 'poveri': [], 'ricchi_non_invertiti': [],
               'poveri_non_invertiti': [], 'quota_inv': []}
    for b in rendiconti.values():
        bl = [x for x in b if x['congelato_lr']['auroc'] is not None
              and x['adattivo_lr']['auroc'] is not None]
        g = lambda L: st.mean(x['adattivo_lr']['auroc'] - x['congelato_lr']['auroc'] for x in L)
        ric = [x for x in bl if x['congelato_lr']['normali'] >= 200]
        pov = [x for x in bl if x['congelato_lr']['normali'] < 10]
        rni = [x for x in ric if x['congelato_lr']['auroc'] >= 0.5]
        pni = [x for x in pov if x['congelato_lr']['auroc'] >= 0.5]
        estremi['ricchi'].append(g(ric))
        estremi['poveri'].append(g(pov))
        estremi['ricchi_non_invertiti'].append(g(rni))
        estremi['poveri_non_invertiti'].append(g(pni))
        estremi['quota_inv'].append(round(100 * (len(ric) - len(rni)) / len(ric)))

    for chiave in ('ricchi', 'poveri', 'ricchi_non_invertiti', 'poveri_non_invertiti'):
        lo, hi = min(estremi[chiave]), max(estremi[chiave])
        for valore in (lo, hi):
            assert presente(testi, dec(valore, 3, segno=True)), \
                f'{chiave}: estremo {dec(valore, 3, True)} non dichiarato'
    assert len(set(estremi['quota_inv'])) == 1
    assert presente(testi, f'{estremi["quota_inv"][0]}%')


def test_i_costi_dichiarati_sono_quelli_registrati(rendiconti, testi):
    """§5: i costi vengono dal rendiconto, non da una stima."""
    with open(CARTELLA_EV / 'replay_C_seme42.json', encoding='utf-8') as f:
        costi = json.load(f).get('costi')
    if not costi:
        pytest.skip('il rendiconto non registra i costi')
    assert presente(testi, f'{round(costi["secondi_replay"])} secondi'), \
        f'i secondi di replay registrati sono {costi["secondi_replay"]}'
    assert presente(testi, str(round(costi['secondi_addestramento_iniziale']))), \
        f'i secondi di addestramento registrati sono {costi["secondi_addestramento_iniziale"]}'
    assert presente(testi, dec(costi['secondi_per_blocco'], 2))
    # i costi variano fra esecuzioni: il documento deve dirlo, altrimenti
    # sembrano una misura ripetibile
    assert presente(testi, 'variano di qualche secondo') or \
        presente(testi, 'non una misura ripetibile'), \
        'i costi sono riportati come se fossero riproducibili'


def test_la_forma_del_terzo_modello_e_dichiarata(testi):
    for nome in ('protocol.md', 'sintesi_pilota.md', 'nota_confini_componenti.md'):
        if nome not in testi:
            continue
        t = testi[nome]
        assert 'additivo' in t, f'{nome}: il terzo modello non e\' nominato'
    unione = '\n'.join(testi.values())
    for pezzo in ('singolo strato', 'edge a B-spline', '400.000', 'nove parametri',
                  'BSplineKANBinary', 'reimplement'):
        assert pezzo in unione, f'la descrizione del modello non dichiara: {pezzo}'


def test_non_si_nega_la_parentela_con_il_componente_del_paper_1(testi):
    """La correzione trovata dall'audit delle richieste.

    Una versione dei documenti diceva che il terzo modello «non e il componente a
    B-spline del Paper 1». Letto il codice, `BSplineKANBinary` e' anch'essa a
    singolo strato con edge a B-spline e la forma funzionale e' la stessa: la
    negazione era vera sul codice e falsa sulla famiglia di modelli, cioe'
    fuorviante. I documenti devono dichiarare che la forma e' la stessa e che la
    differenza sta nel regime di stima.
    """
    for nome, t in testi.items():
        assert 'né il componente a B-spline del Paper 1' not in t, \
            f'{nome}: la negazione fuorviante e rientrata'
    unione = '\n'.join(testi.values())
    assert 'forma funzionale è la stessa' in unione or \
           'La forma funzionale è la stessa' in unione, \
        'nessun documento dichiara che la forma funzionale coincide'
    assert 'regime di stima' in unione, \
        'nessun documento indica dove sta la differenza reale'


def test_il_risparmio_e_dichiarato_di_aggiornamenti_non_di_etichette(testi):
    """La frase del referente, che l'audit aveva trovato assente dai documenti."""
    unione = '\n'.join(testi.values())
    assert 'di aggiornamenti, non di etichette' in unione, \
        'il risparmio non e dichiarato per quello che e'
    assert '91.506' in unione, \
        'manca la misura che lo sostiene: le etichette spese sono le stesse'


# --- il confronto con la soglia scelta su B -------------------------------

def _rendiconti_calibrati():
    presenti = {}
    for s in SEMI:
        p = CARTELLA_EV / f'replay_C_calibrato_seme{s}.json'
        if p.is_file():
            presenti[s] = p
    return presenti if len(presenti) == len(SEMI) else None


@pytest.fixture(scope='module')
def calibrati():
    percorsi = _rendiconti_calibrati()
    if percorsi is None:
        pytest.skip('rendiconti con la soglia scelta su B non presenti')
    fuori = {}
    for s, p in percorsi.items():
        with open(p, encoding='utf-8') as f:
            fuori[s] = json.load(f)
    return fuori


def test_la_soglia_non_cambia_l_auroc(rendiconti, calibrati):
    """L'invariante del §10bis: una soglia non puo' cambiare un ordinamento."""
    confrontate = 0
    forze = set()
    for s in SEMI:
        base, cal = rendiconti[s], calibrati[s]['per_blocco']
        assert len(base) == len(cal)
        for i, (x, y) in enumerate(zip(base, cal)):
            for m in MODELLI:
                for stato in ('congelato', 'adattivo'):
                    k = f'{stato}_{m}'
                    assert x[k]['auroc'] == y[k]['auroc'], \
                        f'seme {s}, blocco {i}, {k}: AUROC diversa'
                    confrontate += 1
            uguali, forza = _impronta().confronta(x, y)
            assert uguali, f'seme {s}, blocco {i}: indici campionati diversi'
            forze.add(forza)
    assert forze and 'assente' not in forze, \
        'gli indici campionati non sono verificabili in qualche blocco'
    assert confrontate == 5 * 916 * 3 * 2 == 27480
    assert presente(_testi_documenti(), '27.480'), 'il conteggio non compare nei documenti'


def _impronta():
    """Il modulo che confronta gli indici campionati, caricato per percorso."""
    for c in (QUI / 'impronta_campione.py', QUI.parent / 'impronta_campione.py',
              QUI.parent / 'ton_temporal' / 'impronta_campione.py'):
        if c.is_file():
            spec = importlib.util.spec_from_file_location('impronta_campione', c)
            m = importlib.util.module_from_spec(spec)
            sys.modules['impronta_campione'] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip('impronta_campione.py non trovato accanto alla suite')


def _testi_documenti():
    return {nome: ' '.join((CARTELLA_DOC / nome).read_text(encoding='utf-8').split())
            for nome in ('protocol.md', 'sintesi_pilota.md')
            if (CARTELLA_DOC / nome).is_file()}


@pytest.mark.parametrize('m', MODELLI)
def test_le_due_aggregazioni_con_la_soglia_su_B_sono_quelle_scritte(calibrati, testi, m):
    dm, dc, dfp = [], [], []
    for s in SEMI:
        bl = calibrati[s]['per_blocco']
        dm.append(fpr_medio(bl, f'adattivo_{m}') - fpr_medio(bl, f'congelato_{m}'))
        ca, fa = fpr_compl(bl, f'adattivo_{m}')
        cc, fc = fpr_compl(bl, f'congelato_{m}')
        dc.append(ca - cc)
        dfp.append(fa - fc)
    for valore, nome in ((st.mean(dm), 'FPR medio per blocco'),
                         (st.mean(dc), 'FPR complessivo')):
        assert presente(testi, dec(valore, 3, segno=True)), \
            f'{m} su B: {nome} {dec(valore, 3, True)} non compare'
    assert presente(testi, mille(round(st.mean(dfp)), segno=True)), \
        f'{m} su B: falsi positivi {mille(round(st.mean(dfp)), True)} non compare'
    assert len({x > 0 for x in dm}) == 1, f'{m}: verso non unanime sulla media per blocco'
    assert len({x > 0 for x in dc}) == 1, f'{m}: verso non unanime sul complessivo'


def test_le_soglie_scelte_sono_quelle_scritte(calibrati, testi):
    per_modello = {m: [] for m in MODELLI}
    pari_merito = []
    for s in SEMI:
        d = calibrati[s]['soglia_di_decisione']
        assert d['modo'] == 'balanced_accuracy_su_B'
        for m, v in d['per_modello'].items():
            per_modello[m].append(v['soglia'])
            pari_merito.append(v['candidate_a_pari_merito'])
    # la LR congelata e' deterministica: una sola soglia nei cinque semi
    assert len(set(round(x, 4) for x in per_modello['lr'])) == 1
    assert presente(testi, dec(per_modello['lr'][0], 4, segno=True))
    for m in ('mlp', 'kan'):
        for valore in (min(per_modello[m]), max(per_modello[m])):
            assert presente(testi, dec(valore, 4, segno=True)), \
                f'{m}: soglia estrema {dec(valore, 4, True)} non compare'
    assert set(pari_merito) == {1}, \
        'la regola di parita\' si e\' attivata: i documenti dicono che non e\' mai servita'
    assert presente(testi, 'non è mai servita') or presente(testi, 'non e\' mai servita')


def test_la_soglia_e_identica_nella_coppia_congelato_adattivo(calibrati):
    """Il requisito esplicito: una soglia per modello, non una per stato."""
    for s in SEMI:
        d = calibrati[s]['soglia_di_decisione']['per_modello']
        for m, v in d.items():
            assert 'soglia' in v and isinstance(v['soglia'], float)
        # il rendiconto ha una voce per modello, non per coppia modello/stato
        assert set(d) == set(MODELLI)


# --------------------------------------------------------------------------
# I numeri dei blocchi incompleti, ricontati dal manifest
# --------------------------------------------------------------------------

def _manifest():
    for c in (CARTELLA_DOC / 'split_manifest.json',
              CARTELLA_DOC.parent / 'split_manifest.json'):
        if c.is_file():
            with open(c, encoding='utf-8') as f:
                return json.load(f)['intervalli']
    pytest.skip('split_manifest.json non trovato')


def test_i_blocchi_incompleti_sono_quelli_del_manifest(testi):
    """Il difetto che il residuo 2 ha fatto emergere.

    La riga sui blocchi dichiarava B 1.199, C 677 e D 9.968: tre numeri che non
    sono `righe_valide mod 10.000`. Non cambiavano un risultato, ma sono i
    numeri con cui si ricalcola la quota di etichette, che era appunto
    sbagliata. Qui si ricontano dal manifest e si pretende che compaiano nei
    documenti, scritti nella convenzione italiana.
    """
    for nome, v in _manifest().items():
        resto = v['righe_valide'] % 10000
        assert resto == v['righe_blocco_finale'], \
            f'{nome}: il manifest non chiude su se stesso'
        completi = v['righe_valide'] // 10000
        assert completi == v['blocchi_completi_da_10000']
        atteso = f'{resto:,}'.replace(',', '.')
        assert presente(testi, atteso), \
            f'{nome}: le {atteso} righe del blocco finale non compaiono'
        assert v['budget_etichette_blocco_finale'] == int(0.01 * resto)


def test_i_vecchi_numeri_dei_blocchi_non_tornano(testi):
    """Le tre cifre corrette non devono rientrare da una copia vecchia."""
    for sbagliato in ('B 568 più 1.199', 'C 915 più 677', 'D 354 più 9.968'):
        for nome, t in testi.items():
            assert sbagliato not in t, f'{nome}: {sbagliato} e rientrato'


def test_nessun_segnaposto_e_rimasto_nei_documenti(testi):
    """Durante la stesura i numeri non ancora misurati sono marcati con `@@`.

    Un segnaposto dimenticato in un documento consegnato sarebbe l'errore
    peggiore fra quelli possibili, perche' sembrerebbe un numero.
    """
    for nome, t in testi.items():
        assert '@@' not in t, f'{nome}: segnaposto non sostituito'


# --------------------------------------------------------------------------
# I numeri dei residui 3 e 5, ricalcolati dall'evidenza
# --------------------------------------------------------------------------

def _tempi(nome):
    p = CARTELLA_EV / 'tempi' / nome
    if not p.is_file():
        pytest.skip(f'{nome} non trovato')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def test_il_costo_per_aggiornamento_e_quello_registrato(testi):
    """Il residuo 3: ogni numero della tabella dei costi viene dal rendiconto."""
    d = _tempi('ogni_blocco_seme42_costi.json')
    c = d['costi']
    for m in MODELLI:
        a = c['tempo_del_singolo_aggiornamento'][m]
        assert presente(testi, dec(a['ms_mediano'], 2)), \
            f'{m}: la mediana {dec(a["ms_mediano"], 2)} ms non compare'
        assert presente(testi, dec(a['secondi_totali'], 1)), \
            f'{m}: i {dec(a["secondi_totali"], 1)} s di aggiornamento non compaiono'
        quota = a['secondi_totali'] / c['secondi_replay']
        assert presente(testi, f'{quota * 100:.1f}%'.replace('.', ',')), \
            f'{m}: la quota {quota:.1%} non compare'


def test_il_conto_fra_le_due_politiche_torna(testi):
    """I totali e il residuo non attribuito sono aritmetica, non stime."""
    o = _tempi('ogni_blocco_seme42_costi.json')['costi']
    e = _tempi('evidenza_seme42_costi.json')['costi']

    def totale(c):
        return sum(c['tempo_del_singolo_aggiornamento'][m]['secondi_totali']
                   + c['tempo_degli_aggiornamenti_saltati'][m]['secondi_totali']
                   for m in MODELLI)

    assert presente(testi, dec(totale(o), 1)), 'il totale di ogni_blocco non compare'
    assert presente(testi, dec(totale(e), 1)), 'il totale dell\'evidenza non compare'
    differenza = o['secondi_replay'] - e['secondi_replay']
    residuo = differenza - (totale(o) - totale(e))
    assert presente(testi, dec(differenza, 1)), 'la differenza fra i replay non compare'
    assert presente(testi, dec(residuo, 1)), 'il residuo non attribuito non compare'
    assert presente(testi, 'non è attribuito') or presente(testi, 'non attribuito'), \
        'il residuo va dichiarato come non attribuito'


def test_la_memoria_dichiarata_e_quella_misurata(testi):
    m = _tempi('ogni_blocco_seme42_costi.json')['costi']['memoria']
    assert presente(testi, f'{m["picco_rss_processo_mib"]:,.0f}'.replace(',', '.'))
    assert presente(testi, f'{m["byte_memoria_fifo"]:,}'.replace(',', '.'))
    assert presente(testi, 'non è la memoria') or presente(testi, 'Non è la memoria')


def _confronto_politiche():
    p = CARTELLA_EV / 'confronto_politiche.json'
    if not p.is_file():
        pytest.skip('confronto_politiche.json non trovato')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def test_i_delta_per_seme_sono_quelli_misurati(testi):
    """Il residuo 5: le tabelle per seme non sono trascritte a mano."""
    d = _confronto_politiche()['per_seme']
    for m in MODELLI:
        c = d[m]['confronti']['evidenza contro casuale']
        for misura in ('auroc', 'fpr_complessivo'):
            for seme, delta in c[misura]['delta_per_seme'].items():
                assert presente(testi, dec(delta, 3, segno=True)), \
                    f'{m} {misura} seme {seme}: {dec(delta, 3, True)} non compare'
            quanti = c[misura]['quanti_semi_su']
            n = quanti.split('/')[0]
            assert presente(testi, f'{n} su 5') or presente(testi, f'{n}/5'), \
                f'{m} {misura}: il conteggio {quanti} non compare'


def test_il_vantaggio_non_e_dichiarato_uniforme_dove_non_lo_e(testi):
    """La frase vietata: «in tutti e tre i modelli» senza «in media» davanti."""
    d = _confronto_politiche()['per_seme']
    non_uniformi = [m for m in MODELLI
                    if not d[m]['confronti']['evidenza contro casuale']['auroc']['uniforme']]
    assert non_uniformi, 'nessun confronto non uniforme: il test non prova nulla'
    for nome, t in testi.items():
        i = 0
        while True:
            i = t.find('batte il controllo', i)
            if i == -1:
                break
            intorno = t[max(0, i - 120):i + 200]
            assert 'in media' in intorno or 'media' in intorno, \
                f'{nome}: «batte il controllo» senza dire che e una media: {intorno[:150]}'
            i += 1
    assert presente(testi, 'seme 44'), 'il seme in cui si rovescia non e nominato'


def test_l_fpr_del_controllo_casuale_sta_accanto_all_auroc(testi):
    """Sull'MLP il sorteggio ha meno falsi allarmi: va scritto, non omesso."""
    tab = _confronto_politiche()['tabella']['mlp']
    assert tab['casuale']['fpr_complessivo'] < tab['evidenza']['fpr_complessivo']
    assert presente(testi, dec(tab['evidenza']['fpr_complessivo'], 4))
    assert presente(testi, dec(tab['casuale']['fpr_complessivo'], 4))


def test_una_sola_pianificazione_casuale_e_dichiarata(testi):
    assert presente(testi, 'una sola pianificazione') or \
        presente(testi, 'un solo sorteggio'), \
        'il controllo casuale e un solo sorteggio per modello e per seme'


def test_il_richiamo_sugli_attacchi_sta_accanto_all_auroc(testi):
    """Il relatore chiede di affiancare FPR **e richiamo** all'AUROC.

    Sul richiamo degli attacchi il controllo casuale fa meglio su due modelli su
    tre: e' il compromesso che la sola AUROC nasconde, e va scritto.
    """
    d = _confronto_politiche()
    tab = d['tabella']
    peggio = [m for m in MODELLI
              if tab[m]['casuale']['richiamo_attacchi'] > tab[m]['evidenza']['richiamo_attacchi']]
    assert peggio, 'nessun modello in cui il casuale ha richiamo migliore: il test non prova nulla'
    for m in peggio:
        for politica in ('evidenza', 'casuale'):
            v = tab[m][politica]['richiamo_attacchi']
            assert presente(testi, dec(v, 4)), \
                f'{m} {politica}: il richiamo {dec(v, 4)} non compare nei documenti'
    for m in MODELLI:
        c = d['per_seme'][m]['confronti']['evidenza contro casuale']['richiamo_attacchi']
        for seme, delta in c['delta_per_seme'].items():
            assert presente(testi, dec(delta, 3, segno=True)), \
                f'{m} richiamo attacchi seme {seme}: {dec(delta, 3, True)} non compare'


def test_i_risultati_sono_dichiarati_esplorativi(testi):
    """Nessuna superiorita generale, nessuna significativita calcolata."""
    assert presente(testi, 'esplorativi')
    assert presente(testi, 'significatività statistica') or \
        presente(testi, 'significativita statistica')


# --------------------------------------------------------------------------
# I blocchi di comando devono essere incollabili
# --------------------------------------------------------------------------

DOCUMENTI_CON_COMANDI = ('protocol.md', 'sintesi_pilota.md', 'ambiente_e_comandi.md',
                         'nota_confini_componenti.md', 'replay_evidenza/LEGGIMI.md')


def _blocchi_recintati(testo):
    """I blocchi fra ``` con il loro numero di riga d'inizio."""
    fuori, dentro, inizio, corrente = [], False, 0, []
    for n, riga in enumerate(testo.split('\n'), 1):
        if riga.startswith('```'):
            if dentro:
                fuori.append((inizio, corrente)); corrente = []
            else:
                inizio = n
            dentro = not dentro
        elif dentro:
            corrente.append((n, riga))
    return fuori, dentro


def test_i_comandi_dei_documenti_sono_incollabili():
    """Il difetto che un riavvolgimento automatico ha introdotto una volta.

    Riavvolgendo i paragrafi per tenerli a 79 colonne, alcune righe **dentro i
    blocchi di comando** sono state fuse fra loro: `cd replay_evidenza Z="..."`
    su una riga, due `python` di fila, una barra di continuazione rimasta sola.
    Incollato, un blocco così esegue comandi diversi da quelli scritti, e il
    lettore non ha modo di accorgersene.

    Qui ogni riga di ogni blocco viene esaminata per i segni di quella fusione.
    Non è una prova sul contenuto dei comandi: è una prova sulla loro forma, che
    è esattamente ciò che un riavvolgimento rompe.
    """
    guai = []
    for nome in DOCUMENTI_CON_COMANDI:
        p = CARTELLA_DOC / nome
        if not p.is_file():
            continue
        testo = p.read_text(encoding='utf-8')
        blocchi, aperto = _blocchi_recintati(testo)
        assert not aperto, f'{nome}: un blocco ``` non è chiuso'
        for _, righe in blocchi:
            for n, r in righe:
                comandi = r.count('python ') + r.count('git ')
                if comandi > 1:
                    guai.append(f'{nome}:{n} due comandi sulla stessa riga: {r[:70]}')
                if r.strip() == '\\':
                    guai.append(f'{nome}:{n} barra di continuazione rimasta sola')
                if '...' in r:
                    guai.append(f'{nome}:{n} ellissi dentro un comando: {r[:70]}')
                if re.match(r'^\$?\w+\s*=', r) and comandi:
                    guai.append(f'{nome}:{n} assegnazione e comando sulla stessa riga')
                if re.match(r'^\w+\s*=', r) and not comandi and r.count('=') > 1:
                    guai.append(f'{nome}:{n} più definizioni sulla stessa riga: {r[:70]}')
    assert not guai, 'blocchi di comando non incollabili:\n  ' + '\n  '.join(guai)


def test_i_comandi_non_tornano_a_spezzarsi_su_piu_righe():
    """Nessuna continuazione con la barra: una riga spezzata, incollata male, è
    il modo più facile per eseguire metà di un comando."""
    for nome in DOCUMENTI_CON_COMANDI:
        p = CARTELLA_DOC / nome
        if not p.is_file():
            continue
        blocchi, _ = _blocchi_recintati(p.read_text(encoding='utf-8'))
        for _, righe in blocchi:
            for n, r in righe:
                assert not r.rstrip().endswith('\\'), \
                    f'{nome}:{n} comando spezzato su più righe: {r[:70]}'


def _comandi_dei_blocchi(nome):
    """(riga, comando, cartella dichiarata) per ogni comando dei blocchi di `nome`.

    La cartella e' quella dichiarata dall'ultimo `cd` incontrato **dentro lo
    stesso blocco**, espressa rispetto alla radice del repository. `None`
    significa che il blocco non l'ha dichiarata.
    """
    p = CARTELLA_DOC / nome
    if not p.is_file():
        return []
    blocchi, _ = _blocchi_recintati(p.read_text(encoding='utf-8'))
    fuori = []
    for k, (_, righe) in enumerate(blocchi):
        cartella = None
        for n, r in righe:
            s = r.strip()
            if s.startswith('cd '):
                cartella = s[3:].strip().strip('"').replace('\\', '/')
                continue
            if s.startswith('python ') or s.startswith('git '):
                fuori.append((n, s, cartella, k))
    return fuori


def test_ogni_blocco_di_comandi_dichiara_la_cartella():
    """Da dove si esegue un comando non e' un dettaglio: lo stesso comando
    riesce da una cartella e fallisce da un'altra, e il documento che non lo
    dice obbliga chi legge a indovinare.
    """
    muti = []
    for nome in DOCUMENTI_CON_COMANDI:
        for n, comando, cartella, _ in _comandi_dei_blocchi(nome):
            if cartella is None:
                muti.append(f'{nome}:{n} {comando[:68]}')
    assert not muti, ('comandi senza cartella dichiarata:\n  '
                      + '\n  '.join(muti))


def test_i_file_citati_nei_comandi_esistono_dove_il_comando_li_cerca():
    """Un comando che cita un file inesistente non e' un comando: e' un errore
    in attesa di essere incollato. Qui ogni percorso viene risolto rispetto
    alla cartella che il blocco dichiara, e cercato sul disco.
    """
    if not (CARTELLA_DOC / 'replay.py').is_file():
        pytest.skip('copia parziale del repository: i percorsi non si risolvono')
    radice = CARTELLA_DOC.parent
    mancanti = []
    for nome in DOCUMENTI_CON_COMANDI:
        prodotti, blocco_corrente = set(), None
        for n, comando, cartella, blocco in _comandi_dei_blocchi(nome):
            if blocco != blocco_corrente:
                prodotti, blocco_corrente = set(), blocco
            if cartella is None:
                continue          # lo dice gia' l'altra prova
            # `<repo>` e' la radice del clone, dichiarata nei documenti. Si
            # traduce il prefisso logico invece di comporre percorsi, perche'
            # la cartella che contiene i documenti puo' avere un altro nome in
            # una copia di lavoro.
            c = cartella.rstrip('/')
            if c == '<repo>/ton_temporal':
                base = CARTELLA_DOC
            elif c.startswith('<repo>/ton_temporal/'):
                base = CARTELLA_DOC / c[len('<repo>/ton_temporal/'):]
            elif c == '<repo>':
                base = radice
            else:
                base = radice / c
            pezzi = comando.split()
            for i, pezzo in enumerate(pezzi):
                q = pezzo.strip('"')
                if not q.endswith('.py') and not q.endswith('.json'):
                    continue
                if '<' in q or '*' in q or '?' in q or '$' in q or '(' in q:
                    continue      # segnaposto, glob o variabile di shell
                # il valore che segue --uscita e' prodotto dal comando, non
                # letto: pretenderlo gia' sul disco sarebbe un controesenso
                if i and pezzi[i - 1] in ('--uscita', '--output', '>'):
                    prodotti.add(q)
                    continue
                # un argomento puo' avere la forma nome=percorso
                q = q.split('=', 1)[1] if '=' in q else q
                if q in prodotti:
                    continue      # lo produce un comando precedente del blocco
                if not (base / q.replace('\\', '/')).exists():
                    mancanti.append(f'{nome}:{n} da {cartella}/ non esiste {q}')
    assert not mancanti, ('file citati dai comandi e non trovati:\n  '
                          + '\n  '.join(mancanti))


def test_nessun_paragrafo_di_prosa_finisce_tronco():
    """Una frase tagliata a meta' non fa cadere niente e si legge male una volta
    sola: e' il difetto piu' facile da lasciare in un documento lungo, e ce n'e'
    stato uno pubblicato — «...`test_guardia_monoclasse.py` con 11. Su» —
    seguito dal paragrafo successivo come se niente fosse.

    Due criteri, perche' il troncamento si presenta in due forme.

    1. Un paragrafo che chiude su una riga vuota senza punteggiatura finale.
       Non vale se il paragrafo introduce un blocco di comandi: li' la frase
       continua legittimamente nel blocco.
    2. Una riga che chiude una frase col punto e lascia una o due parole
       orfane, subito sopra un capoverso nuovo: e' la forma esatta del difetto
       pubblicato. Un criterio piu' largo — «riga piu' corta delle altre» —
       segnalava quindici righe sane, riavvolte a 79 colonne davanti a una
       parola lunga, e sarebbe stato abbandonato al primo falso allarme.
    """
    chiusure = ('.', ':', '?', '!', '\u00bb', ')', ']', '*', '`', '|', '-', ',',
                ';', '"', '\u201d', '\u2014')
    tronchi = []
    for nome in DOCUMENTI_CON_COMANDI + ('nota_confini_componenti.md',):
        p = CARTELLA_DOC / nome
        if not p.is_file():
            continue
        righe = p.read_text(encoding='utf-8').split('\n')
        dentro = False
        for i, riga in enumerate(righe):
            if riga.startswith('```'):
                dentro = not dentro
                continue
            if dentro:
                continue
            s = riga.rstrip()
            # prosa: niente titoli, citazioni, tabelle, elenchi, continuazioni
            if (not s or s != s.lstrip()
                    or s.startswith(('#', '>', '|', '-', '*'))
                    or re.match(r'^\d+\.', s)):
                continue
            if s.endswith(chiusure):
                continue
            dopo = righe[i + 1].rstrip() if i + 1 < len(righe) else ''
            if dopo.strip() == '':
                # un paragrafo che introduce un blocco finisce senza punto
                dopodopo = righe[i + 2].strip() if i + 2 < len(righe) else ''
                if not dopodopo.startswith('```'):
                    tronchi.append(f'{nome}:{i + 1} chiude senza punto: ...{s[-54:]}')
            elif re.match(r'^\*\*[^*]*\.\*\*', dopo.lstrip()):
                # Il residuo tipico di una frase cancellata: la riga chiude
                # una frase col punto, restano una o due parole orfane, e
                # subito sotto comincia un capoverso nuovo. Il capoverso si
                # riconosce dal titolino in grassetto che **contiene** il punto
                # fermo, come «**I 193 non sono il totale.**»: un grassetto
                # senza punto, come «**affiancate**, ciascuna», e' invece una
                # frase che prosegue dalla riga sopra. Senza questa distinzione
                # la prova segnalava tre righe sane ogni una guasta.
                coda = s.rsplit('. ', 1)[-1] if '. ' in s else ''
                if coda and len(coda) <= 15:
                    tronchi.append(
                        f'{nome}:{i + 1} resto di frase «{coda}» prima di un '
                        f'capoverso nuovo')
    assert not tronchi, ('frasi che sembrano troncate:\n  '
                         + '\n  '.join(tronchi))


def test_il_conteggio_delle_prove_dichiarato_e_quello_vero():
    """Il residuo che e' tornato due volte: un numero di prove rimasto indietro.

    Prima «11 test superati» con conteggi di 634 e 643 misurati su un altro
    ramo; poi 193 e 827 dopo che erano diventati 195 e 829. Un numero scritto a
    mano invecchia al commit successivo, e nessuno se ne accorge finche' non lo
    rilegge chi ha chiesto il controllo.

    Qui il numero dichiarato viene confrontato con la raccolta vera, fatta in un
    sottoprocesso con `--collect-only`, che **non esegue** le prove e quindi non
    rientra qui dentro.
    """
    doc = CARTELLA_DOC / 'ambiente_e_comandi.md'
    if not doc.is_file():
        pytest.skip('ambiente_e_comandi.md non presente')
    testo = doc.read_text(encoding='utf-8')
    m = re.search(r'Attesi:\s*\*\*(\d+) test superati\*\*', testo)
    assert m, 'il documento non dichiara piu il numero di prove attese'
    dichiarato = int(m.group(1))

    import subprocess
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q',
         '-p', 'no:cacheprovider', str(CARTELLA_DOC)],
        capture_output=True, text=True, timeout=300)
    raccolti = sum(1 for riga in r.stdout.splitlines() if '::' in riga)
    assert raccolti, 'la raccolta non ha prodotto nulla:\n' + r.stdout[-800:]
    assert dichiarato == raccolti, (
        f'il documento dichiara {dichiarato} prove, la raccolta ne trova '
        f'{raccolti}')

    # e il dettaglio per file, che il documento elenca accanto al totale
    per_file = {}
    for riga in r.stdout.splitlines():
        if '::' in riga:
            f = riga.split('::')[0].split('/')[-1]
            per_file[f] = per_file.get(f, 0) + 1
    sbagliati = []
    for f, quante in sorted(per_file.items()):
        m = re.search(r'`' + re.escape(f) + r'`\s*\n?\s*(\d+)', testo)
        if m and int(m.group(1)) != quante:
            sbagliati.append(f'{f}: dichiarate {m.group(1)}, raccolte {quante}')
    assert not sbagliati, ('conteggi per file non aggiornati:\n  '
                           + '\n  '.join(sbagliati))


SCRITTORI = ('verifica_richieste.py', 'riepiloghi_semi.py',
             'confronto_politiche.py', 'costi_aggiornamento.py',
             'curve_replay.py')


def test_gli_script_dichiarano_l_encoding_della_propria_uscita():
    """Il difetto arrivato dalla macchina Windows, e il terzo di questa serie.

    L'uscita di questi programmi finisce in file versionati — per esempio
    `verifica_richieste.txt` — con un `>` che la redirige. Su Windows Python
    sceglie allora `cp1252`, dove il meno tipografico U+2212 non esiste, e il
    comando cade con `UnicodeEncodeError` a meta' file. A schermo funzionava:
    la console gestisce UTF-8, la redirezione no.

    Oggi solo `verifica_richieste.py` stampa caratteri fuori da cp1252, ma gli
    altri quattro stampano su flussi che possono essere rediretti allo stesso
    modo, e basta un accento aggiunto domani. L'encoding dell'uscita e' quindi
    dichiarato in tutti e cinque, e qui si verifica che la dichiarazione ci sia.
    """
    senza = []
    for nome in SCRITTORI:
        p = CARTELLA_DOC / nome
        if not p.is_file():
            continue
        if 'sys.stdout.reconfigure' not in p.read_text(encoding='utf-8'):
            senza.append(nome)
    assert not senza, ('script che lasciano al sistema l\'encoding della propria '
                       'uscita: ' + ', '.join(senza))


def test_l_uscita_del_controllo_regge_un_encoding_che_non_ha_il_meno():
    """La prova vera: si rilancia il controllo con l'uscita forzata a cp1252,
    che e' esattamente la condizione in cui cadeva su Windows."""
    p = CARTELLA_DOC / 'verifica_richieste.py'
    if not p.is_file() or not (CARTELLA_DOC / 'sovrapposizioni_abcd.json').is_file():
        pytest.skip('copia parziale: il controllo non ha tutto il materiale')
    import subprocess, os, tempfile
    ambiente = dict(os.environ, PYTHONIOENCODING='cp1252')
    with tempfile.TemporaryDirectory() as d:
        destinazione = Path(d) / 'uscita.txt'
        with open(destinazione, 'wb') as f:
            r = subprocess.run([sys.executable, str(p), '--cartella',
                                str(CARTELLA_DOC)],
                               stdout=f, stderr=subprocess.PIPE, env=ambiente,
                               timeout=300)
        assert b'UnicodeEncodeError' not in r.stderr, (
            'l\'uscita cade su un encoding senza il meno tipografico:\n'
            + r.stderr.decode('utf-8', 'replace')[-600:])
        testo = destinazione.read_text(encoding='utf-8')
        assert '−' in testo, (
            'il meno tipografico non e arrivato nel file: o non c\'e piu, o '
            'e stato sostituito in silenzio')
        assert testo.rstrip().endswith('dopo il push'), \
            'il file si ferma prima della fine'


def test_il_rendiconto_dei_controlli_e_un_file_di_testo_utf8():
    """La prova che mancava, e che avrebbe intercettato un difetto pubblicato.

    Le due prove sull'encoding guardavano il **programma**: che dichiarasse
    UTF-8 e che non cadesse sotto `cp1252`. Nessuna guardava il **file
    consegnato**. Cosi' `verifica_richieste.txt` e' stato pubblicato in UTF-16
    con BOM, con il meno tipografico corrotto in `OeaeAE`, perche' la
    redirezione di PowerShell riscrive l'uscita in UTF-16 e la rilegge con una
    code page diversa: git lo trattava come binario e nessuna prova se ne
    accorgeva.

    La correzione sta a monte — il file lo scrive il programma, con `--uscita`,
    non la shell — ma la lezione sta qui: quello che si pubblica va controllato
    sul file pubblicato, non sul programma che lo produce.
    """
    p = CARTELLA_DOC / 'verifica_richieste.txt'
    if not p.is_file():
        pytest.skip('verifica_richieste.txt non presente')
    grezzo = p.read_bytes()

    assert not grezzo.startswith(b'\xff\xfe') and not grezzo.startswith(b'\xfe\xff'), \
        'il file e in UTF-16: probabilmente prodotto con la redirezione della shell'
    assert not grezzo.startswith(b'\xef\xbb\xbf'), 'il file ha un BOM UTF-8'
    try:
        testo = grezzo.decode('utf-8')
    except UnicodeDecodeError as e:
        raise AssertionError(f'il file non e UTF-8 valido: {e}')

    assert '−' in testo, (
        'il meno tipografico non c\'e: o e stato sostituito dal trattino, o '
        'l\'encoding l\'ha corrotto')
    for rovinato in ('ÔêÆ', 'âˆ’', '�'):
        assert rovinato not in testo, (
            f'nel file c\'e la sequenza {rovinato!r}: e il meno tipografico '
            f'passato attraverso una code page sbagliata')
    assert testo.rstrip().endswith('dopo il push'), \
        'il rendiconto non arriva in fondo'
    atteso = sum(1 for r in testo.split('\n') if r.startswith('OK '))
    dichiarato = re.search(r'(\d+) controlli su (\d+) passati', testo)
    assert dichiarato, 'il file non dichiara quanti controlli sono passati'
    assert int(dichiarato.group(1)) == atteso, (
        f'il file dichiara {dichiarato.group(1)} controlli passati ma ne elenca '
        f'{atteso}')

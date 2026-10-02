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
            assert x.get('row_id_campionati_sha') == y.get('row_id_campionati_sha')
    assert confrontate == 5 * 916 * 3 * 2 == 27480
    assert presente(_testi_documenti(), '27.480'), 'il conteggio non compare nei documenti'


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

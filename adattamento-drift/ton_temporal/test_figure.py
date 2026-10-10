"""Le figure: che siano prodotte, e che dicano gli stessi numeri delle tabelle.

Perche' questa suite
--------------------
Una figura e' un'affermazione come un'altra, e puo' essere sbagliata negli stessi
modi: un valore finale che non coincide con la tabella, una media fatta su
blocchi in cui la misura non e' definita, un confronto fra politiche costruito su
copie congelate diverse. Qui si verifica quello che si puo' verificare senza
guardare l'immagine:

  * il valore finale della media progressiva **e'** la media riportata nelle
    tabelle, quindi la figura e il testo non possono divergere;
  * gli attacchi non rilevati cumulati finiscono sulla somma dei falsi negativi
    del rendiconto;
  * gli episodi di inversione sono le sequenze consecutive di blocchi sotto 0,5,
    contate ignorando i blocchi senza AUROC definita, che non aprono e non
    chiudono un episodio;
  * il confronto fra politiche si rifiuta di disegnare se le copie congelate non
    coincidono;
  * i numeri nelle didascalie usano la convenzione dei documenti.

Che l'immagine sia leggibile non lo decide una prova: quello si guarda.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

QUI = Path(__file__).resolve().parent


def _carica(nome, etichetta):
    for c in (QUI / nome, QUI.parent / nome, QUI.parent / 'ton_temporal' / nome):
        if c.is_file():
            spec = importlib.util.spec_from_file_location(etichetta, c)
            m = importlib.util.module_from_spec(spec)
            sys.modules[etichetta] = m
            spec.loader.exec_module(m)
            return m
    pytest.skip(f'{nome} non trovato accanto a questa suite')


CR = _carica('curve_replay.py', 'curve_replay')


def _cartella_evidenza():
    for c in (QUI / 'replay_evidenza', QUI.parent / 'replay_evidenza',
              QUI.parent / 'ton_temporal' / 'replay_evidenza'):
        if (c / 'replay_C_calibrato_seme42.json').is_file():
            return c
    return None


EV = _cartella_evidenza()
if EV is None:
    pytest.skip('rendiconti non trovati', allow_module_level=True)


@pytest.fixture(scope='module')
def calibrato():
    with open(EV / 'replay_C_calibrato_seme42.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def politiche_tutte():
    fuori = {}
    for nome, f in (('ogni_blocco', 'replay_C_calibrato_seme42.json'),
                    ('evidenza', 'replay_C_evidenza_seme42.json'),
                    ('casuale', 'replay_C_casuale_seme42.json')):
        with open(EV / f, encoding='utf-8') as fh:
            fuori[nome] = json.load(fh)
    return fuori


# --------------------------------------------------------------------------
# 1. I numeri della figura sono quelli del rendiconto
# --------------------------------------------------------------------------

def test_la_media_progressiva_finisce_sulla_media_pubblicata(calibrato):
    """Il valore finale va confrontato con la TABELLA, non con se stesso.

    La prima versione di questa prova confrontava `media_corrente(v)[-1]` con
    `nanmean(v)`: la funzione contro la propria definizione, sulla stessa
    serie. Passava per costruzione, e infatti e' passata mentre la frase che
    enunciava — «il valore finale e' la media riportata nelle tabelle» — era
    falsa, perche' le tabelle del §10ter sono medie sui cinque semi mentre la
    figura e' sul solo seme 42. Qui si legge il valore dalla tabella per seme
    pubblicata in `confronto_politiche.json`.
    """
    with open(EV / 'confronto_politiche.json', encoding='utf-8') as f:
        cp = json.load(f)
    seme = str(calibrato['parametri']['seme'])
    for m in ('lr', 'mlp', 'kan'):
        tabella = cp['per_seme'][m]['per_seme'][seme]
        for variante, voce in (('congelato', 'congelato'), ('adattivo', 'ogni_blocco')):
            v = CR.serie(calibrato['per_blocco'], variante, m, 'auroc')
            finale = CR.media_corrente(v)[-1]
            atteso = tabella[voce]['auroc']
            assert abs(round(finale, 4) - atteso) < 5e-5, (
                '%s_%s: la figura finirebbe su %.4f, la tabella per seme dice %.4f'
                % (variante, m, finale, atteso))


def test_il_valore_finale_non_e_la_media_sui_cinque_semi(calibrato):
    """Il contrario di quello che i documenti dicevano, messo per iscritto.

    Serve a impedire che la frase sbagliata torni: se un giorno le due cose
    coincidessero, questa prova cadrebbe e si potrebbe riscrivere la frase.
    """
    with open(EV / 'confronto_politiche.json', encoding='utf-8') as f:
        cp = json.load(f)
    diversi = 0
    for m in ('lr', 'mlp', 'kan'):
        v = CR.serie(calibrato['per_blocco'], 'adattivo', m, 'auroc')
        if abs(round(CR.media_corrente(v)[-1], 4)
               - cp['tabella'][m]['ogni_blocco']['auroc']) > 5e-5:
            diversi += 1
    assert diversi > 0, ('la media del seme 42 e quella sui cinque semi ora '
                         'coincidono: la didascalia va riscritta')


def test_la_media_progressiva_ignora_i_blocchi_senza_misura():
    v = np.array([0.8, np.nan, 0.6, np.nan], dtype=float)
    p = CR.media_corrente(v)
    assert p[0] == 0.8
    assert p[1] == 0.8, 'un blocco senza misura non deve spostare la media'
    assert abs(p[2] - 0.7) < 1e-12
    assert abs(p[3] - 0.7) < 1e-12


def test_la_media_progressiva_prima_di_ogni_misura_e_non_definita():
    p = CR.media_corrente(np.array([np.nan, np.nan, 0.5]))
    assert np.isnan(p[0]) and np.isnan(p[1]) and p[2] == 0.5


def test_gli_attacchi_non_rilevati_sommano_ai_falsi_negativi(politiche_tutte):
    """Tutte le curve cumulate disegnate, non solo le due della figura a due
    serie: anche le quattro del confronto fra politiche."""
    contate = 0
    for politica, d in politiche_tutte.items():
        B = d['per_blocco']
        for m in ('lr', 'mlp', 'kan'):
            for variante in ('congelato', 'adattivo'):
                atteso = sum(b[f'{variante}_{m}']['fn'] for b in B)
                assert CR.non_rilevati(B, variante, m)[-1] == atteso, \
                    f'{politica}/{variante}_{m}'
                contate += 1
    assert contate == 18, f'{contate} curve controllate invece di 18'


# --------------------------------------------------------------------------
# 2. Gli episodi di inversione
# --------------------------------------------------------------------------

def test_gli_episodi_sono_le_sequenze_consecutive_sotto_mezzo():
    v = np.array([0.8, 0.4, 0.3, 0.9, 0.2, 0.7], dtype=float)
    assert CR.episodi_inversione(v) == [1, 4]
    assert CR.lunghezze_episodi(v) == [2, 1]


def test_le_lunghezze_stanno_nello_stesso_ordine_degli_inizi():
    """Il difetto che questa prova presidia e' stato commesso davvero.

    `lunghezze_episodi` restituiva le lunghezze ordinate per grandezza mentre
    `episodi_inversione` restituisce gli inizi in ordine di tempo. Accoppiate,
    attribuivano a un episodio la lunghezza di un altro, e la figura del
    recupero mediava gli episodi sbagliati. Qui la serie e' costruita apposta
    perche' i due ordini **non** coincidano: un episodio corto prima di uno
    lungo.
    """
    #        0    1    2    3    4    5    6    7    8
    v = np.array([0.4, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9], dtype=float)
    inizi = CR.episodi_inversione(v)
    lunghezze = CR.lunghezze_episodi(v)
    assert inizi == [0, 2]
    assert lunghezze == [1, 6], 'le lunghezze devono seguire il tempo, non la grandezza'
    assert lunghezze != sorted(lunghezze, reverse=True), \
        'la serie di prova non distingue i due ordini: non proverebbe nulla'
    assert CR.episodi_lunghi(v, 5) == [2], \
        'con l accoppiamento sbagliato qui uscirebbe l episodio che comincia a 0'


def test_l_accoppiamento_sbagliato_cambia_quali_episodi_si_mediano(calibrato):
    """Sul caso vero in cui il difetto si vedeva: l'MLP del seme 42."""
    cong = CR.serie(calibrato['per_blocco'], 'congelato', 'mlp', 'auroc')
    inizi = CR.episodi_inversione(cong)
    lunghezze = CR.lunghezze_episodi(cong)
    assert lunghezze == [122, 2, 12, 1, 1, 4, 8, 1, 2, 1]
    assert CR.episodi_lunghi(cong, 5) == [0, 174, 752]
    sbagliati = [i for i, lun in zip(inizi, sorted(lunghezze, reverse=True))
                 if lun >= 5]
    assert sbagliati == [0, 171, 174], 'il caso di prova non e piu quello storico'
    assert sbagliati != CR.episodi_lunghi(cong, 5)


def test_un_blocco_senza_auroc_non_apre_e_non_chiude_un_episodio():
    """Il blocco non misurabile viene saltato, non interpretato.

    Se un `NaN` chiudesse l'episodio, due blocchi invertiti separati da un
    blocco senza normali diventerebbero due episodi invece di uno, e il
    conteggio della didascalia sarebbe inventato.
    """
    v = np.array([0.8, 0.4, np.nan, 0.3, 0.9], dtype=float)
    assert CR.episodi_inversione(v) == [1]
    assert CR.lunghezze_episodi(v) == [2]


def test_un_flusso_mai_invertito_non_ha_episodi():
    assert CR.episodi_inversione(np.array([0.8, 0.9, 0.7])) == []
    assert CR.lunghezze_episodi(np.array([0.8, 0.9, 0.7])) == []


def test_un_flusso_sempre_invertito_e_un_episodio_solo():
    v = np.array([0.4, 0.3, 0.2], dtype=float)
    assert CR.episodi_inversione(v) == [0]
    assert CR.lunghezze_episodi(v) == [3]


def test_gli_episodi_del_seme_42_sono_quelli_dichiarati(calibrato):
    """Sul seme 42 la logistica ha 174 blocchi invertiti in 9 episodi."""
    cong = CR.serie(calibrato['per_blocco'], 'congelato', 'lr', 'auroc')
    assert int(np.nansum(cong < 0.5)) == 174
    assert len(CR.episodi_inversione(cong)) == 9
    assert CR.lunghezze_episodi(cong) == [122, 10, 36, 1, 1, 1, 1, 1, 1]
    assert sorted(CR.lunghezze_episodi(cong), reverse=True) == \
        [122, 36, 10, 1, 1, 1, 1, 1, 1], 'e la forma che le didascalie scrivono'


# --------------------------------------------------------------------------
# 3. Il confronto fra politiche non disegna se le premesse cadono
# --------------------------------------------------------------------------

def test_il_confronto_si_ferma_se_le_copie_congelate_differiscono(tmp_path, calibrato):
    import copy
    guasto = copy.deepcopy(calibrato)
    guasto['per_blocco'][3]['congelato_lr']['fn'] += 1
    with pytest.raises(SystemExit, match='non coincide'):
        CR.politiche({'ogni_blocco': calibrato, 'evidenza': guasto}, 'lr',
                     tmp_path / 'x.png')


def test_il_confronto_si_ferma_anche_a_totale_uguale(tmp_path, calibrato):
    """Il caso che una guardia sul solo totale lascerebbe passare.

    Spostare un falso negativo da un blocco a un altro conserva la somma: le
    due curve cumulate si chiudono sullo stesso punto e divergono nel mezzo.
    """
    import copy
    guasto = copy.deepcopy(calibrato)
    guasto['per_blocco'][3]['congelato_lr']['fn'] += 1
    guasto['per_blocco'][4]['congelato_lr']['fn'] -= 1
    atteso = CR.non_rilevati(calibrato['per_blocco'], 'congelato', 'lr')[-1]
    assert CR.non_rilevati(guasto['per_blocco'], 'congelato', 'lr')[-1] == atteso, \
        'il guasto costruito deve conservare il totale, altrimenti non prova nulla'
    with pytest.raises(SystemExit, match='falsi negativi'):
        CR.politiche({'ogni_blocco': calibrato, 'evidenza': guasto}, 'lr',
                     tmp_path / 'x.png')


def test_il_confronto_si_ferma_se_le_auroc_congelate_differiscono(tmp_path, calibrato):
    """Il pannello di destra disegna le AUROC: anche quelle vanno confrontate."""
    import copy
    guasto = copy.deepcopy(calibrato)
    for b in guasto['per_blocco']:
        if b['congelato_lr'].get('auroc') is not None:
            b['congelato_lr']['auroc'] = 0.5
            break
    with pytest.raises(SystemExit, match='AUROC'):
        CR.politiche({'ogni_blocco': calibrato, 'evidenza': guasto}, 'lr',
                     tmp_path / 'x.png')


def test_il_confronto_si_ferma_se_i_blocchi_sono_di_numero_diverso(tmp_path, calibrato):
    import copy
    corto = copy.deepcopy(calibrato)
    corto['per_blocco'] = corto['per_blocco'][:-1]
    with pytest.raises(SystemExit, match='blocchi'):
        CR.politiche({'ogni_blocco': calibrato, 'evidenza': corto}, 'lr',
                     tmp_path / 'x.png')


def test_il_confronto_disegna_quando_le_premesse_tengono(tmp_path, calibrato):
    u = CR.politiche({'ogni_blocco': calibrato}, 'lr', tmp_path / 'politiche.png')
    assert Path(u).is_file() and Path(u).stat().st_size > 10_000


# --------------------------------------------------------------------------
# 4. Le figure si producono, e dichiarano il regime di soglia
# --------------------------------------------------------------------------

def test_le_cinque_forme_si_producono_su_tutti_i_modelli(tmp_path, politiche_tutte):
    prodotte = 0
    for m in ('lr', 'mlp', 'kan'):
        attese = {
            'curve_per_blocco': CR.curve(
                politiche_tutte['ogni_blocco'], m, tmp_path / f'curve_{m}.png'),
            'guadagno_per_fascia': CR.fasce(
                politiche_tutte['ogni_blocco'], m, tmp_path / f'fasce_{m}.png'),
            'attacchi_non_rilevati': CR.attacchi_persi(
                politiche_tutte['ogni_blocco'], m, tmp_path / f'fn_{m}.png'),
            'recupero': CR.recupero(
                politiche_tutte['ogni_blocco'], m, tmp_path / f'rec_{m}.png'),
            'politiche': CR.politiche(politiche_tutte, m, tmp_path / f'pol_{m}.png'),
        }
        for nome, percorso in attese.items():
            if percorso is None:
                assert nome == 'recupero' and m == 'kan', \
                    f'{nome}_{m} non prodotta e l astensione non e prevista'
                continue
            q = Path(percorso)
            assert q.is_file(), f'{nome}_{m} non prodotta'
            assert q.stat().st_size > 10_000, f'{nome}_{m} sospettamente piccola'
            prodotte += 1
    assert prodotte == 14, f'{prodotte} figure invece delle 14 pubblicate'


def test_il_regime_di_soglia_e_dichiarato(calibrato):
    """Gli attacchi non rilevati dipendono dal punto di decisione: la figura
    deve dire quale, perche' lo stesso modello dà conteggi diversi."""
    assert 'soglia scelta su B' in CR.regime_soglia(calibrato)
    assert CR.regime_soglia({'soglia_di_decisione': {'modo': 'zero'}}) == 'soglia a zero'
    assert 'non dichiarata' in CR.regime_soglia({})


def test_i_numeri_delle_didascalie_usano_la_convenzione_dei_documenti():
    assert CR.italiano(9150673) == '9.150.673'
    assert CR.italiano(-1377545) == '−1.377.545'
    assert CR.italiano(0) == '0'


def test_anche_le_tacche_degli_assi_usano_quella_convenzione():
    """Un asse che scrive «0.2» accanto a un testo che scrive «0,2» si legge
    come due misure diverse."""
    assert CR.tacca_italiana(0.2, 1) == '0,2'
    assert CR.tacca_italiana(-0.6, 1) == '−0,6'
    assert CR.tacca_italiana(1000, 0) == '1.000'
    assert CR.tacca_italiana(0, 0) == '0'
    assert CR.tacca_italiana(-1400, 0) == '−1.400'
    assert CR.tacca_italiana(12345.6, 1) == '12.345,6'


def test_le_tacche_di_un_asse_hanno_tutte_le_stesse_cifre():
    """Un asse da 0 a 1 scriveva «0,9» e poi «1»: sembra che cambi unità."""
    assert CR._cifre_decimali([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) == 1
    assert CR._cifre_decimali([0, 200, 400]) == 0
    assert CR._cifre_decimali([0.0, 0.25, 0.5]) == 2
    cifre = CR._cifre_decimali([0.0, 0.5, 1.0])
    assert [CR.tacca_italiana(x, cifre) for x in (0.0, 0.5, 1.0)] == \
        ['0,0', '0,5', '1,0']


def test_le_tacche_dell_asse_vero_sono_coerenti():
    """Non la funzione da sola: l'asse come `stile` lo installa."""
    fig, ax = CR.plt.subplots(figsize=(4.0, 2.0))
    CR.stile(ax)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1000)
    fig.canvas.draw()
    etichette = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert etichette and all(',' in e for e in etichette), \
        'tacche con e senza decimali sullo stesso asse: %s' % etichette
    assert any('.' in t.get_text() for t in ax.get_xticklabels() if t.get_text()), \
        'le migliaia dell asse orizzontale non usano il punto'
    CR.plt.close(fig)


# --------------------------------------------------------------------------
# 5. Nessun testo esce dal foglio
# --------------------------------------------------------------------------

def _figura_di_prova(testo, larghezza=4.0):
    fig, ax = CR.plt.subplots(figsize=(larghezza, 2.0))
    CR.stile(ax)
    ax.plot([0, 1], [0, 1])
    fig.text(0.012, 0.02, testo, fontsize=8.5)
    return fig


def test_la_figura_non_si_salva_con_un_testo_tagliato(tmp_path):
    """La prova del difetto: una didascalia piu' larga del foglio.

    Non e' ipotetico: una riga troppo lunga non fa cadere il disegno, esce dal
    bordo in silenzio e la figura resta pubblicabile con mezza frase.
    """
    fig = _figura_di_prova('una riga molto piu lunga del foglio ' * 6)
    with pytest.raises(SystemExit, match='fuori dal foglio'):
        CR.salva(fig, tmp_path / 'tagliata.png')
    assert not (tmp_path / 'tagliata.png').exists(), \
        'la figura guasta non deve restare sul disco'


def test_la_figura_si_salva_quando_i_testi_stanno_dentro(tmp_path):
    fig = _figura_di_prova('una riga corta')
    assert Path(CR.salva(fig, tmp_path / 'buona.png')).is_file()


def test_le_tacche_fuori_scala_non_contano_come_testo_tagliato(tmp_path):
    """Regressione: la prima versione della guardia cadeva su etichette di
    tacca che matplotlib tiene fra gli oggetti ma non disegna, perche' fuori
    dall'intervallo visibile. Segnalavano un difetto invisibile al lettore."""
    fig, ax = CR.plt.subplots(figsize=(4.0, 2.0))
    CR.stile(ax)
    ax.plot([0, 1000], [0, 1000])
    ax.set_xlim(0, 300)          # le tacche a 400, 600, 800, 1000 restano fuori
    ax.set_ylim(0, 300)
    fig.tight_layout()
    assert CR.testi_fuori_dal_foglio(fig) == []
    CR.plt.close(fig)


def test_i_guadagni_per_fascia_disegnati_sono_quelli_ricalcolati(calibrato):
    """I valori scritti sulle barre compaiono nei documenti: vanno ricalcolati."""
    B = calibrato['per_blocco']
    attesi = {'lr': (-0.132, -0.035, -0.032, 0.409),
              'mlp': (-0.183, -0.100, -0.073, 0.438),
              'kan': (-0.199, -0.086, -0.160, 0.045)}
    limiti = ((0, 10), (10, 50), (50, 200), (200, np.inf))
    conteggi = None
    for m, valori in attesi.items():
        cong = CR.serie(B, 'congelato', m, 'auroc')
        adat = CR.serie(B, 'adattivo', m, 'auroc')
        norm = np.array([b['adattivo_' + m]['normali'] for b in B], dtype=float)
        misurati, quanti = [], []
        for lo, hi in limiti:
            k = (~np.isnan(cong)) & (~np.isnan(adat)) & (norm >= lo) & (norm < hi)
            misurati.append(round(float(np.mean(adat[k] - cong[k])), 3))
            quanti.append(int(k.sum()))
        assert misurati == [round(x, 3) for x in valori], \
            '%s: fasce %s invece di %s' % (m, misurati, list(valori))
        assert sum(x > 0 for x in misurati) == 1 and misurati[-1] > 0, \
            '%s: il guadagno non e positivo nella sola fascia ricca' % m
        if conteggi is None:
            conteggi = quanti
        assert quanti == conteggi, 'le fasce cambiano fra i modelli'
    assert conteggi == [221, 275, 133, 246]
    assert sum(conteggi) == 875, 'le fasce non coprono i blocchi misurabili'


# --------------------------------------------------------------------------
# 6. La didascalia si spezza dove entra
# --------------------------------------------------------------------------

def test_la_didascalia_si_spezza_su_piu_righe_quando_serve():
    fig, _ = CR.plt.subplots(figsize=(4.0, 2.0))
    righe = CR.spezza_didascalia(fig, ['parola ' * 60])
    assert len(righe) > 1
    CR.plt.close(fig)


def test_ogni_riga_della_didascalia_entra_davvero_nel_foglio():
    """La larghezza e' misurata sul motore di disegno, non stimata a caratteri:
    il carattere non e' a larghezza fissa e una stima sbaglia in entrambi i
    versi."""
    from matplotlib.font_manager import FontProperties
    fig, _ = CR.plt.subplots(figsize=(6.0, 2.0))
    righe = CR.spezza_didascalia(fig, ['una didascalia lunga da spezzare ' * 8])
    ren = fig.canvas.get_renderer()
    prop = FontProperties(size=8.5)
    disponibile = CR._larghezza_disponibile(fig, 0.012)
    for r in righe:
        assert ren.get_text_width_height_descent(r, prop, False)[0] <= disponibile, \
            'riga piu larga dello spazio: %r' % r
    CR.plt.close(fig)


# --------------------------------------------------------------------------
# 7. Il recupero si astiene quando gli episodi non bastano
# --------------------------------------------------------------------------

def test_gli_episodi_lunghi_sono_quelli_sopra_il_minimo():
    v = np.array([0.4, 0.4, 0.4, 0.9, 0.4, 0.9], dtype=float)
    assert CR.episodi_lunghi(v, 3) == [0]
    assert CR.episodi_lunghi(v, 1) == [0, 4]
    assert CR.episodi_lunghi(v, 4) == []


def test_il_recupero_non_si_disegna_con_meno_di_due_episodi_lunghi(tmp_path, calibrato):
    """Sul modello additivo gli episodi sono tre, lunghi 7, 2 e 1 blocchi: uno
    solo arriva al minimo. La media di un episodio e' quell'episodio, e
    presentarla come una curva media direbbe piu' del dato."""
    cong = CR.serie(calibrato['per_blocco'], 'congelato', 'kan', 'auroc')
    assert CR.lunghezze_episodi(cong) == [2, 7, 1]
    assert sorted(CR.lunghezze_episodi(cong), reverse=True) == [7, 2, 1]
    assert CR.recupero(calibrato, 'kan', tmp_path / 'r.png') is None
    assert not (tmp_path / 'r.png').exists()


def test_il_recupero_si_disegna_dove_gli_episodi_bastano(tmp_path, calibrato):
    for m in ('lr', 'mlp'):
        u = CR.recupero(calibrato, m, tmp_path / ('r_%s.png' % m))
        assert u is not None and Path(u).is_file()


# --------------------------------------------------------------------------
# 8. La cartella pubblicata e quello che i documenti citano
# --------------------------------------------------------------------------

ATTESE = {'%s_%s.png' % (forma, m)
          for forma in ('curve_per_blocco', 'guadagno_per_fascia',
                        'attacchi_non_rilevati', 'politiche')
          for m in ('lr', 'mlp', 'kan')} | {'recupero_lr.png', 'recupero_mlp.png'}


def _cartella_figure():
    for c in (QUI / 'figure', QUI.parent / 'figure',
              QUI.parent / 'ton_temporal' / 'figure'):
        if c.is_dir():
            return c
    return None


FIG = _cartella_figure()


@pytest.mark.skipif(FIG is None, reason='cartella figure non presente')
def test_la_cartella_contiene_esattamente_le_figure_attese():
    presenti = {p.name for p in FIG.glob('*.png')}
    assert presenti == ATTESE, (
        'mancano %s; in piu %s'
        % (sorted(ATTESE - presenti) or 'niente', sorted(presenti - ATTESE) or 'niente'))


@pytest.mark.skipif(FIG is None, reason='cartella figure non presente')
def test_la_figura_di_cui_il_programma_si_astiene_non_e_pubblicata():
    """Se un giorno comparisse, vorrebbe dire che e' stata prodotta a mano."""
    assert not (FIG / 'recupero_kan.png').exists()


@pytest.mark.skipif(FIG is None, reason='cartella figure non presente')
def test_i_documenti_citano_solo_figure_che_esistono():
    import re
    documenti = [p for p in (QUI / 'protocol.md', QUI / 'sintesi_pilota.md',
                             FIG / 'LEGGIMI.md') if p.is_file()]
    assert documenti, 'nessun documento trovato'
    citate, dove = set(), {}
    for p in documenti:
        for nome in re.findall(r'[a-z_]+_(?:lr|mlp|kan)\.png', p.read_text(encoding='utf-8')):
            citate.add(nome)
            dove.setdefault(nome, set()).add(p.name)
    assert citate, 'nessun documento cita una figura: la scheda le chiede citate'
    # `recupero_kan.png` compare nei documenti per dire che **non** esiste: e'
    # una dichiarazione di astensione, non una citazione. Va esclusa qui e
    # pretesa nella prova seguente.
    mancanti = {n: sorted(dove[n]) for n in citate - {'recupero_kan.png'}
                if not (FIG / n).is_file()}
    assert not mancanti, 'figure citate e non presenti: %s' % mancanti


@pytest.mark.skipif(FIG is None, reason='cartella figure non presente')
def test_i_numeri_che_i_documenti_prendono_dalle_figure_sono_ricalcolati():
    """I valori per blocco sono il punto cieco del controllo numerico generale,
    che non entra nei `per_blocco` dei rendiconti. Qui si ricalcolano dalle
    stesse serie che disegnano le figure e si pretende che i documenti li
    scrivano cosi' come sono, nella convenzione dei documenti.
    """
    sorgenti = {}
    for nome, f in (('ogni_blocco', 'replay_C_calibrato_seme42.json'),
                    ('evidenza', 'replay_C_evidenza_seme42.json'),
                    ('casuale', 'replay_C_casuale_seme42.json')):
        with open(EV / f, encoding='utf-8') as fh:
            sorgenti[nome] = json.load(fh)

    B = sorgenti['ogni_blocco']['per_blocco']
    attesi = {}
    # La differenza di attacchi non rilevati sulla logistica. I documenti la
    # scrivono in valore assoluto perche' il verso lo portano le parole («in
    # meno»), la figura col meno tipografico: si controllano separatamente la
    # grandezza, qui, e il verso, sotto.
    cong = CR.non_rilevati(B, 'congelato', 'lr')[-1]
    adat = CR.non_rilevati(B, 'adattivo', 'lr')[-1]
    assert adat < cong, 'sulla logistica l adattivo non perde meno attacchi del congelato'
    attesi['differenza lr'] = CR.italiano(abs(int(adat - cong)))
    # il caso in cui il sorteggio perde piu' del congelato, sull'additivo
    attesi['congelato kan'] = CR.italiano(
        int(CR.non_rilevati(B, 'congelato', 'kan')[-1]))
    attesi['casuale kan'] = CR.italiano(int(CR.non_rilevati(
        sorgenti['casuale']['per_blocco'], 'adattivo', 'kan')[-1]))

    testi = {p.name: p.read_text(encoding='utf-8')
             for p in (QUI / 'sintesi_pilota.md', QUI / 'rapporto_pilota.md',
                       FIG / 'LEGGIMI.md') if p.is_file()}
    assert testi, 'nessun documento da controllare'
    for quale, valore in attesi.items():
        dove = [n for n, t in testi.items() if valore in t]
        assert dove, ('nessun documento scrive %s per «%s»: o il numero e '
                      'cambiato, o il documento ne scrive un altro'
                      % (valore, quale))

    # e il verso del risultato scomodo, che e' quello che conta
    assert int(CR.non_rilevati(sorgenti['casuale']['per_blocco'], 'adattivo',
                               'kan')[-1]) > int(CR.non_rilevati(B, 'congelato',
                                                                 'kan')[-1]), \
        'il sorteggio sull additivo non perde piu del congelato: i documenti lo dicono'


def test_nelle_didascalie_non_tornano_le_frasi_ritirate(tmp_path, politiche_tutte):
    """Le didascalie sono testo pubblicato come i documenti, e una frase
    corretta nel documento puo' restare sbagliata dentro l'immagine: li' nessun
    controllo sul testo la vede. Queste sono le formulazioni ritirate dopo una
    verifica, con accanto il motivo.
    """
    ritirate = {
        'riportata nelle tabelle':
            'il valore finale e la media del seme, non quella sui cinque semi',
        '\u00a710ter':
            'il \u00a710ter non pubblica le AUROC per seme: stanno in '
            'confronto_politiche.json',
        'per l\u2019intero flusso':
            'sull additivo il congelato non e sopra in ogni punto',
        'primi duecento':
            'nei primi 213 blocchi e al piu una politica a stare davanti',
    }
    import matplotlib.text
    testi = []
    originale = CR.salva

    def spia(fig, uscita):
        for oggetto in fig.findobj(matplotlib.text.Text):
            if oggetto.get_visible() and oggetto.get_text().strip():
                testi.append((Path(uscita).name, oggetto.get_text()))
        return originale(fig, uscita)

    CR.salva = spia
    try:
        for m in ('lr', 'mlp', 'kan'):
            CR.curve(politiche_tutte['ogni_blocco'], m, tmp_path / f'a_{m}.png')
            CR.fasce(politiche_tutte['ogni_blocco'], m, tmp_path / f'b_{m}.png')
            CR.attacchi_persi(politiche_tutte['ogni_blocco'], m, tmp_path / f'c_{m}.png')
            CR.recupero(politiche_tutte['ogni_blocco'], m, tmp_path / f'd_{m}.png')
            CR.politiche(politiche_tutte, m, tmp_path / f'e_{m}.png')
    finally:
        CR.salva = originale

    assert len(testi) > 300, 'non ho raccolto i testi delle figure'
    guasti = [(f, frase, perche) for f, testo in testi
              for frase, perche in ritirate.items() if frase in testo]
    assert not guasti, '\n'.join(
        '%s: «%s» — %s' % g for g in guasti)

    # E una affermazione positiva: i confronti sulle impronte sono il doppio
    # dei blocchi, perche' il primo rendiconto fa da riferimento agli altri
    # due. Chiamarli «blocchi» raddoppierebbe un numero disegnato sull'asse.
    blocchi = len(politiche_tutte['ogni_blocco']['per_blocco'])
    attesa = '%s confronti' % CR.italiano(blocchi * (len(politiche_tutte) - 1))
    didascalie = [testo for f, testo in testi if f.startswith('e_')]
    assert any(attesa in testo for testo in didascalie), \
        'nessuna didascalia del confronto fra politiche dichiara «%s»' % attesa
    assert not any('%s blocchi' % CR.italiano(blocchi * (len(politiche_tutte) - 1))
                   in testo for testo in didascalie), \
        'i confronti sono dichiarati come blocchi: sono il doppio'


@pytest.mark.skipif(FIG is None, reason='cartella figure non presente')
def test_l_astensione_resta_dichiarata_nei_documenti():
    """Una figura che manca senza che nessuno lo dica si legge come una
    dimenticanza. L'assenza va scritta dove il lettore la cerca."""
    for p in (FIG / 'LEGGIMI.md', QUI / 'protocol.md'):
        if not p.is_file():
            continue
        testo = p.read_text(encoding='utf-8')
        assert 'recupero_kan.png' in testo, \
            '%s non dichiara perche la figura del recupero sull additivo manca' % p.name

"""Le due aggregazioni del FPR: che siano distinte, e che possano avere verso opposto.

Perche' questa suite esiste
---------------------------
Il primo riepilogo del pilota riportava una sola aggregazione senza nominarla, e
ne traeva una conclusione assoluta: «i falsi allarmi aumentano sempre». Sui dati
di C quella frase e' vera sulla media per blocco e falsa sul complessivo per due
modelli su tre, perche' i normali sono concentrati in una minoranza di blocchi.

Un errore di questo tipo non si previene rileggendo: si previene calcolando
entrambe le misure e fissando con una prova che possano divergere. I casi qui
sono costruiti a mano, con numeri scelti perche' la divergenza sia verificabile
a mente, e non dipendono da alcun risultato del replay.

Che cosa presidia
-----------------
  - la media per blocco pesa i blocchi, il complessivo pesa i normali;
  - su una distribuzione concentrata le due possono avere SEGNO OPPOSTO;
  - i blocchi senza normali non entrano nella media per blocco;
  - un blocco con AUROC non definita non e' ne' invertito ne' non invertito;
  - la soglia dell'inversione e' stretta: AUROC esattamente 0,5 non e' inversione;
  - i controlli di coerenza sui rendiconti si accorgono di una matrice falsa.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

QUI = Path(__file__).resolve().parent


def _carica():
    for candidato in (QUI / 'riepiloghi_semi.py', QUI.parent / 'riepiloghi_semi.py',
                      QUI.parent / 'ton_temporal' / 'riepiloghi_semi.py'):
        if candidato.is_file():
            spec = importlib.util.spec_from_file_location('riepiloghi_semi', candidato)
            modulo = importlib.util.module_from_spec(spec)
            sys.modules['riepiloghi_semi'] = modulo
            spec.loader.exec_module(modulo)
            return modulo
    pytest.skip('riepiloghi_semi.py non trovato accanto alla suite')


RS = _carica()


def matrice(normali, fp, attacchi=100, vp=None, auroc=0.8):
    """Una cella del rendiconto, coerente per costruzione."""
    vp = attacchi if vp is None else vp
    return {
        'righe': normali + attacchi,
        'normali': normali,
        'attacchi': attacchi,
        'fp': fp,
        'vn': normali - fp,
        'vp': vp,
        'fn': attacchi - vp,
        'falsi_allarmi': (fp / normali) if normali else 0.0,
        'auroc': auroc,
    }


def blocco(numero, normali, fp_cong, fp_adat, auroc_cong=0.8, auroc_adat=0.9,
           modelli=('lr',)):
    b = {'blocco': numero, 'righe': normali + 100}
    for m in modelli:
        b[f'congelato_{m}'] = matrice(normali, fp_cong, auroc=auroc_cong)
        b[f'adattivo_{m}'] = matrice(normali, fp_adat, auroc=auroc_adat)
    return b


def rendiconto(blocchi, seme=1):
    return {'parametri': {'seme': seme}, 'per_blocco': blocchi}


# --- le due aggregazioni sono misure diverse -------------------------------

def test_le_due_aggregazioni_coincidono_quando_i_normali_sono_uniformi():
    """Senza concentrazione non c'e' divergenza: e' il caso di controllo."""
    blocchi = [blocco(i, normali=10, fp_cong=2, fp_adat=5) for i in range(4)]
    medio, quanti = RS.fpr_medio_per_blocco(blocchi, 'congelato_lr')
    compl, fp, vn = RS.fpr_complessivo(blocchi, 'congelato_lr')
    assert quanti == 4
    assert medio == pytest.approx(0.2)
    assert compl == pytest.approx(0.2)
    assert (fp, vn) == (8, 32)


def test_la_media_per_blocco_pesa_i_blocchi_il_complessivo_pesa_i_normali():
    """Un blocco povero sbagliato tutto conta come un blocco ricco perfetto."""
    blocchi = [blocco(0, normali=1, fp_cong=1, fp_adat=1),
               blocco(1, normali=99, fp_cong=0, fp_adat=0)]
    medio, _ = RS.fpr_medio_per_blocco(blocchi, 'congelato_lr')
    compl, fp, vn = RS.fpr_complessivo(blocchi, 'congelato_lr')
    assert medio == pytest.approx(0.5)          # (1,0 + 0,0) / 2
    assert compl == pytest.approx(1 / 100)      # 1 falso positivo su 100 normali
    assert (fp, vn) == (1, 99)


def test_le_due_aggregazioni_possono_avere_verso_opposto():
    """Il caso che la nostra prima lettura aveva mancato, in piccolo.

    Nove blocchi con un normale ciascuno: l'adattamento li peggiora tutti.
    Un blocco con novantuno normali: l'adattamento lo migliora molto.
    La media per blocco sale, il complessivo scende.
    """
    poveri = [blocco(i, normali=1, fp_cong=0, fp_adat=1) for i in range(9)]
    ricco = [blocco(9, normali=91, fp_cong=80, fp_adat=40)]
    blocchi = poveri + ricco

    medio_c, _ = RS.fpr_medio_per_blocco(blocchi, 'congelato_lr')
    medio_a, _ = RS.fpr_medio_per_blocco(blocchi, 'adattivo_lr')
    compl_c, fp_c, _ = RS.fpr_complessivo(blocchi, 'congelato_lr')
    compl_a, fp_a, _ = RS.fpr_complessivo(blocchi, 'adattivo_lr')

    assert medio_a - medio_c > 0, 'sulla media per blocco deve peggiorare'
    assert compl_a - compl_c < 0, 'sul complessivo deve migliorare'
    assert fp_c - fp_a == 31
    # e i conti sono quelli attesi a mente
    assert medio_c == pytest.approx((0 * 9 + 80 / 91) / 10)
    assert medio_a == pytest.approx((1 * 9 + 40 / 91) / 10)
    assert compl_c == pytest.approx(80 / 100)
    assert compl_a == pytest.approx(49 / 100)


def test_i_blocchi_senza_normali_non_entrano_nella_media_per_blocco():
    blocchi = [blocco(0, normali=0, fp_cong=0, fp_adat=0),
               blocco(1, normali=10, fp_cong=5, fp_adat=5)]
    medio, quanti = RS.fpr_medio_per_blocco(blocchi, 'congelato_lr')
    assert quanti == 1, 'il blocco senza normali non e\' misurabile'
    assert medio == pytest.approx(0.5)


# --- bande di ricchezza ----------------------------------------------------

def test_le_bande_coprono_tutti_i_blocchi_e_la_variazione_si_chiude():
    blocchi = [blocco(0, normali=0, fp_cong=0, fp_adat=0),
               blocco(1, normali=5, fp_cong=0, fp_adat=3),
               blocco(2, normali=20, fp_cong=1, fp_adat=4),
               blocco(3, normali=120, fp_cong=10, fp_adat=20),
               blocco(4, normali=400, fp_cong=300, fp_adat=200)]
    righe, normali_totali = RS.bande_ricchezza(blocchi, 'lr')
    assert [r['banda'] for r in righe] == ['0', '1-9', '10-49', '50-199', '>=200']
    assert sum(r['blocchi'] for r in righe) == len(blocchi)
    assert normali_totali == 545
    assert sum(r['delta_fp'] for r in righe) == (3 + 3 + 10 - 100)
    assert righe[0]['delta_fpr_medio'] is None


# --- regime di inversione --------------------------------------------------

def test_auroc_non_definita_non_e_ne_invertito_ne_non_invertito():
    blocchi = [blocco(0, 10, 1, 1, auroc_cong=0.3, auroc_adat=0.9),
               blocco(1, 10, 1, 1, auroc_cong=0.7, auroc_adat=0.6),
               blocco(2, 10, 1, 1)]
    blocchi[2]['congelato_lr']['auroc'] = None
    blocchi[2]['adattivo_lr']['auroc'] = None
    r = RS.regime_inversione(blocchi, 'lr')
    assert r['blocchi_misurabili'] == 2
    assert r['blocchi_senza_auroc'] == 1
    assert r['invertiti'] == 1
    assert r['guadagno_invertiti'] == pytest.approx(0.6)
    assert r['guadagno_altrove'] == pytest.approx(-0.1)


def test_auroc_esattamente_un_mezzo_non_e_inversione():
    """La soglia e' stretta: 0,5 significa ordinamento casuale, non rovesciato."""
    blocchi = [blocco(0, 10, 1, 1, auroc_cong=0.5, auroc_adat=0.9),
               blocco(1, 10, 1, 1, auroc_cong=0.499999, auroc_adat=0.9)]
    r = RS.regime_inversione(blocchi, 'lr')
    assert r['invertiti'] == 1


def test_un_solo_blocco_misurabile_non_lascia_meta_del_riepilogo_vuota():
    blocchi = [blocco(0, 10, 1, 1, auroc_cong=0.2, auroc_adat=0.8)]
    r = RS.regime_inversione(blocchi, 'lr')
    assert r['invertiti'] == 1
    assert r['guadagno_invertiti'] == pytest.approx(0.6)
    assert r['guadagno_altrove'] is None, 'senza blocchi non invertiti deve essere assente'


# --- i controlli sui rendiconti si accorgono di un rendiconto falso --------

def test_una_matrice_che_non_torna_viene_respinta():
    b = blocco(0, normali=10, fp_cong=2, fp_adat=2)
    b['congelato_lr']['vn'] = 99          # fp + vn non fa piu' normali
    with pytest.raises(RS.Incoerenza, match='fp\\+vn'):
        RS.controlla_blocco(b, 0, 'lr')


def test_un_tasso_di_falsi_allarmi_incoerente_viene_respinto():
    b = blocco(0, normali=10, fp_cong=2, fp_adat=2)
    b['adattivo_lr']['falsi_allarmi'] = 0.9     # vero valore 0,2
    with pytest.raises(RS.Incoerenza, match='falsi_allarmi'):
        RS.controlla_blocco(b, 0, 'lr')


def test_le_quattro_celle_devono_sommare_alle_righe():
    b = blocco(0, normali=10, fp_cong=2, fp_adat=2)
    b['congelato_lr']['righe'] = 1000
    with pytest.raises(RS.Incoerenza, match='sommano a righe'):
        RS.controlla_blocco(b, 0, 'lr')


# --- il riepilogo di un seme intero ---------------------------------------

def test_riepiloga_un_seme_restituisce_le_due_aggregazioni_e_il_regime():
    blocchi = []
    for i in range(9):
        blocchi.append(blocco(i, normali=1, fp_cong=0, fp_adat=1,
                              auroc_cong=0.3, auroc_adat=0.9, modelli=RS.MODELLI))
    blocchi.append(blocco(9, normali=91, fp_cong=80, fp_adat=40,
                          auroc_cong=0.8, auroc_adat=0.7, modelli=RS.MODELLI))
    fuori = RS.riepiloga_un_seme(rendiconto(blocchi, seme=7))
    assert fuori['seme'] == 7
    assert fuori['blocchi'] == 10
    for m in RS.MODELLI:
        x = fuori['per_modello'][m]
        assert x['fpr_medio_per_blocco']['delta'] > 0
        assert x['fpr_complessivo']['delta'] < 0
        assert x['fpr_complessivo']['delta_fp'] == -31
        assert x['regime_inversione']['invertiti'] == 9
        assert x['normali_nello_stream'] == 100


def test_due_rendiconti_con_lo_stesso_seme_sono_un_errore(tmp_path):
    import json
    blocchi = [blocco(0, 10, 1, 2, modelli=RS.MODELLI)]
    for nome in ('a.json', 'b.json'):
        (tmp_path / nome).write_text(json.dumps(rendiconto(blocchi, seme=5)),
                                     encoding='utf-8')
    with pytest.raises(RS.Incoerenza, match='semi ripetuti'):
        RS.principale(['--rendiconti', str(tmp_path / 'a.json'), str(tmp_path / 'b.json')])


# --- confronto fra due insiemi di rendiconti ------------------------------

def _due_insiemi():
    """Gli stessi blocchi due volte: cambia solo la decisione, non l'ordinamento."""
    base, conf = [], []
    for i in range(3):
        b1 = blocco(i, normali=10, fp_cong=5, fp_adat=7, modelli=RS.MODELLI)
        b2 = blocco(i, normali=10, fp_cong=1, fp_adat=2, modelli=RS.MODELLI)
        for b in (b1, b2):
            b['row_id_campionati_sha256'] = RS.imp.impronta([1000 + i, 2000 + i])
            b['row_id_campionati_somma_storica'] = 3000 + 2 * i
        base.append(b1)
        conf.append(b2)
    return {7: base}, {7: conf}


def _aggr(blocchi):
    seme = next(iter(blocchi))
    r = RS.riepiloga_un_seme(rendiconto(blocchi[seme], seme=seme))
    return RS.aggrega_sui_semi([r])


def test_il_confronto_passa_quando_cambia_solo_la_decisione():
    b, c = _due_insiemi()
    esito = RS.confronta(b, c, _aggr(b), _aggr(c))
    assert esito['auroc_confrontate'] == 3 * len(RS.MODELLI) * 2
    assert set(esito['per_modello']) == set(RS.MODELLI)


def test_il_confronto_rifiuta_un_auroc_diverso():
    """Il controllo che conta: la soglia non puo' cambiare una misura di ordinamento."""
    b, c = _due_insiemi()
    c[7][1]['adattivo_lr']['auroc'] = 0.123
    with pytest.raises(RS.Incoerenza, match='misura di ordinamento'):
        RS.confronta(b, c, _aggr(b), _aggr(c))


def test_il_confronto_rifiuta_indici_campionati_diversi():
    b, c = _due_insiemi()
    c[7][2]['row_id_campionati_sha256'] = RS.imp.impronta([1, 2, 3])
    with pytest.raises(RS.Incoerenza, match='indici campionati diversi'):
        RS.confronta(b, c, _aggr(b), _aggr(c))


def test_il_confronto_vede_il_digest_quando_c_e():
    """Il confronto dichiara su che cosa si e' basato: qui sul digest."""
    b, c = _due_insiemi()
    esito = RS.confronta(b, c, _aggr(b), _aggr(c))
    assert esito['indici_campionati']['forza_minima'] == RS.imp.DIGEST
    assert esito['indici_campionati']['blocchi_confrontati'] == 3


def test_il_confronto_ripiega_sulla_somma_storica_e_lo_dichiara():
    """Sui rendiconti vecchi il digest non c'e': il confronto resta possibile ma
    piu' debole, e deve dirlo invece di passare in silenzio."""
    b, c = _due_insiemi()
    for insieme in (b, c):
        for voce in insieme[7]:
            del voce['row_id_campionati_sha256']
    esito = RS.confronta(b, c, _aggr(b), _aggr(c))
    assert esito['indici_campionati']['forza_minima'] == RS.imp.SOMMA_STORICA
    assert 'piu\' debole' in esito['indici_campionati']['descrizione']


def test_il_confronto_rifiuta_un_campione_non_verificabile():
    """Senza nessuno dei due campi non c'e' niente da verificare: non si passa."""
    b, c = _due_insiemi()
    for insieme in (b, c):
        for voce in insieme[7]:
            del voce['row_id_campionati_sha256']
            del voce['row_id_campionati_somma_storica']
    with pytest.raises(RS.Incoerenza, match='non sono verificabili'):
        RS.confronta(b, c, _aggr(b), _aggr(c))


def test_il_confronto_rifiuta_semi_diversi():
    b, c = _due_insiemi()
    altro = {8: c[7]}
    with pytest.raises(RS.Incoerenza, match='semi diversi'):
        RS.confronta(b, altro, _aggr(b), _aggr(c))


def test_il_confronto_rifiuta_un_numero_di_blocchi_diverso():
    b, c = _due_insiemi()
    c[7] = c[7][:2]
    with pytest.raises(RS.Incoerenza, match='blocchi contro'):
        RS.confronta(b, c, _aggr(b), _aggr(c))

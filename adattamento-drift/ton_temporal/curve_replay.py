"""Figure temporali del replay: congelato contro adattivo, politiche, recupero.

Produce i PNG dai rendiconti di `replay.py`, senza rileggere i dati grezzi:

  curve_per_blocco.png        AUROC e richiamo sui normali, blocco per blocco,
                              traccia grezza in chiaro e media mobile in evidenza
  guadagno_per_fascia.png     guadagno di AUROC dell'adattivo sul congelato,
                              per fascia di ricchezza di normali del blocco
  attacchi_non_rilevati.png   attacchi non rilevati, cumulati nel tempo, e la
                              differenza fra adattivo e congelato: e' la misura
                              per cui il rilevatore esiste
  recupero.png                le curve di recupero allineate all'inizio di ogni
                              episodio di inversione del congelato, senza
                              scegliere nessuna soglia di recupero
  politiche.png               le quattro politiche sugli stessi blocchi: attacchi
                              non rilevati cumulati e AUROC

Attenzione alla soglia. AUROC non dipende dalla soglia; gli attacchi non rilevati
si'. Le figure che contano i falsi negativi vanno quindi lette insieme al campo
`soglia_di_decisione` del rendiconto, e la didascalia dichiara quale regime e':
`soglia zero` o `soglia scelta su B`.

Scelte dichiarate. Le due misure hanno scale e significati diversi, quindi non
condividono un asse: sono due riquadri affiancati, uno per misura, ciascuno con
un solo asse verticale. Su 916 blocchi la traccia grezza e' illeggibile da sola,
percio' e' disegnata sottile e chiara e sopra passa una media mobile su 25
blocchi; la finestra e' scritta in didascalia. Le serie sono due sole, con
legenda e etichetta diretta a fine linea, quindi l'identita' non dipende dal
colore. I blocchi in cui una misura non e' definita sono esclusi da quella
misura, non posti a zero.

Colori: blu e arancio della tavolozza categorica, coppia verificata (Delta E
24,7 protan, 33,6 a visione normale, contrasto oltre 3:1 sulla superficie).
Per il guadagno, coppia divergente blu/rosso con lo zero come riferimento.
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.text
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter

try:
    import impronta_campione as _imp
except ImportError:          # la figura si fa anche senza, dichiarandolo
    _imp = None

SUPERFICIE = '#fcfcfb'
INCHIOSTRO = '#0b0b0b'
INCHIOSTRO_2 = '#52514e'
GRIGLIA = '#e3e2de'
CONGELATO = '#2a78d6'      # slot 1, blu
ADATTIVO = '#eb6834'       # slot 2, arancio
# quattro politiche: slot 1-4 della tavolozza categorica, nell'ordine fisso.
# Verificata col validatore: banda di luminosita', soglia di croma, separazione
# per deuteranopia e protanopia (peggiore adiacente dE 9,1 protan) e soglia a
# visione normale (22,9) tutte passate; il contrasto di aqua e giallo sulla
# superficie resta sotto 3:1, percio' ogni serie porta l'etichetta diritta a
# fine linea e non si affida al colore.
# Un episodio piu' corto di tanti blocchi non si puo' seguire per venti:
# la finestra dopo il suo inizio cadrebbe quasi tutta fuori dall'episodio.
LUNGHEZZA_MINIMA_EPISODIO = 5
POLITICHE = {'congelato': '#2a78d6', 'ogni_blocco': '#eb6834',
             'evidenza': '#1baf7a', 'casuale': '#eda100'}
NOMI_POLITICHE = {'congelato': 'congelato', 'ogni_blocco': 'ogni blocco',
                  'evidenza': 'su evidenza', 'casuale': 'casuale'}
NEGATIVO = '#e34948'       # polo caldo della coppia divergente
POSITIVO = '#2a78d6'       # polo freddo
FINESTRA = 25


def _cifre_decimali(valori):
    """Quante cifre decimali servono perche' le tacche si somiglino."""
    cifre = 0
    for x in valori:
        testo = '%g' % abs(x)
        if '.' in testo:
            cifre = max(cifre, len(testo.split('.')[1]))
    return cifre


def tacca_italiana(v, cifre=None):
    """Una tacca nella convenzione dei documenti: virgola decimale, punto per le
    migliaia, meno tipografico. Senza questo gli assi scrivono «0.2» e «-0.6»
    mentre il testo accanto scrive «0,2» e «\u22120,6»: due convenzioni nella
    stessa pagina si leggono come due misure diverse.

    Le cifre decimali sono quelle che l'asse richiede nel suo punto piu' fine,
    non quelle che il singolo valore richiederebbe: altrimenti un asse da 0 a 1
    scrive «0,9» e poi «1», e sembra che cambi unita'.
    """
    if not cifre:
        return italiano(int(round(v)))
    s = f'{abs(v):,.{cifre}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    return ('\u2212' if v < 0 else '') + s


def _formattatore(asse):
    def formatta(v, _pos=None):
        return tacca_italiana(v, _cifre_decimali(asse.get_ticklocs()))
    return formatta


def stile(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_formattatore(ax.xaxis)))
    ax.yaxis.set_major_formatter(FuncFormatter(_formattatore(ax.yaxis)))
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, color=GRIGLIA, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lato in ('top', 'right'):
        ax.spines[lato].set_visible(False)
    for lato in ('left', 'bottom'):
        ax.spines[lato].set_color(GRIGLIA)
    ax.tick_params(colors=INCHIOSTRO_2, labelsize=9, length=3, width=0.8)


def mobile(v, n):
    """Media mobile che ignora i valori non definiti."""
    v = np.asarray(v, dtype=float)
    fuori = np.full(len(v), np.nan)
    for i in range(len(v)):
        a, b = max(0, i - n // 2), min(len(v), i + n // 2 + 1)
        f = v[a:b]
        f = f[~np.isnan(f)]
        if len(f):
            fuori[i] = f.mean()
    return fuori


def serie(blocchi, variante, modello, campo):
    return np.array([
        (np.nan if b[variante + '_' + modello][campo] is None
         else b[variante + '_' + modello][campo]) for b in blocchi], dtype=float)


def curve(d, modello, uscita):
    B = d['per_blocco']
    x = np.arange(len(B))
    misure = (('auroc', 'AUROC'), ('richiamo_normali', 'richiamo sui normali'))
    fig, assi = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SUPERFICIE)
    for ax, (campo, titolo) in zip(assi, misure):
        stile(ax)
        fine = {}
        for variante, colore, nome in (('congelato', CONGELATO, 'congelato'),
                                       ('adattivo', ADATTIVO, 'adattivo')):
            v = serie(B, variante, modello, campo)
            ax.plot(x, v, color=colore, linewidth=0.5, alpha=0.13, zorder=2)
            m = mobile(v, FINESTRA)
            ax.plot(x, m, color=colore, linewidth=2.0, zorder=3, label=nome,
                    solid_capstyle='round')
            buoni = np.where(~np.isnan(m))[0]
            if len(buoni):
                fine[nome] = (x[buoni[-1]], float(m[buoni[-1]]), colore)
        # etichette dirette: se i due valori finali sono troppo vicini le
        # separo, altrimenti si sovrappongono e non si legge nessuna delle due
        etichette_dirette(ax, [(n, x_, y_, c_) for n, (x_, y_, c_) in fine.items()])
        ax.set_title(titolo, color=INCHIOSTRO, fontsize=11, loc='left', pad=8)
        ax.set_xlabel('blocco (10.000 righe ciascuno, ordine di feature_ready_time)',
                      color=INCHIOSTRO_2, fontsize=9)
        ax.set_xlim(0, len(B) + 60)
        ax.set_ylim(0, 1)
    assi[0].legend(frameon=False, fontsize=9, loc='lower left',
                   labelcolor=INCHIOSTRO_2)
    fig.suptitle('Replay su C, modello %s: adattivo contro copia congelata'
                 % modello.upper(), color=INCHIOSTRO, fontsize=13, x=0.012,
                 ha='left', y=0.985, fontweight='semibold')
    sotto = didascalia(
        fig,
        'Traccia sottile: valore per blocco. Linea spessa: media mobile su '
        '%d blocchi. I blocchi in cui una misura non è definita sono esclusi '
        'da quella misura. Seme %d.' % (FINESTRA, d['parametri']['seme']))
    fig.tight_layout(rect=(0, sotto, 1, 0.95))
    return salva(fig, uscita)


def fasce(d, modello, uscita):
    B = d['per_blocco']
    cong = serie(B, 'congelato', modello, 'auroc')
    adat = serie(B, 'adattivo', modello, 'auroc')
    norm = np.array([b['adattivo_' + modello]['normali'] for b in B], dtype=float)
    limiti = ((0, 10, 'meno di 10'), (10, 50, 'da 10 a 49'),
              (50, 200, 'da 50 a 199'), (200, np.inf, 'almeno 200'))
    et, val, quanti = [], [], []
    for lo, hi, nome in limiti:
        m = (~np.isnan(cong)) & (~np.isnan(adat)) & (norm >= lo) & (norm < hi)
        if m.sum() < 5:
            continue
        et.append(nome); quanti.append(int(m.sum()))
        val.append(float(np.mean(adat[m] - cong[m])))
    fig, ax = plt.subplots(figsize=(8.6, 3.4), facecolor=SUPERFICIE)
    stile(ax)
    y = np.arange(len(et))
    colori = [POSITIVO if v > 0 else NEGATIVO for v in val]
    ax.barh(y, val, color=colori, height=0.55, zorder=3)
    ax.axvline(0, color=INCHIOSTRO_2, linewidth=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(['%s\n%d blocchi' % (e, q) for e, q in zip(et, quanti)],
                       color=INCHIOSTRO, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(val):
        ax.annotate(('%+.3f' % v).replace('.', ',').replace('-', '\u2212'),
                    (v, y[i]), xytext=(8 if v > 0 else -8, 0),
                    textcoords='offset points', ha='left' if v > 0 else 'right',
                    va='center', color=INCHIOSTRO, fontsize=9.5,
                    fontweight='semibold')
    m = max(abs(min(val)), abs(max(val)))
    ax.set_xlim(-m * 1.45, m * 1.45)
    ax.set_xlabel('guadagno di AUROC dell’adattivo sul congelato',
                  color=INCHIOSTRO_2, fontsize=9)
    ax.set_title('Dove vive il guadagno, modello %s: per ricchezza di normali del blocco'
                 % modello.upper(), color=INCHIOSTRO, fontsize=12, loc='left', pad=10)
    sotto = didascalia(
        fig,
        'Positivo a destra, negativo a sinistra. Medie sui blocchi della '
        'fascia, su %s blocchi in tutto: i %d in cui una delle due copie non ha '
        'AUROC definita sono esclusi, non contati a zero. Seme %d.'
        % (italiano(sum(quanti)), len(B) - sum(quanti), d['parametri']['seme']))
    fig.tight_layout(rect=(0, sotto, 1, 1))
    return salva(fig, uscita)


def etichette_dirette(ax, voci, minimo=0.072):
    """Etichetta ogni serie a fine linea, separando quelle troppo vicine.

    Con quattro serie le etichette finali si sovrappongono quasi sempre, e due
    etichette sovrapposte non identificano niente: qui vengono distanziate in
    coordinate dell'asse di almeno `minimo`, mantenendo l'ordine verticale dei
    valori reali, cosi' la posizione resta un'informazione e non diventa una
    bugia.
    """
    if not voci:
        return
    lo, hi = ax.get_ylim()
    altezza = hi - lo
    ordinate = sorted(voci, key=lambda v: v[2], reverse=True)
    posti = [(v[2] - lo) / altezza for v in ordinate]
    for i in range(1, len(posti)):
        if posti[i - 1] - posti[i] < minimo:
            posti[i] = posti[i - 1] - minimo
    # se la pila sfora in basso, la risale
    scarto = min(0.0, min(posti))
    posti = [q - scarto for q in posti]
    for (nome, x, _, colore), q in zip(ordinate, posti):
        ax.annotate(nome, (x, lo + q * altezza), xytext=(8, 0),
                    textcoords='offset points', color=colore, fontsize=9,
                    va='center', fontweight='medium',
                    annotation_clip=False)


def media_corrente(v):
    """Media progressiva dei valori definiti: al blocco k, la media da 0 a k.

    Sostituisce la media mobile nel confronto fra politiche. Su 916 blocchi
    quattro medie mobili sono quattro tracce intrecciate e illeggibili; la media
    progressiva e' quasi monotona, si legge, e il suo valore finale **e'** la
    media riportata nelle tabelle, quindi la figura e il testo dicono lo stesso
    numero invece di assomigliarsi.
    """
    v = np.asarray(v, dtype=float)
    fuori = np.full(len(v), np.nan)
    somma = conta = 0
    for i, x in enumerate(v):
        if not np.isnan(x):
            somma += x; conta += 1
        if conta:
            fuori[i] = somma / conta
    return fuori


def italiano(n):
    """Un intero nella convenzione dei documenti: punto per le migliaia, meno
    tipografico. Le figure e il testo devono scrivere lo stesso numero allo
    stesso modo, altrimenti sembrano due misure."""
    s = f'{abs(int(n)):,}'.replace(',', '.')
    return ('\u2212' if n < 0 else '') + s


def _larghezza_disponibile(fig, margine):
    return fig.get_figwidth() * fig.dpi * (1 - 2 * margine)


def spezza_didascalia(fig, paragrafi, fontsize=8.5, margine=0.012):
    """Spezza le righe dove entrano davvero, misurandole sul motore di disegno.

    Una didascalia troppo lunga non fa cadere il disegno: esce dal bordo e
    sparisce in silenzio, e la figura resta pubblicabile con meta' della sua
    spiegazione mancante. E' successo. La larghezza quindi non si stima a
    occhio ne' si fissa a un numero di caratteri: si chiede al renderer quanto
    misura ogni riga candidata, con lo stesso corpo con cui verra' scritta.
    """
    ren = fig.canvas.get_renderer()
    prop = FontProperties(size=fontsize)
    disponibile = _larghezza_disponibile(fig, margine)

    def larghezza(s):
        return ren.get_text_width_height_descent(s, prop, False)[0]

    righe = []
    for p in paragrafi:
        corrente = ''
        for parola in p.split():
            prova = parola if not corrente else corrente + ' ' + parola
            if corrente and larghezza(prova) > disponibile:
                righe.append(corrente)
                corrente = parola
            else:
                corrente = prova
        righe.append(corrente)
    return righe


def didascalia(fig, *paragrafi, fontsize=8.5, margine=0.012, base=0.012):
    """Scrive la didascalia in fondo alla figura e torna il margine da lasciarle.

    Il valore di ritorno va passato a `tight_layout(rect=(0, qui, 1, ...))`:
    cosi' lo spazio sotto il disegno segue il numero di righe effettivo invece
    di essere un numero scelto a mano che va corretto ogni volta che il testo
    cambia.
    """
    righe = spezza_didascalia(fig, paragrafi, fontsize=fontsize, margine=margine)
    passo = fontsize * 1.45 / (fig.get_figheight() * 72)
    for k, riga in enumerate(reversed(righe)):
        fig.text(margine, base + k * passo, riga, color=INCHIOSTRO_2,
                 fontsize=fontsize, ha='left')
    return base + len(righe) * passo + 0.010


def testi_fuori_dal_foglio(fig, tolleranza=1.0):
    """I testi che escono dal bordo della figura, con quanto escono.

    Un testo troppo largo non fa cadere il disegno: viene tagliato dal bordo
    senza dire niente, e la figura resta pubblicabile con una frase a meta'.
    Qui si misura dopo l'impaginazione dove finisce davvero ogni testo.
    """
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    foglio = fig.bbox
    fuori = []
    # Le etichette delle tacche fuori dall'intervallo visibile restano fra gli
    # oggetti della figura ma non vengono disegnate: misurarle segnalerebbe un
    # difetto che il lettore non puo' vedere.
    saltare = set()
    for ax in fig.axes:
        for asse, limiti in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(limiti), max(limiti)
            for tacca in list(asse.get_major_ticks()) + list(asse.get_minor_ticks()):
                if not (lo <= tacca.get_loc() <= hi):
                    saltare.add(id(tacca.label1))
                    saltare.add(id(tacca.label2))
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in saltare:
            continue
        e = t.get_window_extent(renderer=ren)
        sbordo = max(foglio.x0 - e.x0, e.x1 - foglio.x1,
                     foglio.y0 - e.y0, e.y1 - foglio.y1)
        if sbordo > tolleranza:
            fuori.append((t.get_text().replace('\n', ' ')[:70], round(sbordo, 1)))
    return fuori


def salva(fig, uscita):
    """Salva solo se nessun testo esce dal foglio."""
    fuori = testi_fuori_dal_foglio(fig)
    if fuori:
        plt.close(fig)
        raise SystemExit(
            'testo fuori dal foglio in %s, la figura sarebbe pubblicata con una '
            'frase tagliata:\n  %s'
            % (Path(uscita).name,
               '\n  '.join('«%s» esce di %s punti' % (t, s) for t, s in fuori)))
    fig.savefig(uscita, dpi=170, facecolor=SUPERFICIE)
    plt.close(fig)
    return uscita


def regime_soglia(d):
    """Come leggere i conteggi di decisione di questo rendiconto."""
    s = d.get('soglia_di_decisione')
    if not isinstance(s, dict):
        return 'soglia non dichiarata nel rendiconto'
    return ('soglia a zero' if s.get('modo') == 'zero'
            else 'soglia scelta su B col massimo della balanced accuracy')


def non_rilevati(blocchi, variante, modello):
    """Attacchi non rilevati per blocco: i falsi negativi, cumulati.

    Non c'e' nessun NA da gestire: `fn` e' un conteggio, definito anche nei
    blocchi senza normali. Dove un blocco non contiene attacchi il conteggio e'
    zero perche' non ce n'erano da rilevare, che e' diverso da una misura non
    definita e non va confuso con essa.
    """
    v = np.array([b[variante + '_' + modello]['fn'] for b in blocchi], dtype=float)
    return np.cumsum(v)


def attacchi_persi(d, modello, uscita):
    """Attacchi non rilevati nel tempo, e la differenza fra le due copie.

    E' la figura che la scheda chiede per prima: «la differenza di attacchi non
    rilevati». Due riquadri, un asse ciascuno — il cumulato e la differenza
    hanno scale diverse e non condividono l'asse. A destra lo zero e' la linea
    di riferimento: sotto lo zero l'adattamento ha perso meno attacchi del
    congelato, sopra ne ha persi di piu'.
    """
    B = d['per_blocco']
    x = np.arange(len(B))
    cong = non_rilevati(B, 'congelato', modello)
    adat = non_rilevati(B, 'adattivo', modello)
    attacchi = sum(b['congelato_' + modello]['attacchi'] for b in B)

    fig, (sx, dx) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SUPERFICIE)
    stile(sx); stile(dx)
    for v, colore, nome in ((cong, CONGELATO, 'congelato'),
                            (adat, ADATTIVO, 'adattivo')):
        sx.plot(x, v / 1000, color=colore, linewidth=2.0, zorder=3, label=nome,
                solid_capstyle='round')
        sx.annotate(nome, (x[-1], v[-1] / 1000), xytext=(7, 0),
                    textcoords='offset points', color=colore, fontsize=9,
                    va='center', fontweight='medium')
    sx.set_title('attacchi non rilevati, cumulati (migliaia)', color=INCHIOSTRO,
                 fontsize=11, loc='left', pad=8)
    sx.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INCHIOSTRO_2)

    diff = (adat - cong) / 1000
    dx.plot(x, diff, color=INCHIOSTRO, linewidth=1.6, zorder=4)
    dx.fill_between(x, 0, diff, where=diff <= 0, color=POSITIVO, alpha=0.22, zorder=3)
    dx.fill_between(x, 0, diff, where=diff > 0, color=NEGATIVO, alpha=0.22, zorder=3)
    dx.axhline(0, color=INCHIOSTRO_2, linewidth=1.0, zorder=5)
    dx.set_title('differenza: adattivo meno congelato (migliaia)', color=INCHIOSTRO,
                 fontsize=11, loc='left', pad=8)
    saldo = int(adat[-1] - cong[-1])
    dx.annotate('%s attacchi non rilevati\nalla fine del flusso'
                % italiano(saldo),
                (0.5, 0.06 if saldo <= 0 else 0.94), xycoords='axes fraction',
                ha='center', va='bottom' if saldo <= 0 else 'top',
                color=INCHIOSTRO, fontsize=10, fontweight='semibold')
    for ax in (sx, dx):
        ax.set_xlabel('blocco (10.000 righe ciascuno, ordine di feature_ready_time)',
                      color=INCHIOSTRO_2, fontsize=9)
        ax.set_xlim(0, len(B) + 60)

    fig.suptitle('Replay su C, modello %s: attacchi non rilevati' % modello.upper(),
                 color=INCHIOSTRO, fontsize=13, x=0.012, ha='left', y=0.985,
                 fontweight='semibold')
    sotto = didascalia(
        fig,
        'Falsi negativi sommati blocco per blocco su %s attacchi: '
        'congelato %s, adattivo %s. Il conteggio dipende dal punto di '
        'decisione, qui %s. Seme %d.'
        % (italiano(attacchi), italiano(cong[-1]), italiano(adat[-1]),
           regime_soglia(d), d['parametri']['seme']))
    fig.tight_layout(rect=(0, sotto, 1, 0.95))
    return salva(fig, uscita)


def episodi_inversione(cong):
    """Gli inizi degli episodi in cui il congelato ordina al contrario.

    Un episodio comincia nel primo blocco con AUROC sotto 0,5 che segue un
    blocco con AUROC definita e sopra 0,5. I blocchi con AUROC non definita non
    aprono e non chiudono un episodio: sono saltati, perche' trattarli come
    «non invertiti» inventerebbe un confine che i dati non danno.
    """
    inizi, dentro = [], False
    for v in cong:
        if np.isnan(v):
            continue
        if v < 0.5 and not dentro:
            inizi.append(True); dentro = True
        else:
            inizi.append(False)
            if v >= 0.5:
                dentro = False
    # rimappa sugli indici originali, saltando i NaN
    fuori, j = [], 0
    for i, v in enumerate(cong):
        if np.isnan(v):
            continue
        if inizi[j]:
            fuori.append(i)
        j += 1
    return fuori


def lunghezze_episodi(cong):
    """Quanti blocchi dura ciascun episodio di inversione, **in ordine di
    tempo**, cioe' nello stesso ordine di `episodi_inversione`.

    L'ordine non e' un dettaglio: le due liste vengono accoppiate per scegliere
    quali episodi mediare, e una versione precedente restituiva qui le lunghezze
    ordinate per grandezza. Accoppiate agli inizi cronologici, attribuivano a un
    episodio la lunghezza di un altro: la figura del recupero dell'MLP mediava
    un episodio di due blocchi — proprio quelli che il metodo dichiara di
    escludere — e ne ometteva uno di otto. Chi vuole le lunghezze in ordine di
    grandezza, per la didascalia, le ordina a valle.
    """
    lun, corrente = [], 0
    for v in cong:
        if np.isnan(v):
            continue
        if v < 0.5:
            corrente += 1
        elif corrente:
            lun.append(corrente); corrente = 0
    if corrente:
        lun.append(corrente)
    return lun


def episodi_lunghi(cong, minimo):
    """Gli inizi dei soli episodi lunghi almeno `minimo` blocchi.

    Serve alla figura del recupero. Un episodio di un blocco solo non si puo'
    seguire per venti blocchi: la finestra dopo il suo inizio e' quasi tutta
    fuori dall'episodio, e mediarla con quella di un episodio di centoventidue
    blocchi non misura il recupero di nessuno dei due.
    """
    inizi = episodi_inversione(cong)
    lunghezze = lunghezze_episodi(cong)
    return [i for i, lun in zip(inizi, lunghezze) if lun >= minimo]


def recupero(d, modello, uscita, prima=5, dopo=20,
             lunghezza_minima=LUNGHEZZA_MINIMA_EPISODIO):
    """Le curve di recupero, allineate all'inizio degli episodi di inversione.

    La scheda chiede le curve e **non** una soglia di recupero scelta dopo aver
    visto il risultato: qui non c'e' nessuna soglia. Ogni episodio di inversione
    del congelato viene allineato al proprio inizio e si riportano le due copie
    da `prima` blocchi avanti a `dopo` blocchi indietro.

    Si mediano i soli episodi lunghi almeno `lunghezza_minima` blocchi, e i
    singoli episodi restano disegnati sotto la media: a un dato ritardo la media
    fra episodi di lunghezza molto diversa mescolerebbe episodi ancora in corso
    e altri finiti da tempo, cioe' risponderebbe a una domanda diversa da
    «quanto ci mette a riprendersi». Il numero di episodi esclusi e la loro
    lunghezza sono scritti nella didascalia, non taciuti.

    Con meno di due episodi lunghi la figura **non viene prodotta** e la
    funzione torna `None`: una media su un episodio solo e' quell'episodio, e
    presentarla come una curva media sarebbe una affermazione piu' forte del
    dato. Il chiamante deve dichiarare l'assenza, non riempirla.
    """
    B = d['per_blocco']
    cong = serie(B, 'congelato', modello, 'auroc')
    adat = serie(B, 'adattivo', modello, 'auroc')
    lun = lunghezze_episodi(cong)
    inizi = episodi_lunghi(cong, lunghezza_minima)
    if len(inizi) < 2:
        return None
    lunghi = [x for x in lun if x >= lunghezza_minima]
    esclusi = [x for x in lun if x < lunghezza_minima]
    scarti = np.arange(-prima, dopo + 1)
    medie, quanti = {}, []
    for nome, v in (('congelato', cong), ('adattivo', adat)):
        m = []
        for s in scarti:
            valori = [v[i + s] for i in inizi
                      if 0 <= i + s < len(v) and not np.isnan(v[i + s])]
            m.append(np.mean(valori) if valori else np.nan)
            if nome == 'congelato':
                quanti.append(len(valori))
        medie[nome] = np.array(m, dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 4.2), facecolor=SUPERFICIE)
    stile(ax)
    ax.axvline(0, color=INCHIOSTRO_2, linewidth=1.0, zorder=2)
    ax.axhline(0.5, color=GRIGLIA, linewidth=1.4, zorder=1)
    # Solo l'etichetta del livello: la spiegazione sta in didascalia, perche'
    # una frase lunga dentro il riquadro finisce per cadere su una curva.
    ax.annotate('0,5', (scarti[0], 0.5), xytext=(2, 5),
                textcoords='offset points', ha='left', color=INCHIOSTRO_2,
                fontsize=8.5)
    # I singoli episodi sotto la media: la media di tre episodi non va letta
    # come una curva senza vedere quanto i tre si somigliano.
    for nome, v, colore in (('congelato', cong, CONGELATO),
                            ('adattivo', adat, ADATTIVO)):
        for i in inizi:
            tratto = [v[i + s] if 0 <= i + s < len(v) else np.nan for s in scarti]
            ax.plot(scarti, tratto, color=colore, linewidth=1.0, alpha=0.30,
                    zorder=2, solid_capstyle='round')

    for nome, colore in (('congelato', CONGELATO), ('adattivo', ADATTIVO)):
        v = medie[nome]
        ax.plot(scarti, v, color=colore, linewidth=2.0, marker='o',
                markersize=4.5, zorder=3, label=nome, solid_capstyle='round')
        buoni = np.where(~np.isnan(v))[0]
        if len(buoni):
            ax.annotate(nome, (scarti[buoni[-1]], v[buoni[-1]]), xytext=(8, 0),
                        textcoords='offset points', color=colore, fontsize=9,
                        va='center', fontweight='medium')
    # Legenda nella riga del titolo, fuori dal riquadro: dentro cadrebbe
    # sulle tracce dei singoli episodi.
    ax.legend(frameon=False, fontsize=9, loc='lower right',
              bbox_to_anchor=(1.0, 1.0), ncol=2, labelcolor=INCHIOSTRO_2,
              columnspacing=1.6, handlelength=1.6)
    ax.set_title('Recupero dopo l\'inizio di un\'inversione, modello %s'
                 % modello.upper(), color=INCHIOSTRO, fontsize=12, loc='left', pad=10)
    ax.set_xlabel('blocchi dall\'inizio dell\'episodio (0 = primo blocco invertito)',
                  color=INCHIOSTRO_2, fontsize=9)
    ax.set_ylabel('AUROC', color=INCHIOSTRO_2, fontsize=9)
    ax.set_xlim(scarti[0] - 0.6, scarti[-1] + 3.2)
    ax.set_ylim(0, 1)
    sotto = didascalia(
        fig,
        'Linee spesse: media sui %d episodi lunghi almeno %d blocchi '
        '(%s blocchi). Linee sottili: i singoli episodi. Esclusi dalla media '
        '%d episodi pi\u00f9 corti (%s): la loro finestra cadrebbe quasi tutta '
        'fuori dall\u2019episodio.'
        % (len(lunghi), lunghezza_minima,
           ', '.join(str(x) for x in sorted(lunghi, reverse=True)),
           len(esclusi),
           (', '.join(str(x) for x in sorted(esclusi, reverse=True)) + ' blocchi')
           if esclusi else 'nessuno'),
        'Sotto la linea a 0,5 l\u2019ordinamento \u00e8 rovesciato. Da %d a %d '
        'episodi per punto. Nessuna soglia di recupero: solo le curve. I '
        'blocchi senza AUROC definita sono esclusi dalla media. Seme %d.'
        % (min(quanti), max(quanti), d['parametri']['seme']))
    fig.tight_layout(rect=(0, sotto, 1, 1))
    return salva(fig, uscita)


def politiche(rendiconti, modello, uscita):
    """Le quattro politiche sugli stessi blocchi.

    `rendiconti` e' una mappa nome -> rendiconto. La copia congelata e' identica
    in tutte le politiche per costruzione, e qui lo si verifica invece di
    assumerlo: se due rendiconti dessero falsi negativi diversi per il congelato
    la figura sarebbe priva di senso, e la funzione si ferma.
    """
    primo = next(iter(rendiconti.values()))
    B0 = primo['per_blocco']
    # Il confronto e' blocco per blocco, non sul totale: due flussi possono
    # chiudere sulla stessa somma di falsi negativi avendoli distribuiti in modo
    # diverso, e in quel caso il pannello di sinistra coinciderebbe alla fine
    # mentre le due curve divergono nel mezzo. Si confrontano anche le AUROC,
    # che sono quello che disegna il pannello di destra.
    atteso_fn = non_rilevati(B0, 'congelato', modello).tolist()
    atteso_auroc = serie(B0, 'congelato', modello, 'auroc')
    for nome, d in rendiconti.items():
        B = d['per_blocco']
        if len(B) != len(B0):
            raise SystemExit(
                'la copia congelata di %s non coincide con le altre: %d blocchi '
                'invece di %d, la figura non e confrontabile'
                % (nome, len(B), len(B0)))
        if non_rilevati(B, 'congelato', modello).tolist() != atteso_fn:
            raise SystemExit(
                'la copia congelata di %s non coincide con le altre nei falsi '
                'negativi: la figura non e confrontabile' % nome)
        suo = serie(B, 'congelato', modello, 'auroc')
        if not np.array_equal(suo, atteso_auroc, equal_nan=True):
            raise SystemExit(
                'la copia congelata di %s non coincide con le altre nelle '
                'AUROC: la figura non e confrontabile' % nome)

    # Le righe etichettate: la didascalia lo afferma, quindi va verificato qui
    # e non altrove. Le politiche differiscono per QUANDO aggiornano, non per
    # quali righe vedono; se vedessero righe diverse il confronto misurerebbe
    # due cose insieme.
    campione = _imp.Riepilogo() if _imp is not None else None
    if campione is not None:
        for nome, d in rendiconti.items():
            if nome == next(iter(rendiconti)):
                continue
            for x, y in zip(B0, d['per_blocco']):
                uguali, forza = _imp.confronta(x, y)
                campione.aggiungi(forza)
                if uguali is False:
                    raise SystemExit(
                        'le righe etichettate di %s non sono quelle delle altre '
                        'politiche: la figura confronterebbe due cose insieme'
                        % nome)

    x = np.arange(len(B0))
    fig, (sx, dx) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SUPERFICIE)
    stile(sx); stile(dx)
    serie_fig = [('congelato', primo, 'congelato')]
    for nome in ('ogni_blocco', 'evidenza', 'casuale'):
        if nome in rendiconti:
            serie_fig.append((nome, rendiconti[nome], 'adattivo'))

    et_sx, et_dx = [], []
    for nome, d, variante in serie_fig:
        colore = POLITICHE[nome]
        v = non_rilevati(d['per_blocco'], variante, modello) / 1000
        sx.plot(x, v, color=colore, linewidth=2.0, zorder=3,
                label=NOMI_POLITICHE[nome], solid_capstyle='round')
        et_sx.append((NOMI_POLITICHE[nome], x[-1], float(v[-1]), colore))
        a = media_corrente(serie(d['per_blocco'], variante, modello, 'auroc'))
        dx.plot(x, a, color=colore, linewidth=2.0, zorder=3,
                label=NOMI_POLITICHE[nome], solid_capstyle='round')
        buoni = np.where(~np.isnan(a))[0]
        if len(buoni):
            et_dx.append((NOMI_POLITICHE[nome], x[buoni[-1]],
                          float(a[buoni[-1]]), colore))
    sx.set_title('attacchi non rilevati, cumulati (migliaia)', color=INCHIOSTRO,
                 fontsize=11, loc='left', pad=8)
    dx.set_title('AUROC, media progressiva dal primo blocco', color=INCHIOSTRO,
                 fontsize=11, loc='left', pad=8)
    dx.set_ylim(0.3, 1)
    for ax, voci in ((sx, et_sx), (dx, et_dx)):
        ax.set_xlabel('blocco (10.000 righe ciascuno, ordine di feature_ready_time)',
                      color=INCHIOSTRO_2, fontsize=9)
        ax.set_xlim(0, len(B0) + 150)
        # Una legenda sola per la figura: i due riquadri hanno le stesse serie
        # con gli stessi colori, e la seconda cadrebbe sulle curve.
        if ax is sx:
            ax.legend(frameon=False, fontsize=9, loc='upper left',
                      labelcolor=INCHIOSTRO_2)
        etichette_dirette(ax, voci)
    aggiornamenti = {n: d['costi']['decisioni_della_politica'][modello].get('applicato', 0)
                     for n, d, v in serie_fig if v == 'adattivo'}
    fig.suptitle('Politiche di aggiornamento su C, modello %s' % modello.upper(),
                 color=INCHIOSTRO, fontsize=13, x=0.012, ha='left', y=0.985,
                 fontweight='semibold')
    sotto = didascalia(
        fig,
        'Stessi blocchi e stesse righe etichettate in tutte le politiche, '
        'confrontate qui e non assunte (%s); anche la copia congelata \u00e8 '
        'confrontata blocco per blocco. Aggiornamenti applicati su questo '
        'seme: %s.'
        % (('%s, su %s confronti'
            % ({'digest': 'SHA-256 della serializzazione dichiarata',
                'somma_storica': 'somma storica dei row_id, controllo pi\u00f9 debole',
                'assente': 'NON verificati, il campo manca'}.get(
                    campione.forza_minima, campione.forza_minima),
               italiano(campione.confronti))) if campione is not None
           else 'confronto degli indici non disponibile',
           ', '.join('%s %d' % (NOMI_POLITICHE[n], q)
                     for n, q in aggiornamenti.items())),
        'A destra la media progressiva, non una media mobile: il suo valore '
        'finale \u00e8 l\u2019AUROC media di questa corsa, cio\u00e8 il valore del seme '
        '%d nel campo per_seme di confronto_politiche.json. Le medie sui cinque '
        'semi sono altre. Falsi negativi contati con %s.'
        % (primo['parametri']['seme'], regime_soglia(primo)))
    fig.tight_layout(rect=(0, sotto, 1, 0.95))
    return salva(fig, uscita)


def uscita_in_utf8():
    """Dichiara UTF-8 sull'uscita standard, invece di affidarsi al sistema.

    Senza questo, su Windows `python ... > file.txt` usa cp1252 e cade con
    UnicodeEncodeError sul primo carattere che quella tabella non ha. E'
    successo davvero, sul meno tipografico U+2212 introdotto per allineare
    l'uscita alla convenzione numerica dei documenti: a schermo si vedeva, ma
    la riga che salva il risultato su file si fermava a meta'. L'uscita di
    questi programmi finisce in file versionati, quindi il suo encoding e' una
    proprieta' da dichiarare, non da ereditare.
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError, OSError):
        pass          # flussi che non si possono riconfigurare: si prosegue


def main(argv=None):
    uscita_in_utf8()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rendiconto', help='uscita JSON di replay.py')
    p.add_argument('--politiche', nargs='+', metavar='NOME=FILE',
                   help='rendiconti delle politiche, per esempio '
                        'ogni_blocco=cal.json evidenza=ev.json casuale=cas.json')
    p.add_argument('--modello', default='lr', choices=('lr', 'mlp', 'kan'))
    p.add_argument('--uscita', required=True, help='cartella dei PNG')
    a = p.parse_args(argv)
    if not a.rendiconto and not a.politiche:
        p.error('serve almeno --rendiconto o --politiche')
    u = Path(a.uscita); u.mkdir(parents=True, exist_ok=True)
    scritti, non_prodotte = [], []

    if a.rendiconto:
        d = json.loads(Path(a.rendiconto).read_text(encoding='utf-8'))
        scritti.append(curve(d, a.modello, u / ('curve_per_blocco_%s.png' % a.modello)))
        scritti.append(fasce(d, a.modello, u / ('guadagno_per_fascia_%s.png' % a.modello)))
        scritti.append(attacchi_persi(
            d, a.modello, u / ('attacchi_non_rilevati_%s.png' % a.modello)))
        r = recupero(d, a.modello, u / ('recupero_%s.png' % a.modello))
        if r is None:
            non_prodotte.append(
                'recupero_%s.png: meno di due episodi di inversione lunghi '
                'almeno %d blocchi, la media non direbbe nulla che il singolo '
                'episodio non dica meglio' % (a.modello, LUNGHEZZA_MINIMA_EPISODIO))
        else:
            scritti.append(r)

    if a.politiche:
        mappa = {}
        for voce in a.politiche:
            if '=' not in voce:
                p.error('--politiche vuole NOME=FILE, ricevuto %r' % voce)
            nome, percorso = voce.split('=', 1)
            if nome not in POLITICHE:
                p.error('nome di politica sconosciuto: %s (ammessi: %s)'
                        % (nome, ', '.join(POLITICHE)))
            mappa[nome] = json.loads(Path(percorso).read_text(encoding='utf-8'))
        scritti.append(politiche(mappa, a.modello, u / ('politiche_%s.png' % a.modello)))

    print('scritti:\n  ' + '\n  '.join(str(s) for s in scritti))
    if non_prodotte:
        print('non prodotte:\n  ' + '\n  '.join(non_prodotte))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

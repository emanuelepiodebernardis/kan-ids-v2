"""Curve per blocco del replay: congelato contro adattivo, e guadagno per fascia.

Produce due PNG dal rendiconto di `replay.py`, senza rileggere i dati grezzi:

  curve_per_blocco.png     AUROC e richiamo sui normali, blocco per blocco,
                           traccia grezza in chiaro e media mobile in evidenza
  guadagno_per_fascia.png  guadagno di AUROC dell'adattivo sul congelato,
                           per fascia di ricchezza di normali del blocco

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

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SUPERFICIE = '#fcfcfb'
INCHIOSTRO = '#0b0b0b'
INCHIOSTRO_2 = '#52514e'
GRIGLIA = '#e3e2de'
CONGELATO = '#2a78d6'      # slot 1, blu
ADATTIVO = '#eb6834'       # slot 2, arancio
NEGATIVO = '#e34948'       # polo caldo della coppia divergente
POSITIVO = '#2a78d6'       # polo freddo
FINESTRA = 25


def stile(ax):
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
        if len(fine) == 2:
            (n1, (x1, y1, c1)), (n2, (x2, y2, c2)) = sorted(
                fine.items(), key=lambda kv: kv[1][1], reverse=True)
            if abs(y1 - y2) < 0.06:
                mezzo = (y1 + y2) / 2
                y1, y2 = mezzo + 0.035, mezzo - 0.035
            for nome, xx, yy, cc in ((n1, x1, y1, c1), (n2, x2, y2, c2)):
                ax.annotate(nome, (xx, yy), xytext=(7, 0),
                            textcoords='offset points', color=cc,
                            fontsize=9, va='center', fontweight='medium')
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
    fig.text(0.012, 0.015,
             'Traccia sottile: valore per blocco. Linea spessa: media mobile su '
             '%d blocchi. I blocchi in cui una misura non è definita sono esclusi '
             'da quella misura. Seme %d.' % (FINESTRA, d['parametri']['seme']),
             color=INCHIOSTRO_2, fontsize=8.5, ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(uscita, dpi=170, facecolor=SUPERFICIE)
    plt.close(fig)
    return uscita


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
        ax.annotate(('%+.3f' % v).replace('.', ','),
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
    fig.text(0.012, 0.02,
             'Positivo a destra, negativo a sinistra. Medie sui blocchi della fascia; '
             'seme %d.' % d['parametri']['seme'],
             color=INCHIOSTRO_2, fontsize=8.5, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(uscita, dpi=170, facecolor=SUPERFICIE)
    plt.close(fig)
    return uscita


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rendiconto', required=True, help='uscita JSON di replay.py')
    p.add_argument('--modello', default='lr', choices=('lr', 'mlp', 'kan'))
    p.add_argument('--uscita', required=True, help='cartella dei PNG')
    a = p.parse_args(argv)
    d = json.loads(Path(a.rendiconto).read_text(encoding='utf-8'))
    u = Path(a.uscita); u.mkdir(parents=True, exist_ok=True)
    f1 = curve(d, a.modello, u / ('curve_per_blocco_%s.png' % a.modello))
    f2 = fasce(d, a.modello, u / ('guadagno_per_fascia_%s.png' % a.modello))
    print('scritti:\n  %s\n  %s' % (f1, f2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

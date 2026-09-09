# `iters` riselezionato sotto il protocollo corretto

**Prima.** `iters=6000` era stato scelto (sezione 16.1) massimizzando la
balanced accuracy calcolata sulle righe target riportate come risultato,
sui seed 42-44. Selezione sul test.

**Ora.** Sweep su 10 seed di **calibrazione** (90-99), disgiunti dai seed di
riporto (42-51), punteggio letto sulla sola **validation**; il test non e'
accessibile al processo (`kanids/valutazione.py`).

| iters | media (validation) | dev.std | delta vs argmax | t appaiato | p (Holm) | MAC/aggiornamento |
|---|---|---|---|---|---|---|
| 2 000 | 0,6500 | 0,0850 | −0,0207 | −1,55 | 0,44 | 6,4 M |
| 4 000 | 0,6558 | 0,0839 | −0,0149 | −1,59 | 0,44 | 12,8 M |
| 6 000 | 0,6632 | 0,0801 | −0,0075 | −1,81 | 0,41 | 19,2 M |
| 8 000 | 0,6668 | 0,0793 | −0,0039 | −1,41 | 0,44 | 25,6 M |
| **12 000** | **0,6707** | 0,0769 | — | — | — | 38,4 M |

## Tre cose che la tabella dice, e che cambiano la sezione 16.1

**1. L'argmax non e' 6000: e' 12000, ed e' il bordo della griglia.** La curva
e' monotona crescente su tutti e cinque i punti. E' esattamente la situazione
che il professore ha segnalato per il rapporto 1:5 nel primo lavoro — un
argmax al bordo con curva monotona non e' un ottimo, e' un limite della
griglia. Non e' escluso che 20 000 o 50 000 facciano ancora meglio, e
nessun esperimento qui lo verifica.

**2. Nessuna configurazione si distingue dalla migliore.** Test t appaiati
per seed, correzione di Holm sulla famiglia dei quattro confronti: tutti i
p corretti stanno sopra 0,41. Su questo criterio la scelta di `iters` **non
e' determinata dall'accuratezza**, e presentarla come tale sarebbe una
formulazione piu' forte del dato.

**3. Applicando la regola 1-SE — la stessa usata per la scelta
dell'architettura nel primo lavoro — si prende 2000.** Tutte e cinque le
configurazioni cadono entro un errore standard dalla migliore, quindi la
regola seleziona la piu' economica: 6,4 M MAC per aggiornamento invece di
19,2 M, **un terzo del calcolo a bordo**, senza una differenza misurabile.

## Cosa se ne fa

La scelta va spostata dall'accuratezza al costo, dichiarandolo. Restano due
opzioni difendibili, ed e' la seconda quella coerente con cio' che il
documento ha gia' misurato:

- **`iters=2000` fisso**, per la regola 1-SE. Costo minimo, ma il seed 90
  mostra il rischio di coda che la sezione 16.1 aveva gia' isolato: passa
  da 0,538 a 0,655 fra 2000 e 12000 iterazioni, cioe' a 2000 non converge.
- **Il passo adattivo (sezione 17a)**, che si dimezza sui plateau e si
  ferma sul punto fisso: spende 1 800-4 200 iterazioni sulle direzioni
  facili e fino a 12 800 su quella difficile. Copre il rischio di coda
  senza pagare il caso peggiore ovunque, ed e' l'unica delle tre che non
  richiede di scegliere un numero.

La griglia va inoltre estesa oltre 12 000 prima di poter scrivere qualunque
cosa sull'argmax, oppure la questione va chiusa adottando il passo
adattivo, per cui l'argmax non serve.

I risultati del protocollo precedente sono in `results/protocollo_v1/`.

# `iters` e `ridge` riselezionati sotto il protocollo corretto

Le due costanti del lavoro erano state scelte guardando le righe riportate come risultato. Rifatta la scelta onestamente, **una cade e una regge** — e il modo in cui reggono e' diverso, non solo il verdetto.

---

## Parte 1 — `iters`: la scelta non regge

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


---

## Parte 2 — `ridge`: la scelta regge, e nettamente

**Prima.** `ridge=0,1` era stato fissato (sezione 16.2) sulla direzione
`ton->bot` e «validato sulle altre cinque», cioe' sui numeri poi riportati.

**Ora.** La valutazione di `drift_graduale.py` e' prequenziale — ogni batch
e' prima valutato con il modello corrente e poi usato per adattarlo — quindi
non esiste un complemento da ritagliare: l'intero stream **e'** il test.
L'unico modo di scegliere una costante senza guardarlo e' sceglierla su
repliche che non entrano in nessuna tabella. Sweep su 10 seed di
calibrazione (90-99), disgiunti dai seed di riporto (42-51).

| ridge | media | dev.std | delta vs argmax | t appaiato | p (Holm) |
|---|---|---|---|---|---|
| 0,001 | 0,7505 | 0,0486 | −0,1698 | −10,87 | **0,0000** |
| 0,001 + clip 5 | 0,7329 | 0,0926 | −0,1874 | −6,42 | **0,0006** |
| 0,01 | 0,8234 | 0,0644 | −0,0968 | −5,07 | **0,0027** |
| **0,1** | **0,9203** | 0,0190 | — | — | — |
| 0,1 + clip 5 | 0,9099 | 0,0394 | −0,0103 | −1,19 | 0,266 |
| 1,0 | 0,9042 | 0,0271 | −0,0161 | −1,98 | 0,158 |
| 10,0 | 0,8379 | 0,0730 | −0,0823 | −3,94 | **0,0102** |

**Il massimo e' interno alla griglia**, e la curva e' unimodale: 0,75 → 0,82
→ **0,92** → 0,90 → 0,84. Non e' il caso di `iters`, dove l'argmax stava al
bordo con curva monotona; qui la griglia contiene davvero l'ottimo e non c'e'
ragione di allargarla.

`ridge=0,1` **batte quattro delle sei alternative in modo significativo dopo
Holm**, e le due che non si distinguono (1,0 e 0,1 con clip) hanno comunque
media piu' bassa e dispersione da due a quattro volte maggiore. La regola
1-SE, che su `iters` non separava nulla, qui seleziona **una sola
configurazione**: la soglia 0,9143 esclude anche 1,0.

## Il confronto fra le due parti e' il punto

Le due costanti erano state scelte con lo stesso metodo sbagliato, e
rifacendo la scelta correttamente si comportano in modo opposto:

| | `iters` | `ridge` |
|---|---|---|
| argmax | 12 000, **al bordo** della griglia | 0,1, **interno**, curva unimodale |
| differenze significative dopo Holm | **nessuna** (p sempre > 0,41) | **quattro su sei** |
| dev.std dell'argmax | 0,0769 | **0,0190** |
| esito della regola 1-SE | tutte dentro → si prende la piu' economica | **una sola** dentro |
| verdetto | la scelta non e' determinata dall'accuratezza | la scelta e' determinata, e resta 0,1 |

Che una selezione fatta sul test si riveli sbagliata per una costante e
giusta per l'altra e' esattamente il motivo per cui il protocollo va
corretto invece che difeso: **guardando i numeri sbagliati non si puo'
sapere in quale dei due casi ci si trova.** La sezione 16.2 arrivava alla
conclusione giusta per la ragione sbagliata; la 16.1 no.

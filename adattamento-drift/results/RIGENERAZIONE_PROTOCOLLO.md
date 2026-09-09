# Cosa e' cambiato rigenerando sotto il protocollo corretto

Sette stage, 10 seed uniformi (42-51), partizione validation/test unica,
2h27m di calcolo. Confronto con quanto `RISULTATI.md` pubblica oggi.

**Tre affermazioni cadono, due reggono e si rafforzano.** Nessuna cade per
la correzione del protocollo in se': cadono tutte perche' erano misurate su
3 seed, o perche' i p-value non erano corretti per confronti multipli.

---

## Cade — «funziona la regola piu' banale: gli n flussi piu' vicini al confine»

Sezione 4. La tabella pubblicata (3 seed) da' la regola a margine a 0,9002
con 8 etichette e 0,9056 con 32 su `ton->bot`, davanti alla regola
adattiva. A 10 seed:

| `ton->bot` | n=8 | n=32 | n=128 | n=512 |
|---|---|---|---|---|
| margine — pubblicato (3 seed) | 0,9002 | 0,9056 | 0,7769 | 0,7805 |
| **margine — 10 seed** | **0,7135** (4/10) | **0,8485** | 0,7998 | 0,8034 |
| **adattiva — 10 seed** | **0,8061** (9/10) | **0,8754** | **0,8917** | **0,9027** |

La regola adattiva batte il margine **a ogni budget**, e il margine
produce un numero solo in 4 seed su 10 a n=8.

Nella direzione facile il verdetto e' piu' duro: il margine e' **la peggiore
di tutte le regole**, sotto il prelievo casuale a ogni budget.

| `bot->ton` | n=8 | n=32 | n=128 | n=512 | normali a n=32 |
|---|---|---|---|---|---|
| margine | 0,6039 (3/10) | 0,6147 (4/10) | 0,5517 | 0,5293 | 0,9 |
| casuale | 0,6862 | 0,8583 | 0,9242 | **0,9323** | 6,7 |
| adattiva | 0,7083 | 0,8141 | 0,8737 | 0,9299 | 7,3 |

Il meccanismo si legge nell'ultima colonna: dove i normali abbondano, il
confine di decisione e' una fetta non rappresentativa e il margine ne
raccoglie meno di uno su 32 etichette. Il limite era gia' dichiarato nel
documento («e' anche la peggiore nell'altra direzione») ma come nota; a 10
seed e' il risultato principale.

**Cosa scrivere invece.** La regola adattiva e' l'unica che funziona in
entrambi i regimi, e non paga nulla per farlo: il suo sondaggio da 8
etichette entra comunque nel training. E' anche l'unica eseguibile a bordo
senza sapere in che regime ci si trova, che era il punto di partenza della
sezione.

**Confermato, invece:** su `ton->bot` prelievo casuale e conformal
raccolgono **zero normali in tutti e 10 i seed, a ogni budget**, e
`strat_z` fa lo stesso fino a n=128 (la prima normale compare a n=512, in 2
seed su 10) — non e' un artefatto del campione piccolo.

---

## Cade — «la KAN ha il miglior ROC-AUC sul target in entrambe le direzioni»

Sezione 1, marcata nel documento come *nota per il paper*: un secondo
argomento, indipendente, a favore della struttura additiva. A 10 seed regge
in una direzione sola.

| ROC-AUC sul target | `bot->ton` | `ton->bot` |
|---|---|---|
| KAN(cat,1L) | **0,8265 ± 0,0355** | 0,5171 ± 0,0424 |
| LightGBM | 0,7680 ± 0,0305 | **0,5516 ± 0,0356** |

Su `ton->bot` LightGBM sta davanti. Il divario non e' significativo
(t=−1,80, p=0,11 appaiato su 10 seed), quindi la formulazione corretta e'
che **le due non si distinguono in quella direzione**, non che la KAN
vince. Su `bot->ton` il vantaggio della KAN resta netto.

Da notare, e da tenere: su `ton->bot` **tutti e sei i modelli** stanno fra
0,38 e 0,55 di ROC-AUC, cioe' al caso o sotto. La diagnosi della sezione 1
— li' il problema non e' la soglia, e' la rappresentazione — ne esce
rafforzata, non indebolita.

---

## Cade — «il rifit completo vince in modo significativo in 4 direzioni su 5»

Sezione 11. Quella conta usava p-value **non corretti** per la famiglia dei
15 confronti (5 direzioni × 3 budget). Con Holm ne sopravvivono due:

| direzione | budget | 13 coeff | rifit | delta | t | p (Holm) |
|---|---|---|---|---|---|---|
| `ton->bot` | 128 | **0,9029** | 0,7657 | **+0,1372** | +14,95 | **<0,0001** |
| `unsw->ton` | 32 | 0,7069 | **0,8145** | −0,1076 | −4,74 | **0,015** |
| tutte le altre 13 | | | | −0,11 … +0,14 | | > 0,11 |

**Il delta medio a n=128 sulle cinque direzioni e' −0,0016**, contro il
−0,002 pubblicato: il «pareggio in valore atteso» non solo regge, ma ora ha
un test corretto dietro invece di un conteggio di vittorie.

La lettura giusta e' quindi piu' semplice di quella nel documento: **fuori
da `ton->bot`, dove i 13 coefficienti vincono di molto, non c'e' evidenza
che i due metodi si distinguano.** Il vantaggio resta di costo — 24 byte
contro 250, nessun riaddestramento a bordo — e non c'e' svantaggio di
accuratezza da compensare.

---

## Regge — la mappa delle sei direzioni

Sezione 11, riprodotta quasi cifra per cifra sotto il protocollo nuovo:

| direzione | non adattato | 8 etich. | 32 | 128 | seed riusciti a 128 |
|---|---|---|---|---|---|
| `ton->bot` | 0,5554 | 0,7698 | 0,8623 | 0,9029 | 9/10 |
| `bot->ton` | 0,6340 | 0,7750 | 0,8004 | 0,8624 | 9/10 |
| `bot->unsw` | 0,4551 | 0,6489 | 0,7381 | 0,7551 | 10/10 |
| `ton->unsw` | 0,2237 | 0,6434 | 0,7161 | 0,7495 | 10/10 |
| `unsw->ton` | 0,2984 | 0,6095 | 0,7069 | 0,8364 | 10/10 |
| `unsw->bot` | 0,7368 | — | — | — | **0/10** |

Il collasso su sei direzioni, il recupero in cinque, il tetto di UNSW-NB15
come bersaglio e **`unsw->bot` che fallisce in tutti e 10 i seed**: tutto
confermato, con gli stessi numeri a meno del terzo decimale.

---

## Regge, con una qualifica — KAN contro l'ultimo strato di una MLP

Sezione 5. Il confronto appaiato per seed, con Holm su sette confronti:

| direzione | budget | KAN (13 par) | MLP (17 par) | delta | p (Holm) | n |
|---|---|---|---|---|---|---|
| `ton->bot` | 128 | 0,8727 | 0,6771 | **+0,1956** | 0,059 | 5 |
| `ton->bot` | 512 | 0,8749 | 0,6699 | **+0,2050** | 0,055 | 7 |
| `bot->ton` | 128 | 0,8788 | 0,9351 | −0,0563 | 0,43 | 8 |
| `bot->ton` | 512 | 0,9137 | 0,9310 | −0,0173 | 0,43 | 9 |

**Dopo Holm nessun confronto e' significativo**, ma le entita' sono
asimmetriche di un ordine di grandezza: dove la KAN vince, vince di 0,20;
dove perde, perde di 0,02-0,06. E i seed utilizzabili sono pochi (5-9)
perche' la selezione fallisce spesso proprio nella direzione difficile.

La conclusione della sezione 12 — «cade: il vantaggio non e'
architetturale» — e' **troppo forte in un verso**, esattamente come quella
originale lo era nell'altro. Con questi dati non si puo' dire ne' che la
KAN domini, ne' che sia alla pari: si puo' dire che nella direzione dove
l'adattamento serve davvero la differenza e' grande e nella direzione dove
tutti convergono e' trascurabile, e che il campione non basta a stabilirlo.

---

## Igiene: due difetti trovati rigenerando

**L'header C usciva con terminatori CRLF.** `Path.write_text` su Windows
traduce ogni `\n` in `\r\n`, quindi l'header rigenerato differiva da quello
committato **per i soli terminatori** — contenuto identico byte per byte
dopo normalizzazione, ma `git status` lo segnava modificato e
`.gitattributes` impone `eol=lf`. E' lo stesso difetto di igiene gia'
corretto una volta nel primo lavoro. Il generatore ora scrive con
`newline="\n"` esplicito.

**La verifica di bit-esattezza e' stata rilanciata** sull'header
rigenerato, g++ 11.4.0 a `-O2`:

    golden vector: 200 / logit diversi: 0 / decisioni diverse: 0
    byte riscritti per l'adattamento: 24 (12 int16)

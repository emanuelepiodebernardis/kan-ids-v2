# Errori dopo aggiornamenti e skip

## Metodo

La decisione presa alla fine del blocco `k` viene associata agli errori del
blocco `k+1`, il primo che puo' usare il nuovo stato. I blocchi senza una
decisione precedente registrata e la decisione finale senza blocco successivo
sono esclusi. Le matrici di confusione sono sommate prima di calcolare recall e
FPR: non si mediano tassi di blocchi di dimensione diversa.

Gli strati descrivono **quando** il controllore ha potuto aggiornare, non
stimano l'effetto causale dell'atto di aggiornare. Stato adattato, composizione
di classe e difficolta' temporale differiscono fra gli strati. I confronti
variante-controllo restano appaiati sullo stesso blocco.

## Risultati

| Fase | Confronto | Modello | Strato | Blocchi | Delta FP | Delta FN | Delta FPR (pp) | Delta recall (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ADAPT-01 | evidenza meno frozen | LR | aggiornamento | 226 | -7 171 | -507 456 | -13.590 | 23.088 |
| ADAPT-01 | evidenza meno frozen | LR | skip | 4349 | -246 605 | -3 916 377 | -30.362 | 9.185 |
| ADAPT-01 | evidenza meno frozen | MLP | aggiornamento | 180 | 4 565 | -671 725 | 7.611 | 38.604 |
| ADAPT-01 | evidenza meno frozen | MLP | skip | 4395 | -79 709 | -13 218 208 | -9.902 | 30.670 |
| ADAPT-01 | evidenza meno frozen | KAN | aggiornamento | 93 | -480 | -91 503 | -1.612 | 10.165 |
| ADAPT-01 | evidenza meno frozen | KAN | skip | 4482 | -33 121 | -335 615 | -3.966 | 0.764 |
| ADAPT-02 | per-classe meno FIFO, evidence | LR | aggiornamento | 151 | -7 107 | 870 179 | -23.061 | -58.828 |
| ADAPT-02 | per-classe meno FIFO, evidence | LR | skip | 4424 | -142 368 | 27 689 716 | -17.067 | -63.861 |
| ADAPT-02 | per-classe meno FIFO, evidence | MLP | aggiornamento | 70 | -1 701 | 242 994 | -9.331 | -35.642 |
| ADAPT-02 | per-classe meno FIFO, evidence | MLP | skip | 4505 | -150 505 | 25 140 445 | -17.774 | -56.935 |
| ADAPT-02 | per-classe meno FIFO, evidence | KAN | aggiornamento | 51 | -1 046 | 241 933 | -5.121 | -49.417 |
| ADAPT-02 | per-classe meno FIFO, evidence | KAN | skip | 4524 | -66 690 | 19 981 678 | -7.896 | -45.056 |
| ADAPT-03 | soglia adattiva meno frozen+B | LR | aggiornamento | 703 | -25 101 | -959 166 | -14.642 | 14.004 |
| ADAPT-03 | soglia adattiva meno frozen+B | LR | skip | 211 | 98 | -77 543 | 7.656 | 3.677 |
| ADAPT-03 | soglia adattiva meno frozen+B | MLP | aggiornamento | 703 | 20 265 | -3 656 308 | 11.821 | 53.383 |
| ADAPT-03 | soglia adattiva meno frozen+B | MLP | skip | 211 | 157 | -1 314 125 | 12.266 | 62.319 |
| ADAPT-03 | soglia adattiva meno frozen+B | KAN | aggiornamento | 703 | 2 456 | -1 354 176 | 1.433 | 19.771 |
| ADAPT-03 | soglia adattiva meno frozen+B | KAN | skip | 211 | 105 | -237 243 | 8.203 | 11.251 |
| ADAPT-03 | pesi+soglia meno pesi+soglia_B | KAN | aggiornamento | 914 | 51 417 | -5 428 963 | 29.770 | 60.605 |
| ADAPT-04 | FIT192+CAL64 meno frozen+CAL64 | KAN | aggiornamento | 474 | -26 528 | -29 697 | -17.190 | 0.649 |
| ADAPT-04 | FIT192+CAL64 meno frozen+CAL64 | KAN | skip | 440 | -1 342 | 263 032 | -7.298 | -6.003 |
| ADAPT-04 | FIT192+CAL64 meno frozen+CAL64 | KAN | tratto_242_664 | 423 | 807 | 219 608 | 12.321 | -5.200 |

## Tratto ADAPT-04 242--664

Il tratto comprende 423 blocchi e viene valutato separatamente
perche' riceve lo stato mantenuto durante lo skip contiguo deciso nei blocchi
241--663. Rispetto a frozen+CAL64, FIT192+CAL64 passa da
309 a 1.116 FP
(`+807`) e da 969.055 a
1.188.663 FN (`+219.608`). La lettura resta descrittiva:
il tratto non e' randomizzato e lo stato incorpora tutti gli aggiornamenti
precedenti.


## Lettura

ADAPT-01 mostra che l'aggiornamento selettivo riduce i falsi allarmi rispetto
al congelato sia nei blocchi successivi a un update sia negli skip, ma il
richiamo non si muove sempre nella stessa direzione. ADAPT-02 rende il
compromesso piu' netto: la memoria per classe riduce FP e FPR, mentre aumenta
FN e riduce il recall in entrambi gli strati. In ADAPT-03 la sola soglia
recupera attacchi al prezzo di piu' falsi allarmi; nel replay pesi+soglia non
esiste uno strato skip e il confronto update/skip non e' quindi stimabile.
ADAPT-04 conferma il compromesso sul disegno 192/64: rispetto a frozen+CAL64,
nei blocchi successivi agli update FIT192+CAL64 riduce sia FP sia FN; negli
skip riduce gli FP ma aumenta i FN. Il peggioramento globale dei FN e' quindi
concentrato negli skip, non in entrambi gli strati.
Nel tratto 242--664, valutato separatamente perche' segue lo skip contiguo
deciso nei blocchi 241--663, la variante produce +807 FP e +219.608 FN.

I conteggi completi e i denominatori sono in `analisi_aggiornamenti_skip.csv` e
`analisi_aggiornamenti_skip.json`.

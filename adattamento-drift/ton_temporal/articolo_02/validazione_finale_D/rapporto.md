# Validazione finale: KAN su D, semi 42--46

## Esecuzione e controlli

La campagna finale usa il commit congelato
`ea216914f756433fe69fd825146de5c1e325cbd8`. Prima di leggere D, il runner ha
addestrato su A e calibrato su B tutti e cinque i modelli iniziali. Sono poi
stati eseguiti cinque replay, uno per ciascun seme 42--46, senza tuning o
modifiche al disegno. A, B, C e D sono verificati tramite SHA-256; D non
interviene nella scelta di soglie, requisiti o criteri.

Ogni replay contiene 355 blocchi: 354 da 10.000 righe e l'ultimo da 9.973,
per 3.549.973 predizioni. Il sottoinsieme non visto in A/B/C contiene 2.228.436
righe. `D_completo` e `D_non_visto_ABC` sono aggregazioni delle stesse
predizioni, non esecuzioni diverse. Per ogni seme, AUROC e balanced accuracy
per blocco hanno denominatore 352/355 su D completo e 350/355 sul sottoinsieme
non visto; i blocchi monoclasse restano nulli e non sono sostituiti con zero.

La suite completa sul commit eseguito ha superato 375 test. Un controllo
indipendente successivo ha ricalcolato dai cinque JSON tutti gli aggregati,
i confronti appaiati, i conteggi update/skip, le somme dei tempi e gli hash:
esito PASS.

## Confronto principale: contributo dei pesi

Il confronto preregistrato e' `FIT192+CAL64 - frozen+CAL64`: stessa
calibrazione recente e stesso calendario, con o senza aggiornamento dei pesi.
I valori seguenti sono medie e deviazioni standard delle differenze appaiate
sui cinque semi; per FN, FP e FPR un valore negativo favorisce FIT192+CAL64.

| ambito | delta FN | delta FP | delta recall (pp) | delta FPR (pp) | delta BA (pp) | delta AUROC media/blocco |
|---|---:|---:|---:|---:|---:|---:|
| D completo | -9.255,2 +/- 97.413,0 | -81.355,4 +/- 6.051,6 | +0,286 +/- 3,006 | -26,271 +/- 1,954 | +13,279 +/- 2,258 | +0,082329 +/- 0,015496 |
| D non visto in A/B/C | -70.828,6 +/- 97.060,6 | -2.655,2 +/- 3.957,2 | +3,375 +/- 4,625 | -2,043 +/- 3,044 | +2,709 +/- 3,625 | +0,111712 +/- 0,027817 |

### Differenze per seme

| seme | FN D completo | FP D completo | FN non visto | FP non visto |
|---:|---:|---:|---:|---:|
| 42 | +54.175 | -84.858 | +4.271 | -3.659 |
| 43 | -115.380 | -89.311 | -165.016 | -6.888 |
| 44 | +125.148 | -73.762 | +59.443 | +3.547 |
| 45 | -40.935 | -77.720 | -129.600 | -4.684 |
| 46 | -69.284 | -81.126 | -123.241 | -1.592 |

L'aggiornamento dei pesi riduce i falsi allarmi su D completo in 5/5 semi e
migliora balanced accuracy e AUROC media per blocco in 5/5. I falsi negativi
diminuiscono pero' soltanto nei semi 43, 45 e 46. Anche sul sottoinsieme non
visto il miglioramento congiunto di FN e FP riguarda 3/5 semi; il seme 44
peggiora entrambi. Non e' quindi soddisfatto il criterio preregistrato di
superiorita' generale, che richiede di non peggiorare ne' FN ne' FP e un segno
concorde in almeno quattro semi su cinque.

## Contributo della soglia

A pesi adattati, `CAL64 - B` riduce in media i FN di 2.610.759,8 ma aggiunge
60.637,6 FP su D completo; recall e FPR aumentano rispettivamente di 80,572 e
19,581 punti percentuali. Sul sottoinsieme non visto riduce i FN di 1.883.015,0
e aggiunge 40.603,6 FP, con +89,733 punti di recall e +31,238 punti di FPR.
L'AUROC resta invariata, come richiesto dall'identita' degli score a pesi
comuni. La calibrazione recente recupera quindi molti attacchi, ma introduce
un aumento consistente dei falsi allarmi: il solo recall non basta per
concludere che la variante sia migliore.

## Update, skip e costi

| seme | update | skip | fit online (s) | cal. frozen (s) | cal. adattata (s) | replay D (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 337 | 17 | 10,672 | 0,103 | 0,059 | 59,837 |
| 43 | 351 | 3 | 10,224 | 0,090 | 0,052 | 52,683 |
| 44 | 347 | 7 | 9,630 | 0,083 | 0,050 | 49,032 |
| 45 | 341 | 13 | 9,632 | 0,094 | 0,055 | 48,622 |
| 46 | 350 | 4 | 10,378 | 0,090 | 0,057 | 53,348 |

Ogni seme registra 354 decisioni dopo il ritardo iniziale. Dei 44 skip, 43
dipendono dall'insufficienza di normali recenti per CAL64; nel seme 43 uno
dipende dall'assenza di una classe nel buffer FIT. I timer di fit e delle due
calibrazioni sono conservati per ciascun aggiornamento e la loro somma e'
riportata per seme; il tempo di replay misura l'intero ciclo su D per quel
seme. Il comando completo, che comprende hash, caricamenti,
inizializzazione A/B dei cinque modelli, replay e costruzione del riepilogo ma
non la serializzazione finale, ha richiesto 1.234,388 secondi.

## Conclusione circoscritta

Su questa campagna, separare pesi e soglia chiarisce due contributi diversi:
l'aggiornamento dei pesi riduce stabilmente i falsi allarmi e migliora AUROC e
balanced accuracy, mentre la soglia recente recupera recall pagando piu'
falsi allarmi. La variabilita' dei FN fra i semi impedisce una conclusione di
superiorita' generale. Il risultato descrive quando ciascun componente aiuta
e dove rimane il compromesso, senza estenderlo oltre KAN, i cinque semi e il
protocollo preregistrato.

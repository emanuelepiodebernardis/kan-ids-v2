# Validazione finale: KAN su D, semi 42--46

## Esecuzione e perimetro

La campagna finale usa il commit congelato
`ea216914f756433fe69fd825146de5c1e325cbd8`. Prima di calcolare l'impronta di D o caricarlo, il
runner ha addestrato su A e calibrato sull'intero B tutti e cinque i modelli
iniziali. Sono poi stati eseguiti una sola volta i semi 42--46, senza tuning o
modifiche del disegno. La suite del commit eseguito ha superato 375 test.

Ogni replay contiene 3.549.973 predizioni in 355 blocchi: 354 da 10.000 righe
e l'ultimo da 9.973. Il sottoinsieme non visto in A/B/C contiene 2.228.436
righe. I due ambiti sono aggregazioni delle stesse predizioni; la maschera non
modifica richieste, memorie o stato. I blocchi monoclasse conservano metriche
nulle e non vengono sostituiti con zero.

D non e' stato usato per scegliere il disegno o gli iperparametri. Le etichette
richieste su D sono pero' entrate online, con il ritardo e la regola FIT/CAL
preregistrati, e hanno potuto modificare soltanto le predizioni successive.

## Risultati assoluti delle quattro tracce

I valori sono media +/- deviazione standard campionaria sui cinque semi. FN e
FP sono conteggi; recall, FPR e BA sono percentuali. La BA della tabella e'
calcolata dalla matrice di confusione globale e non e' una media per blocco.

### D completo

| traccia | FN | FP | recall (%) | FPR (%) | BA globale (%) | AUROC media/blocco |
|---|---:|---:|---:|---:|---:|---:|
| frozen+B | 2.556.210,8 +/- 23.236,3 | 39.136,2 +/- 413,1 | 21,112 +/- 0,717 | 12,638 +/- 0,133 | 54,237 +/- 0,329 | 0,853550 +/- 0,012193 |
| frozen+CAL64 | 353.454,0 +/- 53.161,4 | 144.730,2 +/- 6.318,6 | 89,092 +/- 1,641 | 46,736 +/- 2,040 | 71,178 +/- 1,733 | 0,853550 +/- 0,012193 |
| FIT192+B | 2.954.958,6 +/- 190.791,8 | 2.737,2 +/- 3.083,7 | 8,806 +/- 5,888 | 0,884 +/- 0,996 | 53,961 +/- 3,233 | 0,935879 +/- 0,003423 |
| FIT192+CAL64 | 344.198,8 +/- 49.694,1 | 63.374,8 +/- 6.542,7 | 89,378 +/- 1,534 | 20,465 +/- 2,113 | 84,456 +/- 1,146 | 0,935879 +/- 0,003423 |

Le metriche AUROC e BA calcolate per blocco hanno denominatore `352/355` per seme.

### D non visto in A/B/C

| traccia | FN | FP | recall (%) | FPR (%) | BA globale (%) | AUROC media/blocco |
|---|---:|---:|---:|---:|---:|---:|
| frozen+B | 2.032.985,0 +/- 23.235,3 | 6.887,2 +/- 260,8 | 3,120 +/- 1,107 | 5,299 +/- 0,201 | 48,911 +/- 0,518 | 0,793388 +/- 0,022893 |
| frozen+CAL64 | 255.423,0 +/- 63.982,9 | 44.176,4 +/- 4.223,8 | 87,828 +/- 3,049 | 33,987 +/- 3,250 | 76,921 +/- 2,819 | 0,793388 +/- 0,022893 |
| FIT192+B | 2.067.609,4 +/- 5.672,1 | 917,6 +/- 228,5 | 1,470 +/- 0,270 | 0,706 +/- 0,176 | 50,382 +/- 0,190 | 0,905100 +/- 0,004932 |
| FIT192+CAL64 | 184.594,4 +/- 38.339,5 | 41.521,2 +/- 1.447,6 | 91,203 +/- 1,827 | 31,944 +/- 1,114 | 79,630 +/- 1,012 | 0,905100 +/- 0,004932 |

Le metriche AUROC e BA calcolate per blocco hanno denominatore `350/355` per seme.

## Confronto principale: contributo dei pesi

Il confronto preregistrato e' `FIT192+CAL64 - frozen+CAL64`: stessa
calibrazione recente e stesso calendario, con o senza aggiornamento dei pesi.
Per FN, FP e FPR un delta negativo favorisce FIT192+CAL64.

| ambito | delta FN | delta FP | delta recall (pp) | delta FPR (pp) | delta BA (pp) | delta AUROC media/blocco |
|---|---:|---:|---:|---:|---:|---:|
| D completo | -9.255,2 +/- 97.413,0 | -81.355,4 +/- 6.051,6 | 0,286 +/- 3,006 | -26,271 +/- 1,954 | 13,279 +/- 2,258 | 0,082329 +/- 0,015496 |
| D non visto in A/B/C | -70.828,6 +/- 97.060,6 | -2.655,2 +/- 3.957,2 | 3,375 +/- 4,625 | -2,043 +/- 3,044 | 2,709 +/- 3,625 | 0,111712 +/- 0,027817 |

| seme | FN D completo | FP D completo | FN non visto | FP non visto |
|---:|---:|---:|---:|---:|
| 42 | 54.175 | -84.858 | 4.271 | -3.659 |
| 43 | -115.380 | -89.311 | -165.016 | -6.888 |
| 44 | 125.148 | -73.762 | 59.443 | 3.547 |
| 45 | -40.935 | -77.720 | -129.600 | -4.684 |
| 46 | -69.284 | -81.126 | -123.241 | -1.592 |

Su D completo i FP diminuiscono in 5/5 semi e BA e AUROC media per blocco
migliorano in 5/5; i FN diminuiscono nei semi 43, 45, 46.
Sul sottoinsieme non visto FN e FP migliorano insieme nei semi
43, 45, 46; il seme 44 peggiora entrambi. Il criterio
preregistrato di superiorita' generale, almeno 4/5 senza peggiorare FN o FP,
non e' quindi soddisfatto.

## Soglia e riferimento frozen+B

A pesi adattati, `CAL64 - B` riduce in media i FN di
2.610.759,8 ma aggiunge
60.637,6 FP su D completo; sul non visto
riduce i FN di 1.883.015,0 e aggiunge
40.603,6 FP. Recall e FPR aumentano
insieme, mentre l'AUROC resta invariata perche' gli score a pesi comuni sono
identici.

Rispetto al riferimento esterno `frozen+B`, `FIT192+CAL64` riduce in media i
FN di 2.212.012,0 e aumenta i FP di
24.238,6 su D completo. Sul non visto i
corrispondenti valori sono 1.848.390,6 FN
in meno e 34.634,0 FP in piu'. Neppure
questo confronto indica dominanza: il recupero di attacchi ha un costo in
falsi allarmi.

## Temporalita' su C e D

In ADAPT-04 su C/seed42 sono stati osservati 474
blocchi successivi a un update e 440 successivi a uno skip.
Il tratto 242--664, collegato a 423 skip consecutivi, ha prodotto `+807 FP` e
`+219.608 FN` rispetto a frozen+CAL64.

Su D gli update/skip per i semi 42--46 sono rispettivamente 337/17, 351/3,
347/7, 341/13 e 350/4. Tutti i 44 skip sono iniziali, nei blocchi
1--17, 1--3, 1--7, 1--13, 1--4; dopo il primo update non si osservano altri skip. C e D
hanno lunghezza e composizione diverse: il contrasto localizza periodi
sfavorevoli ma non identifica causalmente una singola ragione della differenza.

Ogni replay registra 354 decisioni. Quella presa nel blocco 354 e' marcata
attiva dal blocco 355, che non esiste, e non influenza quindi altre predizioni
di D. Le ultime 99 etichette richieste restano in attesa a fine stream.

## Costi e confini dei timer

| seme | update | skip | fit online (s) | cal. frozen (s) | cal. adattata (s) | replay D (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 337 | 17 | 10,672 | 0,103 | 0,059 | 59,837 |
| 43 | 351 | 3 | 10,224 | 0,090 | 0,052 | 52,683 |
| 44 | 347 | 7 | 9,630 | 0,083 | 0,050 | 49,032 |
| 45 | 341 | 13 | 9,632 | 0,094 | 0,055 | 48,622 |
| 46 | 350 | 4 | 10,378 | 0,090 | 0,057 | 53,348 |

`secondi_totali` vale 1.234,388 s. Il
timer parte dopo il calcolo degli SHA-256 di A/B/C e termina prima della
serializzazione del riepilogo finale. Include SHA-256 e caricamento di D,
caricamenti A/B/C, preparazione A/B dei cinque modelli, cinque replay,
scrittura dei cinque JSON per seme e costruzione del riepilogo. I timer di
fit e delle due calibrazioni sono sottoinsiemi del replay, non costi ulteriori.

## Analisi descrittive successive e limiti

La riduzione dei FP del confronto principale e' uniforme in 5/5 semi soltanto
su D completo; sul non visto il seme 44 aumenta i FP. Sottraendo il non visto
dal completo, il 96,7%
della riduzione media aggregata dei FP ricade nelle righe con vettori gia'
comparsi in A/B/C. Questa decomposizione e' successiva e descrittiva: non prova
che le ripetizioni causino il beneficio. Inoltre la maschera non elimina le
ripetizioni interne a D. Il FPR assoluto di FIT192+CAL64 sul non visto resta
31,944%.

I cinque semi rappresentano inizializzazioni sullo stesso stream, non cinque
dataset indipendenti; la deviazione standard non e' un intervallo di
confidenza. L'aggiornamento dei pesi migliora stabilmente FP, BA e AUROC su D
completo, mentre la soglia recente recupera recall pagando piu' falsi allarmi.
La variabilita' dei FN e il risultato sul non visto impediscono una conclusione
di superiorita' generale.

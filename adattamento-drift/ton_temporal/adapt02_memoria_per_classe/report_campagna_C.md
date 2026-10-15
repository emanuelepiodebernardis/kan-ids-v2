# ADAPT-02: campagna completa su C

## Ambito

Campagna completa autorizzata dopo il checkpoint ADAPT-02: cinque semi di base,
due strategie di memoria e due politiche. Ogni replay produce LR, MLP e KAN
nello stesso rendiconto.

SHA di avvio e codice eseguito:

`078a9d8447aee8493a9da7b98ba76d1bdec27678`

D non e' stato usato.

## Matrice

Semi: 42, 43, 44, 45, 46.

Strategie di memoria:

- `fifo`
- `per_classe`

Politiche:

- `ogni_blocco`
- `evidenza_inversione`

Replay totali: 20.

La campagna e' stata eseguita in una directory nuova, senza riusare i JSON dello
smoke:

`adattamento-drift/archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8/`

Lo script ha verificato l'appaiamento: per ogni seme e politica coincidono fra
`fifo` e `per_classe` i digest dei `row_id` campionati e le misure frozen per
blocco.

## Archivio esterno

I 20 rendiconti completi restano fuori Git. La directory esterna contiene anche
il riepilogo prodotto dallo script; in Git ne e' conservata la copia leggera,
senza le serie per blocco dei rendiconti completi.

Archivio compresso:

`adattamento-drift/archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8.zip`

Collegamento diretto al file condiviso:

`https://drive.google.com/file/d/1C4uZIcW7rtWk5j1tROcHrCrhdzvzZq5E/view?usp=drive_link`

Dimensione: 7.253.160 byte.

SHA-256:

`212B9270EF47D0B997667E83B38CB6C899A5B6ADD00DF61B881AA14DA905AFE0`

In Git e' conservato il riepilogo leggero:

`adapt02_memoria_per_classe/riepilogo_campagna_C.json`

Analisi leggera derivata dai rendiconti completi:

`adapt02_memoria_per_classe/analisi_campagna_C.json`

Differenze appaiate per singolo seme:

`adapt02_memoria_per_classe/differenze_appaiate_per_seme_C.csv`

## Tempi

Tempo totale della campagna: 10.299,36 secondi, cioe' circa 2 ore, 51 minuti e
39 secondi.

Comando eseguito:

```bash
cd <repo>/adattamento-drift/ton_temporal
python campagna_memoria_per_classe.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --uscita-dir ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8
```

## Convenzione sui tassi

L'AUROC e' calcolata come media sui blocchi in cui sono presenti entrambe le
classi. Su C il denominatore e' 875 blocchi validi su 916, per tutti i
rendiconti della campagna.

Il FPR e' riportato in due modi distinti:

- **FPR complessivo**: FP / normali, con i conteggi sommati sul rendiconto;
- **FPR medio blocchi**: media semplice dei FPR dei blocchi validi.

Il secondo valore puo' essere diverso dal primo perche' i blocchi non hanno lo
stesso numero di normali. Nel caso KAN/per-classe i valori richiamati nella
revisione sono:

| politica | FPR complessivo | FPR medio blocchi |
|---|---:|---:|
| ogni blocco | 0,323% | 0,765% |
| evidence | 1,545% | 0,681% |

## Frozen

Medie sui cinque semi, una sola volta per seme. La dispersione e' la deviazione
standard campionaria sui cinque semi.

| modello | AUROC media+-ds | recall compl. media+-ds | FPR compl. % media+-ds | FPR blocchi % media+-ds | FN media+-ds | FP media+-ds |
|---|---:|---:|---:|---:|---:|---:|
| LR | 0,7149 +- 0,0000 | 0,7213 +- 0,0000 | 48,722 +- 0,000 | 46,727 +- 0,000 | 2.501.546 +- 0 | 84.443 +- 0 |
| MLP | 0,7933 +- 0,0177 | 0,4378 +- 0,0874 | 28,601 +- 0,716 | 14,212 +- 0,512 | 5.046.751 +- 784.237 | 49.570 +- 1.241 |
| KAN | 0,8911 +- 0,0046 | 0,6593 +- 0,0059 | 13,239 +- 0,015 | 3,988 +- 0,126 | 3.058.698 +- 52.541 | 22.945 +- 26 |

## Risultati adattivi

Medie sui cinque semi. La tabella distingue i tassi complessivi dalle medie per
blocco.

| memoria | politica | modello | AUROC media+-ds | recall compl. media+-ds | FPR compl. % media+-ds | FPR blocchi % media+-ds | FN media+-ds | FP media+-ds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FIFO | ogni blocco | LR | 0,7739 +- 0,0108 | 0,8824 +- 0,0089 | 18,542 +- 1,619 | 41,236 +- 1,516 | 1.055.800 +- 80.210 | 32.136 +- 2.806 |
| FIFO | ogni blocco | MLP | 0,7674 +- 0,0511 | 0,8590 +- 0,0603 | 17,069 +- 2,816 | 38,848 +- 3,536 | 1.265.693 +- 541.106 | 29.584 +- 4.880 |
| FIFO | ogni blocco | KAN | 0,8073 +- 0,0126 | 0,7690 +- 0,0196 | 11,350 +- 2,277 | 29,122 +- 1,493 | 2.073.531 +- 175.728 | 19.671 +- 3.947 |
| FIFO | evidence | LR | 0,7756 +- 0,0290 | 0,8199 +- 0,0312 | 19,437 +- 8,295 | 42,887 +- 7,280 | 1.616.779 +- 280.106 | 33.688 +- 14.377 |
| FIFO | evidence | MLP | 0,7939 +- 0,0357 | 0,7473 +- 0,1136 | 19,930 +- 8,525 | 36,463 +- 10,693 | 2.268.765 +- 1.019.960 | 34.541 +- 14.775 |
| FIFO | evidence | KAN | 0,8341 +- 0,0368 | 0,6688 +- 0,0876 | 9,362 +- 3,690 | 23,728 +- 9,509 | 2.973.274 +- 786.101 | 16.225 +- 6.395 |
| per classe | ogni blocco | LR | 0,8250 +- 0,0227 | 0,2361 +- 0,0163 | 1,976 +- 0,176 | 6,118 +- 2,751 | 6.858.037 +- 145.887 | 3.425 +- 304 |
| per classe | ogni blocco | MLP | 0,9093 +- 0,0183 | 0,2816 +- 0,1107 | 1,346 +- 0,393 | 3,413 +- 1,115 | 6.449.138 +- 993.424 | 2.333 +- 681 |
| per classe | ogni blocco | KAN | 0,8915 +- 0,0146 | 0,2024 +- 0,0173 | 0,323 +- 0,117 | 0,765 +- 0,743 | 7.160.018 +- 155.307 | 559 +- 203 |
| per classe | evidence | LR | 0,8237 +- 0,0328 | 0,1836 +- 0,0328 | 2,188 +- 1,062 | 5,346 +- 4,033 | 7.328.758 +- 294.877 | 3.793 +- 1.840 |
| per classe | evidence | MLP | 0,8963 +- 0,0244 | 0,1818 +- 0,0732 | 2,366 +- 1,531 | 2,980 +- 1,076 | 7.345.453 +- 657.015 | 4.100 +- 2.653 |
| per classe | evidence | KAN | 0,8929 +- 0,0303 | 0,2183 +- 0,0694 | 1,545 +- 1,550 | 0,681 +- 0,606 | 7.017.997 +- 623.239 | 2.678 +- 2.687 |

## Matrici di confusione aggregate

Conteggi medi sui cinque semi, aggregati su tutti i blocchi di ciascun replay.
Per il frozen la matrice e' riportata una sola volta per modello.

| memoria | politica | modello | VN | FP | FN | VP |
|---|---|---|---:|---:|---:|---:|
| frozen | - | LR | 88.873 | 84.443 | 2.501.546 | 6.475.811 |
| frozen | - | MLP | 123.746 | 49.570 | 5.046.751 | 3.930.606 |
| frozen | - | KAN | 150.371 | 22.945 | 3.058.698 | 5.918.659 |
| FIFO | ogni blocco | LR | 141.180 | 32.136 | 1.055.800 | 7.921.557 |
| FIFO | ogni blocco | MLP | 143.732 | 29.584 | 1.265.693 | 7.711.664 |
| FIFO | ogni blocco | KAN | 153.645 | 19.671 | 2.073.531 | 6.903.826 |
| FIFO | evidence | LR | 139.628 | 33.688 | 1.616.779 | 7.360.578 |
| FIFO | evidence | MLP | 138.775 | 34.541 | 2.268.765 | 6.708.592 |
| FIFO | evidence | KAN | 157.091 | 16.225 | 2.973.274 | 6.004.083 |
| per classe | ogni blocco | LR | 169.891 | 3.425 | 6.858.037 | 2.119.320 |
| per classe | ogni blocco | MLP | 170.983 | 2.333 | 6.449.138 | 2.528.219 |
| per classe | ogni blocco | KAN | 172.757 | 559 | 7.160.018 | 1.817.339 |
| per classe | evidence | LR | 169.523 | 3.793 | 7.328.758 | 1.648.599 |
| per classe | evidence | MLP | 169.216 | 4.100 | 7.345.453 | 1.631.904 |
| per classe | evidence | KAN | 170.638 | 2.678 | 7.017.997 | 1.959.360 |

I tipi di attacco presenti nei blocchi di C sono `ddos`, `dos`, `injection` e
`password`. I rendiconti registrano la presenza dei tipi per blocco, non una
matrice di confusione separata per tipo; non si attribuiscono quindi FN o recall
alle singole famiglie.

Aggiornamenti e skip restano quelli del riepilogo di campagna:

| memoria | politica | LR agg./skip | MLP agg./skip | KAN agg./skip |
|---|---|---:|---:|---:|
| FIFO | ogni blocco | 520,0 / 395,0 | 520,0 / 395,0 | 520,0 / 395,0 |
| FIFO | evidence | 45,2 / 0,0 | 36,2 / 0,0 | 18,6 / 0,0 |
| per classe | ogni blocco | 915,0 / 0,0 | 915,0 / 0,0 | 915,0 / 0,0 |
| per classe | evidence | 30,4 / 0,0 | 14,0 / 0,0 | 10,2 / 0,0 |

## Differenze appaiate

Differenze `per_classe - FIFO` sugli stessi cinque semi e sulla stessa politica.
I valori di ogni singolo seme sono in
`adapt02_memoria_per_classe/differenze_appaiate_per_seme_C.csv`; la tabella
riporta media e deviazione standard campionaria dei cinque delta. I cinque semi
percorrono lo stesso stream C e non sono trattati come cinque dataset
indipendenti; la dispersione descrive la sensibilita' al seme.

| politica | modello | delta AUROC media+-ds | delta recall media+-ds | delta FPR compl. p.p. media+-ds | delta FP media+-ds | delta FN media+-ds |
|---|---|---:|---:|---:|---:|---:|
| ogni blocco | LR | 0,0511 +- 0,0210 | -0,6463 +- 0,0202 | -16,566 +- 1,516 | -28.711 +- 2.628 | 5.802.238 +- 181.717 |
| ogni blocco | MLP | 0,1419 +- 0,0409 | -0,5774 +- 0,0567 | -15,723 +- 2,700 | -27.251 +- 4.680 | 5.183.445 +- 508.851 |
| ogni blocco | KAN | 0,0843 +- 0,0181 | -0,5666 +- 0,0095 | -11,027 +- 2,172 | -19.111 +- 3.764 | 5.086.487 +- 85.609 |
| evidence | LR | 0,0481 +- 0,0533 | -0,6363 +- 0,0381 | -17,249 +- 7,312 | -29.895 +- 12.674 | 5.711.979 +- 342.317 |
| evidence | MLP | 0,1024 +- 0,0247 | -0,5655 +- 0,0637 | -17,564 +- 8,249 | -30.441 +- 14.297 | 5.076.688 +- 572.131 |
| evidence | KAN | 0,0589 +- 0,0462 | -0,4505 +- 0,1484 | -7,816 +- 4,113 | -13.547 +- 7.129 | 4.044.722 +- 1.331.867 |

## Memoria e costo

Medie sui cinque semi, per combinazione memoria-politica.

| memoria | politica | normali medi | attacchi medi | eta mediana normali | eta mediana attacchi | secondi medi |
|---|---|---:|---:|---:|---:|---:|
| FIFO | ogni blocco | 4,8 | 251,2 | 1,8 | 2,0 | 511,8 |
| FIFO | evidence | 4,8 | 251,2 | 1,8 | 2,0 | 372,1 |
| per classe | ogni blocco | 125,6 | 130,4 | 142,5 | 1,0 | 650,4 |
| per classe | evidence | 125,6 | 130,4 | 142,5 | 1,0 | 523,5 |

## Figure

Le figure nuove della campagna C sono in:

`adapt02_memoria_per_classe/figure_campagna_C/`

Sono rigenerate dai 20 rendiconti completi tramite:

```bash
cd <repo>/adattamento-drift/ton_temporal
python analisi_memoria_per_classe.py
```

Figure prodotte:

- `fpr_kan_complessivo_blocchi.png`
- `differenze_appaiate_auroc.png`
- `differenze_appaiate_recall_attacchi.png`
- `differenze_appaiate_fpr_complessivo.png`
- `differenze_appaiate_fp.png`
- `differenze_appaiate_fn.png`

## Lettura

La memoria per classe cambia drasticamente la composizione del buffer: da una
FIFO quasi tutta Attack a una memoria quasi bilanciata. Questo elimina gli skip
monoclasse nelle corse `ogni_blocco` e riduce molto i falsi allarmi.

Il costo e' pero' un aumento marcato degli attacchi mancati. Il caso KAN e' il
piu' chiaro: con `per_classe` il FPR complessivo e' 0,323% con
`ogni_blocco` e 1,545% con evidence. Le medie per blocco sono invece 0,765% e
0,681%. Il recall degli attacchi resta intorno a 0,20-0,22. La KAN frozen
mantiene AUROC medio 0,8911; la KAN adattiva con `per_classe` arriva a 0,8915
con `ogni_blocco` e 0,8929 con evidence, ma con molti piu' FN.

Il risultato non va quindi formulato come superiorita' generale della memoria
per classe. La campagna mostra un compromesso: meno falsi allarmi e meno skip,
ma piu' attacchi mancati. AUROC, FN, FP e recall devono essere letti insieme.

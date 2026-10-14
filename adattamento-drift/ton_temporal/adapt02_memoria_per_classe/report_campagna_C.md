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

Cartella condivisa da cui scaricare l'archivio:

`https://drive.google.com/drive/folders/1lAsdaKNOWUu9S70Pp8rKlA2kyyfi8ncK?usp=sharing`

Dimensione: 7.253.160 byte.

SHA-256:

`212B9270EF47D0B997667E83B38CB6C899A5B6ADD00DF61B881AA14DA905AFE0`

In Git e' conservato il riepilogo leggero:

`adapt02_memoria_per_classe/riepilogo_campagna_C.json`

Analisi leggera derivata dai rendiconti completi:

`adapt02_memoria_per_classe/analisi_campagna_C.json`

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
| ogni blocco | 0,0032 | 0,0076 |
| evidence | 0,0155 | 0,0068 |

## Frozen

Medie sui cinque semi, una sola volta per seme. La dispersione e' la deviazione
standard campionaria sui cinque semi.

| modello | AUROC media+-ds | recall compl. media+-ds | FPR compl. media+-ds | FPR blocchi media+-ds | FN media+-ds | FP media+-ds |
|---|---:|---:|---:|---:|---:|---:|
| LR | 0,7149 +- 0,0000 | 0,7213 +- 0,0000 | 0,4872 +- 0,0000 | 0,4673 +- 0,0000 | 2.501.546 +- 0 | 84.443 +- 0 |
| MLP | 0,7933 +- 0,0177 | 0,4378 +- 0,0874 | 0,2860 +- 0,0072 | 0,1421 +- 0,0051 | 5.046.751 +- 784.237 | 49.570 +- 1.241 |
| KAN | 0,8911 +- 0,0046 | 0,6593 +- 0,0059 | 0,1324 +- 0,0001 | 0,0399 +- 0,0013 | 3.058.698 +- 52.541 | 22.945 +- 26 |

## Risultati adattivi

Medie sui cinque semi. La tabella distingue i tassi complessivi dalle medie per
blocco.

| memoria | politica | modello | AUROC media+-ds | recall compl. media+-ds | FPR compl. media+-ds | FPR blocchi media+-ds | FN media+-ds | FP media+-ds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FIFO | ogni blocco | LR | 0,7739 +- 0,0108 | 0,8824 +- 0,0089 | 0,1854 +- 0,0162 | 0,4124 +- 0,0152 | 1.055.800 +- 80.210 | 32.136 +- 2.806 |
| FIFO | ogni blocco | MLP | 0,7674 +- 0,0511 | 0,8590 +- 0,0603 | 0,1707 +- 0,0282 | 0,3885 +- 0,0354 | 1.265.693 +- 541.106 | 29.584 +- 4.880 |
| FIFO | ogni blocco | KAN | 0,8073 +- 0,0126 | 0,7690 +- 0,0196 | 0,1135 +- 0,0228 | 0,2912 +- 0,0149 | 2.073.531 +- 175.728 | 19.671 +- 3.947 |
| FIFO | evidence | LR | 0,7756 +- 0,0290 | 0,8199 +- 0,0312 | 0,1944 +- 0,0830 | 0,4289 +- 0,0728 | 1.616.779 +- 280.106 | 33.688 +- 14.377 |
| FIFO | evidence | MLP | 0,7939 +- 0,0357 | 0,7473 +- 0,1136 | 0,1993 +- 0,0853 | 0,3646 +- 0,1069 | 2.268.765 +- 1.019.960 | 34.541 +- 14.775 |
| FIFO | evidence | KAN | 0,8341 +- 0,0368 | 0,6688 +- 0,0876 | 0,0936 +- 0,0369 | 0,2373 +- 0,0951 | 2.973.274 +- 786.101 | 16.225 +- 6.395 |
| per classe | ogni blocco | LR | 0,8250 +- 0,0227 | 0,2361 +- 0,0163 | 0,0198 +- 0,0018 | 0,0612 +- 0,0275 | 6.858.037 +- 145.887 | 3.425 +- 304 |
| per classe | ogni blocco | MLP | 0,9093 +- 0,0183 | 0,2816 +- 0,1107 | 0,0135 +- 0,0039 | 0,0341 +- 0,0112 | 6.449.138 +- 993.424 | 2.333 +- 681 |
| per classe | ogni blocco | KAN | 0,8915 +- 0,0146 | 0,2024 +- 0,0173 | 0,0032 +- 0,0012 | 0,0076 +- 0,0074 | 7.160.018 +- 155.307 | 559 +- 203 |
| per classe | evidence | LR | 0,8237 +- 0,0328 | 0,1836 +- 0,0328 | 0,0219 +- 0,0106 | 0,0535 +- 0,0403 | 7.328.758 +- 294.877 | 3.793 +- 1.840 |
| per classe | evidence | MLP | 0,8963 +- 0,0244 | 0,1818 +- 0,0732 | 0,0237 +- 0,0153 | 0,0298 +- 0,0108 | 7.345.453 +- 657.015 | 4.100 +- 2.653 |
| per classe | evidence | KAN | 0,8929 +- 0,0303 | 0,2183 +- 0,0694 | 0,0155 +- 0,0155 | 0,0068 +- 0,0061 | 7.017.997 +- 623.239 | 2.678 +- 2.687 |

Aggiornamenti e skip restano quelli del riepilogo di campagna:

| memoria | politica | LR agg./skip | MLP agg./skip | KAN agg./skip |
|---|---|---:|---:|---:|
| FIFO | ogni blocco | 520,0 / 395,0 | 520,0 / 395,0 | 520,0 / 395,0 |
| FIFO | evidence | 45,2 / 0,0 | 36,2 / 0,0 | 18,6 / 0,0 |
| per classe | ogni blocco | 915,0 / 0,0 | 915,0 / 0,0 | 915,0 / 0,0 |
| per classe | evidence | 30,4 / 0,0 | 14,0 / 0,0 | 10,2 / 0,0 |

## Differenze appaiate

Differenze `per_classe - FIFO` sugli stessi cinque semi e sulla stessa politica.

| politica | modello | delta AUROC media+-ds | delta recall media+-ds | delta FPR compl. media+-ds | delta FP media+-ds | delta FN media+-ds |
|---|---|---:|---:|---:|---:|---:|
| ogni blocco | LR | 0,0511 +- 0,0210 | -0,6463 +- 0,0202 | -0,1657 +- 0,0152 | -28.711 +- 2.628 | 5.802.238 +- 181.717 |
| ogni blocco | MLP | 0,1419 +- 0,0409 | -0,5774 +- 0,0567 | -0,1572 +- 0,0270 | -27.251 +- 4.680 | 5.183.445 +- 508.851 |
| ogni blocco | KAN | 0,0843 +- 0,0181 | -0,5666 +- 0,0095 | -0,1103 +- 0,0217 | -19.111 +- 3.764 | 5.086.487 +- 85.609 |
| evidence | LR | 0,0481 +- 0,0533 | -0,6363 +- 0,0381 | -0,1725 +- 0,0731 | -29.895 +- 12.674 | 5.711.979 +- 342.317 |
| evidence | MLP | 0,1024 +- 0,0247 | -0,5655 +- 0,0637 | -0,1756 +- 0,0825 | -30.441 +- 14.297 | 5.076.688 +- 572.131 |
| evidence | KAN | 0,0589 +- 0,0462 | -0,4505 +- 0,1484 | -0,0782 +- 0,0411 | -13.547 +- 7.129 | 4.044.722 +- 1.331.867 |

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
- `differenze_appaiate_fp.png`
- `differenze_appaiate_fn.png`

## Lettura

La memoria per classe cambia drasticamente la composizione del buffer: da una
FIFO quasi tutta Attack a una memoria quasi bilanciata. Questo elimina gli skip
monoclasse nelle corse `ogni_blocco` e riduce molto i falsi allarmi.

Il costo e' pero' un aumento marcato degli attacchi mancati. Il caso KAN e' il
piu' chiaro: con `per_classe` il FPR complessivo scende a 0,0032 con
`ogni_blocco` e a 0,0155 con evidence. Le medie per blocco sono invece 0,0076 e
0,0068. Il recall degli attacchi resta intorno a 0,20-0,22. La KAN frozen
mantiene AUROC medio 0,8911; la KAN adattiva con `per_classe` arriva a 0,8915
con `ogni_blocco` e 0,8929 con evidence, ma con molti piu' FN.

Il risultato non va quindi formulato come superiorita' generale della memoria
per classe. La campagna mostra un compromesso: meno falsi allarmi e meno skip,
ma piu' attacchi mancati. AUROC, FN, FP e recall devono essere letti insieme.

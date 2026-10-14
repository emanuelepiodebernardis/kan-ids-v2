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

Dimensione: 7.253.160 byte.

SHA-256:

`212B9270EF47D0B997667E83B38CB6C899A5B6ADD00DF61B881AA14DA905AFE0`

In Git e' conservato il riepilogo leggero:

`adapt02_memoria_per_classe/riepilogo_campagna_C.json`

## Tempi

Tempo totale della campagna: 10.299,36 secondi, cioe' circa 2 ore, 51 minuti e
39 secondi.

Comando eseguito:

```bash
cd <repo>/adattamento-drift/ton_temporal
python campagna_memoria_per_classe.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --uscita-dir ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8
```

## Frozen

Medie sui cinque semi, una sola volta per seme.

| modello | AUROC | recall attacchi | FPR | FN medi | FP medi |
|---|---:|---:|---:|---:|---:|
| LR | 0,7149 | 0,7204 | 0,4673 | 2.501.546 | 84.443 |
| MLP | 0,7933 | 0,4376 | 0,1421 | 5.046.751 | 49.570 |
| KAN | 0,8911 | 0,6585 | 0,0399 | 3.058.698 | 22.945 |

## Risultati adattivi

Medie sui cinque semi.

| memoria | politica | modello | AUROC | recall attacchi | FPR | FN medi | FP medi | aggiornamenti | skip |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FIFO | ogni blocco | LR | 0,7739 | 0,8786 | 0,4124 | 1.055.800 | 32.136 | 520,0 | 395,0 |
| FIFO | ogni blocco | MLP | 0,7674 | 0,8556 | 0,3885 | 1.265.693 | 29.584 | 520,0 | 395,0 |
| FIFO | ogni blocco | KAN | 0,8073 | 0,7660 | 0,2912 | 2.073.531 | 19.671 | 520,0 | 395,0 |
| FIFO | evidence | LR | 0,7756 | 0,8179 | 0,4289 | 1.616.779 | 33.688 | 45,2 | 0,0 |
| FIFO | evidence | MLP | 0,7939 | 0,7459 | 0,3646 | 2.268.765 | 34.541 | 36,2 | 0,0 |
| FIFO | evidence | KAN | 0,8341 | 0,6668 | 0,2373 | 2.973.274 | 16.225 | 18,6 | 0,0 |
| per classe | ogni blocco | LR | 0,8250 | 0,2378 | 0,0612 | 6.858.037 | 3.425 | 915,0 | 0,0 |
| per classe | ogni blocco | MLP | 0,9093 | 0,2822 | 0,0341 | 6.449.138 | 2.333 | 915,0 | 0,0 |
| per classe | ogni blocco | KAN | 0,8915 | 0,2040 | 0,0076 | 7.160.018 | 559 | 915,0 | 0,0 |
| per classe | evidence | LR | 0,8237 | 0,1864 | 0,0535 | 7.328.758 | 3.793 | 30,4 | 0,0 |
| per classe | evidence | MLP | 0,8963 | 0,1834 | 0,0298 | 7.345.453 | 4.100 | 14,0 | 0,0 |
| per classe | evidence | KAN | 0,8929 | 0,2195 | 0,0068 | 7.017.997 | 2.678 | 10,2 | 0,0 |

## Memoria e costo

Medie sui cinque semi, per combinazione memoria-politica.

| memoria | politica | normali medi | attacchi medi | eta mediana normali | eta mediana attacchi | secondi medi |
|---|---|---:|---:|---:|---:|---:|
| FIFO | ogni blocco | 4,8 | 251,2 | 1,8 | 2,0 | 511,8 |
| FIFO | evidence | 4,8 | 251,2 | 1,8 | 2,0 | 372,1 |
| per classe | ogni blocco | 125,6 | 130,4 | 142,5 | 1,0 | 650,4 |
| per classe | evidence | 125,6 | 130,4 | 142,5 | 1,0 | 523,5 |

## Lettura

La memoria per classe cambia drasticamente la composizione del buffer: da una
FIFO quasi tutta Attack a una memoria quasi bilanciata. Questo elimina gli skip
monoclasse nelle corse `ogni_blocco` e riduce molto i falsi allarmi.

Il costo e' pero' un aumento marcato degli attacchi mancati. Il caso KAN e' il
piu' chiaro: con `per_classe` il FPR scende fino a 0,0076 con `ogni_blocco` e a
0,0068 con evidence, ma il recall degli attacchi resta intorno a 0,20-0,22. La
KAN frozen mantiene AUROC medio 0,8911; la KAN adattiva con `per_classe` arriva
a 0,8915 con `ogni_blocco` e 0,8929 con evidence, ma con molti piu' FN.

Il risultato non va quindi formulato come superiorita' generale della memoria
per classe. La campagna mostra un compromesso: meno falsi allarmi e meno skip,
ma piu' attacchi mancati. AUROC, FN, FP e recall devono essere letti insieme.

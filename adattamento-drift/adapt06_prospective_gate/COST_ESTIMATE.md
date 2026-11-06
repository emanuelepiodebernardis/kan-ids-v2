# Stima del costo

La stima non è una misura ADAPT-06: usa tempi registrati dal Paper 2 e sarà
sostituita dai tempi dello smoke, senza cambiare il protocollo.

## Etichette

Il costo operativo per un deployment/seed è fissato, non stimato:

| Dataset | Richieste totali | CONTROL | FIT | EXPLORE ACTIVE | EXPLOIT ACTIVE |
|---|---:|---:|---:|---:|---:|
| TON | 35.499 | 8.874 | 26.625 | 5.325 | 21.300 |
| UNSW | 10.160 | 2.540 | 7.620 | 1.524 | 6.096 |

Le repliche non rappresentano label aggiuntive in un singolo deployment, ma
moltiplicano il costo sperimentale. Non sono previste seconde annotazioni nel
protocollo primario.

Con la prevalenza osservata degli stream e CONTROL uniforme, in 50 blocchi
pieni sono attesi circa 109 normali/1.141 attacchi su TON e 989 normali/261
attacchi su UNSW. Il primo look 100/100 è quindi plausibile ma non garantito:
il runner deve usare i conteggi effettivi e restare in `WAIT` quando una classe
è insufficiente. Sull'intero stream le attese sono 774/8.100 per TON e
2.009/531 per UNSW; non vengono usate per sostituire osservazioni mancanti.

## Calcolo

Tempi medi registrati sui medesimi stream e cinque seed:

| Dataset | GAM spline | LR | ARF |
|---|---:|---:|---:|
| TON | 25,62 s | 16,26 s | 231,63 s |
| UNSW | 6,65 s | 4,17 s | 63,92 s |

Le fonti sono
`adapt05_occurrence/results/updates_and_costs.csv` per il runner GAM di
ADAPT-05 e `review_extension/analysis/resources_and_labels.csv` per GAM, LR e
ARF del Paper 2. Sono misure di replay su runner già esistenti, non benchmark
del metodo proposto.

La matrice proposta contiene, per dataset/seed, quattro traiettorie GAM,
quattro LR e due ARF: 20 traiettorie adattive nel pilot sui due dataset e 100
nello sviluppo a cinque seed. I riferimenti congelati o threshold-only vengono
riusati quando le predizioni compatibili sono già disponibili e non sono
conteggiati come nuovi fit.

La somma seriale diretta dei tempi tabellati per il pilot è:

| Dataset | 4 × GAM | 4 × LR | 2 × ARF | Subtotale |
|---|---:|---:|---:|---:|
| TON | 102,48 s | 65,04 s | 463,26 s | 630,78 s |
| UNSW | 26,60 s | 16,68 s | 127,84 s | 171,12 s |
| **Totale** | **129,08 s** | **81,72 s** | **591,10 s** | **801,90 s** |

Il riferimento seriale corretto è **801,9 secondi = 13,365 minuti**. Non è un
wall time: con due worker il limite ideale dato dalla sola somma è 400,95
secondi = 6,6825 minuti, prima di sbilanciamento, I/O e overhead.
Per cinque seed la stessa base vale 4.009,5 secondi = 66,825 minuti seriali e
almeno 33,4125 minuti ideali a due worker.

Le implementazioni dovranno condividere trasformazioni, score invarianti e
richieste invece di duplicare lavoro. Acquisizione farthest-first, shadow
scoring, gate e log aggiungono costo non ancora misurato; per questo la riserva
prudenziale sotto non deriva dividendo semplicemente la somma seriale.

Riserva prudenziale:

- test sintetici: meno di 10 minuti per esecuzione completa;
- pilot seed 42, due dataset: 30–45 minuti, due worker, BLAS a un thread;
- sviluppo a cinque seed: 2–3 ore, due worker;
- analisi e confezionamento: 30–60 minuti;
- spazio temporaneo: 1 GB, da rivedere dopo lo smoke.

La previsione sarà aggiornata dopo il futuro pilot con wall time, CPU time,
picco di memoria, byte dello stato persistente, numero di fit e costo del
selettore. Equal-label-budget non sarà descritto come equal-resource.

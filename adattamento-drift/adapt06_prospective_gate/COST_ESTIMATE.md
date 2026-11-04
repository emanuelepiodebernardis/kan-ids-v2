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
conteggiati come nuovi fit. Le implementazioni dovranno condividere
trasformazioni, score invarianti e richieste invece di duplicare lavoro. Il
lower bound basato sui runner precedenti è circa 10,7 minuti per il pilot seed
42 sui due dataset, dominato dalle due ARF. Acquisizione farthest-first, shadow
scoring, gate e log aggiungono costo non ancora misurato.

Riserva prudenziale:

- test sintetici: meno di 10 minuti per esecuzione completa;
- pilot seed 42, due dataset: 30–45 minuti, due worker, BLAS a un thread;
- sviluppo a cinque seed: 2–3 ore, due worker;
- analisi e confezionamento: 30–60 minuti;
- spazio temporaneo: 1 GB, da rivedere dopo lo smoke.

La previsione sarà aggiornata dopo il futuro pilot con wall time, CPU time,
picco di memoria, byte dello stato persistente, numero di fit e costo del
selettore. Equal-label-budget non sarà descritto come equal-resource.

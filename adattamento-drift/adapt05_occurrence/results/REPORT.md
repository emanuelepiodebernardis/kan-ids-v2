# ADAPT-05 — confronto completo

Commit eseguito: `3d4b4cb6f4ca08f7095785dc433c43919b4dc85f`. Matrice: 2 dataset × 5 seed × 3 sali = 30 combinazioni.
Protocollo, budget, memorie e obiettivi sono quelli fissati prima del checkpoint.
TON e UNSW restano due stream di sviluppo; i cinque seed non sono cinque catture.

## Copertura della calibrazione

La copertura qui riportata e' una diagnostica cumulativa retrospettiva: il denominatore comprende tutte le richieste normali dell'intero stream, incluse quelle del blocco terminale la cui label era ancora pendente alla chiusura del replay. Non descrive la CAL64 online, che e' limitata, soggetta a espulsione e disponibile soltanto dopo il ritardo causale.

| Dataset | Ruolo | Copertura vettori normali ponderata | Copertura vettori unici | Normali assegnati a CAL |
|---|---|---:|---:|---:|
| TON | vector | 28.14% ± 7.95 | 25.16% ± 1.22 | 28.14% |
| TON | occurrence | 69.51% ± 1.15 | 29.48% ± 1.67 | 24.81% |
| UNSW | vector | 22.18% ± 1.31 | 24.99% ± 0.58 | 22.18% |
| UNSW | occurrence | 44.65% ± 0.66 | 26.97% ± 0.77 | 25.15% |

Le label ancora pendenti alla chiusura sono 60–99 richieste per combinazione; sono incluse soltanto nella diagnostica retrospettiva cumulativa.

I digest dei vettori normali sono elencati per frequenza in `type_coverage.csv`; assegnazioni CAL/FIT ed esclusione da CAL sono aggregate sulle 15 celle seed/sale di ciascun dataset e modalita', senza introdurre una soglia post hoc.

## Stato CAL64 disponibile online

Le righe seguenti sintetizzano gli snapshot registrati dopo il feedback. Lo snapshot del blocco k diventa utilizzabile dalle predizioni del blocco k+1, non da quelle gia' emesse in k. `normal` conta i normali fisicamente presenti in CAL64; `recent_normal` conta quelli entro l'eta' massima di 50 blocchi. L'intervallo d'eta' registrato riguarda l'intera vista ammissibile (normali e attacchi): i log non conservano l'eta' separata per classe.

| Dataset | Ruolo | Normali presenti | Normali recenti | Blocchi senza normali | Eta' massima ammissibile |
|---|---|---:|---:|---:|---:|
| TON | vector | 29.81 ± 6.43 | 22.65 ± 8.82 | 18/5310 | 45.0 [1, 50] |
| TON | occurrence | 29.99 ± 6.24 | 22.17 ± 8.42 | 20/5310 | 46.0 [1, 50] |
| UNSW | vector | 32.98 ± 5.05 | 32.98 ± 5.05 | 0/1515 | 8.0 [1, 26] |
| UNSW | occurrence | 33.07 ± 4.66 | 33.07 ± 4.66 | 0/1515 | 6.0 [1, 15] |

La serie completa per blocco e' in `cal64_memory_by_block.csv`; `cal64_memory.png` ne mostra occupazione normale e inviluppo d'eta'.

## Vantaggio residuo della testa

Differenze appaiate head − frozen, medie sulle 15 combinazioni di ciascun dataset:

| Dataset | Ruolo | Obiettivo | Δ recall | Δ FPR | Δ BA |
|---|---|---|---:|---:|---:|
| TON | vector | lambda1 | -1.51 pp | -28.20 pp | +13.35 pp |
| TON | vector | lambda2 | +1.37 pp | -15.19 pp | +8.28 pp |
| TON | vector | fpr0.05 | +33.19 pp | -10.45 pp | +21.82 pp |
| TON | vector | fpr0.1 | +27.48 pp | -9.12 pp | +18.30 pp |
| TON | vector | fpr0.2 | +12.75 pp | -8.56 pp | +10.65 pp |
| TON | occurrence | lambda1 | -0.74 pp | -25.74 pp | +12.50 pp |
| TON | occurrence | lambda2 | +1.31 pp | -0.17 pp | +0.74 pp |
| TON | occurrence | fpr0.05 | +37.77 pp | +6.51 pp | +15.63 pp |
| TON | occurrence | fpr0.1 | +33.93 pp | +6.65 pp | +13.64 pp |
| TON | occurrence | fpr0.2 | +16.11 pp | +7.44 pp | +4.34 pp |
| UNSW | vector | lambda1 | -0.28 pp | -0.79 pp | +0.25 pp |
| UNSW | vector | lambda2 | -0.31 pp | -0.65 pp | +0.17 pp |
| UNSW | vector | fpr0.05 | +0.07 pp | -0.88 pp | +0.48 pp |
| UNSW | vector | fpr0.1 | -0.40 pp | +0.34 pp | -0.37 pp |
| UNSW | vector | fpr0.2 | -0.50 pp | +0.66 pp | -0.58 pp |
| UNSW | occurrence | lambda1 | +0.16 pp | +0.64 pp | -0.24 pp |
| UNSW | occurrence | lambda2 | -0.12 pp | +0.48 pp | -0.30 pp |
| UNSW | occurrence | fpr0.05 | +4.59 pp | +0.60 pp | +2.00 pp |
| UNSW | occurrence | fpr0.1 | +0.54 pp | +0.55 pp | -0.00 pp |
| UNSW | occurrence | fpr0.2 | +0.33 pp | +0.44 pp | -0.06 pp |

## Aggiornamenti e obiettivi

| Dataset | Ruolo | Aggiornamenti applicati |
|---|---|---:|
| TON | vector | 347.5 ± 3.1 |
| TON | occurrence | 347.6 ± 3.6 |
| UNSW | vector | 99.9 ± 1.2 |
| UNSW | occurrence | 100.7 ± 0.7 |

Le violazioni del target FPR sono 305 su 360 punti operativi. I valori raggiunti, non quelli richiesti, sono usati nelle figure e nei confronti.

## Costi osservati

| Dataset | Replay per combinazione, secondi | Intervallo |
|---|---:|---:|
| TON | 20.70 ± 0.34 | 20.11–21.29 |
| UNSW | 5.66 ± 0.12 | 5.51–5.96 |

I tempi sono misurati dal runner Linux con due celle concorrenti e descrivono questa esecuzione; non sono benchmark seriali.

## Aderenza alle analisi previste

La tabella completa, inclusi i riferimenti agli output, e' in `analysis_adherence.csv`.

| Analisi prevista | Stato | Nota |
|---|---|---|
| Recall, FPR raggiunto, FP, FN e BA | complete | Metriche globali per tutte le politiche. |
| Violazioni dei target FPR | complete | Sono usati i valori raggiunti, non i target nominali. |
| Andamento temporale e primi blocchi separati | complete | I primi 30 blocchi sono esportati senza riaggregazione post hoc. |
| Confronti a FPR raggiunto comune | not_estimable | La griglia discreta non definisce tolleranza o interpolazione preregistrata; nessun confronto post hoc e' introdotto. |
| Copertura ponderata dei vettori normali | complete | Diagnostica cumulativa retrospettiva sull'intero stream, incluse le label terminali pendenti. |
| Vettori normali frequenti esclusi da CAL | complete | Una riga per dataset, modalita' e digest, aggregata sulle 15 celle seed/sale. |
| Normali disponibili ed eta' CAL64 | partial | I log danno conteggi normali e intervallo d'eta' dell'intera vista ammissibile; non serializzano l'eta' per classe. |
| Ricambio esatto della memoria | not_estimable | I report non serializzano gli observation_id trattenuti per blocco, quindi ingressi ed espulsioni individuali non sono ricostruibili dai log. |
| Update e skip | complete | Tutti gli stati registrati sono conservati. |
| Costi | complete | Tempi osservati del replay, non benchmark seriali. |
| Visto/non visto in A/B | not_estimable | I report consegnati non contengono un inventario di digest A/B; non viene inferito senza una regola aggiuntiva. |
| Ripetizioni interne a D | complete | La frequenza del digest distingue vettori singoli e ripetuti tra le richieste normali. |
| Dispersione tra seed e sali | complete | Le due fonti di variabilita' sono separate. |
| Conservazione di tutti gli esiti | complete | Nessun esito e' filtrato per segno o significativita'. |

## Interpretazione e limiti

Il cambio di chiave FIT/CAL viene interpretato come intervento sulla copertura, non come nuovo algoritmo. I calendari di aggiornamento possono differire fra le modalità e sono riportati separatamente; per questo le differenze non sono attribuite automaticamente alla sola rappresentatività.

Il vantaggio della testa è valutato all'interno di ciascuna modalità con confronti appaiati. Risultati nulli o sfavorevoli sono inclusi. Due stream e cinque randomizzazioni non dimostrano superiorità generale né sostituiscono una conferma su catture indipendenti.

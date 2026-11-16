# ADAPT-06B-MATRIX — robustezza GAM/RANDOM, seed 42–46

La matrice riutilizza senza modifiche il pilot seed 42 e aggiunge i 16 replay preregistrati per seed 43–46. Il confronto principale è GATED decoupled meno threshold-only autonoma; i seed appartengono agli stessi stream e non sono catture indipendenti.

## Differenze appaiate

| Dataset | Seed | Δ recall (pp) | Δ FPR (pp) | Δ TP | Δ FN | Δ FP | Promozioni |
|---|---:|---:|---:|---:|---:|---:|---:|
| TON | 42 | +0.000 | +0.006 | +0 | +0 | +18 | 0 |
| TON | 43 | -0.004 | -0.667 | -118 | +118 | -2064 | 0 |
| TON | 44 | -0.003 | -0.315 | -92 | +92 | -977 | 0 |
| TON | 45 | -0.002 | -0.380 | -72 | +72 | -1176 | 0 |
| TON | 46 | -0.000 | +0.004 | -1 | +1 | +12 | 0 |
| UNSW | 42 | +1.167 | -1.491 | +2478 | -2478 | -11987 | 1 |
| UNSW | 43 | +5.691 | -1.328 | +12080 | -12080 | -10677 | 0 |
| UNSW | 44 | +1.969 | -2.354 | +4180 | -4180 | -18919 | 0 |
| UNSW | 45 | +0.546 | -2.198 | +1159 | -1159 | -17668 | 0 |
| UNSW | 46 | +4.647 | -1.881 | +9864 | -9864 | -15122 | 0 |

## Controlli diagnostici

| Dataset | Seed | Promozioni | Δ recall vs matched | Δ FPR vs matched | Δ recall vs Frozen/B | Δ FPR vs Frozen/B |
|---|---:|---:|---:|---:|---:|---:|
| TON | 42 | 0 | +0.000 | +0.000 | -0.209 | -7.930 |
| TON | 43 | 0 | +0.000 | +0.000 | -0.209 | -8.153 |
| TON | 44 | 0 | +0.000 | +0.000 | -0.208 | -7.786 |
| TON | 45 | 0 | +0.000 | +0.000 | -0.209 | -8.636 |
| TON | 46 | 0 | +0.000 | +0.000 | -0.209 | -8.599 |
| UNSW | 42 | 1 | +1.490 | +1.010 | +1.490 | +1.322 |
| UNSW | 43 | 0 | +0.000 | +0.000 | -0.232 | +0.422 |
| UNSW | 44 | 0 | +0.000 | +0.000 | +0.000 | +0.356 |
| UNSW | 45 | 0 | +0.000 | +0.000 | +0.000 | +0.212 |
| UNSW | 46 | 0 | +0.000 | +0.000 | +0.000 | +0.074 |

## Sintesi per dataset

| Dataset | Δ recall media ± ds (pp) | Intervallo | Δ FPR media ± ds (pp) | Intervallo | Recall favorevole | FPR favorevole | Entrambe vs autonoma | Seed con promozioni | Entrambe vs matched | Seed peggiore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TON | -0.002 ± 0.002 | [-0.004, +0.000] | -0.270 ± 0.284 | [-0.667, +0.006] | 0/5 | 3/5 | 0/5 | 0/5 | 0/5 | 43 |
| UNSW | +2.804 ± 2.247 | [+0.546, +5.691] | -1.851 ± 0.441 | [-2.354, -1.328] | 5/5 | 5/5 | 5/5 | 1/5 | 0/5 | 45 |

## Gate, CAL e stato terminale

Gli indici di blocco sono 0-based. La contabilità riporta label uniche, distinte per ruolo, e label già richieste ma ancora in transito alla fine dello stream.

| Dataset | Seed | Prima CAL/applicazione | Cambi soglia | Decisioni | Label uniche (CONTROL/RANDOM_FIT) | Label/batch pendenti | Candidato pendente |
|---|---:|---:|---:|---|---:|---:|---:|
| TON | 42 | 275/276 | 2 | FREEZE=3, REJECT_HARM=2, WITHHOLD_INSUFFICIENT=1 | 35499 (8874/26625) | 99/1 | False |
| TON | 43 | 275/276 | 2 | FREEZE=3, REJECT_HARM=2, WAIT=1 | 35499 (8874/26625) | 99/1 | False |
| TON | 44 | 275/276 | 2 | FREEZE=3, REJECT_HARM=2, WITHHOLD_INSUFFICIENT=1 | 35499 (8874/26625) | 99/1 | False |
| TON | 45 | 274/275 | 2 | FREEZE=2, REJECT_HARM=1, WAIT=1, WITHHOLD_INSUFFICIENT=1 | 35499 (8874/26625) | 99/1 | False |
| TON | 46 | 274/275 | 2 | FREEZE=2, REJECT_HARM=1, WAIT=1, WITHHOLD_INSUFFICIENT=1 | 35499 (8874/26625) | 99/1 | False |
| UNSW | 42 | 17/18 | 3 | ACCEPT=1, FREEZE=3, REJECT_HARM=1, WAIT=1, WITHHOLD_STREAM_END_AMBIGUOUS=1 | 10160 (2540/7620) | 60/1 | False |
| UNSW | 43 | 18/19 | 2 | FREEZE=3, REJECT_HARM=2, WAIT=1, WITHHOLD_STREAM_END_AMBIGUOUS=1 | 10160 (2540/7620) | 60/1 | False |
| UNSW | 44 | 19/20 | 4 | FREEZE=4, REJECT_HARM=3, WITHHOLD_STREAM_END_INSUFFICIENT=1 | 10160 (2540/7620) | 60/1 | False |
| UNSW | 45 | 17/18 | 2 | FREEZE=3, REJECT_HARM=1, WAIT=2, WITHHOLD_STREAM_END_INSUFFICIENT=1 | 10160 (2540/7620) | 60/1 | False |
| UNSW | 46 | 18/19 | 2 | FREEZE=2, WAIT=3, WITHHOLD_STREAM_END_AMBIGUOUS=1 | 10160 (2540/7620) | 60/1 | False |

## Costi osservati

| Dataset | Replay legacy medio (s) | Replay decoupled medio (s) | RSS legacy massimo (MiB) | RSS decoupled massimo (MiB) | Payload legacy massimo (B) | Payload decoupled massimo (B) |
|---|---:|---:|---:|---:|---:|---:|
| TON | 198.849 | 210.155 | 523.7 | 521.1 | 145,654 | 144,656 |
| UNSW | 57.583 | 45.153 | 255.5 | 252.7 | 145,654 | 144,656 |

Il launcher seriale delle otto coppie nuove ha richiesto 2442.741 s (40.712 min), inclusi caricamenti, scrittura e verifiche fra i processi.

## Limiti e lettura

Su TON non avviene alcuna promozione in cinque seed e GATED decoupled coincide sempre con threshold-only matched: non emerge quindi un beneficio dei pesi. Su UNSW il confronto primario è favorevole su entrambe le metriche in 5/5 seed, ma una promozione avviene soltanto nel seed 42. Nei seed 43–46 GATED decoupled coincide con matched; il vantaggio rispetto alla baseline autonoma deriva dalla diversa cadenza della soglia, non dai pesi. Nel seed 42 la promozione aumenta il recall rispetto a matched ma aumenta anche l'FPR.

Le differenze di recall e FPR devono essere lette congiuntamente. I controlli matched, Frozen/B e legacy restano nel CSV completo e non vengono sostituiti dal solo confronto primario. Un risultato favorevole contro threshold-only autonoma non dimostra da solo un vantaggio dei pesi a parità di FPR.

Tutti i seed, comprese repliche nulle o sfavorevoli, sono conservati. Le statistiche descrivono robustezza rispetto all'inizializzazione sugli stessi stream, non generalizzazione a nuove catture né superiorità generale.

# ADAPT-06I — verifica TON sui cinque seed

Il seed 42 proviene dal pilot approvato. I seed 43–46 usano i rispettivi modelli A/B e le richieste ADAPT-06E.
Per ciascun seed sono valutati i blocchi 20–354 della stessa cattura, compreso l’ultimo blocco parziale.

| Seed | Frozen/B TP / FP | Memoria fissata TP / FP | Memoria aggiornata TP / FP | ΔTP / ΔFP vs Frozen/B | ΔTP / ΔFP vs memoria fissata | Episodi favorevoli (blocchi) | Chiavi qualificanti | Quota della prima chiave nelle righe ammissibili | Quota della prima chiave nelle decisioni cambiate |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 42 | 662.716 / 38.848 | 676.492 / 40.965 | 678.411 / 31.918 | 15.695 / -6.930 | 1.919 / -9.047 | 3 (59–60, 281, 295) | 5 | 18.694/74.403 (25,13%) | 8.522/25.809 (33,02%) |
| 43 | 643.629 / 39.631 | 643.629 / 39.631 | 661.213 / 33.690 | 17.584 / -5.941 | 17.584 / -5.941 | 8 (21–25, 28–30, 33, 41–47, 53, 56–57, 59–60, 282–283) | 4 | 18.686/77.201 (24,20%) | 7.352/26.347 (27,90%) |
| 44 | 642.381 / 39.180 | 656.157 / 41.297 | 656.822 / 29.076 | 14.441 / -10.104 | 665 / -12.221 | 1 (59–60) | 1 | 18.721/75.800 (24,70%) | 11.593/27.523 (42,12%) |
| 45 | 634.565 / 38.680 | 648.341 / 40.797 | 646.648 / 35.826 | 12.083 / -2.854 | -1.693 / -4.971 | 1 (59) | 1 | 18.686/67.961 (27,50%) | 7.074/16.787 (42,14%) |
| 46 | 632.621 / 38.526 | 646.397 / 40.643 | 647.803 / 28.668 | 15.182 / -9.858 | 1.406 / -11.975 | 2 (59, 281) | 2 | 11.593/66.724 (17,37%) | 11.593/28.510 (40,66%) |

Nei totali il vantaggio rispetto a entrambi i controlli si osserva in 4 seed su cinque (42, 43, 44, 46).
Il criterio del pilot, che richiede anche due episodi distinti e due chiavi, è soddisfatto in 3 seed su cinque (42, 43, 46).
La chiave più frequente sull’intera finestra pesa 49.389/3.349.973 (1,47%) ed è la stessa per tutti i seed.

Gli episodi favorevoli corrispondono a 9 intervalli distinti nella cattura. Gli indici dei blocchi partono da zero. Gli intervalli sovrapposti o consecutivi tra seed sono contati una volta.
I cinque seed misurano la sensibilità al calendario delle richieste sulla stessa cattura di sviluppo.

SEEDS.csv riporta anche FN, TN, i recuperi e le perdite lorde, i digest e i denominatori delle chiavi.
EPISODES.csv conserva i contributi appaiati per intervallo e seed. UNSW resta al pilot seed 42.

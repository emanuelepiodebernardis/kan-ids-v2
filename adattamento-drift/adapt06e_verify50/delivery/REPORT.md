# ADAPT-06E — risultati VERIFY 50/100

Commit di esecuzione: `28a81d5de70642d6f76d31a79d865103e4eb5bd2`.

La matrice comprende TON e UNSW completi, seed 42–46. Il budget resta all'1% e gli ID richiesti coincidono, blocco per blocco, con ADAPT-06B. La sola differenza è l'assegnazione temporanea della testa RANDOM_FIT a VERIFY_EXTRA mentre un candidato è aperto. Tutti gli esiti sono inclusi.

## Decisione preregistrata

- **TON: NO-GO** — 0/5 seed soddisfano simultaneamente ΔTP>0 e ΔFP≤0 sia contro ADAPT-06B sia contro matched; seed: nessuno. Guardia ΔFPR≤1 pp contro ADAPT-06B: PASS.
- **UNSW: NO-GO** — 0/5 seed soddisfano simultaneamente ΔTP>0 e ΔFP≤0 sia contro ADAPT-06B sia contro matched; seed: nessuno. Guardia ΔFPR≤1 pp contro ADAPT-06B: PASS.

## Confronti per seed

| Stream | Seed | vs ADAPT-06B ΔTP / ΔFP | vs matched ΔTP / ΔFP | Promozioni |
|---|---:|---:|---:|---:|
| TON | 42 | +0 / -3 | +0 / +0 | 0 |
| TON | 43 | +0 / +0 | +0 / +0 | 0 |
| TON | 44 | +0 / +0 | +0 / +0 | 0 |
| TON | 45 | +0 / +0 | +0 / +0 | 0 |
| TON | 46 | +0 / +0 | +0 / +0 | 0 |
| UNSW | 42 | -3162 / -8347 | +0 / +0 | 0 |
| UNSW | 43 | +1320 / +1004 | +825 / -11890 | 1 |
| UNSW | 44 | +0 / +1282 | +0 / +0 | 0 |
| UNSW | 45 | +321 / +1077 | +321 / +936 | 1 |
| UNSW | 46 | +30 / +1464 | +30 / +26 | 1 |

Su TON non avviene alcuna promozione: la variante coincide con il controllo matched in tutti i seed e con ADAPT-06B in quattro seed; nel seed 42 cambia soltanto di −3 FP. Aumentare VERIFY non produce quindi un beneficio dei pesi.

Su UNSW avvengono tre promozioni. Nel seed 43, rispetto al matched, la promozione aggiunge 825 TP e riduce 11.890 FP, ma rispetto ad ADAPT-06B aggiunge 1.320 TP e 1.004 FP: il vincolo ΔFP≤0 non è soddisfatto. Nei seed 45 e 46 i piccoli guadagni di TP sono accompagnati da aumenti di FP contro entrambi i riferimenti. Nei seed 42 e 44 non vi è promozione e la variante coincide con matched.

## Riferimenti esterni al criterio primario

Contro Frozen/B, TON riduce gli FP ma perde TP in tutti i seed. Su UNSW la variante non riduce mai gli FP: nel seed 42 aggiunge 2.281 FP senza TP; negli altri seed aggiunge fra 2.057 e 4.392 FP per guadagni fra 0 e 827 TP. Contro la threshold-only storica, UNSW riduce gli FP in tutti i seed, ma nel seed 42 perde 684 TP; gli altri quattro seed guadagnano TP. Questi confronti restano diagnostici e non sostituiscono il criterio preregistrato.

## Decisioni, costo e casi estremi

Le promozioni sono 0 su TON e 3 su UNSW. I tempi per ciclo e gli esiti REJECT/WITHHOLD sono in DECISION_TIMING.csv e CYCLES.csv; le variazioni successive a ciascuna promozione sono in PROMOTIONS.csv. TRAJECTORY.csv ricostruisce FIT, CAL, candidato, soglia e decisioni dal primo cambiamento di ruolo.

I dieci replay richiedono 2418.8 s seriali; il picco RSS per processo varia da 236.7 a 505.3 MiB. Sono registrate 228,295 richieste uniche, di cui 18,360 VERIFY_EXTRA; 850 arrivano dopo la chiusura del ciclo e 60 restano pendenti alla fine dello stream. Nessuna di queste etichette entra in FIT o CAL.

Il caso descrittivamente migliore per variazione dell'errore totale contro ADAPT-06B è UNSW seed 42 (ΔTP=-3162, ΔFP=-8347); resta un compromesso perché perde TP. Il peggiore è UNSW seed 46 (ΔTP=+30, ΔFP=+1464). Questa graduatoria è descrittiva e non modifica il NO-GO.

## Conclusione limitata

Una quota VERIFY maggiore accelera o completa alcune decisioni, ma non soddisfa il criterio operativo in alcun seed. Il risultato chiude questa ipotesi nel perimetro preregistrato; non dimostra inefficacia generale del controllo prospettico né autorizza tuning o ADAPT-07. Le etichette complete sono state usate soltanto per valutare predizioni già emesse.

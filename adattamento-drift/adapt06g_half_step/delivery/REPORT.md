# ADAPT-06G — aggiornamento parziale dei pesi

Commit eseguito: `cb651dd2bc0bd0ffe70384f0ac9b00a46d7c0f5b`. Coefficiente preregistrato: `0,5`.

## Esito

**NO-GO** — Il mezzo passo preregistrato non soddisfa la regola per ADAPT-07.

- TON: 0/5 seed favorevoli contro sia ADAPT-06E sia il controllo matched; 0 intervalli locali nell'insieme dei seed e 0 episodi qualificanti nei seed favorevoli; criterio ADAPT-07 FAIL.
- UNSW: 0/5 seed favorevoli contro sia ADAPT-06E sia il controllo matched; 1 intervallo locale nell'insieme dei seed e 0 episodi qualificanti nei seed favorevoli; criterio ADAPT-07 FAIL.

## Dieci confronti

| Stream | Seed | TP | FP | FN | FPR | Prom. | ΔTP/ΔFP vs 06E | vs matched | vs Frozen/B | vs 06B | Esito |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| TON | 42 | 691071 | 14305 | 2549229 | 4.619% | 0 | +0/+0 | +0/+0 | -6779/-24561 | +0/-3 | FAIL |
| TON | 43 | 672017 | 14402 | 2568283 | 4.651% | 0 | +0/+0 | +0/+0 | -6781/-25247 | +0/+0 | FAIL |
| TON | 44 | 670764 | 15088 | 2569536 | 4.872% | 0 | +0/+0 | +0/+0 | -6752/-24110 | +0/+0 | FAIL |
| TON | 45 | 662923 | 11954 | 2577377 | 3.860% | 0 | +0/+0 | +0/+0 | -6775/-26744 | +0/+0 | FAIL |
| TON | 46 | 661013 | 11915 | 2579287 | 3.848% | 0 | +0/+0 | +0/+0 | -6773/-26629 | +0/+0 | FAIL |
| UNSW | 42 | 201381 | 19367 | 10901 | 2.410% | 1 | +926/+1557 | +926/+1007 | +926/+3838 | -2236/-6790 | FAIL |
| UNSW | 43 | 200817 | 19698 | 11465 | 2.451% | 0 | -826/-549 | +0/+0 | +1/+3843 | +494/+455 | FAIL |
| UNSW | 44 | 201270 | 19483 | 11012 | 2.424% | 1 | +426/-365 | +426/+889 | +426/+3780 | +426/+917 | FAIL |
| UNSW | 45 | 196644 | 18245 | 15638 | 2.270% | 2 | -4535/-345 | -4214/-17 | -4214/+2433 | -4214/+732 | FAIL |
| UNSW | 46 | 200930 | 18065 | 11352 | 2.248% | 1 | +498/+400 | +528/+426 | +529/+2457 | +528/+1864 | FAIL |

Su TON non si osservano promozioni: il mezzo passo coincide con ADAPT-06E e con il controllo matched in tutti i cinque seed. Su UNSW nessun seed soddisfa il criterio contro entrambi i riferimenti. Il solo intervallo locale favorevole è il blocco 89 del seed 46; il risultato complessivo di quel seed aumenta sia TP sia FP e quindi resta sfavorevole.

La tabella `RESULTS.csv` conserva integralmente entrambi gli stream e tutti i seed, inclusi esiti nulli o sfavorevoli. `TEMPORAL_BLOCKS.csv` localizza le differenze per blocco; `EPISODES.csv` elenca gli intervalli contigui, senza trattare lo stesso intervallo ripetuto su più seed come più episodi indipendenti.

Le traiettorie half-step e ADAPT-06E possono assegnare ruoli diversi dopo decisioni differenti. Le colonne dedicate rendono esplicite tali differenze; gli ID richiesti restano identici in tutti i blocchi, il budget resta invariato ed entrambi sono verificati nei report.

Non sono stati eseguiti nuovi fit iniziali, tuning, ACTIVE, altri modelli o nuovi dataset. La conclusione riguarda soltanto il coefficiente 0,5 preregistrato sui due stream di sviluppo.

# ADAPT-06 — matrice completa del gate prospettico

## Perimetro

- commit eseguito: `2b2ec4d02672be8432f9e462e7011359bf5c66d8`;
- stream completi TON e UNSW; seed 42–46;
- 60 report JSON e 100 traiettorie adattive;
- RANDOM/ACTIVE per GAM, LR e ARF FIT-only; ALWAYS/GATED per GAM e LR;
- stati A/B archiviati ricaricati in ogni processo; nessun riuso dello stato del pilot;
- protocollo, budget, ritardo, memorie e criteri invariati; nessun tuning.

## Esito primario preregistrato

| Dataset | Esito | Seed supportati | FPR_PASS | Δ recall medio GATED−ALWAYS |
|---|---:|---:|---:|---:|
| TON | NO-GO | 5/5 | 0/5 | 9,26 pp |
| UNSW | NO-GO | 5/5 | 2/5 | 24,38 pp |

**Esito complessivo: NO-GO**.

L'esito usa GAM ACTIVE, confusioni aggregate sull'intero stream e tutti i seed supportati, compresi quelli con FPR sfavorevole. GO richiede almeno quattro seed supportati, almeno quattro FPR_PASS e un aumento medio di recall di almeno 2 punti percentuali in entrambi i dataset.

## Confronti appaiati sui cinque seed

| Dataset | Famiglia | Acquisizione | Δ recall medio ± DS (pp) | Δ FPR medio ± DS (pp) | Recall migliore | |ΔFPR| ≤ 1 pp |
|---|---|---|---:|---:|---:|---:|
| TON | SPLINE | RANDOM | 3,67 ± 6,78 | 6,06 ± 2,05 | 5/5 | 0/5 |
| TON | SPLINE | ACTIVE | 9,26 ± 7,10 | 6,53 ± 1,90 | 5/5 | 0/5 |
| TON | LR | RANDOM | 8,31 ± 7,74 | 49,72 ± 1,80 | 5/5 | 0/5 |
| TON | LR | ACTIVE | 6,04 ± 6,46 | 50,39 ± 2,79 | 5/5 | 0/5 |
| UNSW | SPLINE | RANDOM | -0,50 ± 1,39 | -2,94 ± 3,08 | 2/5 | 2/5 |
| UNSW | SPLINE | ACTIVE | 24,38 ± 8,33 | -2,37 ± 2,69 | 5/5 | 2/5 |
| UNSW | LR | RANDOM | -0,79 ± 2,87 | 2,05 ± 8,11 | 1/5 | 2/5 |
| UNSW | LR | ACTIVE | 29,29 ± 19,96 | 1,44 ± 8,28 | 5/5 | 0/5 |

Le dispersioni sono deviazioni standard campionarie sulle cinque differenze appaiate. Un Δ positivo indica GATED maggiore di ALWAYS. La colonna |ΔFPR| ≤ 1 pp descrive soltanto la distanza; l'esito primario richiede anche FPR GATED non superiore al 6%.

## Inventario e controlli

- report presenti e completi: 60/60;
- traiettorie adattive: 100/100;
- provenienza input e ordine dei blocchi: PASS su 60/60 report;
- conservazione dei ruoli rispetto al budget: PASS su 60/60 report;
- promozioni GATED: 5;
- withholding per supporto insufficiente: 46;
- candidati ancora pendenti dopo la chiusura dello stream: 0.

Assenza di promozioni, insufficienze e risultati sfavorevoli sono conservati come esiti; non vengono rimossi come dati mancanti.

## Costi effettivi

- somma seriale dei wall time delle 60 celle: 39.771,128 s (11,048 h);
- wall time osservato con 2 worker: 19.937,904 s (5,538 h);
- somma dei tempi di replay: 38.709,057 s (10,753 h);
- picco RSS massimo per processo: 551,7 MiB;
- massimo payload online logico: 145.782 byte.

## Tabelle e figure

`PRIMARY_SEED_OUTCOMES.csv` e `AGGREGATE_DECISION.json` implementano la regola primaria. `PAIRED_SEED_COMPARISONS.csv` conserva i 40 confronti GATED−ALWAYS per GAM/LR e RANDOM/ACTIVE; `PAIRED_GROUP_SUMMARY.csv` ne riporta media, dispersione e supporto sui cinque seed. `ADAPTIVE_TRAJECTORIES.csv` contiene le 100 traiettorie; `CANDIDATE_DECISIONS.csv` conserva ogni freeze, look, promozione, rifiuto e withholding.

Le figure mostrano differenze primarie appaiate, livelli di recall/FPR, decisioni del gate e costi per cella. Sono descrittive e non modificano i criteri preregistrati.

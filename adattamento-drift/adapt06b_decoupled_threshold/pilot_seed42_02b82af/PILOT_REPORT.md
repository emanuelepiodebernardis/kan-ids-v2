# ADAPT-06B — pilot completo GAM/RANDOM, seed 42

## Perimetro e provenienza

I quattro replay sono stati eseguiti dal commit `02b82af36ae8151e4ac83d806f434704e4557abd` sugli stream completi TON e UNSW, partendo dagli stati A/B archiviati. Legacy e decoupled sono stati avviati in processi distinti con BLAS a un thread. Non sono stati eseguiti nuovi fit iniziali, tuning, ACTIVE, altri seed o ADAPT-07.

| Dataset | Blocchi | Righe | CONTROL | RANDOM_FIT | Label totali | Prima CAL eleggibile | Prima soglia applicata |
|---|---:|---:|---:|---:|---:|---:|---:|
| TON | 355 | 3,549,973 | 8,874 | 26,625 | 35,499 | 275 | 276 |
| UNSW | 102 | 1,016,018 | 2,540 | 7,620 | 10,160 | 17 | 18 |

## Confronto principale con threshold-only autonoma

| Dataset | Politica | Recall | FPR | Δ recall | Δ FPR | Promozioni pesi | Variazioni soglia | Soglia iniziale | Soglia finale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TON | threshold_only_autonomous | 21.327% | 4.615% | +0.000 pp | +0.000 pp | 0 | 2 | 7.163567 | 12.604135 |
| TON | gated_decoupled | 21.327% | 4.620% | +0.000 pp | +0.006 pp | 0 | 2 | 7.163567 | 12.604135 |
| UNSW | threshold_only_autonomous | 94.751% | 4.746% | +0.000 pp | +0.000 pp | 0 | 23 | -0.320670 | -0.320758 |
| UNSW | gated_decoupled | 95.918% | 3.254% | +1.167 pp | -1.491 pp | 1 | 3 | -0.320670 | -0.631922 |

## Decisioni e costi osservati

| Dataset | Decisioni decoupled | Replay legacy | Replay decoupled | RSS legacy | RSS decoupled | Payload legacy | Payload decoupled |
|---|---|---:|---:|---:|---:|---:|---:|
| TON | FREEZE=3, REJECT_HARM=2, WITHHOLD_INSUFFICIENT=1 | 120.176 s | 121.501 s | 523.4 MiB | 520.9 MiB | 145,654 B | 144,645 B |
| UNSW | FREEZE=3, ACCEPT=1, REJECT_HARM=1, WAIT=1, WITHHOLD_STREAM_END_AMBIGUOUS=1 | 48.383 s | 46.219 s | 255.4 MiB | 252.3 MiB | 102,490 B | 131,006 B |

## Sintesi descrittiva

- **TON:** non è stato promosso alcun candidato. GATED decoupled e threshold-only matched coincidono; rispetto alla threshold-only autonoma il recall è invariato e l'FPR differisce di +0.006 pp. Il forte calo dell'FPR rispetto a Frozen/B deriva quindi dalla manutenzione della soglia, non dall'adattamento dei pesi.
- **UNSW:** è stato promosso un candidato. Rispetto alla threshold-only autonoma GATED decoupled mostra +1.167 pp di recall e -1.491 pp di FPR. Rispetto al controllo threshold-only matched, il guadagno di recall è +1.490 pp con un aumento dell'FPR di +1.010 pp.
- Il pilot contiene un solo seed per dataset: il risultato UNSW è favorevole nel confronto primario, ma non dimostra stabilità né superiorità generale. TON costituisce un risultato nullo per il contributo dei pesi.
- Le soglie sono espresse nella scala degli score del relativo stato del modello. Dopo una promozione, la soglia operativa finale si riferisce quindi ai pesi promossi e non va interpretata come una variazione isolata dello stesso score.

Tutti i cicli risultano chiusi alla fine degli stream: non rimangono candidati o aggiornamenti di soglia pendenti.

I tempi riportati sono tempi di replay per singolo processo; caricamento degli input e del modello, scrittura e verifica dei file restano distinti nei JSON. I valori RSS sono indipendenti perché ogni replay è stato eseguito in un processo nuovo.

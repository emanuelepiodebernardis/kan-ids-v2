# ADAPT-06 — pilot prospettico seed 42, prefisso di 100 blocchi

## Perimetro congelato

- merge di partenza: `56cbfd9de62569bb1b72b63c0ec1640e4df2d318`;
- commit effettivamente eseguito: `ceaf0e14a2619bbc38189a7cee046abcb0647994`;
- stream: TON e UNSW; seed 42; primi 100 blocchi (`--max-blocks 100`);
- 12 report JSON e 20 traiettorie adattive: RANDOM/ACTIVE per GAM, LR e ARF FIT-only; ALWAYS/GATED per GAM e LR; riferimenti Frozen/B e threshold-only;
- inizializzazione A/B, budget, ritardo, memorie e margini del protocollo invariati; nessun tuning e nessun altro seed eseguito.

La verifica di provenienza legge l'intero contenitore D per ricalcolare schema, forme, dtype e SHA-256 degli otto array. La valutazione usa soltanto il prefisso autorizzato di 100 blocchi (1.000.000 di righe per report). I byte del contenitore NPZ ricostruito differiscono dall'archivio, ma gli otto array scientifici coincidono esattamente; entrambi gli hash del contenitore restano registrati nei JSON.

## Controlli e funzionamento

- inventario e identità: PASS su 12/12 report; 20/20 traiettorie adattive presenti; commit, comando, ordine 0–99 e provenienza verificati;
- contabilità per report: 10.000 richieste, 9.900 label arrivate nel prefisso e 100 ancora pendenti per il ritardo di un blocco;
- promozioni GATED: 0 (nessuna); rifiuti per danno: 12;
- supporto FIT/CAL insufficiente a fine prefisso in 4 report GAM/LR; candidati ancora pendenti in 4 report;
- ARF è il riferimento FIT-only: non usa CAL né il gate prospettico e applica 99 batch arrivati per report.

Su TON, GAM e LR non congelano candidati entro il prefisso perché la memoria CAL, di capacità 256, non raggiunge il minimo di 100 normali e 100 attacchi. Su UNSW, GAM e LR congelano candidati; i look conclusi sono rifiutati dal vincolo di danno e l'ultimo candidato resta `PENDING_AT_PILOT_END`. Il prefisso non forza una decisione terminale e non simula la fine dello stream.

## Costi misurati

- somma seriale dei wall time delle 12 celle: 3462.052 s (57.701 min);
- somma dei tempi di replay registrati: 3229.378 s (53.823 min);
- picco RSS massimo osservato: 551.3 MiB;
- massimo payload online logico contabilizzato: 101974 byte; esclude overhead dei contenitori/interprete e array/cache di input come dichiarato nei JSON.

`REPORT_SUMMARY.csv` separa wall time, replay, predizione e fit/calibrazione per cella. `TRACK_METRICS.csv` riporta anche Frozen/B e threshold-only; `ADAPTIVE_TRAJECTORIES.csv` contiene le 20 traiettorie adattive; `CANDIDATE_DECISIONS.csv` conserva i freeze, i look, i rifiuti e gli stati pendenti.

## Interpretazione limitata al pilot

Il pilot verifica il funzionamento e misura i costi; non autorizza una conclusione sui cinque seed. L'assenza di promozioni GATED nel prefisso è un esito osservato, non una prova di inefficacia generale. La matrice completa resta subordinata a una decisione successiva.

# ADAPT-06 — pilot prospettico seed 42, prefisso di 100 blocchi

## Perimetro congelato

- merge di partenza: `56cbfd9de62569bb1b72b63c0ec1640e4df2d318`;
- commit effettivamente eseguito: `ceaf0e14a2619bbc38189a7cee046abcb0647994`;
- stream: TON e UNSW; seed 42; primi 100 blocchi (`--max-blocks 100`);
- 12 report JSON e 20 traiettorie adattive: RANDOM/ACTIVE per GAM, LR e ARF FIT-only; ALWAYS/GATED per GAM e LR; riferimenti Frozen/B e threshold-only;
- inizializzazione A/B, budget, ritardo, memorie e margini del protocollo invariati; nessun tuning e nessun altro seed eseguito.

La verifica di provenienza legge l'intero contenitore D per ricalcolare schema, forme, dtype e SHA-256 degli otto array. La valutazione usa soltanto il prefisso autorizzato di 100 blocchi (1.000.000 di righe per report). I byte del contenitore NPZ ricostruito differiscono dall'archivio, ma gli otto array scientifici coincidono esattamente; entrambi gli hash del contenitore restano registrati nei JSON.

## Controlli e funzionamento

- suite: 48 test nel preflight eseguito prima del pilot; 50 test nella verifica finale, inclusi i due nuovi controlli su ruoli e interpretazione;
- inventario e identità: PASS su 12/12 report; 20/20 traiettorie adattive presenti; commit, comando, ordine 0–99 e provenienza verificati;
- contabilità per report: 10.000 richieste, 9.900 label arrivate nel prefisso e 100 ancora pendenti per il ritardo di un blocco;
- ruoli ricostruiti dai `query_audit`: ogni RANDOM contiene 2.500 CONTROL e 7.500 RANDOM_FIT, senza richieste ACTIVE; ogni ACTIVE contiene 2.500 CONTROL, 1.500 ACTIVE_EXPLORE e 6.000 ACTIVE_EXPLOIT, senza RANDOM_FIT;
- conservazione dei ruoli: PASS su 12/12 report; la somma dei ruoli coincide con le 10.000 richieste in entrambe le politiche;
- promozioni GATED: 0 (nessuna); rifiuti per danno: 12;
- supporto FIT/CAL insufficiente a fine prefisso in 4 report GAM/LR; candidati ancora pendenti in 4 report;
- ARF è il riferimento FIT-only: non usa CAL né il gate prospettico e applica 99 batch arrivati per report.

Su TON la memoria CAL, di capacità 256, termina con 19 normali e 237 attacchi e non raggiunge il minimo 100/100. Nessun candidato viene quindi congelato: per GAM e LR, GATED coincide esattamente con ALWAYS e Frozen/B.

Su UNSW il supporto CAL raggiunge 128/128. Il gate rifiuta gli aggiornamenti peggiorativi e non promuove candidati; GATED resta quindi identico a Frozen/B. L'ultimo candidato di ciascuna cella GAM/LR resta `PENDING_AT_PILOT_END`: il prefisso non forza una decisione terminale e non simula la fine dello stream.

Per GAM ACTIVE su UNSW, GATED/Frozen-B conserva un recall superiore ad ALWAYS di 15,52 punti percentuali, ma la distanza di FPR è 6,51 punti percentuali (ALWAYS più alto). La differenza di FPR supera il margine di comparabilità del prefisso: il vantaggio di recall non soddisfa quindi il criterio primario per un confronto a FPR comparabile.

ARF FIT-only su UNSW raggiunge recall elevato ma anche FPR molto alto: 98,00% con RANDOM e 95,03% con ACTIVE. Come preregistrato, gli aggiornamenti FIT conservano la soglia B; il runner verifica che la soglia adattiva usata in ogni blocco coincida con quella Frozen/B.

## Costi misurati

- somma seriale dei wall time delle 12 celle: 3.462,052 s (57,701 min);
- somma dei tempi di replay registrati: 3.229,378 s (53,823 min);
- picco RSS massimo osservato: 551,3 MiB;
- massimo payload online logico contabilizzato: 101.974 byte; esclude overhead dei contenitori/interprete e array/cache di input come dichiarato nei JSON.

`REPORT_SUMMARY.csv` separa wall time, replay, predizione e fit/calibrazione per cella. `TRACK_METRICS.csv` riporta anche Frozen/B e threshold-only; `ADAPTIVE_TRAJECTORIES.csv` contiene le 20 traiettorie adattive; `CANDIDATE_DECISIONS.csv` conserva i freeze, i look, i rifiuti e gli stati pendenti.

## Interpretazione limitata al pilot

Il pilot verifica il funzionamento e misura i costi; non autorizza da solo una conclusione sui cinque seed. L'assenza di promozioni GATED nel prefisso è un esito osservato, non una prova di inefficacia generale. La matrice completa, autorizzata separatamente dopo questa correzione, dovrà partire dagli stati A/B archiviati in una directory nuova e non riutilizzare lo stato finale del pilot.

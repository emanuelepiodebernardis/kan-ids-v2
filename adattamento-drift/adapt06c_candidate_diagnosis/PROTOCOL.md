# Protocollo di lettura — ADAPT-06C

## Perimetro

La diagnosi è retrospettiva sui due stream di sviluppo TON e UNSW, GAM/RANDOM,
seed 42–46. Non è una conferma indipendente e non modifica soglie, margini,
budget o stati. `WAIT` è un passaggio interno e non un candidato aggiuntivo.

## Unità e confronto

La chiave del ciclo è `dataset/seed/freeze_block/candidate_digest`. Il ciclo si
apre con `FREEZE` e termina con `ACCEPT`, `REJECT_HARM` o `WITHHOLD*`. Il
confronto usa l'incumbent operativo congelato all'apertura, sulle stesse
osservazioni e sullo stesso intervallo VERIFY del candidato.

Per ogni look completo si ricalcolano sui conteggi interi:

- guadagno recall: `(TP_candidate - TP_incumbent) / attacks ≥ 0,02`;
- delta FPR: `(FP_candidate - FP_incumbent) / normals ≤ 0,01`;
- FPR candidato: `FP_candidate / normals ≤ 0,06`;
- danno recall: `(TP_incumbent - TP_candidate) / attacks ≥ 0,02`.

Tutte le condizioni simultanee sono conservate. `ACCEPT` richiede i primi tre
vincoli; `REJECT_HARM` scatta per superamento di almeno un limite FPR o per
danno recall. In assenza di entrambi si registra `WAIT`, oppure
`WITHHOLD_AMBIGUOUS` alla frontiera terminale prevista.

## Supporto incompleto

Una chiusura senza look completo conserva supporti e confusioni come dato
diagnostico, ma lascia vuoti delta e condizioni decisionali. Anche un ultimo
prefisso parziale dopo uno o più look è marcato `terminal_partial_diagnostic`:
non viene trasformato retrospettivamente in una decisione.

L'attesa fino alla prima CAL eleggibile è distinta da VERIFY. Su TON la prima
calibrazione viene proposta a fine blocco 274–275 e si applica dal blocco
275–276; la durata di un ciclo parte solo dal relativo `FREEZE`.

## Interpretazione

Le categorie sono: accettato, dannoso secondo i margini fissati, ambiguo e
supporto insufficiente. I quadranti TP/FP descrivono compromessi empirici ma
non una prestazione controfattuale sull'intero stream. Se i log non separano
qualità del FIT, composizione VERIFY e drift, la nota dichiara il limite senza
attribuzioni causali.

# ADAPT-06 — matrice completa

Questa directory conserva i riepiloghi leggeri della matrice autorizzata dopo
la verifica del pilot. L'esecuzione parte dal commit congelato
`2b2ec4d02672be8432f9e462e7011359bf5c66d8`, ricarica per ogni cella gli stati
A/B archiviati e non riutilizza lo stato finale del pilot.

Il disegno comprende gli stream completi TON e UNSW, i seed 42--46,
RANDOM/ACTIVE per GAM, LR e ARF FIT-only, ALWAYS/GATED per GAM e LR e i
riferimenti previsti dal protocollo. I 60 report JSON corrispondono a 100
traiettorie adattive.

## Esito preregistrato

Il confronto primario GAM ACTIVE è **NO-GO** in entrambi i dataset. Il recall
medio GATED−ALWAYS aumenta di 9,26 punti percentuali su TON e di 24,38 punti
su UNSW, ma il criterio FPR_PASS è rispettato in 0/5 seed TON e 2/5 seed UNSW.
Tutti i seed, comprese le repliche con FPR sfavorevole, entrano nell'esito.
Il risultato non dimostra una superiorità generale del gate.

Sono registrate cinque promozioni GATED nell'intera matrice, 46 withholding per
supporto insufficiente e nessun candidato ancora pendente dopo la fine reale
dello stream. Le assenze di promozione e gli esiti sfavorevoli sono conservati,
non esclusi come dati mancanti.

## Materiali versionati

- `FULL_MATRIX_REPORT.md`: rapporto interpretativo e costi effettivi;
- `AGGREGATE_DECISION.json`: applicazione della regola GO/NO-GO/INCONCLUSIVE;
- `PRIMARY_SEED_OUTCOMES.csv`: confronto primario per dataset e seed;
- `PAIRED_GROUP_SUMMARY.csv`: media e dispersione delle differenze appaiate;
- `DECISION_COUNTS.csv`: decisioni, insufficienze e casi senza promozione;
- `REPORT_SUMMARY.csv`: inventario sintetico dei 60 report;
- `figures/`: quattro figure descrittive riproducibili.

I JSON completi, le 100 traiettorie, i confronti per seed, i log, i comandi,
l'ambiente e il codice di riproduzione sono conservati nello ZIP esterno
descritto in `ARCHIVE.md`. Nessun nuovo fit iniziale o tuning è stato eseguito.

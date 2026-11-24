# ADAPT-06E — VERIFY 50/100

Questa cartella implementa il confronto preregistrato fra ADAPT-06B e una
variante che, soltanto mentre un candidato è aperto, assegna a VERIFY metà
delle richieste di etichetta. Il budget totale, gli ID richiesti e tutti gli
altri parametri restano invariati.

## Provenienza ed esecuzione

- base `main`: `fe18b0f5a83d51076c07a621430b69225c026ccd`;
- protocollo e codice fissati prima dei risultati:
  `1530a81b87881c7659371551c9aff346067afb51`;
- commit effettivamente eseguito: `28a81d5de70642d6f76d31a79d865103e4eb5bd2`;
- riferimento ADAPT-06B riprodotto su TON e UNSW, seed 42: PASS;
- matrice: TON e UNSW completi, seed 42--46, dieci report.

La stima registrata prima dei risultati era di 15 minuti per i primi due
report. Il primo report TON/seed42 ha richiesto 424,269 s e il primo report
UNSW/seed42 114,190 s: 538,459 s seriali complessivi, circa 9,0 minuti. I dieci
replay hanno richiesto 2.418,8 s seriali, circa 40,3 minuti, esclusi analisi e
confezionamento. Il picco RSS per processo è compreso fra 236,7 e 505,3 MiB.

## Esito

La regola preregistrata produce **NO-GO su entrambi gli stream**. Nessun seed
soddisfa contemporaneamente l'aumento dei TP senza aumento degli FP sia contro
ADAPT-06B sia contro il controllo matched a pesi fissi. La guardia di un punto
percentuale sull'FPR rispetto ad ADAPT-06B passa in tutti i seed.

Su TON non avviene alcuna promozione. Su UNSW avvengono tre promozioni, ma
nessuna determina il beneficio simultaneo richiesto rispetto a entrambi i
controlli primari. Tutti gli esiti nulli e sfavorevoli sono conservati.

Il riesame multipiattaforma localizza una differenza circoscritta in
UNSW/seed45. FIT e CAL del candidato congelato al blocco 41 hanno gli stessi
digest su Windows e Linux; la soglia applicata dal blocco 64 è invece
`-0.10554843372708511` su Windows e `-0.09784439655829225` su Linux. La
confusione complessiva passa da 201.179 TP / 18.590 FP a 201.178 TP / 18.631
FP. Il vettore dei pesi Linux non è riportato nella scheda: i nove pesi
Windows sono conservati senza attribuire la divergenza al fit o alla
calibrazione.

## Materiali

- `PROTOCOL.md` e `config.json`: disegno e parametri congelati;
- `engine.py` e `runner.py`: motore isolato e ingresso della variante;
- `preflight.py` e `verify_reference.py`: controlli prima dei risultati;
- `run_matrix.py`: launcher dei dieci replay completi;
- `verify_report.py`: validatore autonomo dei report;
- `analyze.py`: confronti appaiati, costi e decisione preregistrata;
- `delivery/`: CSV, rapporto e manifest leggeri destinati a Git;
- `publish_delivery.py`: esportazione verificata dei prodotti leggeri;
- `package_archive.py`: archivio deterministico dei report completi;
- `audit_unsw45.py`: estrazione verificata del candidato UNSW/45;
- `UNSW45_WINDOWS_AUDIT.json`, `UNSW45_FREEZE41_IDS.csv` e
  `UNSW45_PLATFORM_NOTE.md`: riscontro multipiattaforma dai log originali;
- `ARCHIVE.md`: collegamento, dimensione, SHA-256 e commit del pacchetto
  completo;
- `test_verify50.py`, `test_delivery.py`, `test_package.py` e
  `test_platform_audit.py`: controlli sintetici, negativi, di confezionamento
  e del riscontro multipiattaforma;
- `COMMANDS.md`: comandi PowerShell effettivi.

I dieci JSON completi, i due riferimenti freschi e i prodotti di analisi sono
conservati nell'archivio esterno. Nessun nuovo fit iniziale, tuning, ACTIVE o
ADAPT-07 appartiene a questa consegna.


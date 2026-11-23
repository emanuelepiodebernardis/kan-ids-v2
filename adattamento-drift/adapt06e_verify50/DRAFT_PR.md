# adapt-06e: destinare 50 richieste su 100 a VERIFY durante i cicli aperti

## Obiettivo

ADAPT-06E verifica se, a budget invariato, una quota VERIFY temporaneamente
più ampia renda le decisioni sui candidati più tempestive e produca un
beneficio operativo successivo. La variante conserva GAM/RANDOM, FIT192,
CAL256, budget dell'1%, ritardo, margini, scadenza e ID richiesti di ADAPT-06B.

## Provenienza

- base `main`: `fe18b0f5a83d51076c07a621430b69225c026ccd`;
- protocollo e prima implementazione:
  `1530a81b87881c7659371551c9aff346067afb51`;
- commit effettivamente eseguito:
  `28a81d5de70642d6f76d31a79d865103e4eb5bd2`;
- stream: TON e UNSW completi; seed 42--46;
- nessun nuovo fit iniziale, tuning, ACTIVE o ADAPT-07.

Il preflight verifica input, stati A/B, configurazione e provenienza. Prima
della variante sono stati rieseguiti i riferimenti ADAPT-06B TON/UNSW seed42:
ID richiesti, decisioni e confusioni operative coincidono con gli artefatti
archiviati.

## Controlli causali

I test coprono conservazione del budget e degli ID, selezione prima delle
etichette, ritardo di un blocco, separazione FIT/CAL/VERIFY, attribuzione a un
solo ciclo, arrivi tardivi, fine stream, frontiera operativa delle promozioni,
regressione ADAPT-06B e controprove negative sul validatore. Il motore della
variante è isolato: il percorso storico non è stato modificato.

## Risultati

La matrice comprende dieci replay completi e produce **NO-GO su TON e UNSW**.
La guardia `delta FPR <= 1 pp` rispetto ad ADAPT-06B passa in tutti i seed, ma
nessun seed soddisfa simultaneamente `delta TP > 0` e `delta FP <= 0` sia
contro ADAPT-06B sia contro il controllo matched a pesi fissi.

| Stream | Seed | vs ADAPT-06B delta TP / delta FP | vs matched delta TP / delta FP | Promozioni |
|---|---:|---:|---:|---:|
| TON | 42 | 0 / -3 | 0 / 0 | 0 |
| TON | 43 | 0 / 0 | 0 / 0 | 0 |
| TON | 44 | 0 / 0 | 0 / 0 | 0 |
| TON | 45 | 0 / 0 | 0 / 0 | 0 |
| TON | 46 | 0 / 0 | 0 / 0 | 0 |
| UNSW | 42 | -3162 / -8347 | 0 / 0 | 0 |
| UNSW | 43 | +1320 / +1004 | +825 / -11890 | 1 |
| UNSW | 44 | 0 / +1282 | 0 / 0 | 0 |
| UNSW | 45 | +321 / +1077 | +321 / +936 | 1 |
| UNSW | 46 | +30 / +1464 | +30 / +26 | 1 |

Su TON non avviene alcuna promozione. Su UNSW le tre promozioni non producono
un beneficio stabile rispetto a entrambi i controlli primari. Contro Frozen/B
la variante perde TP su TON e non riduce mai gli FP su UNSW. La threshold-only
storica conserva un confronto diagnostico distinto; nessuno di questi
confronti sostituisce la regola preregistrata.

## Contabilità e costi

I dieci replay registrano 228.295 richieste uniche, di cui 18.360
VERIFY_EXTRA; 850 etichette aggiuntive arrivano dopo la chiusura del ciclo e
60 restano pendenti alla fine dello stream. I replay richiedono 2.418,8 s
seriali. La stima preregistrata per i primi due risultati era di 15 minuti;
TON/seed42 e UNSW/seed42 hanno richiesto 538,459 s complessivi (circa 9,0
minuti). Il picco RSS per processo varia da 236,7 a 505,3 MiB.

In Git entrano rapporto, CSV e manifest leggeri. L'archivio esterno conserva
preflight, due riferimenti freschi, dieci JSON completi, analisi e manifest con
SHA-256. Collegamento e SHA-256 vengono aggiunti dopo il caricamento del file
verificato.


# Comandi riproducibili

Eseguire da PowerShell nella radice del clone. Impostare `$adaptPython` e
`$adaptData` con i propri percorsi. `$adaptData` contiene le sottocartelle
`ton/` e `unsw/`. Per un nuovo replay scegliere un `$adaptRun` separato.

```powershell
$adaptPython = 'C:\percorso\ambiente\Scripts\python.exe'
$adaptData = 'C:\percorso\dati-verificati'
$adaptRun = 'adattamento-drift/archivio_esterno/adapt06g_half_step/nuovo-run'
$adaptSaved = 'adattamento-drift/archivio_esterno/adapt06g_half_step/cb651dd'
& $adaptPython -m pytest adattamento-drift/adapt06e_verify50 adattamento-drift/adapt06g_half_step -q
ruff check adattamento-drift/adapt06e_verify50/engine.py adattamento-drift/adapt06g_half_step
```

Il preflight verifica configurazione, A/B/D, i dieci modelli iniziali e i dieci
report ADAPT-06E originali prima dei nuovi replay:

```powershell
& $adaptPython adattamento-drift/adapt06g_half_step/preflight.py `
  --data-root $adaptData `
  --model-root adattamento-drift/review_extension/models `
  --reference-root adattamento-drift/archivio_esterno/adapt06e_verify50/28a81d5/reports `
  --out "$adaptRun/PREFLIGHT.json"
```

La matrice usa un processo nuovo per ciascun replay, due processi concorrenti e
un solo thread BLAS/OpenMP per processo. Il launcher rifiuta una directory già
esistente e un commit o una configurazione diversi dal preflight.

```powershell
& $adaptPython adattamento-drift/adapt06g_half_step/run_matrix.py `
  --data-root $adaptData `
  --model-root adattamento-drift/review_extension/models `
  --preflight "$adaptRun/PREFLIGHT.json" `
  --out "$adaptRun/reports" `
  --workers 2
```

L'analisi ricalcola i confronti dai report verificati:

```powershell
& $adaptPython adattamento-drift/adapt06g_half_step/analyze.py `
  --half-step-root "$adaptSaved/reports" `
  --adapt06e-root adattamento-drift/archivio_esterno/adapt06e_verify50/28a81d5/reports `
  --adapt06b-root adattamento-drift/adapt06b_decoupled_threshold `
  --out adattamento-drift/adapt06g_half_step/delivery
```

## Esecuzione consegnata

- commit del protocollo: `66228d22cbaa29f84902cd69730294531f24ce0e`;
- commit eseguito: `cb651dd2bc0bd0ffe70384f0ac9b00a46d7c0f5b`;
- directory dei report dalla radice del clone: `adattamento-drift/archivio_esterno/adapt06g_half_step/cb651dd/reports`;
- report prodotti: 10;
- processi concorrenti: 2, con un thread per processo;
- wall time della matrice: 1.191,062 secondi.

Il comando esatto di ogni replay, incluso il percorso locale dell'input, è
registrato in `EXECUTION_LOG.json`; ogni JSON conserva inoltre il proprio
comando, commit, ambiente e hash degli input e del modello.

Per rieseguire la matrice con l'identità storica, preparare preflight e replay
da un checkout separato del commit eseguito. Per rigenerare la consegna usare
lo script di analisi del commit di consegna con i report conservati.

Lo ZIP `adapt06g_half_step_cb651dd.zip` contiene la directory `cb651dd/`,
con `PREFLIGHT.json`, dieci JSON, log e manifest del run. Estrarre la directory
sotto `adattamento-drift/archivio_esterno/adapt06g_half_step/`.
Dimensione 15.014.360 byte, SHA-256
`4606bdd9f1a8eab3333d8ac014a60c9edbb5a2205defaeea2f9f2e929cfaa7d9`.

Il riferimento ADAPT-06E corrisponde allo ZIP `adapt06e_verify50_full_6ee5eb3.zip`,
SHA-256 `91c2cf5c3fae62b1b6009714ed4a33560e5077e8cad647d87ad1c1cb9fb30eef`.
Usare il contenuto di `evidence/reports/`, compresa la versione Windows di
UNSW/45 (201.179 TP e 18.590 FP). I dodici file di quella directory sono stati
verificati contro lo ZIP.

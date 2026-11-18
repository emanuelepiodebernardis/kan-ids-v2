# Comandi preregistrati — ADAPT-06B-MATRIX

I comandi si eseguono dalla radice del clone, su Windows, dal commit di
esecuzione che contiene protocollo, configurazione, launcher e test. Le
directory `<matrix-root>` e `<matrix-root>/new_replays` non devono esistere.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

py -3.13 -m pytest adattamento-drift/adapt06_prospective_gate `
  adattamento-drift/adapt06b_decoupled_threshold -q

py -3.13 adattamento-drift/adapt06b_decoupled_threshold/preflight_matrix.py `
  --data-root <data-root> `
  --model-root adattamento-drift/review_extension/models `
  --out <matrix-root>/PREFLIGHT.json

py -3.13 adattamento-drift/adapt06b_decoupled_threshold/run_matrix.py `
  --data-root <data-root> `
  --model-root adattamento-drift/review_extension/models `
  --preflight <matrix-root>/PREFLIGHT.json `
  --out <matrix-root>/new_replays

py -3.13 adattamento-drift/adapt06b_decoupled_threshold/analyze_matrix.py `
  --matrix-root <matrix-root>/new_replays `
  --out <matrix-root>/analysis
```

Il preflight deve precedere i replay e deve riportare lo stesso HEAD usato da
`run_matrix.py`. Il launcher esegue serialmente le otto coppie nuove; dentro
ogni coppia legacy e decoupled sono processi distinti. Il pilot seed 42 viene
soltanto letto dall'analisi e non è mai una destinazione di scrittura.

## Stima pre-run

Dai tempi osservati nel pilot, la proiezione seriale dei soli replay è circa
22,4 minuti: 4 × (120,176 + 121,501) secondi per TON e 4 × (48,383 + 46,219)
secondi per UNSW. Con preflight, caricamenti, scrittura, verifiche e analisi, la
stima operativa è 25–30 minuti sullo stesso dispositivo. I tempi effettivi
saranno riportati separatamente e non sostituiranno questa previsione.

# Comandi PowerShell

I comandi si eseguono dalla radice del clone. I percorsi locali sono variabili
esplicite e non vengono registrati nei prodotti destinati a Git.

```powershell
$Python = '<ambiente-python>\Scripts\python.exe'
$DataRoot = '<archivio-dati>\adapt05\data'
$ModelRoot = 'adattamento-drift\review_extension\models'
$RunRoot = 'adattamento-drift\archivio_esterno\adapt06e_verify50\28a81d5'

$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

& $Python -m pytest adattamento-drift\adapt06e_verify50\test_verify50.py -q

& $Python adattamento-drift\adapt06e_verify50\preflight.py `
  --data-root $DataRoot --model-root $ModelRoot `
  --out "$RunRoot\PREFLIGHT.json"

& $Python adattamento-drift\adapt06e_verify50\verify_reference.py `
  --data-root $DataRoot --model-root $ModelRoot `
  --preflight "$RunRoot\PREFLIGHT.json" `
  --out "$RunRoot\reference_seed42"

& $Python adattamento-drift\adapt06e_verify50\run_matrix.py `
  --data-root $DataRoot --model-root $ModelRoot `
  --preflight "$RunRoot\PREFLIGHT.json" `
  --reference-verification "$RunRoot\reference_seed42\REFERENCE_VERIFICATION.json" `
  --out "$RunRoot\reports"

& $Python adattamento-drift\adapt06e_verify50\analyze.py `
  --variant-root "$RunRoot\reports" `
  --out "$RunRoot\analysis_v2"

& $Python adattamento-drift\adapt06e_verify50\publish_delivery.py `
  --analysis "$RunRoot\analysis_v2" `
  --preflight "$RunRoot\PREFLIGHT.json" `
  --reference-verification "$RunRoot\reference_seed42_rerun\REFERENCE_VERIFICATION.json" `
  --run-manifest "$RunRoot\reports\RUN_MANIFEST.json" `
  --out adattamento-drift\adapt06e_verify50\delivery

& $Python adattamento-drift\adapt06e_verify50\package_archive.py `
  --preflight "$RunRoot\PREFLIGHT.json" `
  --reference-dir "$RunRoot\reference_seed42_rerun" `
  --reports "$RunRoot\reports" `
  --analysis "$RunRoot\analysis_v2" `
  --delivery adattamento-drift\adapt06e_verify50\delivery `
  --execution-commit 28a81d5de70642d6f76d31a79d865103e4eb5bd2 `
  --out "$RunRoot\adapt06e_verify50_full.zip"
```

`run_matrix.py` verifica che HEAD coincida con il commit registrato dal
preflight; se codice o configurazione cambiano, l'esecuzione viene rifiutata.
`package_archive.py` rifiuta la sovrascrittura, genera un manifest interno e
riapre lo ZIP per verificare elenco, dimensioni, CRC e SHA-256 di ogni file.


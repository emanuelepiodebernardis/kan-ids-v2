# Riproduzione della campagna ADAPT-05

Eseguire da `<repo>` con Python 3.12.14, le dipendenze di
`adattamento-drift/review_extension/requirements.txt` e BLAS a un thread.
Protocollo, matrice e sali sono in questa directory. La directory di output
deve essere nuova o vuota: i JSON di campagne precedenti non vengono
riutilizzati.

## Dati

```powershell
python adattamento-drift/review_extension/fetch_data.py ton --out <download>
python adattamento-drift/review_extension/fetch_data.py unsw --out <download>
python adattamento-drift/review_extension/prepare_data.py ton `
  --raw <download>/ton_raw --out <data>/ton
python adattamento-drift/review_extension/prepare_data.py unsw `
  --raw <download>/unsw_raw --out <data>/unsw
```

## Test, matrice e analisi

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONHASHSEED='0'
python -m pytest -q `
  adattamento-drift/adapt05_occurrence/test_runner.py `
  adattamento-drift/adapt05_occurrence/test_campaign_analysis.py `
  adattamento-drift/review_extension/test_experiment.py `
  adattamento-drift/review_extension/review2/test_runner.py `
  adattamento-drift/review_extension/review3/test_matched_memory.py `
  adattamento-drift/review_extension/calibration_audit/test_calibration_audit.py `
  adattamento-drift/ton_temporal/test_calibrazione_recente.py `
  adattamento-drift/ton_temporal/test_soglia.py `
  adattamento-drift/ton_temporal/test_replay_validazione_finale.py
python adattamento-drift/adapt05_occurrence/campaign.py `
  --data-root <data> --out <campaign> --workers 2
python adattamento-drift/adapt05_occurrence/analyze.py `
  --reports <campaign>/reports --data-root <data> `
  --out <campaign>/analysis
Copy-Item adattamento-drift/adapt05_occurrence/REPRODUCTION.md `
  <campaign>/REPRODUCTION.md
python adattamento-drift/adapt05_occurrence/package_campaign.py `
  --root <campaign> --out <archive>/adapt05_occurrence_full.zip
```

Il workflow `.github/workflows/adapt05-smoke-linux.yml`, avviato con modalità
`campaign`, esegue gli stessi passi.
I cinque seed sono randomizzazioni condizionate ai due stream e non cinque
catture indipendenti. Nessun risultato viene escluso in base al segno o alla
significatività dell'effetto.

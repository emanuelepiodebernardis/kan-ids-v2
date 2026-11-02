# ADAPT-05 — occurrence-wise FIT/CAL roles

Questa directory è separata e additiva: non modifica runner, modelli, report o
pacchetti DTRAP. Leggere `PROTOCOL.md` prima di eseguire codice.

## Test sintetici

Da `<repo>`:

```powershell
python -m pytest -q adattamento-drift/adapt05_occurrence/test_runner.py
```

I test coprono compatibilità vettoriale, codifica versionata, ripetizioni,
disgiunzione per `observation_id`, indipendenza dalle label, delay, quote,
espulsione, età, classi mancanti, determinismo/restart, query comuni e ultimo
blocco parziale.

## Smoke preregistrato

I dati ricostruiti non sono conservati in Git. Con BLAS a un thread:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python adattamento-drift/adapt05_occurrence/runner.py `
  --data <data-root>/ton --dataset ton --seed 42 --salt default `
  --max-blocks 30 --out <out>/ton_seed42_default.json
python adattamento-drift/adapt05_occurrence/runner.py `
  --data <data-root>/unsw --dataset unsw --seed 42 --salt default `
  --max-blocks 30 --out <out>/unsw_seed42_default.json
```

Il runner fallisce se query, stato head, soglie o aggiornamenti del controllo
vettoriale divergono dai rendiconti archiviati del Paper 2.

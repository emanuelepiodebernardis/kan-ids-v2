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
blocco parziale. Verificano inoltre la provenienza per-array degli input e un
replay sintetico end-to-end con le quattro tracce appaiate.

## Smoke preregistrato

I dati ricostruiti non sono conservati in Git. La verifica di compatibilità
numerica usa l'ambiente archiviato (Python 3.12.14 e dipendenze di
`review_extension/requirements.txt`) con BLAS a un thread. Il workflow Linux
registra anche `OPENBLAS_CORETYPE=Zen2`: il kernel e' fissato prima del replay,
per evitare che la scelta automatica basata sul processore cambi il fit:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_CORETYPE='Zen2'
python adattamento-drift/adapt05_occurrence/runner.py `
  --data <data-root>/ton --dataset ton --seed 42 --salt default `
  --max-blocks 30 --out <out>/ton_seed42_default.json
python adattamento-drift/adapt05_occurrence/runner.py `
  --data <data-root>/unsw --dataset unsw --seed 42 --salt default `
  --max-blocks 30 --out <out>/unsw_seed42_default.json
```

Il runner fallisce se query, stato head, soglie o aggiornamenti del controllo
vettoriale divergono dai rendiconti archiviati del Paper 2.

I due rendiconti consegnati, gli hash, l'esito della compatibilità e la stima
dei tempi della matrice completa sono in [`smoke/`](smoke/README.md).

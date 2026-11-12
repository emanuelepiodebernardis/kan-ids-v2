# Comandi preparati per il pilot ADAPT-06B

Questi comandi sono **pianificati e non eseguiti al checkpoint A**. Devono
essere lanciati soltanto dopo il riesame del protocollo e del codice, dalla
radice del clone e dal commit congelato della futura autorizzazione.

## Controlli prima del pilot

```powershell
git status --short
git rev-parse HEAD
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
py -3.13 -m pytest adattamento-drift/adapt06_prospective_gate `
  adattamento-drift/adapt06b_decoupled_threshold -q
```

Le directory di uscita non devono esistere. I due comandi usano gli stream
completi: non è previsto `--max-blocks`.

## TON, GAM/RANDOM, seed 42

```powershell
py -3.13 adattamento-drift/adapt06b_decoupled_threshold/run_pilot.py `
  --data <data-root>/ton `
  --model-root adattamento-drift/review_extension/models `
  --out <output-root>/ton_seed42 `
  --dataset ton `
  --seed 42
```

## UNSW, GAM/RANDOM, seed 42

```powershell
py -3.13 adattamento-drift/adapt06b_decoupled_threshold/run_pilot.py `
  --data <data-root>/unsw `
  --model-root adattamento-drift/review_extension/models `
  --out <output-root>/unsw_seed42 `
  --dataset unsw `
  --seed 42
```

Ogni comando produce un JSON legacy, un JSON decoupled, il confronto appaiato
e un manifest con SHA-256. I due dataset possono essere eseguiti in terminali
separati con BLAS a un thread; la modalità di esecuzione effettiva dovrà essere
registrata nel manifest della consegna.


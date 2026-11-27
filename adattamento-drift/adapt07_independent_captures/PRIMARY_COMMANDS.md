# ADAPT-07 — riproduzione del confronto primario

Il [preflight della prima coppia](PREFLIGHT.json) conserva l’esito della
1-1. L’[emendamento](AMENDMENT_35_1.md) che fissa la coppia 35-1 e 48-1
è al commit `9a3166849592cf6bde1396392d282a7770accd8e`. Il
[preflight emendato](PREFLIGHT_35_1.json) passa per entrambe le sessioni.
Il codice del confronto è stato congelato al commit
`c49ed036cba8fc4a2fa92d2eb32a31e5733f88a8` prima dei fit e delle predizioni.

## Log e ambiente

Salvare dal publisher i log Zeek etichettati come `iot23-35-1.log` e
`iot23-48-1.log` in una cartella esterna al clone. Il preflight
registra gli SHA-256 e le dimensioni. Il runner richiede gli stessi
file. [Log 35-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-35-1/bro/conn.log.labeled) ·
[Log 48-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-48-1/bro/conn.log.labeled)

L’ambiente Windows usa Python 3.12.7, NumPy 2.3.5 e scikit-learn
1.7.2. I quattro limiti a un thread sono richiesti dal runner.
Il calcolo dei supporti e dei digest è separato dalle misure
di tempo e dal picco RSS del processo. Il picco dopo la seconda
cattura è cumulativo, perché entrambe sono elaborate nello stesso
processo in sequenza.

## PowerShell

Eseguire da un checkout che contiene il commit del runner, dalla
radice del clone. Scegliere una directory di risultati nuova.

```powershell
$adaptPython = 'C:\percorso\ambiente\Scripts\python.exe'
$adaptLogs = 'C:\percorso\iot23'
$adaptWork = Join-Path ([System.IO.Path]::GetTempPath()) ('adapt07-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $adaptWork -ErrorAction Stop | Out-Null
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

& $adaptPython adattamento-drift/adapt07_independent_captures/preflight.py --replacement --root $adaptLogs --out (Join-Path $adaptWork 'preflight.json')
if ($LASTEXITCODE -ne 0) { throw 'Preflight fallito' }
& $adaptPython adattamento-drift/adapt07_independent_captures/run_primary.py --root $adaptLogs --out (Join-Path $adaptWork 'primary')
if ($LASTEXITCODE -ne 0) { throw 'Confronto primario fallito' }
```

Il runner legge il preflight versionato e verifica i log prima dei fit.
Il preflight rigenerato serve come controllo indipendente dei supporti.
L’HEAD e i tempi possono differire tra esecuzioni.

## Bash

```bash
set -eu
adapt_python='/percorso/ambiente/bin/python'
adapt_logs='/percorso/iot23'
adapt_work=$(mktemp -d)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$adapt_python" adattamento-drift/adapt07_independent_captures/preflight.py --replacement --root "$adapt_logs" --out "$adapt_work/preflight.json"
"$adapt_python" adattamento-drift/adapt07_independent_captures/run_primary.py --root "$adapt_logs" --out "$adapt_work/primary"
```

## Prodotti

`RESULTS.csv` riporta per ogni cattura e politica TP sul numero
di attacchi, FP sul numero di benigni, FN, TN, recall e FPR.
`BLOCKS.csv`, `KEYS.csv` ed `EPISODES.csv` conservano i confronti
temporali e per chiave. `TYPES.csv` distingue le categorie viste
in A/B da quelle nuove. `FEEDBACK.csv` rende verificabili budget e
frontiera temporale di FIT192. I due file `AB_*.npz` sono salvati
prima delle predizioni su D e contengono trasformazione, modello e
soglia. Il manifest registra gli hash di tutti i prodotti, le versioni,
i limiti ai thread e i tempi per fase.

La figura di copertura si rigenera a partire da `BLOCKS.csv` con
Matplotlib 3.10.7.

```powershell
& $adaptPython adattamento-drift/adapt07_independent_captures/plot_primary.py --blocks (Join-Path $adaptWork 'primary/BLOCKS.csv') --out (Join-Path $adaptWork 'primary/TEMPORAL.png')
```

Il PNG consegnato ha SHA-256
`40bb8084038de1edc61258cf74d2c815dcb343488440e804ff3b9fa531f93349`.

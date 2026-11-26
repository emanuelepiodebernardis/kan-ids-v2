# ADAPT-06I — riproduzione del pilot e dell’estensione TON

Eseguire da PowerShell nella radice del clone, che contiene
`adattamento-drift/`. Il commit del protocollo è
`4f97a0b73b50ebe85e00a3ace0ac1987640f2d58` e quello eseguito è
`d7f33c6e9392667d88d54b55ab02c1106ebf1db7`.

## Input e ambiente

- Impostare `$pilotSource` alla cartella contenente l’archivio originale
  `adapt06e_verify50_full_6ee5eb3.zip`. Il suo SHA-256 è
  `91c2cf5c3fae62b1b6009714ed4a33560e5077e8cad647d87ad1c1cb9fb30eef`.
- `$pilotData` contiene `ton/D.npz`, `ton/D.json`, `unsw/D.npz` e
  `unsw/D.json`, con gli array e l’ordine delle osservazioni verificati
  contro la provenienza ADAPT-06E. Si accetta la ricompressione del
  contenitore NPZ quando gli array coincidono. Per ricostruire gli input
  dalle catture originali, seguire la sezione
  [Dati di ADAPT-05](../adapt05_occurrence/REPRODUCTION.md#dati).
- I modelli GAM e i metadati A/B registrati sono in
  `adattamento-drift/review_extension/models`.
- L’esecuzione di riferimento usa Windows 11 e Python 3.12.7 con
  NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.7.2 e joblib 1.6.0.
  I test sono stati eseguiti con pytest 8.4.2.

Il pacchetto sorgente è quello già condiviso per ADAPT-06E.
[Archivio ADAPT-06E](https://drive.google.com/file/d/1OCTJSaD9blogZM2zcQ4aXeGaZPWJgjCt/view?usp=drive_link)

## Pilot TON e UNSW, seed 42

Il comando verifica l’archivio, i report e le identità degli input e scrive
in una directory nuova. I due report JSON e il manifest si possono leggere
direttamente dalla struttura estratta in `evidence/reports/`.

```powershell
$pilotPython = 'C:\percorso\ambiente\Scripts\python.exe'
$pilotSource = 'C:\percorso\adapt06e_verify50\28a81d5'
$pilotData = 'C:\percorso\adapt05\data'
$pilotZip = Join-Path $pilotSource 'adapt06e_verify50_full_6ee5eb3.zip'
$pilotTemp = Join-Path ([System.IO.Path]::GetTempPath()) ('adapt06i-' + [guid]::NewGuid().ToString('N'))
$pilotExtract = Join-Path $pilotTemp 'source'
$pilotOut = Join-Path $pilotTemp 'results'

Expand-Archive -LiteralPath $pilotZip -DestinationPath $pilotExtract -ErrorAction Stop
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
& $pilotPython adattamento-drift/adapt06i_recurring_feedback/run.py `
  --reports (Join-Path $pilotExtract 'evidence/reports') `
  --archive $pilotZip `
  --data-root $pilotData `
  --model-root adattamento-drift/review_extension/models `
  --out $pilotOut
if ($LASTEXITCODE -ne 0) { throw 'Pilot ADAPT-06I fallito.' }
```

Il manifest in `delivery/` registra i sei prodotti scientifici con
dimensione e SHA-256. Il manifest rigenerato documenta anche l’ambiente
locale, quindi si confrontano i prodotti elencati in `files`.
Il controllo Frozen/B richiede gli stessi conteggi TP, FP, FN e TN
degli archivi in ogni blocco. Se uno dei quattro conteggi differisce,
il runner interrompe la rigenerazione e indica lo stream e il blocco.

```powershell
$pilotManifest = Get-Content -LiteralPath 'adattamento-drift/adapt06i_recurring_feedback/delivery/MANIFEST.json' -Raw | ConvertFrom-Json
foreach ($file in $pilotManifest.files.PSObject.Properties) {
    $actual = Get-FileHash -LiteralPath (Join-Path $pilotOut $file.Name) -Algorithm SHA256 -ErrorAction Stop
    if ($actual.Hash -ne $file.Value.sha256) { throw "SHA-256 diverso per $($file.Name)" }
    Write-Output "PASS $($file.Name)"
}
```

## Controlli

I test sintetici e le verifiche di coerenza dei prodotti consegnati
si eseguono anche senza archivi esterni.

```powershell
& $pilotPython -m pytest -q adattamento-drift/adapt06i_recurring_feedback
ruff check adattamento-drift/adapt06i_recurring_feedback
```

Il controllo di regressione comprende anche i moduli usati come riferimenti.

```powershell
& $pilotPython -m pytest -q `
  adattamento-drift/adapt06b_decoupled_threshold `
  adattamento-drift/adapt06e_verify50 `
  adattamento-drift/adapt06f_next_block `
  adattamento-drift/adapt06g_half_step `
  adattamento-drift/adapt06i_recurring_feedback
```

## Estensione TON ai cinque seed

L’estensione usa lo stesso archivio e la directory dei modelli indicati
sopra. Per gli input grezzi bastano `ton/D.npz` e `ton/D.json`. Il
comando riproduce i quattro replay TON e verifica i prodotti originali
del seed 42 conservati in `delivery/` nel checkout.
Il [protocollo dell’estensione](TON_EXTENSION.md) è nel commit
`852c12e4adda860728ead6d4072978487ad7f092`.
La prima esecuzione è al commit
`0b1d9944113b71d1d87d9adebbe2a22944c7982d`. Il manifest di consegna
registra la riproduzione con la nota aggiornata al commit
`47e539df5e116774e3fffb0ee683f2b8432ccfe2`.

Usare le variabili `$pilotPython`, `$pilotData`, `$pilotZip`,
`$pilotExtract` e i limiti ai thread della procedura sopra. È sufficiente
estrarre l’archivio. I prodotti del primo pilot sono già nel checkout.

```powershell
$tonOut = Join-Path ([System.IO.Path]::GetTempPath()) ('adapt06i-ton-' + [guid]::NewGuid().ToString('N'))
& $pilotPython adattamento-drift/adapt06i_recurring_feedback/run_ton_extension.py `
  --reports (Join-Path $pilotExtract 'evidence/reports') `
  --archive $pilotZip `
  --data-root $pilotData `
  --model-root adattamento-drift/review_extension/models `
  --out $tonOut
if ($LASTEXITCODE -ne 0) { throw 'Estensione TON fallita.' }

$tonManifest = Get-Content -LiteralPath 'adattamento-drift/adapt06i_recurring_feedback/ton_matrix/MANIFEST.json' -Raw | ConvertFrom-Json
foreach ($file in $tonManifest.files.PSObject.Properties) {
    $actual = Get-FileHash -LiteralPath (Join-Path $tonOut $file.Name) -Algorithm SHA256 -ErrorAction Stop
    if ($actual.Hash -ne $file.Value.sha256) { throw "SHA-256 diverso per $($file.Name)" }
    Write-Output "PASS $($file.Name)"
}
```

Il controllo confronta i sei prodotti elencati nel manifest. Il manifest
rigenerato registra il checkout, l’ambiente e i tempi dell’esecuzione locale.

### Riproduzione con Bash

Dalla radice del clone, impostare l’interprete, lo ZIP della campagna
originale e la directory che contiene `ton/`. La sezione
[Dati di ADAPT-05](../adapt05_occurrence/REPRODUCTION.md#dati) indica
come ricostruire gli input.

```bash
set -eu
ton_python='/percorso/ambiente/bin/python'
ton_zip='/percorso/adapt06e_verify50_full_6ee5eb3.zip'
ton_data='/percorso/adapt05/data'
ton_work=$(mktemp -d)
ton_out="$ton_work/results"

"$ton_python" -m zipfile -e "$ton_zip" "$ton_work/source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$ton_python" adattamento-drift/adapt06i_recurring_feedback/run_ton_extension.py \
  --reports "$ton_work/source/evidence/reports" \
  --archive "$ton_zip" \
  --data-root "$ton_data" \
  --model-root adattamento-drift/review_extension/models \
  --out "$ton_out"

"$ton_python" - "$ton_out" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
reference = Path("adattamento-drift/adapt06i_recurring_feedback/ton_matrix/MANIFEST.json")
manifest = json.loads(reference.read_text(encoding="utf-8"))
for name, expected in manifest["files"].items():
    payload = (output / name).read_bytes()
    if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise SystemExit(f"Dimensione o SHA-256 diverso per {name}")
    print(f"PASS {name}")
PY
```

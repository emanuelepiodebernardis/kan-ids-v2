# Riproduzione di ADAPT-06H

Eseguire da PowerShell nella radice del clone. Sono richiesti l’ambiente
Python del progetto, i modelli archiviati, i flussi con i relativi metadati
e lo ZIP ADAPT-06E con i report originali. I percorsi di esempio assumono
l’albero seguente.

```text
adattamento-drift/archivio_esterno/adapt06e_verify50/28a81d5/
  adapt06e_verify50_full_6ee5eb3.zip
  reports/
    RUN_MANIFEST.json
    ton_seed42_verify50.json ... unsw_seed46_verify50.json
```

Ottenere i report estraendo `evidence/reports/` dall’archivio identificato
nella [PR #20](https://github.com/emanuelepiodebernardis/kan-ids-v2/pull/20).
`$adaptData` punta alla cartella che contiene `ton/D.npz`, `ton/D.json`,
`unsw/D.npz` e `unsw/D.json` già usati in ADAPT-06F. Il CSV e il manifest
ADAPT-06F sono versionati. Il codice controlla gli hash prima dello scoring.

La rigenerazione della consegna è stata verificata su Windows con Python
3.12.7, NumPy 2.3.5 e scikit-learn 1.7.2. L’ambiente completo della corsa
è identificato nel `delivery/MANIFEST.json`.

```powershell
$adaptPython = 'C:\percorso\ambiente\Scripts\python.exe'
$adaptData = 'C:\percorso\dati-verificati'
$adaptSource = 'adattamento-drift/archivio_esterno/adapt06e_verify50/28a81d5'
$adapt06F = 'adattamento-drift/adapt06f_next_block/delivery'
$adaptOut = Join-Path ([System.IO.Path]::GetTempPath()) ('adapt06h-' + [guid]::NewGuid().ToString('N'))
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

& $adaptPython adattamento-drift/adapt06h_equal_false_alarms/run.py `
  --reports "$adaptSource/reports" `
  --archive "$adaptSource/adapt06e_verify50_full_6ee5eb3.zip" `
  --cycles-06f "$adapt06F/NEXT_BLOCK_CYCLES.csv" `
  --manifest-06f "$adapt06F/MANIFEST.json" `
  --data-root $adaptData `
  --model-root adattamento-drift/review_extension/models `
  --out $adaptOut
if ($LASTEXITCODE -ne 0) { throw 'Rigenerazione ADAPT-06H fallita.' }
```

Confrontare i cinque prodotti scientifici con gli hash della consegna. Il
manifest prodotto durante la rigenerazione registra anche l’ambiente locale.

```powershell
$adaptDelivery = 'adattamento-drift/adapt06h_equal_false_alarms/delivery'
$manifest = Get-Content -LiteralPath "$adaptDelivery/MANIFEST.json" -Raw | ConvertFrom-Json
foreach ($name in @('CYCLES.csv', 'SEEDS.csv', 'EPISODES.csv', 'DECISION.json', 'NOTE.md')) {
    $actual = Get-FileHash -LiteralPath (Join-Path $adaptOut $name) -Algorithm SHA256 -ErrorAction Stop
    if ($actual.Hash -ne $manifest.files.$name.sha256) { throw "SHA-256 diverso per $name" }
    Write-Output "PASS $name"
}

& $adaptPython -m pytest adattamento-drift/adapt06h_equal_false_alarms -q
```

`CYCLES.csv` distingue il tetto `oracle_fp_cap` sul blocco osservato da
`cal_fp_cap` stimato al FREEZE. Le colonne `*_fp_on_cal` sono conteggi sul
buffer storico. Le colonne `*_fp` misurano il blocco di valutazione. Una
soglia `-inf` ammette tutti i punteggi finiti. I quattro cicli senza blocco
hanno campi scientifici vuoti. `EPISODES.csv` distingue gli episodi descrittivi
dall’unione sui soli semi qualificanti.

Per un controllo a partire dal solo checkout, eseguire il test seguente.
Verifica gli SHA-256, l’allineamento con ADAPT-06F, le confusioni, i totali
appaiati e la regola per gli episodi. I report originali, i flussi e i
modelli sono richiesti per la rigenerazione precedente.

```powershell
& $adaptPython -m pytest adattamento-drift/adapt06h_equal_false_alarms/test_delivery.py -q
```

## Riscontro sui due confini CAL

Eseguire con le stesse variabili e i limiti di thread dell’esempio precedente.
Lo script produce un JSON separato per TON/44/FREEZE 302 e UNSW/43/FREEZE 37.
Per ogni caso esporta gli ID e gli score normali vicini ai confini, in decimale
e binary64 esadecimale, oltre alle soglie e ai conteggi dei due bracci.

```powershell
$adaptAudit = Join-Path ([System.IO.Path]::GetTempPath()) ('adapt06h-audit-' + [guid]::NewGuid().ToString('N') + '.json')
& $adaptPython adattamento-drift/adapt06h_equal_false_alarms/audit_platform.py `
  --reports "$adaptSource/reports" `
  --archive "$adaptSource/adapt06e_verify50_full_6ee5eb3.zip" `
  --data-root $adaptData `
  --model-root adattamento-drift/review_extension/models `
  --out $adaptAudit
if ($LASTEXITCODE -ne 0) { throw 'Riscontro sui confini CAL fallito.' }
Get-Content -LiteralPath $adaptAudit -Raw
```

In ambiente Windows il risultato di riferimento è `PLATFORM_AUDIT_WINDOWS.json`.
La prova a tetto alternativo va letta nello stesso ambiente che l’ha prodotta.
Vedere `PLATFORM_NOTE.md` per l’interpretazione e la provenienza dei confronti.

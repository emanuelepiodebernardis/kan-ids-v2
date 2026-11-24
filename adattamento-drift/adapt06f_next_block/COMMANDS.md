# Comandi di riproduzione — Windows PowerShell

I comandi partono dalla radice del clone, cioè dalla directory che contiene
`adattamento-drift/`. Impostare i tre percorsi locali senza spostare o
rinominare i file sorgente.

```powershell
$run = '<percorso>/adapt06e_verify50/28a81d5'
$data = '<percorso>/adapt05/data'
$python = '<percorso>/python.exe'
$out = Join-Path ([System.IO.Path]::GetTempPath()) ('adapt06f-next-block-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null

& $python adattamento-drift/adapt06f_next_block/analyze_next_block.py `
  --reports "$run/reports" `
  --cycles adattamento-drift/adapt06e_verify50/delivery/CYCLES.csv `
  --archive "$run/adapt06e_verify50_full_6ee5eb3.zip" `
  --data-root $data `
  --model-root adattamento-drift/review_extension/models `
  --out $out
if ($LASTEXITCODE -ne 0) { throw 'Rigenerazione ADAPT-06F fallita.' }
```

Lo script accetta una directory di output vuota. La consegna verificata
rimane in `adattamento-drift/adapt06f_next_block/delivery/`.

Confrontare i tre file scientifici `NEXT_BLOCK_CYCLES.csv`, `SUMMARY.json` e
`NOTE.md` con gli SHA-256 in `files` del `MANIFEST.json` consegnato. Il comando
genera anche un nuovo `MANIFEST.json` per documentare la provenienza della
rigenerazione. I suoi metadati di commit e ambiente possono variare tra
esecuzioni, quindi il manifest resta fuori dal confronto byte per byte.

```powershell
$reference = 'adattamento-drift/adapt06f_next_block/delivery/MANIFEST.json'
$manifest = Get-Content -LiteralPath $reference -Raw | ConvertFrom-Json
foreach ($name in @('NEXT_BLOCK_CYCLES.csv', 'SUMMARY.json', 'NOTE.md')) {
    $actual = Get-FileHash -LiteralPath (Join-Path $out $name) -Algorithm SHA256 -ErrorAction Stop
    if ($actual.Hash -ne $manifest.files.$name.sha256) {
        throw "SHA-256 diverso per $name"
    }
    Write-Output "PASS $name"
}
```

Test mirati:

```powershell
& $python -m pytest adattamento-drift/adapt06f_next_block/test_next_block.py -q
```

Controlli statici eseguiti sul checkout:

```powershell
ruff check adattamento-drift/adapt06f_next_block
```

Nessun comando di questa analisi avvia training o replay. Le etichette di
`d+1` sono lette soltanto per ricavare le confusioni dei tre stati già salvati.
Un `D.npz` ricompresso è ammesso: il programma non richiede lo stesso SHA-256
del contenitore, ma verifica digest, forma e dtype di ogni array e l'ordine
completo di `row_id`.


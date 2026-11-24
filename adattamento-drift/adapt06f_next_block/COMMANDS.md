# Comandi di riproduzione — Windows PowerShell

I comandi partono dalla radice del clone, cioè dalla directory che contiene
`adattamento-drift/`. Impostare i tre percorsi locali senza spostare o
rinominare i file sorgente.

```powershell
$run = '<percorso>/adapt06e_verify50/28a81d5'
$data = '<percorso>/adapt05/data'
$python = '<percorso>/python.exe'

& $python adattamento-drift/adapt06f_next_block/analyze_next_block.py `
  --reports "$run/reports" `
  --cycles adattamento-drift/adapt06e_verify50/delivery/CYCLES.csv `
  --archive "$run/adapt06e_verify50_full_6ee5eb3.zip" `
  --data-root $data `
  --model-root adattamento-drift/review_extension/models `
  --out adattamento-drift/adapt06f_next_block/delivery
```

Lo script rifiuta una directory di output non vuota. Per una rigenerazione,
usare una nuova directory e confrontare i quattro file con gli SHA-256 del
manifest consegnato; non sovrascrivere la consegna verificata.

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


# Materiali per il secondo articolo

Questa cartella organizza le evidenze gia' verificate di ADAPT-01--04 e la
validazione finale gia' eseguita su D. I materiali documentali non avviano
nuovi training o replay.

- `tabella_evidenze.csv`: confronti appaiati e gruppi di comparabilita'.
- `appendice_evidenze.md`: tabella leggibile, regole di confronto e provenienza.
- `sintesi.md`: sintesi scientifica breve (massimo due pagine di testo).
- `analisi_aggiornamenti_skip.*`: errori nel blocco successivo a update o skip.
- `protocollo_validazione_finale.md`: proposta minima prima dell'apertura di D.
- `../replay_validazione_finale.py`: runner congelabile per le quattro tracce.
- `../test_replay_validazione_finale.py`: controprove sintetiche, senza D.
- `manifest.json`: SHA-256 di fonti e prodotti.
- `validazione_finale_D/`: rapporto, aggregazioni e manifest della campagna
  finale verificata.
- `manuscript_methods_results.md`: prima bozza manuale delle sole sezioni
  Materials and Methods e Results.
- `manuscript_assets/`: due tabelle, due figure e manifest riproducibile della
  bozza.
- `../genera_materiali_manoscritto.py`: generatore che legge soltanto JSON e
  manifest versionati; non apre i flussi e non esegue fit o replay.

## Riproduzione

Dalla cartella `adattamento-drift/ton_temporal`:

```powershell
cd <repo>/adattamento-drift/ton_temporal
python costruisci_materiali_articolo_02.py
python -m pytest test_materiali_articolo_02.py -q
python -m pytest test_replay_validazione_finale.py -q
python genera_materiali_manoscritto.py
python -m pytest test_materiali_manoscritto.py -q
```

I raw completi devono essere presenti nella cartella documentata
`../archivio_esterno/`. Il generatore non li modifica. I riepiloghi versionati
sono letti dalle cartelle ADAPT-01--04; i conteggi update/skip sono ricalcolati
dai raw. Ogni decisione del blocco `k` e' associata al blocco `k+1`, nel quale
puo' diventare effettiva.

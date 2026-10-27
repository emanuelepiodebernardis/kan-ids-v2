# Materiali per il secondo articolo

Questa cartella organizza le evidenze gia' verificate di ADAPT-01--04. Non
contiene nuovi training o replay e non usa D.

- `tabella_evidenze.csv`: confronti appaiati e gruppi di comparabilita'.
- `appendice_evidenze.md`: tabella leggibile, regole di confronto e provenienza.
- `sintesi.md`: sintesi scientifica breve (massimo due pagine di testo).
- `analisi_aggiornamenti_skip.*`: errori nel blocco successivo a update o skip.
- `protocollo_validazione_finale.md`: proposta minima prima dell'apertura di D.
- `../replay_validazione_finale.py`: runner congelabile per le quattro tracce.
- `../test_replay_validazione_finale.py`: controprove sintetiche, senza D.
- `manifest.json`: SHA-256 di fonti e prodotti.

## Riproduzione

Dalla cartella `adattamento-drift/ton_temporal`:

```powershell
python costruisci_materiali_articolo_02.py
python -m pytest test_materiali_articolo_02.py -q
python -m pytest test_replay_validazione_finale.py -q
```

I raw completi devono essere presenti nella cartella documentata
`../archivio_esterno/`. Il generatore non li modifica. I riepiloghi versionati
sono letti dalle cartelle ADAPT-01--04; i conteggi update/skip sono ricalcolati
dai raw. Ogni decisione del blocco `k` e' associata al blocco `k+1`, nel quale
puo' diventare effettiva.

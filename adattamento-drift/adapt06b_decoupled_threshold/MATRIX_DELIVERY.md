# ADAPT-06B-MATRIX — consegna seed 42–46

Il protocollo, la configurazione, il launcher e i test sono stati congelati
nel commit di esecuzione
`ed50e86805c6591b967b39a7be982f4e8d30eeab`. Da quel commit sono state
eseguite, in una directory nuova, soltanto le otto coppie preregistrate dei
seed 43–46. Il pilot seed 42 non è stato rieseguito né sovrascritto.

La consegna completa è in `matrix_seed42_46_ed50e86/`:

- `new_replays/` contiene i 16 report e gli otto confronti nuovi;
- `analysis/PILOT_INDEX.json` indicizza i quattro report e i due confronti
  seed 42 già verificati;
- `analysis/MATRIX_ALL_CONTROLS.csv` contiene le 50 righe previste;
- `analysis/MATRIX_PAIRED.csv` contiene i dieci confronti appaiati;
- gli altri CSV, le due figure e `MATRIX_REPORT.md` documentano decisioni,
  intervalli temporali, costi e interpretazione;
- `analysis/MATRIX_ENVIRONMENT.json` verifica l'identità dell'ambiente di base
  dei venti replay, distingue i campi registrati dai due runner e documenta
  l'isolamento dei processi;
- `analysis/MATRIX_DELIVERY_MANIFEST.json` registra dimensioni e SHA-256 degli
  output e dei file di provenienza.

`test_matrix_delivery.py` ricalcola confronti, tabelle e rapporto dai report,
verifica il manifest, controlla che il pilot sia rimasto invariato e conserva
esplicitamente gli esiti nulli o sfavorevoli. Il commit di consegna è distinto
dal commit di esecuzione ed è identificato dall'HEAD della PR che contiene il
manifest.

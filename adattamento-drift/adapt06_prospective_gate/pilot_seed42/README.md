# Pilot seed 42 — prefisso di 100 blocchi

Questa directory conserva i riepiloghi leggeri del solo pilot autorizzato su
TON e UNSW. L'esecuzione usa il commit
`ceaf0e14a2619bbc38189a7cee046abcb0647994`, derivato dal merge
`56cbfd9de62569bb1b72b63c0ec1640e4df2d318`, e fissa `--max-blocks 100`.

## Inventario

- [`PILOT_REPORT.md`](PILOT_REPORT.md): perimetro, controlli, funzionamento e
  costi misurati;
- [`REPORT_SUMMARY.csv`](REPORT_SUMMARY.csv): una riga per ciascuno dei 12
  report, con label, supporti, decisioni, tempi e memoria;
- [`TRACK_METRICS.csv`](TRACK_METRICS.csv): metriche di tutte le tracce,
  compresi Frozen/B, threshold-only e i confronti diagnostici;
- [`ADAPTIVE_TRAJECTORIES.csv`](ADAPTIVE_TRAJECTORIES.csv): le 20 traiettorie
  adattive previste dal protocollo;
- [`CANDIDATE_DECISIONS.csv`](CANDIDATE_DECISIONS.csv): freeze, look, rifiuti
  e candidati pendenti;
- [`PILOT_MANIFEST.json`](PILOT_MANIFEST.json): ambiente, commit, hash dei 12
  JSON e dei file di analisi;
- [`ARCHIVE.md`](ARCHIVE.md): posizione, dimensione e SHA-256 del pacchetto
  completo esterno.

I 12 JSON completi, i comandi effettivi, i log dei test, lo script di
esecuzione e quello di verifica non sono duplicati in Git. Sono contenuti
nell'archivio esterno descritto in `ARCHIVE.md`.

Il pilot misura funzionamento e costi. Non autorizza la matrice completa sui
cinque seed e non sostiene una conclusione generale sull'efficacia del gate.

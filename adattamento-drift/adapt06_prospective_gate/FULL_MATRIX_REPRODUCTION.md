# Riproduzione della matrice completa ADAPT-06

Eseguire dalla radice del clone su Windows, dopo il PASS dei test e da un
commit congelato. La directory indicata con `<output-nuovo>` non deve esistere.

```powershell
py -3.13 -m pytest adattamento-drift/adapt06_prospective_gate -q

py -3.13 adattamento-drift/adapt06_prospective_gate/run_full_matrix.py `
  --data-root <data-root> `
  --model-root adattamento-drift/review_extension/models `
  --out <output-nuovo> `
  --workers 2
```

L'esecutore crea 60 processi indipendenti, al massimo due contemporanei. Ogni
processo ricarica gli stati A/B archiviati. I thread BLAS sono fissati a uno;
comandi, ambiente, input, modelli, tempi e SHA del commit vengono registrati.
Non viene passato `--max-blocks`.

Dopo i 60 PASS:

```powershell
py -3.13 adattamento-drift/adapt06_prospective_gate/analyze_full_matrix.py `
  --root <output-nuovo> `
  --archive <archivio-esterno>/adapt06_full_matrix_<sha>.zip
```

L'analisi deve riportare `PASS reports=60 adaptive_trajectories=100`, verificare
inventario, provenienza, ordine dei blocchi e conservazione delle richieste,
quindi produrre confronto appaiato, costi, decisione aggregata, quattro figure,
manifest, `SHA256SUMS.txt`, ZIP e relativo SHA-256.

Il pacchetto estratto contiene sotto `reproduction/adapt06_prospective_gate/`
il codice, i test e i documenti congelati; le due dipendenze storiche sono in
`reproduction/ton_temporal/`. Dopo avere verificato
`SHA256SUMS.txt`, le analisi salvate si rigenerano dalla radice estratta senza
accedere ai flussi originali:

```powershell
py -3.13 reproduction/adapt06_prospective_gate/analyze_full_matrix.py `
  --root .
```

La rigenerazione ricalcola tabelle, decisione e figure dai 60 JSON conservati;
non esegue nuovi fit, tuning o replay.

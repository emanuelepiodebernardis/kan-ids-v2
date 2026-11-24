# Comandi riproducibili

I comandi sono eseguiti da PowerShell nella radice del clone. I segnaposto
`<data-root>` e `<python>` indicano rispettivamente la directory contenente i
dati verificati e l'interprete dell'ambiente documentato. Ogni directory di
uscita deve essere nuova.

```powershell
<python> -m pytest adattamento-drift/adapt06e_verify50 adattamento-drift/adapt06g_half_step -q
ruff check adattamento-drift/adapt06e_verify50/engine.py adattamento-drift/adapt06g_half_step
```

Il preflight verifica configurazione, A/B/D, i dieci modelli iniziali e i dieci
report ADAPT-06E originali prima dei nuovi replay:

```powershell
<python> adattamento-drift/adapt06g_half_step/preflight.py `
  --data-root <data-root> `
  --model-root adattamento-drift/review_extension/models `
  --reference-root adattamento-drift/archivio_esterno/adapt06e_verify50/28a81d5/reports `
  --out adattamento-drift/archivio_esterno/adapt06g_half_step/<sha>/PREFLIGHT.json
```

La matrice usa un processo nuovo per ciascun replay e imposta a uno i thread
BLAS/OpenMP. Il launcher rifiuta una directory già esistente e un commit o una
configurazione diversi dal preflight.

```powershell
<python> adattamento-drift/adapt06g_half_step/run_matrix.py `
  --data-root <data-root> `
  --model-root adattamento-drift/review_extension/models `
  --preflight adattamento-drift/archivio_esterno/adapt06g_half_step/<sha>/PREFLIGHT.json `
  --out adattamento-drift/archivio_esterno/adapt06g_half_step/<sha>/reports
```

L'analisi ricalcola i confronti dai report verificati:

```powershell
<python> adattamento-drift/adapt06g_half_step/analyze.py `
  --half-step-root adattamento-drift/archivio_esterno/adapt06g_half_step/<sha>/reports `
  --adapt06e-root adattamento-drift/archivio_esterno/adapt06e_verify50/28a81d5/reports `
  --adapt06b-root adattamento-drift/adapt06b_decoupled_threshold `
  --out adattamento-drift/adapt06g_half_step/delivery
```

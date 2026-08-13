# Come caricare questa cartella sul repository

Questa cartella contiene il progetto **completo**: il contenuto originale di
`kan-ids-v2` più tutto il lavoro della fase di consolidamento. Non contiene
la cronologia git, i dataset (non redistribuibili) né le cache rigenerabili.

## Opzione A — sovrascrivere il repository esistente

```bash
cd /percorso/del/tuo/clone/kan-ids-v2
git checkout main && git pull

# copia il contenuto di questa cartella sopra il clone
rsync -a --delete --exclude='.git' /percorso/di/questa/cartella/ .

git add -A
git commit -m "Protocollo v2: pipeline leakage-free, cross-domain BoT-IoT, inferenza integer-only verificata"
git push
```

`--delete` rimuove dal repository i file che qui non ci sono più: è voluto,
perché alcuni risultati sono stati spostati in `results/protocol_v1/`.
Se preferisci non cancellare nulla, togli `--delete`.

## Opzione B — partire da zero

```bash
cd /percorso/di/questa/cartella
git init && git branch -M main
git remote add origin https://github.com/emanuelepiodebernardis/kan-ids-v2.git
git add -A
git commit -m "Protocollo v2: consolidamento sperimentale"
git push -u origin main --force   # ATTENZIONE: riscrive la storia remota
```

Sconsigliata se vuoi conservare la cronologia dei commit precedenti.

## Se stai aggiornando un clone gia' allineato

Rispetto alla versione online mancano solo questi file. Se preferisci non
sovrascrivere tutto, copia solo loro:

**nuovi**: `requirements-lock.txt`, `scripts/nested_cv.py`,
`tools/audit_richieste.py`, `results/nested_cv_folds_binary.csv`,
`results/nested_cv_summary_binary.csv`

**modificati**: `README.md`, `reproduce.py`, `.gitignore`,
`scripts/make_report.py`, `tests/test_leakage.py`,
`tests/test_reproducibility.py`, `report_KAN-IDS_fase2.pdf`

## Prima di pubblicare, una verifica in 30 secondi

```bash
pip install -r requirements.txt
python reproduce.py --stage smoke
```

Deve stampare `25 passed` e terminare senza errori. Non serve alcun dataset:
lo stage usa dati sintetici con lo schema di TON_IoT.

Poi la verifica dei sei punti della revisione, con le evidenze:

```bash
python reproduce.py --stage audit
```

Deve chiudersi con `22/23 requisiti verificati come soddisfatti` (l'unico
mancante e' CIC-IoT-2023, obiettivo secondario).

Per verificare che il firmware sia compilabile e bit-esatto senza dataset:

```bash
cd mcu_pio/host_check
g++ -O2 -I../include -o run_e2e_check run_e2e_check.cpp && ./run_e2e_check
g++ -O2 -I../include -o run_mc_e2e_check run_mc_e2e_check.cpp && ./run_mc_e2e_check
```

Entrambi devono riportare 200/200.

## Cosa NON è incluso, e perché

| Non incluso | Motivo |
|---|---|
| `data/*.csv` | TON_IoT e BoT-IoT non sono redistribuibili; `data/README.md` spiega dove scaricarli |
| `artifacts/` | cache rigenerabile, ricostruita da `reproduce.py` |
| `.git/` | la cronologia resta quella del tuo clone |
| checkpoint multiclass | 25 MB di stato dell'ottimizzatore, rigenerabile (vedi `models/MANIFEST.json`) |

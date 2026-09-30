# Ambiente, costi e comandi riproducibili

ADAPT-01, 30 settembre 2026. Tutti i risultati di questo PR sono prodotti dai
comandi qui sotto, nell'ordine indicato.

## Ambiente

Registrato automaticamente dentro ogni rendiconto del replay, campo `ambiente`:

- Python 3.11.15, CPython
- numpy 2.4.4, scikit-learn 1.8.0
- Linux x86_64
- **solo CPU**: nessuna libreria di accelerazione, nessun uso di GPU

La suite di test è stata eseguita anche su Windows con Python 3.13, dove passa
da 634 a 643 test superati con 1 saltato; su Linux con Python 3.11 da 609 a 618
con 26 saltati. In entrambi i casi nove in più, che sono la nuova suite.

## Costi misurati

Dal campo `costi` del rendiconto, corsa di riferimento con seme 42 su C:

- addestramento iniziale dei tre modelli su A: **27 secondi** in tutto
- replay su 916 blocchi: **86 secondi**, cioè **0,09 secondi per blocco**
- etichette richieste: **91.506**, cento per blocco completo
- aggiornamenti applicati: **522** per modello; saltati per memoria monoclasse:
  **393**

L'inventario completo dei 23 file costa circa 13 minuti; la ricostruzione delle
impronte, 2 secondi.

## Comandi, nell'ordine

I dati grezzi non sono redistribuiti: `<dati>` è la cartella con i 23
`Network_dataset_*.csv`, `<lab>` un clone di `ids-update-lab`, `<noto>` il
`train_test_network.csv` il cui SHA-256 è
`26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974`.

**1. Ricostruire le impronte degli input già visti**

```
python impronte_note.py --sorgente <noto> \
    --audit <lab>/research028/vendor/audit_ton_full.py \
    --uscita impronte_note.npz --esigi-sorgente
```

Esce diverso da zero se la sorgente o la cardinalità non coincidono con i
metadati dichiarati: 92.330 impronte uniche da 211.043 righe.

**2. Inventario dei 23 file**

```
python inventario_ton.py --dati <dati> \
    --audit <lab>/research028/vendor/audit_ton_full.py \
    --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json \
    --impronte impronte_note.npz --uscita inventario --riprendi
```

Ripartibile: rilanciandolo salta i file già fatti. Produce
`data_inventory.csv`, `daily_counts.csv`, `hourly_counts.csv`,
`hourly_counts_frt.csv`, `diagnostics.json`, `feature_ready_time.json` e
`per_file/`.

**3. Copertura, con lo strumento del referente, senza modifiche**

```
python <lab>/research028/tools/coverage028.py \
    --input inventario/hourly_counts_frt.csv --output inventario/copertura_frt
```

**4. Costruire i flussi ordinati per feature_ready_time**

```
python costruisci_flusso.py --dati <dati> \
    --audit <lab>/research028/vendor/audit_ton_full.py \
    --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json \
    --inizio 2019-04-02 --fine-esclusa 2019-04-05 --giorni-extra 2019-04-23 \
    --uscita flussi/A.npz
```

Per B: `--inizio 2019-04-24 --fine-esclusa 2019-04-25`.
Per C: `--inizio 2019-04-25 --fine-esclusa 2019-04-27`.
D non è stato costruito.

**5. Il replay**

```
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz \
    --uscita replay_C_seme42.json
```

I parametri predefiniti sono quelli della scheda: blocchi di 10.000, budget 1%,
memoria 256, ritardo di un blocco, seme 42. Per le ripetizioni, `--seme 43` e
seguenti. Per la variante scartata, `--campionamento strati_punteggio`.

**6. Le curve**

```
python curve_replay.py --rendiconto replay_C_seme42.json \
    --modello lr --uscita figure
```

**7. La suite**

```
python -m pytest ton_temporal -q
```

Attesi: 11 test superati.

## Note di riproducibilità

Il seme controlla il campionamento delle etichette, l'inizializzazione della
memoria e il sottocampione usato per l'addestramento iniziale di MLP e KAN. La
regressione logistica è addestrata su tutto A ed è quindi deterministica: il suo
AUROC congelato vale 0,7149 in tutti e cinque i semi.

Le corse dei semi 43, 44, 45 e 46 non sono incluse nel PR per non gonfiare il
repository; si riproducono con il comando 5 cambiando `--seme`. Il consolidato è
in `replay_evidenza/ripetizioni_cinque_semi.json`.

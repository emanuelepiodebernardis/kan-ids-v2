# Ambiente, costi e comandi riproducibili

ADAPT-01, aggiornato al 3 ottobre 2026. Tutti i risultati di questo PR sono
prodotti dai comandi qui sotto, nell'ordine indicato.

## Ambiente

Registrato automaticamente dentro ogni rendiconto del replay, campo `ambiente`,
insieme all'hardware letto dal sistema in `ambiente.hardware_misurato`:

- Python 3.11.15, CPython
- numpy 2.4.4, scikit-learn 1.8.0
- Linux x86_64, **solo CPU**: nessuna libreria di accelerazione, nessun uso di GPU
- un Intel Xeon a 2,80 GHz, due CPU utilizzabili dal processo, 8.031 MiB di RAM
- nessuna variabile `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` o
  `MKL_NUM_THREADS` impostata: i thread di BLAS sono quelli predefiniti

La macchina è virtuale e le CPU sono condivise. Questo basta per i tempi come
ordine di grandezza e per i rapporti fra modelli, **non** come misura di
latenza di un sistema in esercizio: a parità di modello il massimo del tempo di
un singolo aggiornamento sta al minimo come 29 a 1 sulle corse dedicate, e come
178 a 1 sulle corse eseguite mentre girava altro lavoro.
`costi_aggiornamento.py` riporta quel rapporto per ciascun modello.

## La suite di test

```
python -m pytest ton_temporal -q
```

Attesi: **193 test superati**. Erano **139** al commit precedente; i 54 in più
sono le due suite nuove di questo ciclo — `test_impronta_campione.py`, 16 test,
e `test_misure_di_costo.py`, 17 — più i 21 aggiunti a quelle esistenti:
`test_documenti_coerenti.py` passa da 44 a 56, `test_politiche.py` da 17 a 23,
`test_riepiloghi_semi.py` da 18 a 21. Invariate `test_soglia.py` con 26,
`test_sovrapposizioni.py` con 23 e `test_guardia_monoclasse.py` con 11. Su
questo ramo `python -m pytest -q` dalla radice del repository raccoglie gli
stessi test, perché i soli file `test_*.py` presenti sono quelli di
`ton_temporal/`.

Una versione precedente di questo documento dichiarava «11 test superati» e
citava conteggi di 634 e 643 test misurati su un altro ramo: nessuno dei due
numeri era verificabile qui, ed entrambi sono stati sostituiti con il conteggio
misurato su questo ramo.

## Costi misurati

Dal campo `costi` del rendiconto, corsa di riferimento con seme 42 su C
(`replay_C_seme42.json`):

- addestramento iniziale dei tre modelli su A: **33 secondi**
- replay su 916 blocchi: **87 secondi**, cioè **0,09
  secondi per blocco**
- etichette richieste: **91.506**, cento per blocco completo e sei sul blocco
  finale di 673 righe; la quota è `91.506 / 9.150.673`, registrata nel campo
  `quota_etichette_effettiva` accanto a `righe_scorse`
- aggiornamenti applicati: **522** per modello; saltati per memoria monoclasse:
  **393**

I tempi variano di qualche secondo fra esecuzioni: sono un ordine di grandezza,
non una misura ripetibile.

**Costo del singolo aggiornamento**, mediana fra i cinque semi della mediana per
corsa, dalla politica `ogni blocco` con 520 aggiornamenti:

| | ms per aggiornamento | secondi in aggiornamento | quota del replay |
|---|---:|---:|---:|
| logistica | 5,19 ms | 4,9 s | 5,6% |
| MLP | 5,27 ms | 3,1 s | 3,5% |
| additivo | 11,08 ms | 6,8 s | 7,8% |

**Memoria.** Picco di memoria residente del processo: circa 1.338 MiB,
dominato dagli array dello stream che il banco di prova tiene in memoria. Lo
stato che l'adattamento muove è 10.240 byte di memoria FIFO e, per modello,
8.264 byte di parametri per la logistica, 16.520 per l'MLP e 16.456 per
l'additivo. Il numero di aggiornamenti **non dimostra un guadagno di latenza né
di memoria**: il punteggio di ogni blocco si paga in tutte le politiche, e il
risparmio sugli aggiornamenti è di 15 secondi su 86,6 secondi di replay.

L'inventario completo dei 23 file costa circa 13 minuti; la ricostruzione delle
impronte, 2 secondi. Le venti corse di questo ciclo, in sequenza su questa
macchina, circa 48 minuti.

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
Per D il comando sarebbe `--inizio 2019-04-27 --fine-esclusa 2019-04-30`, ma
**D non è stato costruito**: le sue righe sono state soltanto lette e contate
dai CSV originali da `sovrapposizioni_abcd.py`, per il controllo delle
sovrapposizioni del §5bis. D non è stato usato né per addestrare né per
valutare, e il flusso si costruirà quando la valutazione finale sarà
autorizzata.

**4bis. Le sovrapposizioni dirette fra A, B, C e D**

```
python sovrapposizioni_abcd.py --csv <dati> \
    --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json \
    --audit <lab>/research028/vendor/audit_ton_full.py \
    --uscita sovrapposizioni_abcd.json
```

Scorre i 23 CSV una volta, assegna ogni riga al proprio intervallo per
`feature_ready_time` e confronta le impronte dei vettori di feature fra gli
intervalli, usando `normalize_raw` e `fingerprint_rows` del modulo di
riferimento senza modificarli.

**5. Il replay: le quattro configurazioni, per ogni seme**

```
B="--iniziale flussi/A.npz --flusso flussi/C.npz" python replay.py $B --seme
<n> --uscita replay_C_seme<n>.json python replay.py $B --calibrazione
flussi/B.npz --seme <n> \
    --uscita replay_C_calibrato_seme<n>.json
python replay.py $B --calibrazione flussi/B.npz --politica evidenza_inversione
\
    --seme <n> --uscita replay_C_evidenza_seme<n>.json
python replay.py $B --calibrazione flussi/B.npz --politica casuale \
    --quanti-aggiornamenti "lr=54,mlp=20,kan=23" \
    --ammissibili replay_C_calibrato_seme<n>.json --seme <n> \
    --uscita replay_C_casuale_seme<n>.json
```

I parametri predefiniti sono quelli della scheda: blocchi di 10.000, budget 1%,
memoria 256, ritardo di un blocco. I semi sono 42, 43, 44, 45 e 46, e **tutte
le venti corse sono pubblicate** in `replay_evidenza/`. I conteggi da passare
al controllo casuale sono quelli applicati dalla politica su evidenza **dello
stesso seme**, che si leggono in `costi.decisioni_della_politica`: cambiarli
renderebbe il controllo non pareggiato. Per la variante scartata,
`--campionamento strati_punteggio`.

**6. I candidati della soglia e la regola di confronto**

```
python documenta_soglie.py --iniziale flussi/A.npz --calibrazione flussi/B.npz
\
    --semi 42 43 44 45 46 --uscita replay_evidenza/soglie_candidati.json
```

**7. I riepiloghi, i confronti e i costi**

```
cd replay_evidenza Z="replay_C_seme42.json ... replay_C_seme46.json"
B="replay_C_calibrato_seme42.json ... replay_C_calibrato_seme46.json"

python ../riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json
--verifica python ../riepiloghi_semi.py --rendiconti $Z --confronto $B \
    --uscita confronto_soglia.json --verifica
python ../confronto_politiche.py --ogni-blocco $B \
    --evidenza replay_C_evidenza_seme4*.json --casuale replay_C_casuale_seme4*.json \
    --uscita confronto_politiche.json
python ../costi_aggiornamento.py --rendiconti replay_C_*seme4*.json \
    --uscita costi_aggiornamento.json
```

**8. Le curve**

```
python curve_replay.py --rendiconto replay_C_seme42.json \
    --modello lr --uscita figure
```

**9. Le richieste del referente, controllate sul materiale**

```
python verifica_richieste.py --cartella .
```

Traduce ogni richiesta delle lettere e del riesame in un controllo eseguibile
sui file del ramo, ed elenca a parte quello che un programma non può
verificare.

## Note di riproducibilità

Il seme controlla il campionamento delle etichette, l'inizializzazione della
memoria e il sottocampione usato per l'addestramento iniziale di MLP e KAN. La
regressione logistica è addestrata su tutto A ed è quindi deterministica: il
suo AUROC congelato vale 0,7149 in tutti e cinque i semi.

Le righe campionate sono le stesse in tutte le configurazioni dello stesso
seme, e lo si verifica sullo SHA-256 registrato per blocco in
`row_id_campionati_sha256` — non più sulla somma
`row_id_campionati_somma_storica`, che è un controllo più debole e resta solo
per continuità con i rendiconti pubblicati prima di questo ciclo.

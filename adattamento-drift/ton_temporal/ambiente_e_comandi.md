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
cd <repo>
python -m pytest ton_temporal -q
```

Attesi: **239 test superati**, e sono i test **di ADAPT-01**, cioè i soli file
`test_*.py` della cartella `ton_temporal/`. Per file: `test_documenti_coerenti.py`
64, `test_figure.py` 38, `test_soglia.py` 26, `test_politiche.py` 23,
`test_sovrapposizioni.py` 23, `test_riepiloghi_semi.py` 21,
`test_misure_di_costo.py` 17, `test_impronta_campione.py` 16,
`test_guardia_monoclasse.py` 11.

La storia dei conteggi, perché il numero cambia a ogni commit: **139** al commit
`c5e6d02`, **193** a `7ea4a94`, **195** a `5922992`, **239** ora. I 44 di questo
ciclo sono le 38 di `test_figure.py` e 6 aggiunte a `test_documenti_coerenti.py`
— la cartella dichiarata in ogni blocco di comandi, l'esistenza dei file che i
comandi citano, le frasi troncate, il confronto fra il numero di prove
dichiarato qui e quello che la raccolta trova davvero — perché questo conteggio
è già rimasto indietro due volte — e l'encoding con cui gli script scrivono la
propria uscita, dopo che `verifica_richieste.py > verifica_richieste.txt` è
caduto su Windows perché `cp1252` non ha il meno tipografico.

**I 239 non sono il totale del repository.** Lanciando `pytest` dalla radice del
repository si raccolgono anche le suite delle altre linee di lavoro: al commit
`5922992` quel totale era **829 passate e 1 saltata**, di cui 195 erano queste,
quindi ora ci si attendono **873 passate e 1 saltata**. I due numeri vanno
citati separatamente, perché misurano cose diverse: 239 è ADAPT-01, 873 è tutto
il repository. Il conteggio della radice dipende da quali altre cartelle sono
presenti nella copia di lavoro, quindi è una previsione e non una proprietà del
ramo.

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

**Costo del singolo aggiornamento.** I numeri qui sotto vengono da **una misura
dedicata**, non da una media sui semi: una sola corsa, **seme 42**, politica
`ogni blocco`, **522 aggiornamenti applicati** e 393 tentativi fermati dalla
guardia, eseguita con la macchina altrimenti inattiva e pubblicata in
`replay_evidenza/tempi/ogni_blocco_seme42_costi.json`. La colonna che conta è la
mediana.

| misura dedicata, seme 42, 522 aggiornamenti | ms per aggiornamento, mediana | secondi in aggiornamento | quota del replay |
|---|---:|---:|---:|
| logistica | 5,19 ms | 4,9 s | 5,6% |
| MLP | 5,27 ms | 3,1 s | 3,5% |
| additivo | 11,08 ms | 6,8 s | 7,8% |

Le stesse misure su tutti e venti i rendiconti — dove gli aggiornamenti
applicati sono 520 in media — si ottengono con `costi_aggiornamento.py` e stanno
in `replay_evidenza/costi_aggiornamento.json`.

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

I dati grezzi non sono redistribuiti, quindi quattro percorsi dipendono dalla
macchina e sono le **sole** cose da sostituire: `<repo>` è la cartella del
clone di questo repository, `<dati>` è la cartella con i 23
`Network_dataset_*.csv`, `<lab>` un clone di `ids-update-lab`, `<noto>` il
`train_test_network.csv` il cui SHA-256 è
`26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974`. Tutto il
resto dei comandi è letterale e si incolla così com'è.

**Ogni blocco comincia con il `cd` da cui va eseguito**, perché lo stesso
comando riesce da una cartella e fallisce da un'altra: senza quella riga chi
legge deve indovinarla. I comandi sono su una riga sola di proposito: una riga
spezzata, incollata, diventa due comandi sbagliati.

**1. Ricostruire le impronte degli input già visti**

```
cd <repo>/ton_temporal
python impronte_note.py --sorgente <noto> --audit <lab>/research028/vendor/audit_ton_full.py --uscita impronte_note.npz --esigi-sorgente
```

Esce diverso da zero se la sorgente o la cardinalità non coincidono con i
metadati dichiarati: 92.330 impronte uniche da 211.043 righe.

**2. Inventario dei 23 file**

```
cd <repo>/ton_temporal
python inventario_ton.py --dati <dati> --audit <lab>/research028/vendor/audit_ton_full.py --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json --impronte impronte_note.npz --uscita inventario --riprendi
```

Ripartibile: rilanciandolo salta i file già fatti. Produce
`data_inventory.csv`, `daily_counts.csv`, `hourly_counts.csv`,
`hourly_counts_frt.csv`, `diagnostics.json`, `feature_ready_time.json` e
`per_file/`.

**3. Copertura, con lo strumento del referente, senza modifiche**

```
cd <repo>/ton_temporal
python <lab>/research028/tools/coverage028.py --input inventario/hourly_counts_frt.csv --output inventario/copertura_frt
```

**4. Costruire i flussi ordinati per feature_ready_time**

```
cd <repo>/ton_temporal
python costruisci_flusso.py --dati <dati> --audit <lab>/research028/vendor/audit_ton_full.py --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json --inizio 2019-04-02 --fine-esclusa 2019-04-05 --giorni-extra 2019-04-23 --uscita flussi/A.npz
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
cd <repo>/ton_temporal
python sovrapposizioni_abcd.py --csv <dati> --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json --audit <lab>/research028/vendor/audit_ton_full.py --uscita sovrapposizioni_abcd.json
```

Scorre i 23 CSV una volta, assegna ogni riga al proprio intervallo per
`feature_ready_time` e confronta le impronte dei vettori di feature fra gli
intervalli, usando `normalize_raw` e `fingerprint_rows` del modulo di
riferimento senza modificarli.

**5. Il replay: le quattro configurazioni, per ogni seme**

Le quattro corse del **seme 42**, da incollare così come sono. Per gli altri
semi si cambiano i due `42` di ciascuna riga — il valore di `--seme` e quello
nel nome del file — e i conteggi di `--quanti-aggiornamenti`, che sono quelli
della politica su evidenza dello stesso seme.

```
cd <repo>/ton_temporal
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --seme 42 --uscita replay_C_seme42.json
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --seme 42 --uscita replay_C_calibrato_seme42.json
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --politica evidenza_inversione --seme 42 --uscita replay_C_evidenza_seme42.json
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --politica casuale --quanti-aggiornamenti "lr=54,mlp=20,kan=23" --ammissibili replay_C_calibrato_seme42.json --seme 42 --uscita replay_C_casuale_seme42.json
```

I conteggi da passare al controllo casuale, letti in
`costi.decisioni_della_politica` della corsa su evidenza dello stesso seme:
seme 42 `lr=54,mlp=20,kan=23`; seme 43 `lr=43,mlp=35,kan=16`; seme 44
`lr=44,mlp=45,kan=24`; seme 45 `lr=49,mlp=30,kan=13`; seme 46
`lr=36,mlp=39,kan=21`.

I parametri predefiniti sono quelli della scheda: blocchi di 10.000, budget 1%,
memoria 256, ritardo di un blocco. I semi sono 42, 43, 44, 45 e 46, e **tutte
le venti corse sono pubblicate** in `replay_evidenza/`. I conteggi da passare
al controllo casuale sono quelli applicati dalla politica su evidenza **dello
stesso seme**, che si leggono in `costi.decisioni_della_politica`: cambiarli
renderebbe il controllo non pareggiato. Per la variante scartata,
`--campionamento strati_punteggio`.

**6. I candidati della soglia e la regola di confronto**

```
cd <repo>/ton_temporal
python documenta_soglie.py --iniziale flussi/A.npz --calibrazione flussi/B.npz --semi 42 43 44 45 46 --uscita replay_evidenza/soglie_candidati.json
```

**7. I riepiloghi, i confronti e i costi**

Su Linux e macOS, da dentro `replay_evidenza`:

```
cd <repo>/ton_temporal/replay_evidenza
Z="replay_C_seme42.json replay_C_seme43.json replay_C_seme44.json replay_C_seme45.json replay_C_seme46.json"
B="replay_C_calibrato_seme42.json replay_C_calibrato_seme43.json replay_C_calibrato_seme44.json replay_C_calibrato_seme45.json replay_C_calibrato_seme46.json"
python ../riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json --verifica
python ../riepiloghi_semi.py --rendiconti $Z --confronto $B --uscita confronto_soglia.json --verifica
python ../confronto_politiche.py --ogni-blocco $B --evidenza replay_C_evidenza_seme4?.json --casuale replay_C_casuale_seme4?.json --uscita confronto_politiche.json
python ../costi_aggiornamento.py --rendiconti replay_C_*seme4?.json --uscita costi_aggiornamento.json
```

In PowerShell, dove le liste si scrivono con le virgole e il globbing va risolto
prima:

```
cd <repo>/ton_temporal/replay_evidenza
$Z = "replay_C_seme42.json","replay_C_seme43.json","replay_C_seme44.json","replay_C_seme45.json","replay_C_seme46.json"
$B = "replay_C_calibrato_seme42.json","replay_C_calibrato_seme43.json","replay_C_calibrato_seme44.json","replay_C_calibrato_seme45.json","replay_C_calibrato_seme46.json"
python ..\riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json --verifica
python ..\riepiloghi_semi.py --rendiconti $Z --confronto $B --uscita confronto_soglia.json --verifica
python ..\confronto_politiche.py --ogni-blocco $B --evidenza (Get-Item replay_C_evidenza_seme4*.json).Name --casuale (Get-Item replay_C_casuale_seme4*.json).Name --uscita confronto_politiche.json
python ..\costi_aggiornamento.py --rendiconti (Get-Item replay_C_*seme4?.json).Name --uscita costi_aggiornamento.json
```

Il glob `replay_C_*seme4?.json` prende le venti corse e lascia fuori la variante
scartata, il cui nome finisce in `_strati_punteggio`.

**8. Le figure**

```
cd <repo>/ton_temporal
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello lr --uscita figure
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello mlp --uscita figure
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello kan --uscita figure
```

Tre comandi, uno per modello, che producono i quattordici PNG di `figure/`.
Il terzo stampa anche l'astensione dichiarata su `recupero_kan.png`: sul modello
additivo gli episodi di inversione sono tre, lunghi 7, 2 e 1 blocchi, e uno solo
arriva al minimo di cinque, quindi la figura non viene prodotta.
`figure/LEGGIMI.md` dichiara per ciascuna figura la provenienza e l'impronta dei
rendiconti.

La versione precedente di questo passo era **un comando solo e sbagliato**:
`python curve_replay.py --rendiconto replay_C_seme42.json`. Lo script sta in
`ton_temporal/`, il rendiconto in `ton_temporal/replay_evidenza/`, quindi quel
comando falliva da entrambe le cartelle; e citava la corsa a soglia zero, che è
la ragione per cui le sei figure pubblicate erano state prodotte da quella
corsa invece che dalle corse calibrate.

**9. Le richieste del referente, controllate sul materiale**

```
cd <repo>/ton_temporal
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

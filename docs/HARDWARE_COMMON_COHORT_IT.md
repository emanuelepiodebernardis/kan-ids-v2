# Coorte hardware comune: implementazione di lavoro

Traduzione italiana del protocollo dell'8 settembre 2026: stati e verifiche riportati sono storici; per lo stato attuale prevale `docs/INTEGRATION_20260912_IT.md`.

Il confronto principale dei cinque kernel binari utilizza 500 flussi originali
distinti: 250 attack e 250 normal, alternati in un ordine fisso.
La provenienza completa, lo SHA256 del CSV originale, la selezione senza score e
la ricostruzione del preprocessing conservato sono documentati in
`artifacts/finalization/hardware_cohort.json`.
Questo è un carico di lavoro per le misure hardware; la sua balanced accuracy
non sostituisce i risultati sull'intero test. Questa coorte non ha sostituito
i golden vector né i pesi addestrati. Gli esperimenti binary E2E e multiclass
restano separati.

## File e confini della misura

- `hardware_flow_ids.csv`: ordine, ID della riga nei dati grezzi, label e type.
- `hardware_raw_flows.csv`: record originali di questa coorte, con ID e ordine.
- `hardware_cohort.npz`: input Q12/categorie KAN, input/categorie baseline
  per MLP e input Q7 per DT. Il row ID è la posizione della riga di dati,
  esclusa l'intestazione CSV, a partire da zero.
- `mcu_pio/include/hardware_cohort/`: input generati separatamente per
  KAN, KAN-ML, MLP e DT; KAN coefficients e LUT utilizzano gli stessi input KAN.
- `mcu_pio/src/main_common_latency.cpp`: un unico harness per tutti i cinque kernel.

La latenza inizia dopo il caricamento delle feature preparate nella RAM e
termina dopo l'ottenimento della decisione binaria. Feature engineering,
caricamento degli input da Flash a RAM, verifica del risultato e Serial sono
fuori dall'intervallo cronometrato. Gli accessi del kernel ai pesi/alle tabelle
in Flash rientrano nella misura. Il flush UART prima di ogni misura svuota la
coda della riga precedente. Per tutti i modelli sono identici i 64 inference
di warmup, le 500 righe distinte e il loro ordine; il risultato contiene raw ID,
y_true, pred e il riferimento C atteso. L'overhead del timer è registrato
separatamente e non viene sottratto automaticamente. Il timer del target ha
risoluzione finita; i valori host non vengono utilizzati come latenza hardware.

La modalità energia `EB_COMMON_COHORT` utilizza le prime 20 di queste stesse
righe: 10 attack + 10 normal, già in RAM, ripetute all'interno del batch.
È un carico di lavoro separato con cache calda; la sua energia non può essere
descritta come misura su 500 righe distinte. La configurazione dello strumento
e del batch deve essere registrata secondo
`docs/HARDWARE_ENERGY_PROTOCOL_IT.md`.

## Esportazione e verifica senza flash

```bash
python scripts/export_hardware_cohort.py --source-csv /path/to/train_test_network.csv
python scripts/check_hardware_cohort_host.py
python -m pytest -q tests/test_hardware_cohort_export_gate.py
```

Prima di generare il riferimento C, l'esportazione confronta con il manifest
di accompagnamento gli SHA256 di NPZ, raw ID ordinati, CSV originale,
fixture dei flussi grezzi e preprocessing. Una modifica al NPZ mantenendo
il vecchio manifest viene bloccata. `hardware_cohort_export.json` contiene
gli hash degli header di modello/kernel/generati utilizzati. Un nuovo kernel
o header LUT richiede di ripetere esportazione e controlli host.

Le predizioni attese sono state calcolate sull'host con i kernel C di deployment
conservati. È una verifica di replay rispetto alla stessa implementazione C,
non un oracolo indipendente. L'equivalenza sui golden vector e la qualità dei
modelli vengono verificate separatamente.

Nuovi environment PlatformIO: `<board>_common_<model>` per la latenza e
`<board>_common_energy_<model>` per l'energia; board=`megaatmega2560` o `esp32c3`,
model=`coeff`, `lut14`, `mlcoeff`, `mlp` o `dt5`. Gli environment originali
sono conservati come esperimenti historical/golden separati. Questa procedura
non esegue build per il target né upload.

## Verifiche eseguite direttamente

Dopo l'aggiornamento del kernel MLP e della LUT selezionata solo sul training,
sono stati eseguiti 20/20 controlli host di compilazione/esecuzione: cinque
modelli × due harness × due rami di architettura simulati.
In ogni replay di latenza sono stati confermati 500 ID ordinati, le label e
le predizioni del riferimento C; per l'energia sono stati verificati i primi
20 ID e il checksum batch40.
Test di provenienza separati: 4/4 passed, inclusi NPZ obsoleto, CSV diverso
e row ID riordinati. I log host e il riepilogo sono conservati in
`artifacts/finalization/host_hardware_cohort_checks/`.

Questi risultati non costituiscono build per target AVR/ESP32, prova del numero
di esecuzioni, né misure hardware di latenza, energia, SRAM o Flash. Lo stato
verificato indipendentemente è indicato in `summary.json`; i log delle build
reali e le registrazioni degli strumenti saranno aggiunti dopo la fase hardware
autorizzata.

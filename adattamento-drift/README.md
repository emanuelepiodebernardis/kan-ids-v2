# adattamento-drift — adattamento al drift cross-domain di una KAN integer-only

Sottoprogetto autonomo. **Non condivide i moduli con la radice del
repository**: contiene una propria copia di `kanids/` e `src/`, e va eseguito
da dentro questa cartella. La ragione è che il lavoro ha richiesto modifiche
a `kanids/harmonized.py` e `kanids/datasets.py` (spazio armonizzato a tre e
quattro domini, caricatori per UNSW-NB15 e CIC-IoT-2023) e un modulo nuovo,
`kanids/int_adapt.py`, che non sono state riportate nella radice per non
toccare gli script e i risultati già pubblicati. Le due copie sono quindi
**divergenti per costruzione**; se un giorno si fondono, la direzione è da
qui verso la radice, non il contrario.

## Cosa c'è dentro

`RISULTATI.md` è il documento principale, ed è scritto per essere leggibile
senza il codice: ogni numero ha accanto il metodo che l'ha prodotto, ogni
conclusione la sua misura, e i tentativi falliti sono riportati insieme a
quelli riusciti. Se leggi una cosa sola, leggi quello.

| | |
|---|---|
| `scripts/drift_*.py` | gli esperimenti, uno per sezione di `RISULTATI.md` |
| `scripts/cross_domain.py`, `tre_domini.py` | caricamento armonizzato e terne di domini |
| `scripts/sweep_iperparametri.py` | sweep di calibrazione (`iters`, `ridge`) sulla sola direzione `ton->bot` |
| `kanids/int_adapt.py` | primitive intere: sigmoide a LUT, stima dei guadagni, martingala conformal |
| `mcu/kan_int_adapt.h` | tabelle Q15 e 200 golden vector |
| `mcu/run_int_adapt_check.cpp` | verifica di bit-esattezza contro il riferimento Python |
| `results/*.csv` | un file riassuntivo e uno `*_runs.csv` con un record per run |

Gli script sono checkpointati e riprendibili: interrompere e rilanciare non
ricalcola quello che è già in `artifacts/*.jsonl`.

## Dati

I dataset non sono versionati (~1,3 GB). Servono quattro fonti:

| dominio | file |
|---|---|
| TON_IoT | `train_test_network.csv` |
| BoT-IoT | `UNSW_2018_IoT_Botnet_Full5pc_1..4.csv` |
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| CIC-IoT-2023 | `test.csv` |

Vanno messi in una cartella indicata da `KANIDS_DATA`:

    export KANIDS_DATA=/percorso/ai/dataset
    python scripts/cross_domain.py

**Attenzione a `test.csv`.** Nella famiglia CIC la colonna `Duration` è il
TTL del pacchetto (mediana 64, massimo 255) e `IAT` è corrotta — mescola
timestamp Unix assoluti a veri tempi di interarrivo. Solo il vero
CIC-IoT-2023 ha una colonna `flow_duration` utilizzabile; CICIoMT2024, che
si riconosce dagli attacchi MQTT e dai nomi `*_test.pcap.csv`, non ce l'ha.
`load_harmonized(spazio_cic=...)` sceglie fra lo spazio minimo (3+2 feature,
senza durata) e quello ridotto (6+2, con `flow_duration`) di conseguenza:
verificare quale dei due file si ha in mano prima di interpretare i numeri.

## Verifica dell'header C

    g++ -O2 -I mcu -o /tmp/check mcu/run_int_adapt_check.cpp && /tmp/check
    # golden vector: 200 / logit diversi: 0 / decisioni diverse: 0
    # byte riscritti per l'adattamento: 24 (12 int16)

## Limiti dichiarati

- Quasi tutti i numeri sono su **3 seed**. Dove una conclusione dipende da
  un margine piccolo, `RISULTATI.md` lo dice e indica cosa servirebbe.
- Le misure su microcontrollore (tempo e memoria dell'aggiornamento su
  ATmega2560 ed ESP32-C3) **non sono state fatte**: `RISULTATI.md` riporta
  un modello di costo in operazioni al loro posto.
- Le righe di CIC-IoT-2023 sono finestre scorrevoli di pacchetti, non flussi
  bidirezionali come negli altri tre domini. È una differenza di unità di
  osservazione che nessuna correzione allo spazio delle feature elimina.

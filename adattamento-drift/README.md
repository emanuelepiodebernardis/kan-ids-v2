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

`MECCANISMI.md` è il livello interpretativo: per ogni risultato dice il
**meccanismo** che lo produce e la **giustificazione** della procedura che
l'ha misurato. È il documento da cui si scrive l'articolo, e i punti dove il
meccanismo è già noto in letteratura sono marcati `[rif.]` perché le
citazioni si innestino senza riscrivere il testo.

`CRONOLOGIA.md` conserva le versioni superate delle sezioni riscritte e
perché sono cadute: è il materiale della sezione «limiti» dell'articolo, e
serve perché si possa verificare che nulla è sparito in silenzio.

`RISULTATI.md` è il registro dei numeri — stato corrente, ogni affermazione
una volta sola — ed è scritto per essere leggibile senza il codice: ogni numero ha accanto il metodo che l'ha prodotto, ogni
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

## Protocollo di valutazione

`kanids/valutazione.py` divide il target in **validation (30 %)** e **test
(70 %)**, e da entrambe toglie le righe spese per l'adattamento. La
partizione dipende dal solo seed, non da quali righe la regola ha
selezionato: dentro uno stesso seed ogni metodo e ogni budget vedono lo
stesso test set, quindi i confronti appaiati sono davvero appaiati.

**Le costanti si scelgono sulla validation, mai sul test.** Chi esegue uno
sweep gira in modo selezione (`KANIDS_MODO=selezione`) e leggere il test
solleva `AccessoAlTestVietato`: e' un errore di esecuzione, non una
convenzione da ricordare. Dove non esiste un complemento da ritagliare --
`drift_graduale.py` valuta in modo prequenziale, ogni batch prima valutato
e poi usato per adattare, quindi l'intero stream e' il test -- la scelta si
fa su **seed di calibrazione** (90-99) disgiunti dai seed di riporto
(42-51), e lo sweep si ferma se gli si passano questi ultimi.

I risultati prodotti prima di questa correzione sono in
`results/protocollo_v1/` e `artifacts/protocollo_v1/`. Restano validi come
valutazione -- il complemento delle righe selezionate non era contaminato --
ma stanno su un test set diverso, quindi non vanno mescolati con i numeri
nuovi. Cosa era stato scelto guardando i numeri sbagliati, e cosa succede
rifacendo la scelta onestamente, e' in
`results/RISELEZIONE_IPERPARAMETRI.md`.

## Rigenerare i risultati

    python rigenera.py --dati /percorso/ai/dataset --lista   # cosa farebbe
    python rigenera.py --dati /percorso/ai/dataset           # stage del paper
    python rigenera.py --dati /percorso/ai/dataset --tutto   # anche i secondari

In alternativa a `--dati`, la variabile `KANIDS_DATA` -- con la sintassi
della shell in uso, che su Windows non e' la stessa fra le due:

    $env:KANIDS_DATA = "C:\percorso\ai\dataset"   # PowerShell (`set` non basta)
    set KANIDS_DATA=C:\percorso\ai\dataset         # cmd.exe
    export KANIDS_DATA=/percorso/ai/dataset       # bash/zsh

Gli stage sono ordinati dal piu' economico al piu' caro e gli script sono
checkpointati: interrompere e rilanciare riprende senza ricalcolare.

## Test

    python -m pytest tests/ -q

Fra questi, le regressioni che impediscono al difetto di rientrare: la build
fallisce se uno script torna a usare un complemento unico come insieme di
valutazione, se lo sweep esce dal modo selezione, o se la guardia sulla
scrittura dell'header C perde una delle sue condizioni.

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

# mcu_pio — Benchmark firmware KAN-IDS (integer-only)

Progetto **PlatformIO** che replica il protocollo di benchmark su
microcontrollore del paper *Electronics 2026, 15, 2869*: **500 inferenze
temporizzate** per modello (250 con input di classe **attacco** + 250 di
classe **normale**, con vettori pre-normalizzati in Flash), statistiche di
latenza calcolate **a bordo**, misura di **SRAM**, verifica delle
predizioni contro i valori attesi, e hook opzionale per la misura di
**energia** via INA219.

Ogni sorgente in `src/` copre entrambi i target tramite `#ifdef`.

---

## 1. Cosa contiene

```
mcu_pio/
├── platformio.ini          # 13 env su 2 schede: megaatmega2560, esp32c3
├── src/                    # 7 firmware, uno per variante di modello
│   ├── main.cpp            # KAN-LUT integer (env di default)
│   ├── main_coeff.cpp      # KAN single-layer a coefficienti (254 B)
│   ├── main_mlcoeff.cpp    # KAN multi-layer (5,2 KB)
│   ├── main_mc.cpp         # KAN multiclasse 10 classi (8,3 KB)
│   ├── main_e2e.cpp        # catena end-to-end binaria dai contatori grezzi
│   ├── main_mc_e2e.cpp     # catena end-to-end a 10 classi
│   └── main_dt5.cpp        # albero profondo 5, il concorrente sul Pareto
├── include/                # header dei modelli + golden vector
├── host_check/             # verifica offline con g++ (no MCU necessario)
│   ├── arduino_stub.h      # stub minimale di Arduino.h
│   ├── avr/pgmspace.h      # stub PROGMEM (solo per il check host)
│   ├── Wire.h              # stub I2C (solo per il check host)
│   └── run_*_check.cpp     # 6 harness, uno per kernel
├── wokwi.toml              # simulazione senza hardware (vedi §8)
├── diagram.json            # schema Wokwi: Arduino Mega 2560
├── diagram.esp32c3.json    # schema Wokwi: ESP32-C3-DevKitM-1
└── README.md
```

> Gli header in `include/` erano copie non modificate di quelli in `../mcu/`.
> Non lo sono piu': quattro di essi (`dt5_model.h`, `kan_e2e_int.h`,
> `kan_mc_e2e_int.h`, `test_vectors.h`) sono stati corretti con `PROGMEM` e i
> relativi `pgm_read_*`, perche' senza quella qualificazione su AVR le tabelle
> finivano in SRAM invece che in Flash — 6,3 KB e 7,3 KB sugli 8 KB del Mega,
> cioe' due firmware che non partivano. Se rigeneri i modelli, riporta la
> correzione anche nella copia in `../mcu/`.

### Scelta del modello: variante INT

Il firmware usa **`kan_ids_layer_int.h`** (integer-only), non la variante
float, perché:

1. è **autoconsistente**: `KANI_TABLE` contiene tutti i valori già
   pre-scalati (int16), nessuna dipendenza da tabelle esterne;
2. segue il vincolo di **eliminare il float dall'inferenza**: lookup int16
   + interpolazione intera + accumulo int32 + soglia intera;
3. la sigmoid non serve per la decisione binaria:
   `sigmoid(z) >= 0.5  ⇔  z >= 0`, quindi la predizione è un confronto
   intero. L'unica operazione float è la conversione iniziale di ogni
   input in Q16.16 (una per input).

Verificato su host: la variante int riproduce **esattamente** le decisioni
della variante float (stessa accuratezza, stessi mismatch — vedi §6).

---

## 2. Requisiti

- [PlatformIO Core](https://platformio.org/install/cli) (`pip install platformio`)
- Per il flash reale: una scheda **Arduino Mega 2560** o
  **ESP32-C3-DevKitM-1** e il relativo cavo USB.
- (Opzionale, energia) un modulo **INA219** collegato via I2C.

---

## 3. Compilare

> **Stato della verifica, dichiarato.** Senza hardware sono stati controllati:
> i sei kernel di inferenza contro il riferimento Python (200/200 bit-esatti
> ciascuno), la compilazione di tutti e sette i firmware in entrambi i rami
> `#ifdef` con g++, e la compilazione reale per ATmega2560 con `avr-gcc`, da
> cui vengono le cifre di Flash e SRAM piu' sotto. **Non** e' stato eseguito
> `pio run` con i toolchain veri: il registro PlatformIO non era raggiungibile
> dall'ambiente di sviluppo. Le due cose non coincidono — il core
> Arduino-ESP32 porta header e macro che uno stub non riproduce — quindi la
> prima build su una macchina con rete va considerata parte della verifica,
> non una formalita'.

```bash
cd mcu_pio

# Arduino Mega 2560
pio run -e megaatmega2560

# ESP32-C3
pio run -e esp32c3
```

Al primo avvio PlatformIO scarica piattaforma e toolchain da
`registry.platformio.org`. **In reti con proxy restrittivo il download può
fallire (HTTP 403)**: in tal caso usa la verifica offline con g++ (§6).

---

## 4. Flashare

Collega la scheda via USB, poi:

```bash
# Mega 2560
pio run -e megaatmega2560 -t upload

# ESP32-C3
pio run -e esp32c3 -t upload
```

Se la porta non viene rilevata automaticamente, indicala:

```bash
pio run -e esp32c3 -t upload --upload-port /dev/ttyUSB0    # Linux
pio run -e esp32c3 -t upload --upload-port COM5            # Windows
```

Su ESP32-C3, se l'upload non parte, tieni premuto **BOOT** all'inizio del
flash. Su Mega 2560 di solito non serve alcun pulsante.

---

## 5. Raccogliere i dati

Il benchmark gira **una volta all'avvio** (in `setup()`) e stampa un CSV
su seriale a **115200 baud**. Apri il monitor:

```bash
pio device monitor -e megaatmega2560          # o -e esp32c3
```

Per salvare l'output su file:

```bash
pio device monitor -e esp32c3 --baud 115200 | tee run_esp32c3.csv
```

> Riavvia la scheda (tasto RESET) per rieseguire il benchmark, così puoi
> catturare l'output completo dall'inizio.

### Formato dell'output

Righe di **commento** (iniziano con `#`) con metadati e SRAM, poi una riga
di **header CSV**, 500 righe **dati**, una riga **SUMMARY**:

```
# KAN-IDS benchmark (integer-only) — Electronics 2026,15,2869
# target=ESP32C3
# model E=10 K=8 L=64 FP_BITS=9
# sram_free_before_bytes=...
# sram_free_after_bytes=...
# sram_model_cost_bytes=...
# n_attack_vectors=20
# n_normal_vectors=20
# energy=disabled
phase,idx,vec_index,label_expected,pred,logit_int,match,latency_us
ATTACK,0,0,1,1,1250,1,12
...
NORMAL,249,38,0,0,-930,1,11
SUMMARY,n_inferences=500,correct=...,accuracy_pct=...,lat_mean_us=...,lat_std_us=...,lat_min_us=...,lat_max_us=...,sram_model_cost_bytes=...,sram_free_after_bytes=...,energy_total_mJ=NA,energy_per_inf_uJ=NA
# END
```

Colonne dati: fase (`ATTACK`/`NORMAL`), indice nel blocco, indice del
vettore usato, etichetta attesa, predizione, logit intero, `match` (1/0),
latenza in µs. La riga `SUMMARY` è `key=value` separati da virgola, facile
da parsare.

### Cosa aspettarsi

- **Predizioni**: sui 40 vettori di riferimento il modello ottiene ~97.5%
  di accuratezza (un vettore normale borderline viene classificato come
  attacco — è un errore reale del modello, presente anche nella variante
  float, non un artefatto di quantizzazione). Il benchmark cicla su
  20 vettori attacco e 20 normali per riempire i due blocchi da 250.
- **Latenza**: pochi µs per inferenza su ESP32-C3 (32-bit, clock alto),
  significativamente di più sul Mega 2560 (AVR 8-bit @16 MHz, LUT in
  PROGMEM). I valori esatti li fornisce la riga SUMMARY.
- **SRAM**: `sram_model_cost_bytes` è il costo in RAM delle strutture
  costruite a runtime (soli 2 array di indici `int8`). Il modello e i test
  vector stanno in **Flash/PROGMEM** — cosa vera per costruzione solo dopo la
  correzione descritta in §1: prima lo era per le varianti `_coeff` e
  `_mlcoeff`, non per `_dt5`, `_e2e` e l'env di default. Il
  firmware usa **statistiche in streaming** (somma e somma dei quadrati),
  senza mai allocare un buffer da 500 campioni → rispetta il budget di
  8 KB del Mega.

---

## 6. Verifica offline con g++ (senza MCU / senza rete)

Se i toolchain PlatformIO non sono scaricabili, puoi comunque:

### a) Compile-check del firmware (entrambe le varianti #ifdef)

```bash
cd mcu_pio

# variante AVR
g++ -fsyntax-only -std=c++11 -Iinclude -Ihost_check -DHOST_CHECK -D__AVR__ \
    -include host_check/arduino_stub.h src/main.cpp

# variante ESP32
g++ -fsyntax-only -std=c++11 -Iinclude -Ihost_check -DHOST_CHECK -DARDUINO_ARCH_ESP32 \
    -include host_check/arduino_stub.h src/main.cpp

# entrambe con ENABLE_INA219 (deve compilare anche con energia attiva)
g++ -fsyntax-only -std=c++11 -Iinclude -Ihost_check -DHOST_CHECK -D__AVR__ -DENABLE_INA219 \
    -include host_check/arduino_stub.h src/main.cpp
```

### b) Harness sui test vector reali (g++ vero, eseguibile)

Verifica che l'inferenza — la **stessa funzione pura** usata dal firmware —
dia le predizioni attese:

```bash
cd mcu_pio/host_check
g++ -O2 -I../include run_host_check.cpp -o run_host_check
./run_host_check
```

Stampa, per ogni vettore, etichetta attesa vs predizione + `OK/MISMATCH` e
un riepilogo di accuratezza.

---

## 7. Misura di energia (INA219, opzionale)

Il firmware compila **anche senza** energia (default). Per abilitarla,
aggiungi il flag `-DENABLE_INA219` nell'env desiderato in `platformio.ini`:

```ini
[env:esp32c3]
platform = espressif32
board = esp32-c3-devkitm-1
framework = arduino
build_flags = ${env.build_flags} -O2 -DENABLE_INA219
```

Il blocco `#ifdef ENABLE_INA219` legge l'INA219 via **I2C raw** (registri
`0x00..0x05`, nessuna libreria esterna), integra `potenza × dt` durante il
loop di inferenza e aggiunge alla riga SUMMARY:

```
...,energy_total_mJ=<mJ totali>,energy_per_inf_uJ=<µJ per inferenza>
```

**Collegamento INA219 (I2C):**

| INA219 | Mega 2560 | ESP32-C3 |
|--------|-----------|----------|
| VCC    | 5V        | 3V3      |
| GND    | GND       | GND      |
| SDA    | pin 20    | GPIO 8   |
| SCL    | pin 21    | GPIO 9   |

Lo shunt (V+ / V−) va **in serie** sull'alimentazione della scheda sotto
misura. Calibra `INA219_CURRENT_LSB_A`, `INA219_POWER_LSB_W` e
`INA219_CAL_VALUE` in `src/main.cpp` in base al tuo shunt (default: shunt
0.1 Ω, range 32V/2A).

> **Alternativa con libreria Adafruit:** se preferisci non usare l'accesso
> raw, aggiungi `lib_deps = adafruit/Adafruit INA219` in `platformio.ini` e
> sostituisci le funzioni `ina219_*` con le chiamate della libreria
> (`Adafruit_INA219 ina; ina.begin(); ina.getPower_mW();`). L'integrazione
> `potenza × dt` resta identica.

## 8. Verifica preliminare su Wokwi (senza hardware)

`wokwi.toml` + `diagram.json` sono pronti per l'estensione Wokwi di VS Code.
Servono a controllare che il firmware parta davvero, stampi il CSV sulla
seriale e che le predizioni coincidano con i golden vector — non a misurare la
latenza, che in simulazione non è quella del silicio, né l'energia, che non è
simulabile.

```bash
cd mcu_pio
pio run -e megaatmega2560_coeff        # il binario deve esistere prima
# VS Code: F1 -> "Wokwi: Start Simulator"
```

Per l'ESP32-C3: `pio run -e esp32c3_coeff`, poi copiare
`diagram.esp32c3.json` su `diagram.json` e scambiare le due righe
`firmware`/`elf` in `wokwi.toml` con quelle del blocco commentato.

La variante di default è `_coeff` perché è l'unica che sta comodamente su
entrambe le schede e porta a bordo i 200 golden vector con le predizioni
attese, quindi il confronto è automatico e non serve leggere i numeri a mano.

---

## Terza variante: coefficienti B-spline full-integer (254 B)

La variante `main_coeff.cpp` implementa la compilazione a coefficienti della
KAN binaria a 14 feature (F1 0.9826, modello da 254 byte nell'header C
compilato — 250 secondo lo script di compilazione, che non conta i 4 byte
della tabella di offset categorici; zero float):

```
pio run -e megaatmega2560_coeff -t upload    # oppure -e esp32c3_coeff
pio device monitor --baud 115200
```

- Modello: `include/kan14_coeff_int8.h` (generato da `scripts/export_kan14_coeff_c.py`)
- Kernel: `include/kan14_coeff_infer.h` — B-spline cubica in forma matriciale,
  basi Q15 via Horner intero, coefficienti int8, tabelle categoriche int8,
  decisione a segno. Traduzione 1:1 della simulazione numpy verificata.
- Test vector: 200 flussi reali (100 attacco + 100 normale) con predizioni
  attese dalla simulazione bit-fedele.
- Verifica host: `g++ -O2 -o check host_check/run_coeff_check.cpp && ./check`
  → atteso 200/200.

Confronto atteso con le altre varianti on-board: stessa accuratezza della
LUT (99.95% agreement col float) con 1/22 della memoria; latenza da misurare
(lookup+interp vs Horner: ~2× operazioni per edge, entrambe O(1)).

## Varianti 4 e 5: multi-layer (5 KB) e multiclass (8 KB)

- `main_mlcoeff.cpp` — multi-layer binario F1 0.9974, full-integer (~5 KB):
  `pio run -e megaatmega2560_mlcoeff -t upload` (o `esp32c3_mlcoeff`)
- `main_mc.cpp` — multiclass 10 classi macro-F1 0.9409, full-integer (~8 KB):
  `pio run -e esp32c3_mc -t upload`
- Verifica offline: `g++ -O2 host_check/run_ml_coeff_check.cpp && ./a.out`
  (atteso 200/200) e idem con `run_mc_coeff_check.cpp`.
- Header rigenerabili con `scripts/export_kan14_ml_coeff_c.py` e
  `scripts/export_kan14_mc_coeff_c.py` dai pesi in `models/`.

# mcu_pio — Benchmark firmware KAN-IDS (integer-only)

> **Finalizzazione Paper 1 — NOT_HARDWARE_MEASURED (8 settembre 2026).**
> I build, conteggi e check RC3 descritti nelle sezioni storiche qui sotto
> sono evidenze salvate dell'autore. I nuovi header e gli environment common
> richiedono build target aggiornati; nessuna misura fisica nuova e' stata
> acquisita. Host e Wokwi servono alla verifica funzionale, non forniscono
> latenza, energia o peak RAM fisici. I comandi di upload nelle ricette RC3
> non costituiscono autorizzazione a flashare: serve conferma del supervisore.
>
> Per il confronto corrente usare gli environment `<board>_common_<model>`
> e `<board>_common_energy_<model>`, dove `<board>` e' `megaatmega2560` oppure
> `esp32c3`, e `<model>` e' `coeff`, `lut14`, `mlcoeff`, `mlp` oppure `dt5`.
> Il cohort contiene 500 flow ID unici (250+250), trasformati separatamente
> dai preprocessori fissati dei modelli. La misura e' prepared features in
> RAM → decisione; l'energia ripete i primi 20 flussi (10+10), warm cache.
> La provenienza e' in `../artifacts/finalization/hardware_cohort.npz` e
> `../artifacts/finalization/hardware_cohort_export.json`.
>
> La LUT corrente e' L=1025, 20.554 B: la scelta storica L=257/5.194 B usava
> margini dal test. Il nuovo protocollo e' `../results/lut_selection_protocol.json`.
> La configurazione h=16/g=8 e' una preferenza di deployment dichiarata.
> Protocollo cohort: [HARDWARE_COMMON_COHORT_IT.md](../docs/HARDWARE_COMMON_COHORT_IT.md).
> Protocollo e prerequisiti strumentali:
> [HARDWARE_ENERGY_PROTOCOL_IT.md](../docs/HARDWARE_ENERGY_PROTOCOL_IT.md).

Progetto **PlatformIO** che replica il protocollo di benchmark su
microcontrollore del paper *Electronics 2026, 15, 2869*: **500 inferenze
temporizzate** per modello (250 con input di classe **attacco** + 250 di
classe **normale**, con vettori pre-normalizzati in Flash), statistiche di
latenza calcolate **a bordo**, misura di **SRAM**, verifica delle
predizioni contro i valori attesi, e hook opzionale per la misura di
**energia** via INA219 (hook storico, escluso dai numeri del Paper 1).

Ogni sorgente in `src/` copre entrambi i target tramite `#ifdef`.

---

## 1. Cosa contiene

```
mcu_pio/
├── platformio.ini          # 29 env su 2 schede: megaatmega2560, esp32c3
├── src/                    # 10 firmware, uno per variante di modello
│   ├── main.cpp            # KAN-LUT integer (env di default)
│   ├── main_coeff.cpp      # KAN single-layer a coefficienti (254 B)
│   ├── main_mlcoeff.cpp    # KAN multi-layer (5,1 KB)
│   ├── main_mc.cpp         # KAN multiclasse 10 classi (8,1 KB)
│   ├── main_e2e.cpp        # catena end-to-end binaria dai contatori grezzi
│   ├── main_mc_e2e.cpp     # catena end-to-end a 10 classi
│   ├── main_dt5.cpp        # albero profondo 5, il concorrente sul Pareto
│   ├── main_mlp.cpp        # MLP piccolo 16 nascosti, la baseline densa
│   └── main_lut14.cpp      # la stessa KAN single-layer, campionata (5,2 KB)
├── include/                # header dei modelli + golden vector
├── host_check/             # verifica offline con g++ (no MCU necessario)
│   ├── arduino_stub.h      # stub minimale di Arduino.h
│   ├── avr/pgmspace.h      # stub PROGMEM (solo per il check host)
│   ├── Wire.h              # stub I2C (solo per il check host)
│   └── run_*_check.cpp     # 8 harness, uno per kernel
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

> **Stato della verifica.** **Tutti** gli environment di `platformio.ini`
> compilano con PlatformIO, senza warning, e le loro dimensioni di Flash e
> SRAM stanno in `results/firmware_size.csv`: le misura
> `python reproduce.py --stage firmware-size`, che le scrive anche nella
> tabella del README della radice (sezione *Flash and SRAM per variant*). Il
> numero degli environment non e' ripetuto qui apposta: era scritto a mano in
> quattro punti e ne ha sbagliati tre. Sono state verificate anche, senza
> hardware: **sei** kernel di inferenza contro il riferimento Python,
> bit-esatti su 200 golden vector ciascuno (`coeff`, `ml_coeff`, `mc_coeff`,
> `e2e`, `mc_e2e`, `lut`) piu' l'MLP. L'harness della LUT campionata verifica
> in piu' che le sue decisioni coincidano con quelle della versione a
> coefficienti su tutti e 200 i vettori. L'harness `run_host_check` e' invece
> di natura diversa: confronta le predizioni della LUT **storica** del paper
> (`kan_ids_layer_int.h`, un altro modello) con le **etichette reali** e da' 39/40, dove l'unico scarto e'
> un errore del modello presente anche nella variante float (vedi §6), non un
> difetto del kernel. Chiamarlo bit-esatto sarebbe scorretto. Verificata
> inoltre la compilazione di tutti i firmware in entrambi i rami `#ifdef`
> con g++, e la compilazione per
> ATmega2560 con `avr-gcc`, che misura il firmware **senza** il core Arduino e
> serve ad attribuire una variazione al codice invece che al runtime. E' stato
> ricompilato con PlatformIO anche lo stato precedente alla correzione
> `PROGMEM` (worktree su 4a9b235), per misurare il prima e il dopo con la
> stessa toolchain invece di derivarne uno: 79,2% e 92,0% della SRAM contro
> 2,5%. Restano fuori solo latenza ed energia, che richiedono le schede.

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

## 7. Misura di energia: protocollo corrente

Usare [HARDWARE_ENERGY_PROTOCOL_IT.md](../docs/HARDWARE_ENERGY_PROTOCOL_IT.md)
e il template `../templates/energy_acquisition.json`. Prima di collegare lo
strumento occorrono modello e revisione delle board, alimentazione e rail
misurato, modello dello strumento, shunt/range, frequenza di campionamento,
precisione, esportazione delle tracce e compatibilita' dei marker digitali.
I default Mega 22/24 ed ESP32-C3 GPIO3/GPIO4 descrivono il sorgente, non uno
schema di collegamento validato per lo strumento del supervisore.

La metrica principale e' `integral_active(V(t)*I(t) dt) / N`. Il reference
loop e' un busy loop con branch, decrement e nop, non idle a energia nulla.
L'eventuale differenza rispetto a quel riferimento usa le durate effettive;
puo' essere negativa e non va interpretata o ritagliata come energia fisica
negativa. Un solo marker attivo individua la finestra attiva, ma tutto il
resto della traccia non puo' essere assunto reference senza sincronizzazione.

Il ciclo attivo usa un accumulatore locale e uno store volatile alla fine.
`checksum_ok`, calibrazione e flag delle finestre sono controlli necessari;
un checksum da solo non prova formalmente che il compilatore abbia eseguito
ogni chiamata prevista. Conservare versione del compilatore, flag/LTO,
source/binary hash e verifica del loop compilato. Non disabilitare
l'ottimizzazione del modello per ottenere uno speedup atteso.

Il nuovo workload `_common_energy_` usa gli stessi venti flow ID (10+10),
lo stesso ordine, batch e ripetizioni per i modelli confrontati. Il costo
include il ciclo della misura e l'accesso ai vettori in RAM; le tabelle dei
modelli possono essere lette da Flash. I percorsi E2E e multiclass richiedono
un protocollo e una tabella separati.

L'hook `ENABLE_INA219` nei firmware storici di latenza integra anche
intervalli con Serial/I2C. E' mantenuto per provenienza e non deve produrre i
valori energetici dell'articolo. I risultati vanno ricavati dalle tracce
esterne sincronizzate, con `../tools/aggregate_energy_trace.py`.

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

La latenza e' da misurare sullo stesso cohort. La LUT storica di default
appartiene a un altro modello. Per la LUT della stessa KAN usare il protocollo
di selezione corrente e i byte dell'header attuale; non dedurre la latenza
da un conteggio approssimato delle operazioni.

## Varianti 4 e 5: multi-layer (5 KB) e multiclass (8 KB)

- `main_mlcoeff.cpp` — multi-layer binario F1 0.9974, full-integer (~5 KB):
  `pio run -e megaatmega2560_mlcoeff -t upload` (o `esp32c3_mlcoeff`)
- `main_mc.cpp` — multiclass 10 classi, full-integer (8.268 B; macro-F1
  0.9384 dello stato in `results/kan_ml_cat_mc_real.csv`, 0.9388 misurato
  all'export sulla simulazione intera; lo stato canonico e' committato in
  `models/kan14_multiclass_multilayer.pkl`, vedi il paragrafo 9):
  `pio run -e esp32c3_mc -t upload`
- Verifica offline: `g++ -O2 host_check/run_ml_coeff_check.cpp && ./a.out`
  (atteso 200/200) e idem con `run_mc_coeff_check.cpp`.
- Header rigenerabili con `scripts/export_kan14_ml_coeff_c.py` e
  `scripts/export_kan14_mc_coeff_c.py` dai pesi in `models/`.

## Variante 6: MLP piccolo denso (760 B)

`main_mlp.cpp` — la baseline che mancava sul dispositivo. Fino alla revisione
il confronto on-board era fra albero, KAN single-layer, KAN multi-layer e
LUT: la rete densa, cioè proprio l'architettura che la KAN vuole sostituire,
esisteva solo in cross-validation e con i byte **stimati** a un byte per
parametro. Adesso è compilata e misurabile come le altre.

```
pio run -e megaatmega2560_mlp -t upload      # oppure -e esp32c3_mlp
pio device monitor --baud 115200
```

- Modello: `include/mlp16_int8.h` (generato da `scripts/export_mlp_int_c.py`)
- Kernel: `include/mlp16_infer.h` — 10 ingressi in Q12, ReLU, accumulatori
  int32, decisione a segno. Traduzione 1:1 della simulazione numpy.
- Verifica host: `g++ -O2 -o check host_check/run_mlp_check.cpp && ./check`
  → atteso 200/200.

Due dettagli che riguardano cosa viene misurato, non la matematica.

**Il one-hot non esiste a bordo.** Il design di scikit-learn ha 42 colonne:
10 numeriche più 32 colonne one-hot per le quattro feature categoriche. A
bordo ogni categorica seleziona *una riga* di `MLP16_CAT`: una somma per
neurone nascosto invece di 32 moltiplicazioni. Misurare il one-hot esplicito
avrebbe misurato una trasposizione ingenua, non l'MLP.

**Niente aritmetica a 64 bit.** L'attivazione nascosta viene ridotta di
`MLP16_HSHIFT` bit prima del secondo layer, così ogni accumulatore sta in
int32. Con l'accumulatore a 64 bit il kernel chiamerebbe `__adddi3`,
`__ashrdi3` e `__mulsidi3` di libgcc su AVR, e la latenza misurata sarebbe
quella di un tipo che il processore non ha. Lo shift è fissato dal bound
calcolato all'export sui pesi quantizzati e su |xq| ≤ 2¹², non scelto sui
dati, e costa pochi bit su ventitré.

> **La stessa cura è arrivata dopo sui kernel KAN.** Quando questa nota è
> stata scritta, i tre kernel a coefficienti chiudevano ancora ogni edge con
> `((int64_t)acc * MULT) >> 15`: dieci chiamate a `__mulsidi3` + `__ashrdi3`
> per inferenza nella single-layer, centosettantasei nel multi-layer. Il
> confronto di latenza fra KAN e MLP sarebbe stato fra un kernel scritto per
> il target e uno no. Ora la moltiplicazione Q15 passa da
> `include/q15_mul.h`, che calcola `(a·m) >> 15` in solo int32 con
> l'identità `(a>>15)·m + ((a & 32767)·m >> 15)`: **esatta**, non
> approssimata — stesso intero su 200.000 ingressi casuali, verificato
> confrontando i due kernel compilati. Sul programma di prova il firmware
> linkato passa da 2.008 a 1.858 B di `.text`, perché le due routine a 64 bit
> non vengono più linkate affatto.

**Byte: 760, non 705.** La stima table-driven contava un byte per parametro.
Il conteggio sull'header che il compilatore compila davvero è 760: i bias del
primo layer sono int32 (64 B) e la tabella categorica ha una riga per codice.
È lo stesso genere di scarto già visto sull'albero, dove la stima diceva 141 B
e la misura 285. Il numero si legge da `results/footprint.csv`, che a sua
volta lo legge dall'header con `scripts/c_footprint.py`.

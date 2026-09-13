# KAN-IDS Paper 1: protocollo di misura hardware dell'energia

Traduzione italiana del protocollo dell'8 settembre 2026: stati e verifiche riportati sono storici; per lo stato attuale prevale `docs/INTEGRATION_20260912_IT.md`.

Versione del protocollo: 2026-09-08, di lavoro. La base sono i sorgenti esaminati
`mcu_pio/src/main_energy.cpp`, `mcu_pio/platformio.ini`, `tests/test_energy_firmware.py`
e il documento di handoff `04_HARDWARE_AND_EVIDENCE_PLAN_RU.md`.
Questo documento non contiene risultati di misura di Mega 2560 o ESP32-C3.
Le impostazioni dello strumento e la durata finale delle finestre saranno
fissate dopo aver chiarito quale attrezzatura è disponibile.

## Informazioni necessarie prima della fase hardware

1. Modelli e revisioni esatti di entrambe le schede, foto delle marcature;
   per ESP32-C3: SuperMini, DevKitM-1 o un'altra scheda. Il target PlatformIO
   attuale `esp32-c3-devkitm-1` da solo non conferma la compatibilità di un'altra scheda.
2. Nome e modello del misuratore di corrente/energia, oscilloscopio/registratore,
   gamme di corrente disponibili o shunt, frequenza di esportazione, banda,
   acquisizione simultanea di tensione e corrente, formato grezzo supportato.
   È utile un breve esempio di esportazione senza eseguire il modello.
3. Disponibilità di due ingressi/canali digitali per i marcatori, loro
   sincronizzazione con i campioni analogici, tensioni ammesse agli ingressi.
   I GPIO di Mega ed ESP32-C3 hanno livelli differenti; lo schema di collegamento
   non può essere scelto soltanto in base al nome della scheda.
4. Alimentazione della scheda e punto di misura della corrente: ingresso
   dell'intera scheda, linea di alimentazione separata della MCU, USB o sorgente
   esterna; tensione, collegamento USB/Serial durante l'acquisizione, presenza
   di altri percorsi di alimentazione e periferiche. Servono le porte COM
   attuali per uno smoke test concordato, ma il flash richiede una conferma separata.

Questi sono gli ingressi effettivamente mancanti. Per il protocollo energetico
non occorre inviare nuovi dataset o addestrare nuovamente i modelli.

## Comportamento effettivo del firmware

Valori predefiniti: `EB_BATCH=2000`, `EB_REPS=5`, `EB_CACHE=20`, `EB_TOLL_PERMILLE=50`.
Sono disponibili varianti coefficients 1L, coefficients ML, binary E2E, DT5,
MLP16, sampled-LUT e multiclass coefficients. L'ultima è prevista soltanto
per ESP32-C3. Al momento non esiste un environment energia separato per multiclass E2E.

| Fase | Comportamento dell'harness originale |
| --- | --- |
| Preparazione | Serial 115200, ritardo di 1500 ms; caricamento degli input in RAM; calibrazione del reference; checksum atteso; 64 inference di warmup |
| Input | Nella modalità originale, i primi 10 vettori della prima metà del set golden specifico della variante e i primi 10 della seconda; indici identici non dimostrano che i flussi originali siano gli stessi tra i modelli |
| Calibrazione | Lo stesso `eb_nop_loop` eseguito successivamente nel reference: decremento di `volatile uint32_t` e `nop`; il numero di iterazioni raddoppia a partire da 20000 fino a ottenere una finestra di almeno 50 ms; la velocità è memorizzata in Q8 |
| Calibrazione incompleta | Dopo un numero limitato di tentativi viene utilizzato un fallback e `calibration_ok=0`; un reference di questo tipo non è valido per il risultato incrementale principale |
| Marcatori | Active e reference sono impulsi high separati; per impostazione predefinita Mega digital 22/24, ESP32-C3 GPIO 3/4. `EB_NO_PIN` disabilita entrambi; questa modalità non è adatta al presente protocollo di acquisizione della traccia |
| Prima delle finestre | Inizializzazione dei marcatori LOW, stampa dell'intestazione, `Serial.flush()`, ritardo di 200 ms; l'acquisizione deve contenere proprio le coppie complete successive, non i transitori di avvio/calibrazione |
| Finestra active | HIGH active; `micros()`; kernel del batch, indice e accumulatore del checksum; `micros()`; LOW active. I dati di input sono in RAM, ma su AVR i kernel leggono pesi/tabelle da PROGMEM |
| Reference | HIGH reference; `micros()`; busy loop calibrato; `micros()`; LOW reference. Il numero di iterazioni è determinato dalla durata active misurata immediatamente prima |
| Ripetizioni | Active/reference si alternano; nessuna chiamata Serial/I2C tra le coppie; l'ordine è sempre active, poi reference. Interrupt e servizi in background non vengono disabilitati |
| Dopo le finestre | Tutte le righe CSV e SUMMARY; `loop()` è vuoto. In un singolo boot ci sono più ripetizioni, ma non sono riavvii indipendenti della scheda |

All'interno delle finestre l'harness non chiama Serial/Wire. Questo non dimostra
l'assenza di attività in background del runtime ESP32. Lo stato di radio,
clock, periferiche e USB deve essere registrato per ogni build/scheda.
Il busy reference non corrisponde allo sleep o a un carico fisicamente nullo
e può consumare più dell'active.

Il numero di vettori, la composizione dei flussi e il loro ordine vengono
fissati prima del confronto. Per il confronto principale servono un insieme
comune di flussi originali e le rappresentazioni corrispondenti di ogni modello;
le feature preparate vengono confrontate separatamente dalla catena E2E
da contatori a decisione. Una fixture bilanciata è utile per un confronto
controllato, ma non rappresenta la frequenza degli attacchi in una rete reale.
Nel release occorre conservare i flow ID ordinati, l'hash dell'insieme e l'hash
delle predizioni attese specifiche di ciascuna variante. La nuova modalità
comune non va mescolata con le vecchie misure su golden vector eterogenei.

Nella versione di lavoro la modalità comune è già implementata con il flag
`EB_COMMON_COHORT`. Gli environment
`<board>_common_energy_{coeff,lut14,mlcoeff,mlp,dt5}` utilizzano le prime 20 righe
di `artifacts/finalization/hardware_cohort.npz`: si alternano 10 attack e
10 normal degli stessi raw-flow ID usati nella latenza comune.
`<board>` è `megaatmega2560` oppure `esp32c3`. Gli input di ogni modello sono
preparati con il suo preprocessing conservato; il caricamento da Flash a RAM
avviene prima delle finestre. `EB_CACHE=20` è fisso per questa modalità.
Serial stampa lo SHA256 dell'intera coorte e gli ID ordinati di queste 20 righe.
Ogni modello, inclusa la sampled-LUT, ha le proprie predizioni attese: sono
il riferimento del kernel C conservato, compilato sull'host. La concordanza
con esso non sostituisce i test indipendenti di equivalenza sui golden vector.
File e SHA256 sono documentati in `hardware_cohort_export.json`.

`checksum_ok=1` indica che la somma delle predizioni coincide con la somma
attesa. Gli errori di predizioni diverse possono compensarsi. Non significa
classifier accuracy=100%, non dimostra ogni singolo risultato e non garantisce
l'esecuzione di esattamente N chiamate al kernel dopo l'ottimizzazione.
Prima delle misure vengono eseguiti una verifica separata di ciascun vettore
e un controllo del codice compilato per il target. I vecchi test host di
compilazione/checksum confermano soltanto le proprietà esplicitamente verificate
dei sorgenti/dell'esecuzione host; i loro tempi non diventano risultati hardware.

## Grandezze misurate

Siano N il numero di inference nel batch, A e R gli intervalli completi active
e reference definiti dai marcatori esterni, e T_A e T_R le loro durate.
Tensione e corrente sono misurate al confine elettrico dichiarato della scheda:

\[
E_A=\int_A V(t)I(t)\,dt,\quad E_R=\int_R V(t)I(t)\,dt.
\]

La grandezza principale è l'**energia totale della scheda attiva per inference**:

\[
e_{\mathrm{active,board}}=E_A/N.
\]

La grandezza aggiuntiva è l'**incremento rispetto al busy reference misurato**:

\[
e_{\mathrm{incremental}}=
\frac{E_A-(E_R/T_R)T_A}{N}
=\frac{(\overline P_A-\overline P_R)T_A}{N}.
\]

Non si può sottrarre `E_A-E_R` senza correggere per le durate. Entrambi gli
integrali sono calcolati sulle rispettive finestre esterne, non sulla promessa
di tempi uguali. La tolleranza firmware del 5% è una verifica della calibrazione,
non una correzione dell'energia. L'incremento non è l'«energia pura dell'algoritmo»
né riguarda soltanto la MCU: dipende dal reference scelto, dalla scheda, dal clock
e dal carico. Un valore negativo viene conservato e interpretato tenendo conto
dell'incertezza; non viene azzerato né trasformato in valore assoluto.

L'active comprende il kernel, la gestione del batch, l'accumulatore e l'attività
della scheda nel circuito misurato. L'intervallo del marcatore è leggermente
più ampio di `t1-t0`, perché comprende le operazioni del timer/dei confini.
Le formule principali dell'energia utilizzano le durate dei marcatori esterni.
La differenza rispetto alle durate UART viene registrata come controllo di
sincronizzazione/overhead, ma non viene sottratta sulla base di un'ipotesi.

## Acquisizione e criteri di accettazione

Dopo aver identificato lo strumento, scegliere il batch in modo che ogni
finestra contenga abbastanza campioni **indipendenti**, considerando la banda
reale e la media applicata dallo strumento. Il template propone almeno 100
intervalli e una tolleranza relativa del 2% tra tempo esterno e UART, più due
intervalli tra campioni per la discretizzazione dei fronti. Sono parametri
preliminari, non una specifica universale di accuratezza; i parametri finali
vengono fissati prima delle misure. Un singolo campione INA219 non sincronizzato
o la lettura di un USB meter senza traccia temporale non fornisce l'energia
di un kernel breve. Con uno strumento lento si aumenta prima il batch,
non il numero di cifre dichiarate.

Nella copia di lavoro è stato corretto l'overflow dei prodotti intermedi a
32-bit `duration_us*1000` e `sum_duration_us*1000` nella stampa di ns/inference.
In precedenza la soglia era di circa 4.295 s per una singola finestra o per
la somma delle finestre. Le durate di `micros()` restano uint32, quindi ogni
finestra deve essere più breve di un periodo di questo contatore. La durata
utile del batch dipende da strumento, memoria, watchdog/servizi in background
e smoke test; un batch arbitrariamente lungo non è automaticamente ammissibile.

Per ogni modello conservare il log completo di boot/Serial, la traccia grezza
dello strumento, la traccia normalizzata, la registrazione
`templates/energy_acquisition.json`, il build log e gli hash di binary/model/cohort.
I marcatori si collegano a ingressi compatibili ad alta impedenza; i pin scelti
vanno verificati sulla scheda reale. La separazione dei canali GPIO da sola
non misura la corrente.

Prima dell'elaborazione devono esserci esattamente EB_REPS coppie complete,
modello, batch e ordine corretti, timestamp continui e finiti, senza
sovrapposizione dei marcatori. `checksum_ok=1` e tutti gli `ok=1` sono obbligatori
per la normalizzazione per inference. Per l'incrementale occorrono inoltre
`calibration_ok=1`, `windows_ok=1`, `windows_match=1`. Se il reference non
soddisfa i criteri, è possibile conservare separatamente un risultato
total-active valido; l'incrementale non viene pubblicato come confermato.
Non scegliere il boot migliore dopo aver visto i risultati. Le ripetizioni
di uno stesso avvio e i riavvii indipendenti vengono analizzati separatamente;
la SD tra finestre descrive la ripetibilità, non l'incertezza dello strumento.

## Elaborazione riproducibile

`tools/aggregate_energy_trace.py` è un analizzatore offline indipendente basato
sulla libreria standard di Python. Accetta un CSV in unità SI con questi esatti
nomi di colonna:

```text
time_s,voltage_v,current_a,marker_active,marker_reference
```

`time_s` è strettamente crescente; voltage/current sono campioni simultanei;
il marker vale `0` o `1`. Lo stato digitale di una riga vale fino alla riga
successiva. La potenza `voltage_v*current_a` viene integrata con il metodo
dei trapezi su questi intervalli. Servono campioni LOW prima del primo e dopo
l'ultimo impulso. È una convenzione esplicita per normalizzare l'esportazione;
non nasconde l'incertezza dovuta alla discretizzazione dei fronti. Se lo
strumento memorizza separatamente i tempi dei fronti o fornisce campioni di
potenza già mediati su intervalli, occorre un adattatore concordato per il suo
formato, non una rinomina silenziosa delle colonne.

```bash
python tools/aggregate_energy_trace.py --trace evidence/RUN/trace_si.csv --serial-log evidence/RUN/serial.log --acquisition evidence/RUN/acquisition.json --out evidence/RUN/energy_analysis.json
```

L'output contiene entrambi gli integrali e le durate di ogni coppia,
total/incremental, indicatori di qualità, media/SD descrittive, metadati e
SHA256 degli input/dell'analizzatore. Un campionamento insufficiente o un
checksum errato bloccano i corrispondenti valori per inference. L'incrementale
negativo viene conservato. Gli input sintetici sono marcati esplicitamente
`evidence_kind=synthetic` e non vengono mai accettati come hardware.
Anche metadati completi non verificano lo schema di collegamento o la
calibrazione reali; `publication_ready` è intenzionalmente sempre false:
la revisione viene eseguita dal responsabile scientifico.

`tests/test_energy_trace_aggregation.py` verifica l'integrale analitico di
una potenza lineare, la correzione per durate disuguali, l'incremento negativo
e il rifiuto degli input non validi. È una verifica sintetica dell'aritmetica
e della validazione degli input, senza misurazione dell'hardware.

# Changelog

## Joint training, generalizzazione e CIC-IoT-2023 (agosto 2026)

Richiesta del Prof. Kuznetsov per chiudere il primo progetto: joint training
TON_IoT + BoT-IoT, generalizzazione a un terzo dominio mai visto, CIC-IoT-2023
come quarto per nome. `adattamento-drift/` resta un progetto separato, non
integrato in questa pipeline; il README lo documenta come tale.

**Joint training** (`scripts/joint_training.py`, nuovo). Ordine imposto e
verificato da un test (`tests/test_joint_training.py`): split train/test per
dominio, bilanciamento a pari dimensione e pari rapporto normal/attack
(`balance_joint`, vincolato da BoT-IoT: ~382 normali in training su 3,67 M
flussi), unione, solo allora preprocessing e fit. **Il rapporto principale è
1:5, non 1:50**: la configurazione storica è stata testata per prima insieme
a 1:20/1:100, ma tre modelli su sei (LightGBM, XGBoost, MLP) degradavano in
modo significativo all'aumentare del rapporto senza segni di arresto fino al
pavimento testato — la griglia è stata estesa a 1:10/1:5 e la scelta rifatta
sui dati (paired t-test su 10 seed). Non spinto a 1:1: quel regime è dove
adattamento-drift ha già documentato un collasso della selezione delle
etichette.

**Generalizzazione a UNSW-NB15** (`--eval-extra unsw`), congelando feature,
preprocessing e iperparametri: nessun retraining. Il soffitto — UNSW-NB15
arriva a 0,8184 anche in-domain in questo spazio (`adattamento-drift/RISULTATI.md`,
sezione 11) — è dichiarato prima di ogni numero.

**`cross_domain.py` rilanciato a 10 seed** per TON→BoT e BoT→TON (BoT→BoT
incluso per rigore). Una conclusione della fase 2 non regge più: a 3 seed il
KAN multi-layer sembrava il peggior modello cross-domain; a 10 è MLP(16), sia
in valore assoluto sia in δ. Il pattern generale (capacità in-domain costa
transfer) si rafforza: riguarda due famiglie di modelli, non una.
TON_IoT→TON_IoT resta al protocollo originale (3 seed × 5 fold), mai
segnalato come insufficiente.

**CIC-IoT-2023** (`kanids/harmonized.py::build_ridotto_cic`, portato dalla
versione già esistente in `adattamento-drift/kanids/`; verificato — non solo
dichiarato — che nessuna feature mancante è riempita con un valore inventato:
solleva `KeyError` se le colonne vere non ci sono, e sul file reale non
scatta mai). Lo spazio comune ai quattro dataset è **6+2, non 3+2**:
`test.csv` ha una `flow_duration` genuina (mediana 26,1 s benigni contro 0,0 s
attacchi), a differenza di `Duration` che è il TTL. Il costo della riduzione,
misurato sugli stessi tre domini già analizzati nello spazio ricco con t
appaiati per seed, non contando i segni: su UNSW-NB15 lo spazio ridotto vince
in modo significativo per 3 modelli su 6 (KAN single-layer, KAN multi-layer,
LightGBM; p da <0,0001 a 0,049) e non perde in modo significativo per
nessuno — il contrario di quanto assunto in precedenza, sui modelli dove la
differenza è distinguibile dal rumore. Su TON_test la direzione non è
uniforme: KAN single-layer preferisce il ridotto (p<0,0001), LightGBM e MLP
preferiscono il ricco (p=0,0006, p=0,027) — costo comunque piccolo (≤0,036)
ma non lo stesso segno per tutti i modelli. CIC-IoT-2023
stesso resta vicino al caso (0,41–0,51 di balanced accuracy) per un modello
congiunto zero-shot, non in tensione con la misura di adattamento-drift (che
riguarda un pipeline diverso: cross-domain a singolo dominio, con
adattamento).

Un bug corretto durante il lavoro: il CSV di bilanciamento non univa con la
versione già su disco, perdendo silenziosamente un seed quando un run isolato
precedeva quello completo — stessa classe di errore già vista tre volte in
adattamento-drift. Un secondo bug (KAN costruita con `in_dim` fisso a 10
invece del numero reale di feature selezionate) è emerso solo nello spazio
ridotto, dove le candidate sono 6: corretto prima di qualunque run lungo.

Dettagli, tabelle e riproduzione: `README.md`, sezioni "Joint training" e
"A fourth dataset, in a smaller space".

## Protocollo v2 — consolidamento sperimentale (agosto 2026)

Fase di consolidamento richiesta dal Prof. Kuznetsov. Nessuna nuova variante
della KAN: solo rigore metodologico, un secondo dataset e verifica del
deployment. I sei punti della revisione e il loro stato sono in
`report_KAN-IDS_fase2.pdf`.

### 1. Protocollo leakage-free

* Selezione delle feature per mutual information, vocabolari delle categoriche
  e quantili sono ora fittati **esclusivamente sul training** di ogni fold
  (`kanids/preprocessing.py`).
* Introdotto lo **slot UNK** (indice 0) in ogni tabella categorica: rende
  l'edge una funzione totale, necessario per il cross-domain e per il runtime
  su MCU. Costo: +4 byte sul modello binario compilato (246 → 250 B).
* Il difetto era replicato in **tredici script**. Ora la selezione passa da un
  solo punto (`kanids.preprocessing.rank_by_mi`) e un test fa fallire la build
  se qualcuno ne apre un'altra copia.
* **Effetto misurato del difetto: nullo.** La selezione per-fold sceglie le
  stesse 10 feature in 15 fold su 15 e non esiste alcuna categoria non vista
  in-domain (`results/leakage_audit_stability.csv`). I numeri precedenti non
  sono invalidati; il protocollo andava corretto comunque.

### 2. Validazione uniforme, 5-fold × 3 seed

Quindici fit per modello, media ± deviazione standard, stratificazione sempre
sull'etichetta a 10 classi così che binario e multiclass condividano i fold.

| Modello | F1 binario | Macro-F1 10 classi |
|---|---|---|
| LightGBM | 0,9991 ± 0,0001 | 0,9680 ± 0,0021 |
| XGBoost | 0,9989 ± 0,0001 | 0,9666 ± 0,0021 |
| KAN multi-layer | 0,9976 ± 0,0002 | 0,9374 ± 0,0036 |
| MLP (16) | 0,9964 ± 0,0009 | 0,9182 ± 0,0107 |
| Decision Tree (d=5) | 0,9944 ± 0,0004 | 0,7633 ± 0,0033 |
| KAN single-layer | 0,9835 ± 0,0007 | 0,8767 ± 0,0014 |

### 3. Cross-domain TON_IoT ↔ BoT-IoT

Spazio armonizzato a 13 feature candidate calcolate con la stessa formula sui
due dataset; nel cross-domain il target entra solo nella valutazione. Quattro
direzioni, più l'ablation senza categoriche.

**Il transfer collassa**: TON→BoT lascia ogni modello fra 0,40 e 0,56 di
balanced accuracy, cioè al caso o sotto. Il δ del paper precedente era ≤ 5,95
punti; qui è di 40–59 punti.

### 4. Correzioni alle conclusioni precedenti

Due claim del README sono risultati non sostenibili e sono stati corretti:

* Il confronto con le baseline metteva la KAN sullo spazio grezzo a 14 feature
  e gli alberi su quello derivato a 10 del paper. **Con input identici
  LightGBM fa 0,9991 contro 0,9835.**
* A parità di regola di conteggio l'albero profondo 5 occupa **141 byte** con
  F1 0,9944: più piccolo *e* più accurato del modello KAN da 250 byte, che è
  **dominato** sulla frontiera di Pareto in-domain (`results/footprint.csv`).

Cosa regge: la multi-layer sta sulla frontiera (5,2 KB, 0,9976, contro l'MLP
TFLite del paper a 13 KB e 0,9959 con 95 feature); e nel cross-domain la
classifica si ribalta.

### 5. Inferenza integer-only end-to-end, in C

Entrambe le catene — binaria e a 10 classi — vanno dai contatori grezzi alla
decisione in aritmetica intera, verificate **bit per bit** contro il
riferimento Python su 200 golden vector, con **zero istruzioni in virgola
mobile** nell'assembly del percorso di inferenza (`tools/check_no_float.sh`).

Il percorso precedente (`mcu_e2e/`) interpolava 10.000 knot del
QuantileTransformer in doppia precisione: end-to-end nella struttura, non
integer-only nel runtime. È conservato solo per riferimento.

Difetti emersi dal porting, tutti invisibili in Python:
* moltiplicatore di scala che quantizza a esattamente 32768 → overflow
  silenzioso in `int16` (due occorrenze distinte);
* divisione intera: Python arrotonda verso −∞, il C tronca verso zero;
* soglie a 64 bit necessarie perché `src_bytes` e `dst_bytes` superano 2³¹.

### 6. Riproducibilità

* 42 riferimenti a `/tmp` in 21 script → `artifacts/` dentro il repository.
* Cache con **invalidazione automatica** per impronta di configurazione.
* `reproduce.py` con stage dichiarati; `--stage smoke` esegue l'intera catena
  su dati sintetici in ~20 s **senza scaricare nulla**.
* `models/` versionata con `MANIFEST.json` (protocollo, seed, spazio di
  feature, metriche); `artifacts/` è cache cancellabile.
* **30 test** impediscono la ricomparsa dei difetti: percorsi `/tmp`, percorsi
  assoluti, dipendenze non vincolate, selezione feature fuori da `kanids`,
  virgola mobile negli header integer, risultati v1 mescolati ai correnti.
* I risultati prodotti prima della correzione sono in `results/protocol_v1/`
  con un README che spiega quali numeri sono superati.

### 7. Indipendenza della stima, misurata

`scripts/nested_cv.py` esegue una **cross-validation annidata**: dentro ogni
fold esterno una CV interna sceglie k sul solo training, e la valutazione
avviene su dati che non hanno partecipato alla scelta. La differenza rispetto
alla stima piatta e' l'ottimismo di selezione.

**Risultato: l'ottimismo e' negativo.** 0,9845 ± 0,0006 contro 0,9835 per la
KAN single-layer (−0,0009) e 0,9992 contro 0,9991 per LightGBM (−0,0001). I
numeri pubblicati sono semmai conservativi.

La misura smentisce pero' un altro claim: la selezione interna **non sceglie
mai k = 10**, prende tutte e 16 le feature candidate in 15 fold su 15. La curva
e' monotona, non ha un picco. **k = 10 e' una scelta di deployment** — dieci
statistiche di flusso a bordo invece di sedici — e costa 0,0009 di F1.

### 8. La catena end-to-end e' anche deployata, non solo verificata

`mcu_pio/src/main_e2e.cpp` e' una variante di firmware che parte dai
**contatori grezzi** ed esegue a bordo l'intera catena, feature engineering
compreso, sotto il protocollo di benchmark del paper. Tutte le altre varianti
ricevono vettori gia' normalizzati fuori dal dispositivo: senza questa, la
catena integer sarebbe dimostrata corretta ma non sarebbe mai stata *la*
pipeline in esecuzione sull'MCU.

Firmware e harness di verifica includono lo **stesso** kernel
(`mcu_pio/include/kan_e2e_infer.h`): cio' che e' verificato bit per bit e'
cio' che gira sulla board, non una copia che puo' divergere.

### 9. Verifiche che impediscono le regressioni

* `tests/test_leakage.py::test_crossdomain_target_does_not_influence_training`
  fitta la pipeline due volte sullo stesso source con target radicalmente
  diversi e pretende che tutto l'appreso sia identico bit per bit. Introdurre
  la violazione classica (fit su source ∪ target) lo fa fallire.
* `requirements-lock.txt` con le versioni esatte dell'ambiente che ha prodotto
  i numeri pubblicati, piu' un test che impedisce alla promessa di tornare a
  vuoto: prima `requirements.txt` citava un lock che non esisteva.
* `tools/audit_richieste.py` (`reproduce.py --stage audit`) verifica
  meccanicamente i sei punti della revisione e stampa, requisito per
  requisito, l'evidenza che lo sostiene. **26 su 27**; l'unico non fatto e'
  CIC-IoT-2023, indicato come obiettivo secondario.

La suite e' passata da 22 a **30 test**, e l'audit da 23 a **27 controlli**.

### 10. Ogni modello e' esportato in C e flashabile

Prima alcuni modelli esistevano come header ma senza un `main` che li usasse:
esportati sulla carta, non testabili fisicamente. Ora sono **sette su sette**,
ognuno con il suo environment PlatformIO, verificato da un test che fallisce se
qualcuno aggiunge un header senza il firmware corrispondente.

Fra questi il **Decision Tree profondo 5** (`main_dt5.cpp`), che non era mai
stato esportato pur essendo il modello che **domina** la KAN da 250 byte sulla
frontiera in-domain. Senza, l'obiezione piu' seria al lavoro non sarebbe stata
chiudibile su hardware. L'export ha gia' prodotto un risultato: quantizzare le
soglie a Q7 per il dispositivo costa all'albero **0,0028 di F1**
(0,9944 -> 0,9916, agreement 99,55%), e il divario con la KAN compilata scende
da 0,0109 a **0,0081**. La KAN paga la quantizzazione meno dell'albero, ed e'
un argomento che prima non avevamo perche' confrontavamo un albero float con
una KAN gia' quantizzata.

Aggiunto anche il firmware per la catena end-to-end a 10 classi
(`main_mc_e2e.cpp`), che esisteva solo come verifica offline.

### 11. Coerenza degli artefatti deployati

L'export in C del modello multiclass era rimasto al **protocollo v1**: tabelle
categoriche con 28 righe (3+9+13+3), senza slot UNK, contro le 32 (4+10+14+4)
del protocollo attuale. Il firmware girava quindi su un modello incompatibile
con il preprocessing v2 — senza errori visibili, perche' modello e vettori di
test erano coerenti fra loro. Rigenerato, e un test ora vieta la presenza di
qualunque header con tabelle a 28 righe.

Corretto anche `main.cpp`, che era l'unico firmware non compilabile su host e
quindi l'unico che nessuno poteva verificare prima di flasharlo. Un test ora
compila tutti e sette i firmware a ogni esecuzione della suite.

### Cosa resta aperto

* Benchmark fisici su Mega 2560 ed ESP32-C3: latenza, SRAM, energia, code size.
  Tutti i firmware necessari esistono e sono compilabili; manca l'esecuzione
  sulle board. `results/latency_benchmark.csv` contiene le misure del paper
  precedente, relative alle varianti v1.
* Larghezza nascosta, grado e clip non sono stati riselezionati dentro il
  ciclo: restano ereditati dalla fase precedente. L'effetto complessivo
  dell'esposizione ai dati e' pero' misurato ed e' sotto il millesimo di F1,
  in direzione conservativa (vedi punto 7).
* Analisi di sensibilità sul rapporto di undersampling (fissato a 1:50).
* CIC-IoT-2023 come terzo dataset (obiettivo secondario).

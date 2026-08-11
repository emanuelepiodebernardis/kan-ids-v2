# Changelog

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
* **22 test** impediscono la ricomparsa dei difetti: percorsi `/tmp`, percorsi
  assoluti, dipendenze non vincolate, selezione feature fuori da `kanids`,
  virgola mobile negli header integer, risultati v1 mescolati ai correnti.
* I risultati prodotti prima della correzione sono in `results/protocol_v1/`
  con un README che spiega quali numeri sono superati.

### Cosa resta aperto

* Benchmark fisici su Mega 2560 ed ESP32-C3: latenza, SRAM, energia, code size.
  `results/latency_benchmark.csv` contiene le misure del paper precedente.
* Un held-out mai toccato da alcuna decisione (gli iperparametri sono stati
  scelti sugli stessi dati usati nella cross-validation).
* Analisi di sensibilità sul rapporto di undersampling (fissato a 1:50).
* CIC-IoT-2023 come terzo dataset (obiettivo secondario).

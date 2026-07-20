# KAN-IDS — Simulazione Wokwi End-to-End (Passo 5)

Questo sketch esegue la **catena completa** su ESP32-C3:

```
dati grezzi → log1p (SKEW) → QT sklearn-compatible → clip ±3.5
            → clip [-1,1] → eval_l1 (Q16) → tanh_lut → eval_l2 → argmax
```

Macro-F1 su 12 000 sample di test: **0.9118** (identico al riferimento Python).

---

## File necessari per Wokwi

Tutti i file si trovano nelle cartelle `mcu/` e `mcu_e2e/`:

| File | Dimensione | Contenuto |
|---|---|---|
| `mcu_e2e/main_kan_e2e_wokwi.cpp` | 8.3 KB | Sketch principale (preprocessing + forward) |
| `mcu/kan_ml_layer1.h` | 315 KB | LUT layer 1 (int16, 160×512) |
| `mcu/kan_ml_layer2.h` | 340 KB | LUT layer 2 (int16, 160×512) |
| `mcu/kan_ml_tanh.h` | 12 KB | LUT tanh (int16, 2048) |
| `mcu_e2e/kan_ml_prep.h` | 148 KB | Knot QT + references (double) |
| `mcu_e2e/test_vectors_e2e.h` | 3 KB | 40 vettori di test in feature grezze |

**Flash totale stimata**: ~800 KB su 4 MB disponibili sull'ESP32-C3. ✓

---

## Passi su Wokwi

### 1. Crea il progetto

Vai su [wokwi.com](https://wokwi.com/), accedi, crea un nuovo progetto **ESP32-C3**.

### 2. Incolla lo sketch

Copia il contenuto di `mcu_e2e/main_kan_e2e_wokwi.cpp` nel file principale (`sketch.ino`).

### 3. Aggiungi i file header

Nel pannello file di Wokwi (icona "+"), crea questi 5 file con i rispettivi contenuti (i nomi devono essere esatti):

- `kan_ml_layer1.h`
- `kan_ml_layer2.h`
- `kan_ml_tanh.h`
- `kan_ml_prep.h`
- `test_vectors_e2e.h`

### 4. Avvia e leggi

Premi ▶ Play. Apri il Serial Monitor (115200 baud). Output atteso:

```
=== KAN-IDS END-TO-END (preprocessing on-chip) ===
features=10 hidden=16 classes=10
idx,label,pred,match,latency_us
0,9,9,Y,<us>
...
--- riepilogo ---
accuratezza: 95.0%
latenza media (us): <us>
min/max (us): <min> / <max>
```

**Accuratezza attesa**: 95.0% (38/40) — gli stessi 2 errori del modello Python.

---

## Nota sul preprocessing

Il preprocessing replica **esattamente** `sklearn.preprocessing.QuantileTransformer` con:
- output distribution: `normal`
- n_quantiles: 1000
- random_state: 42

La formula bidirezionale implementata:
```c
r = 0.5 * (interp_fwd(x, knots, refs) - interp_rev(x, knots, refs))
qt = norm_ppf(r)   // Acklam rational approximation
```

Fix applicato: `INTERP_EPS = 1e-14` nella binary search per gestire la differenza
di 1 ULP tra `log1p` di runtime C e i knot generati da numpy.

---

## Risultati Passo 5 (host C++)

| Metrica | Valore |
|---|---|
| Macro-F1 | **0.9118** |
| Accuracy | **0.9623** (11547/12000) |
| Riferimento Python | 0.9118 ✓ |

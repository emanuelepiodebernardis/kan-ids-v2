# KAN-IDS Integer-Only v4 — Deployment ESP32-C3

## File

| File | Dimensione | Contenuto |
|---|---|---|
| `main.cpp` | 6.5 KB | Sketch principale — preprocessing + forward + benchmark |
| `qt_int_v4_lut.h` | 109 KB | Preprocessing LUT O(1): QT_KLO, QT_FRAC, QT_PPF |
| `kan_ml_layer1_v4.h` | 322 KB | LUT KAN layer 1 (int16, 160 edge) |
| `kan_ml_layer2_v4.h` | 341 KB | LUT KAN layer 2 (int16, 160 edge) |
| `kan_ml_tanh_v4.h` | 13 KB | LUT tanh int16 + costanti HMAX, TL, S2 |
| `test_vectors_int_v4.h` | 5.5 KB | 40 vettori di test (feature grezze float32) |

## Pipeline

```
raw float32 (valori sensore)
  → logf(x+1) × 10 feature          [float, ~2 µs su HW reale]
  → lookup QT_KLO + QT_FRAC          [int O(1), ~7 µs]
  → interpolazione PPF float32        [~1 µs]
  → to_q16() × 10                    [int, ~1 µs]
  → KAN forward LUT int16             [int, ~675 µs]
  → argmax logits[10]                 [int, ~1 µs]
  → classe (0..9)
```

## Risultati verificati

| Metrica | Valore |
|---|---|
| Macro-F1 (12k test) | **0.9044** |
| Accuratezza 40 sample | **95.0%** (38/40) |
| Latenza Wokwi (simulata) | ~2440 µs (logf simulato lento) |
| **Latenza HW reale stimata** | **~685 µs** |
| Variabilità latenza HW | **~1.02×** (quasi costante) |
| Flash | **346 KB** / 3584 KB ESP32-C3 |
| RAM preprocessing | **43 KB** (copiato da flash a setup()) |

## Classi

| ID | Classe |
|---|---|
| 0 | backdoor |
| 1 | ddos |
| 2 | dos |
| 3 | injection |
| 4 | mitm |
| 5 | normal |
| 6 | password |
| 7 | ransomware |
| 8 | scanning |
| 9 | xss |

## Feature (ordine top-10 MI)

`src_ip_bytes`, `dst_port`, `dst_ip_bytes`, `src_port`, `duration`,
`src_bytes`, `dst_bytes`, `dst_pkts`, `src_pkts`, `dns_qtype`

## Come usare su ESP32-C3

1. Rinomina `main.cpp` → `sketch.ino` (o usa PlatformIO)
2. Carica tutti i 6 file nella stessa cartella del progetto
3. Compila e flasha
4. Per classificare un pacchetto reale, sostituisci `TEST_RAW_V4[i]`
   con il tuo array di 10 feature float32

## Note su Wokwi

Su Wokwi la latenza misurata è ~2440 µs perché il simulatore emula
`logf()` in software (~174 µs/call). Su hardware reale ESP32-C3
`logf()` usa la FPU hardware (~0.2 µs/call).

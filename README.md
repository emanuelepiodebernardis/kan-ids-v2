# KAN-IDS: Kolmogorov–Arnold Networks for Embedded Intrusion Detection

**Sub-kilobyte neural intrusion detection on microcontrollers — with integer-only inference, statistical guarantees, and a readable closed form.**

This repository extends the published pipeline of
[*Lightweight Machine Learning Intrusion Detection for IoT/IIoT Networks*](https://doi.org/10.3390/electronics15132869)
(De Bernardis, Kuznetsov, Arnesano, Zhansaya, Sydykova — *Electronics* 2026, 15, 2869)
with a fourth model family: **Kolmogorov–Arnold Networks (KAN)** compiled for
microcontroller deployment, building on the
[`lut-kan`](https://github.com/KuznetsovKarazin/lut-kan) quantisation framework.

All results below are measured on the full **TON_IoT** network dataset
(211,043 real flows), with every number backed by a script in `scripts/` and an
artifact in `results/`.

---

## Headline results

| Model | Accuracy (TON_IoT, held-out) | Deployed size | Arithmetic |
|---|---|---|---|
| Binary, single-layer + categorical edges | **F1 = 0.9837 ± 0.0007** (5-fold × 3-seed CV) | **246 B** | integer-only (int8 / Q15) |
| Binary, multi-layer (16 hidden) | **F1 = 0.9974**, ROC-AUC 0.9998 | **5.05 KB**, lossless (ΔF1 = 0.0000) | integer-only |
| Multiclass, 10 attack classes | macro-F1 = 0.9409 (weighted 0.9809) | 8.3 KB (inference) / **11.1 KB end-to-end** | integer-only, raw counters → decision |
| Symbolic form of the binary model | F1 = 0.9835 | a printable 10-term equation + 4 lookup tables | — |

For reference, the strongest neural baseline of the original paper — an MLP
via TensorFlow Lite Micro — reaches F1 = 0.9959 using **95 features and 13 KB**.
The multi-layer KAN here surpasses it (0.9974) using **14 features and 5 KB**,
and the 246-byte single-layer model outperforms tree ensembles cross-validated
on the same deployable feature space (RF: 0.9819 ± 0.0006).

---

## What is new, and why it matters

### 1. A 14-feature space that matches 95 features
Feature-count ablation (`scripts/feature_curve_driver.py`) shows accuracy
peaks at exactly **10 numeric features** for both tasks — more features add
noise, not signal. The missing multiclass information turns out to be
**categorical** (`proto`, `service`, `conn_state`, `dns_rejected`): LightGBM on
10 numeric + 4 categorical features reaches macro-F1 0.9675, statistically
indistinguishable from 0.9694 obtained with all 95 features in the original
paper. This cuts on-device feature-extraction cost ~7× for *any* model family.

### 2. Categorical edges for KANs (novel)
KANs have no native way to consume categorical inputs; one-hot encoding wastes
polynomial bases. We introduce **tabular categorical edges**:
φ(category) = a learned table row, trained jointly by backpropagation, indexed
by category ID at inference. In a LUT-compiled KAN this is *free* — a
categorical edge is already a lookup table (560 bytes for all four features).
Effect: multiclass macro-F1 climbs 0.858 → 0.875 (single layer)
→ **0.941** (with depth); binary F1 climbs 0.971 → **0.984**.

### 3. Hybrid compilation: train-Chebyshev, deploy-B-spline (novel)
Chebyshev bases train more accurately; B-splines quantise more faithfully
(10× lower logit error — local support, no oscillation). We get both by
**re-fitting the learned edge functions with cubic B-splines and storing the
quantised spline coefficients** (19 per edge) instead of a sampled LUT:

| Compilation | Size | ΔF1 | Agreement vs float |
|---|---|---|---|
| Sampled LUT, L = 64 (baseline) | 5,476 B | −0.0001 | 99.95 % |
| Spline coefficients, int16 | 492 B | **0.0000 (lossless)** | **100.000 %** |
| Spline coefficients, int8, full-integer | **246 B** | −0.0002 | 99.95 % |

Uniform (unclamped) knots give a closed matrix form per segment, evaluated
with **integer-only Horner (Q15)** — no floating point at inference.

### 4. End-to-end integer pipeline: raw counters → decision
Everything between the packet counters and the decision is integer:
logarithms via a 512 B LUT (using the identity `log1p(a/b) = ln(a+b) − ln(b)`),
integer divisions for asymmetry ratios, and the z-score/clip normalisation
**absorbed into the spline segment mapping** (two affine constants per
feature). For the multiclass model, sklearn's quantile-normal preprocessing is
replaced by **empirical per-feature threshold tables built offline from the
fitted transformer** (quantile knots + most-frequent values — exact on
discrete masses, 3 KB total). Verified end-to-end on all 42,209 test flows:
binary F1 0.9646 in ~842 B total; 10-class macro-F1 0.9384 (agreement 99.5 %)
in ~11.1 KB total.

### 5. Conformal prediction in under 1 KB
Split-conformal calibration (marginal and per-class/Mondrian) is applied
**to the deployed integer model**, so the coverage guarantee holds for what
actually runs on the MCU. Measured coverage: 99.07 / 95.09 / 89.93 % at
targets 99 / 95 / 90 %; 93 % of predictions are decisive singletons at
α = 0.01. On-device cost: **two float thresholds**. Uncertain flows
(two-class sets) can be escalated to a gateway — triage with a statistically
controlled error rate.

### 6. The IDS as one readable equation
Each learned edge is fitted with elementary primitives (sin, tanh, gaussian,
polynomials — weighted by the data density). The result is a 10-term printable
formula plus four small category tables, reaching **F1 0.9835 — slightly above
the network itself** (the smooth primitives act as a regulariser). See
`results/kan14_symbolic_real.txt`.

### 7. Rigour
The main results carry **5-fold × 3-seed cross-validation** (mean ± std),
and every design choice is backed by a measured alternative — see
[Negative results and design justifications](#negative-results-and-design-justifications)
below for the full list with numbers.

---

## Negative results and design justifications

Not everything we tried helped — and each null result settles a design
question that a careful reader (or reviewer) would otherwise raise. All are
measured on the full dataset, with artifacts in `results/`:

| Experiment | Result | What it settles |
|---|---|---|
| **B-spline as *training* basis** (`basis_comparison_unified_real.csv`) | F1 0.9383 vs 0.9672 for Chebyshev at equal parameter count (0.9279 with class-weighted loss — the gap is structural, not a loss artifact) | Why training uses Chebyshev, even though B-splines quantise 10× more faithfully — motivating the hybrid train/deploy split |
| **Re-fit → sampled LUT** (`hybrid_compile_real.csv`) | Statistically identical to direct LUT at every resolution L (e.g. 94.54% vs 94.58% agreement at L=8) | Why the hybrid gain lives in *coefficient storage*, not in smoothing the LUT: uniform-grid sampling is the bottleneck, and re-fitting cannot remove it |
| **Doubling capacity (32 hidden units)** (`ml_binary_real.csv`) | Plateau at F1 0.9778, below the 16-hidden result (0.9784), at 2× the parameters | Why the deployed multi-layer uses 16 hidden units; the bottleneck is input information, not model capacity |
| **More numeric features (k = 12–16)** (`feature_curve_real.csv`) | F1 flat or slightly worse beyond k = 10, on both tasks | Why the feature space stops at 10 numeric features: additional ones add noise, not signal |
| **Lower layer-2 degree (4 vs 8)** (`kan_ml_cat_deg4_real.csv`) | macro-F1 0.9374 vs 0.9409; LUT/coefficient memory does not depend on degree | Why degree 8 is kept: the cheaper variant saves nothing where it matters |
| **Focal loss (γ = 2) for the rare MITM class** (`kan_ml_cat_focal_real.csv`) | macro-F1 0.9401 vs 0.9409; MITM F1 0.572 vs 0.571 | The MITM weakness is not a loss-design problem |
| **SMOTENC oversampling (10× MITM)** (`kan_ml_cat_smote_real.csv`) | macro-F1 0.9377; MITM F1 0.541 (worse than baseline) | Synthetic interpolation adds no real information. Together with focal loss and class weighting, three independent remedies fail: the MITM ceiling is **information-limited**, not methodological |
| **Analytical replication of sklearn's quantile-normal transform in integer arithmetic** | Two attempts (single-sided and two-sided quantile interpolation) left errors up to 0.8σ on discrete-mass features and broke the multiclass pipeline | Why the integer preprocessing uses **empirical per-feature threshold tables sampled offline from the fitted transformer** — exact on discrete masses by construction, 3 KB total |

Two further checks worth knowing about: the accelerated training path used
for large experiments is proven identical to the reference implementation to
machine precision (max coefficient difference 2·10⁻¹⁶ after 60 epochs), and
the focal-loss gradient was verified against numerical differentiation
(max error < 10⁻⁹) before use.

## Repository structure

```
kan-ids/
├── src/                    KAN model implementations (NumPy)
│   ├── kan_chebyshev.py            single-layer Chebyshev KAN (binary)
│   ├── kan_chebyshev_multiclass.py softmax multiclass variant
│   ├── kan_bspline.py              B-spline KAN + basis utilities
│   ├── kan_multilayer_numpy.py     multi-layer forward (verification)
│   └── quantization_export.py, embedded_model_io.py, fixed_point_quantile.py
├── preprocessing/          unified 10-feature engineering (from the paper)
├── scripts/                every experiment, one script each (see index below)
├── results/                43 CSV/TXT artifacts backing every number above
├── mcu_pio/                PlatformIO firmware (Arduino Mega 2560 + ESP32-C3)
│   ├── src/main.cpp              variant 1–2: sampled-LUT inference benchmark
│   ├── src/main_coeff.cpp        variant 3: spline-coefficient full-integer
│   ├── include/                  model headers + 200 real test vectors
│   └── host_check/               offline g++ verification harness
├── models/                 trained model weights (.npz) — ready to compile/test
├── mcu/, mcu_e2e/          earlier firmware experiments (Wokwi, kept for reference)
├── data/                   dataset download instructions (data not redistributed)
└── utils.py                shared models/metrics (from the published pipeline)
```

### Script index

**Training** — `kan14_binary.py` (14-feature binary + categorical edges),
`kan14_ml_binary.py` (multi-layer binary), `kan_categorical_mc.py` /
`kan_ml_cat_mc.py` (multiclass with categorical edges), `cv_multiseed.py` +
`cv_driver.py` and `kan14_cv_driver.py` (cross-validation).

**Compilation & deployment** — `export_lut.py` / `export_lut_fast.py`
(sampled LUT), `hybrid_coeff_full.py` and `kan14_compile.py` /
`kan14_ml_compile.py` (spline-coefficient compilation, int16/int8),
`coeff_int_inference.py` (full-integer kernel), `e2e_int_pipeline.py` and
`kan14_mc_e2e_int.py` (end-to-end integer pipelines),
`export_kan14_coeff_c.py` (C header generation for the firmware).

**Analysis & ablations** — `feature_curve_driver.py`, `ablation_L.py`,
`basis_comparison_unified.py`, `quant_basis_comparison.py`,
`hybrid_compile.py`, `kan_ml_cat_focal.py`, `kan_ml_cat_smote.py`,
`kan_ml_cat_deg.py`, `conformal_ids.py` / `kan14_conformal_symbolic.py`,
`symbolic_extract.py`.

Long-running scripts (`*_driver.py`, `kan*_ml_*.py`) are **checkpointed**:
they save state after each unit/epoch chunk and can be re-invoked until they
print `DONE` — convenient on shared or time-limited machines.

---

## Getting started

```bash
git clone https://github.com/emanuelepiodebernardis/kan-ids-v2.git
cd kan-ids
pip install -r requirements.txt          # numpy, pandas, scikit-learn, scipy
pip install xgboost lightgbm imbalanced-learn   # optional: baselines, SMOTENC
git clone https://github.com/KuznetsovKarazin/lut-kan.git   # OPTIONAL: legacy LUT export scripts only
```

**Pre-trained models.** `models/` ships the trained weights (`.npz`) of the
three headline models plus the feature-space metadata, and `mcu_pio/include/`
ships their C headers — you can compile, verify (host checks below) and flash
**without the dataset and without retraining**. The dataset is only needed to
reproduce training and the evaluation tables.

**Dataset.** Download `train_test_network.csv` (TON_IoT, UNSW Canberra —
see `data/README.md`) and place it in the repository root. It is not
redistributed here.

**Reproduce the headline numbers:**

```bash
# 1. Binary 14-feature model (also builds the shared data cache)
python scripts/kan14_binary.py                    # F1 0.971 -> 0.983 with cat edges

# 2. Its compilations (float / int16 / int8 / full-integer)
python scripts/kan14_compile.py                   # 246 B full-integer

# 3. Multi-layer binary (checkpointed - rerun until DONE)
python scripts/kan14_ml_binary.py 300             # F1 0.9974
python scripts/kan14_ml_compile.py                # 5.05 KB lossless

# 4. Cross-validation (checkpointed)
python scripts/kan14_cv_driver.py 300             # 0.9837 +/- 0.0007

# 5. Conformal + symbolic
python scripts/kan14_conformal_symbolic.py

# 6. Multiclass with categorical edges (checkpointed) + end-to-end integer
python scripts/kan_categorical_mc.py 300
python scripts/kan_ml_cat_mc.py 300
python scripts/kan14_mc_e2e_int.py                # 11.1 KB, raw counters -> 10 classes
```

Intermediate caches and training checkpoints are written to `/tmp`
(Linux/macOS; on Windows use WSL).

---

## Firmware

`mcu_pio/` is a PlatformIO project targeting the two boards of the original
paper — **Arduino Mega 2560** (AVR, 8 KB SRAM) and **ESP32-C3** (RISC-V) —
implementing the same benchmark protocol (500 timed inferences: 250 attack +
250 normal, on-board statistics, prediction verification, SRAM report,
CSV over serial):

```bash
cd mcu_pio
pio run -e megaatmega2560 -t upload           # variant 1-2: LUT inference
pio run -e megaatmega2560_coeff -t upload     # variant 3: 246 B binary (F1 0.983)
pio run -e megaatmega2560_mlcoeff -t upload   # variant 4: 5 KB multi-layer (F1 0.9974)
pio run -e esp32c3_mc -t upload               # variant 5: 8 KB multiclass (macro-F1 0.941)
pio device monitor --baud 115200
```

All five variants embed **real test vectors with expected predictions** from
the bit-exact reference simulation, so on-board correctness is verified
automatically at every run.

Without hardware, the exact inference kernels can be verified on any host:

```bash
cd mcu_pio
g++ -O2 -o c1 host_check/run_host_check.cpp     && ./c1   # LUT variant
g++ -O2 -o c2 host_check/run_coeff_check.cpp    && ./c2   # binary 246 B: 200/200
g++ -O2 -o c3 host_check/run_ml_coeff_check.cpp && ./c3   # multi-layer:  200/200
g++ -O2 -o c4 host_check/run_mc_coeff_check.cpp && ./c4   # multiclass:   200/200
```

The C kernel in `include/kan14_coeff_infer.h` is a line-by-line translation of
the bit-exact NumPy integer simulation and matches it on 200/200 real test
vectors. An optional INA219 hook (`-DENABLE_INA219`) measures energy per
inference.

---

## Results index

Every claim maps to an artifact in `results/` (`*_real.csv` = full-dataset
runs). Highlights: `kan14_cv_summary_real.csv` (CV), `kan14_compile_real.csv`
and `kan14_ml_compile_real.csv` (compilations), `kan14_mc_e2e_int_real.csv`
(end-to-end multiclass), `kan14_conformal_real.csv`,
`kan14_symbolic_real.txt`, `ablation_L_real.csv`, `feature_curve_real.csv`,
`quant_basis_comparison_real.csv`, `cv_multiseed_summary_real.csv`
(unified-10 baseline space).

**Known limitations.** The MITM class (208 test samples) stays at F1 ≈ 0.57:
three independent remedies (class weighting, focal loss, SMOTENC) show the
limit is informational, not methodological. Cross-dataset evaluation
(CIC-IoT-2023) is restricted to the 10 harmonised numeric features — the
categorical features that boost in-domain accuracy are capture-tool specific.
Physical latency/energy benchmarks require the boards and are the next step.

---

## Credits & citation

- LUT quantisation framework: [`lut-kan`](https://github.com/KuznetsovKarazin/lut-kan) (O. Kuznetsov) — see also arXiv:2601.03332 and arXiv:2601.08044.
- Baseline pipeline, preprocessing and reference models: [`iot-audit`](https://github.com/emanuelepiodebernardis/iot-audit).
- Dataset: TON_IoT, Moustafa et al., UNSW Canberra (CC BY 4.0).

If you use this work, please cite:

> De Bernardis, E.P.; Kuznetsov, O.; Arnesano, M.; Zhansaya, P.; Sydykova, M.
> *Lightweight Machine Learning Intrusion Detection for IoT/IIoT Networks:
> Quantisation Strategies and Physical Deployment on Resource-Constrained
> Microcontrollers.* Electronics 2026, 15, 2869.

License: MIT (see `LICENSE`).


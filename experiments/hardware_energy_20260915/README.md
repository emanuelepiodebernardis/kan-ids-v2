# Physical kernel-energy pilots, 15 September 2026

This experiment records **20 physical acquisitions: five frozen models, two
acquisitions per model, on each of Arduino Mega 2560 and ESP32-C3**. These are
USB-input, whole-board estimates for repeated prepared-feature inference.
They are not model retraining, a new detection-quality benchmark, or an
end-to-end network IDS measurement.

## Reproduce the accepted scalar results

From the repository root, with Python 3.11 or newer (standard library only):

```console
python experiments/hardware_energy_20260915/verify_measurements.py
```

The command checks imported byte identities, the repackaged firmware source
manifests, all twenty serial DONE/event records, the twenty native CFNs and
the ten equal-weight means. It recomputes each power estimate from VBUS*IBUS,
using a clipped trapezoidal integral with linear interpolation at the two
accepted window boundaries. It checks `results/all_20_acquisitions.csv` and
`results/board_model_means.csv`; it writes nothing. `--write-results`
explicitly rewrites those two CSVs from the same inputs.

The recorded MCU event times define a roughly 120 s active batch. LED pulse
codes before and after that batch were fitted to each CFN independently.
The retained power interval is the **middle 60 MCU seconds**, mapped into the
CFN clock using the accepted marker fit. Mean call time is active MCU time
divided by the counted calls. Estimated energy per call is mean V*I in the
central interval times mean call time. It is not a directly integrated,
individually resolved energy pulse. No idle subtraction is applied.

`measurements/*/accepted_analysis.json` preserves the original fitted
boundaries, fit diagnostics and sensitivity checks. This lightweight verifier
does not refit the markers or replace the original binary audits. The complete
checkpoint includes those scripts, diagnostic plots and the actual measured
ELF/BIN/HEX files, identified in `CHECKPOINT_REFERENCE.json`. Verify its
identity with `--checkpoint PATH_TO_ZIP`. The checkpoint is currently an
author-held review archive; a permanent public URL must be added when released.

## Scope and interpretation

- The same first twenty raw flow identities (ten benign, ten attack) from the
  frozen common cohort are repeated. Each model retains its own fixed input
  transformation. Millions of calls are not millions of distinct flows.
- The five models are coefficient KAN, its sampled-LUT representation,
  MLP16, multilayer KAN and DT5. The coefficient/LUT pair isolates the
  deployment representation. Timing differences between other models do not
  establish equivalent detection quality.
- The FNB58 files use a 10 samples/s stored time grid. Its internal ADC rate,
  effective bandwidth, absolute calibration and a full instrument uncertainty
  budget were not established by this campaign. The plots include marker
  transitions broadened by approximately one second; the central interval
  avoids those transitions. Repeat differences are descriptive repeatability,
  not absolute measurement accuracy or confidence intervals.
- The supply path includes the board and downstream cable losses. It includes
  indexing, checksum and normal system overhead. It excludes preprocessing,
  packet acquisition and networking. C3 ran at 160 MHz with Wi-Fi/Bluetooth
  uninitialised, no active-loop yield and interrupts enabled. In the actual
  logs no watchdog subscription was removed (`suspended=0`).
- The two C3 acquisitions per model share one boot; the model order is fixed.
  Mega acquisitions are separately uploaded runs. Do not treat samples within
  a CFN, repeated inputs, or the C3 pair as independent board specimens.
- This campaign and historical 500-flow latency/factorial experiments have
  different protocols. Keep their tables and conclusions separate.
- Linker static RAM, model arrays, firmware image size and physical peak RAM
  are different quantities. **Peak RAM was not measured.**

## Source and evidence layout

| Path | Content |
|---|---|
| `measurements/` | Twenty unchanged native CFNs, run records, serial JSONL/raw bytes and accepted trace analyses |
| `firmware/mega/{coeff,lut,mlp,kanml,dt5}/` | Five standalone PlatformIO projects and exact acquisition runners |
| `firmware/c3/` | One standalone PlatformIO project with five environments and the exact suite runner |
| `results/` | All acquisition values, ten means and descriptive representation/board ratios |
| `provenance/` | Original source identities, C3 suite record, binary/provenance/trace audit results |
| `IMPORTED_FILES.json` | Original archive/member provenance and SHA-256 for every imported file |
| `MANIFEST.json` | Complete compact experiment export identity |

The firmware and runner **bytes** are those used in the recorded acquisitions.
Their new `SOURCE_MANIFEST.json` files describe the compact Git packaging;
they are not the original acquisition manifests. Original manifests are
retained separately and the complete source packages remain in the checkpoint.
No measured source has been edited merely to fit repository formatting rules.
Build products, historical development tests, logs and nested archives are
kept in the checkpoint rather than duplicated in Git. Six compact manifests
cover all project headers, firmware sources and runner dependencies.

Building uses PlatformIO 6.1.19. AVR uses atmelavr 5.1.0;
C3 uses espressif32 6.12.0 and Arduino ESP32 3.20017.241212.
For example, build all five C3 environments without flashing:

```console
python -m platformio run -d experiments/hardware_energy_20260915/firmware/c3/project
python -m platformio run -d experiments/hardware_energy_20260915/firmware/mega/coeff/project
```

For acquisition options, invoke the corresponding `run_mega_energy_pilot.py`
or `run_c3_energy_suite.py` with `--help`. Their original board-identity and
source checks remain enabled. On Windows use Python 3.11 and set
`PYTHONIOENCODING=utf-8` when running PlatformIO commands outside the runners.
A different board requires a separately documented protocol; do not disable
the identity checks just to collect additional rows.

## Relationship to existing results

`mcu_pio/`, `results/firmware_size.csv` and
`results/firmware_size_pending.csv` retain their existing protocol and history.
The pilot environments above are distinct. Successful pilot builds do not
close the old twenty-environment build register. Likewise, this experiment
does not modify the saved ML/transfer results or the Paper 2 branches.

# HW500 publication integration

This directory derives accepted results from the original Mega and C3 continuous sessions. It does not acquire new hardware measurements. Five campaigns per model are included; the separate coefficient pilot on each board is retained and excluded from means.

## Repository files

- `build_integration.py`: standard-library only; independently recalculates ten mean/SD rows from extracted original session CSV files and compares original JSON summaries. Generates publication data, 52 raw run rows including pilots, and bilingual tables.
- `replay_hw500.py`: unchanged portable retained numerical audit; validates 29 packet manifests per board, original source identity, per-run records/software payloads, and recomputes marker registration and all 26 integrals per board.
- `HW500_PUBLICATION_DATA.json`, `HW500_RUNS.csv`: full precision values and explicit scope/limitations. Main manuscript table rounds for display.
- `HW500_TABLE_EN.tex`, `HW500_TABLE_RU.tex`, `HW500_PARAGRAPHS_EN.tex`, `HW500_PARAGRAPHS_RU.tex`: editorial inputs; original twenty-flow pilot should remain labelled as historical, not pooled.
- `INTEGRATION_AUDIT.json`, `replay_mega/AUDIT.json`, `replay_c3/AUDIT.json`: numerical replay evidence.
- The complete original v0.13.3 acquisition/analysis source kit, including manifest, protocol, frozen headers, bootstrap, PlatformIO firmware, tests and analysis scripts, is needed to reproduce acquisition or replay. Preserve original kit manifest bytes. It can reside in the repository under a versioned directory or as an immutable release archive.

## Release evidence

Publish unchanged original ZIPs (hashes in `INTEGRATION_AUDIT.json`) as research release assets. They contain continuous CFNs, all UART bytes/JSONL, run/protocol/source manifests, flashed binaries, and build/upload logs. Use a lightweight hash/pointer record in Git rather than multiple expanded copies of identical artifacts. No raw session may be relabelled as a fresh experiment. Historical energy and timing datasets remain separate assets with their original protocol labels.

## Reproduce without connected hardware

Extract both original result ZIPs into one `raw_sessions` directory. Extract the unchanged `KAN_IDS_HW500_CONTINUOUS_20260916_v0.13.3` kit separately. The original pinned analysis environment is Python 3.11, numpy 2.3.5, scipy 1.17.0, pyserial 3.5 and PlatformIO 6.1.19. All computations here were rerun using Python 3.12, numpy 2.3.5 and scipy 1.17.0; serial and PlatformIO packages are not required for this read-only replay.

```text
python replay_hw500.py --session raw_sessions/MEGA_HW500_CONTINUOUS_20260916T090249Z_b3712c5a --kit <unchanged-v0.13.3-directory> --out <fresh-mega-replay-directory>
python replay_hw500.py --session raw_sessions/C3_HW500_CONTINUOUS_20260916T105447Z_5118ea3a --kit <unchanged-v0.13.3-directory> --out <fresh-c3-replay-directory>
python build_integration.py --sessions raw_sessions --output <fresh-publication-output-directory>
```

`replay_hw500.py` requires a fresh output directory outside the original session. The raw evidence is never rewritten. Reanalysis passed for both boards; maximum absolute energy differences from the saved acquisition-side analysis were 6.035e-9 µJ (Mega) and 3.105e-11 µJ (C3).

## Fixed acceptance/exclusion rules

The versioned protocol requires correct board identity and all 500 replay predictions before acquisition, whole-cohort batches with correct accumulated checksum, active durations 60–180 s, unambiguous chronological association of two LED groups per run, valid input voltage throughout the run/markers, and bounded marker-fit quality. The marker conditions are amplitude at least 0.05 mA, normalized RMS at most 0.15, maximum edge residual 0.4 s, edge RMS 0.2 s, and affine clock scale within 0.995–1.005. Acquisition stops on software/identity failure; a restart of the recorder requires a new session. Trace acceptance occurs only after the full continuous file is saved. No failed attempt is silently resumed or pooled. In these two accepted sessions there were no failed campaign attempts; each separate pilot was excluded by design. Mega had one unmatched auxiliary LEDTEST group outside working intervals, retained and documented; all 52 working marker groups were uniquely matched.

## Remaining limits

No independent calibration of meter or clocks exists, so these are descriptive whole-board USB engineering estimates. The 0.1 s storage grid is not 10 Hz bandwidth. Sample SD describes repeatability on one board per type, not instrument accuracy. The common ±2 s endpoint-shift sensitivity is at most 0.03358% for Mega and 0.60438% for C3; it is not a full uncertainty budget. No idle subtraction or NRG multiplier is used. The hardware models are the historical frozen exports, not the Stage2 retrained pair-disjoint models. HW500 energy firmware static allocation differs from the later RAM diagnostic firmware; their memory figures must not be combined.

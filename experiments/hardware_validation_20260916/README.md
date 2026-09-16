# Accepted follow-up evidence, 16 September 2026

This integration adds three distinct experiments to Paper 1. It does not replace the historical protocols or merge their results.

| Evidence | Models and input | Accepted scope |
|---|---|---|
| `paired_software/` | Newly fitted KAN, DT5, MLP16 and GAM; one pair-disjoint TON_IoT split; five seeds | Software quality audit on 38,087 held-out rows. No integer-export certificate or hardware measurement is inherited by these new fits. |
| `hw500/` | Frozen canonical integer models; shared 500-flow cohort | Time and board USB energy from the same prepared-feature batch; 5 technical repetitions/model/board. One pilot/board is excluded from the 50 publication intervals. |
| `ram500/` | Frozen canonical integer models; same prepared hardware cohort | 15 fresh uploads/board; 3 passes/upload. Observed diagnostic RAM components and positive controls, not a global worst-case bound. |

## Fast verification (no hardware, no training)

From the repository root, Python 3.11+:

```sh
python experiments/hardware_validation_20260916/verify_evidence.py
python -m unittest discover -s experiments/hardware_validation_20260916/ram500/tests -v
```

The verifier reconstructs the new table values from committed per-seed/per-run numbers and RAM records. It explicitly reports that external raw archives were not replayed. A successful source-only check is not a fresh hardware experiment or an independent metrological calibration.

## Original evidence and full replay

Raw ZIPs, ELF images, native CFNs, serialized fits and training data are external release artifacts, not duplicated in Git. Exact archive SHA-256 values are listed in `paired_software/INPUT_ARCHIVES.json`, `hw500/INTEGRATION_AUDIT.json`, and `ram500/PROVENANCE.json`. The artifact map is `evidence/review_v015/ARTIFACT_MAP.md` at the repository root.

- Pair split/freeze audit: `paired_software/audit_pair.py --stage1 <extracted-stage1> --stage2 <extracted-stage2> --kit <original-stage2-kit> --out <new-report.json>`.
- Pair saved-score replay: `paired_software/verify_saved_study.py --run <extracted-stage2> --csv <original-source.csv> --out <new-report.json>`. This requires NumPy and does not load fitted models or train.
- HW500 CFN replay: `hw500/replay_hw500.py --session <extracted-session> --kit <original-v0.13.3-kit> --out <new-directory>`. Run separately for Mega and C3. Use the original kit, not the maintained source copy below, because its historical manifest is verified.
- RAM archive/UART replay: `ram500/verify_ram.py --artifacts-dir <directory-containing-the-two-original-RAM-ZIPs>`. It checks archive SHA/CRC, packet manifests, UART/JSONL consistency, protocol controls and reconstructs both summaries. It does not rebuild binary disassembly. Original full binary audits are retained under `ram500/audit/` and require the historical release/toolchain indicated by their `--help`.

## Maintained source versus measured firmware

`ram500/runner/` and `hw500/runner/` retain executable runner, bootstrap, firmware, frozen headers, source controls and regression tests. Their `SOURCE_ORIGIN.json` records the upstream release, original source hashes and exact firmware identity. Their `UPSTREAM_MANIFEST.json` is archival evidence; their new `KIT_MANIFEST.json` describes the maintained source integration, with bulky historical build evidence removed.

RAM firmware `project/src/main.cpp` is byte-identical to accepted v0.14.2. Mega was measured with v0.14.1; the accepted binaries and their hashes remain in the original archive. The only maintained RAM post-link correction is the exact diagnostic-symbol filter: the original substring match incorrectly counted six SDK symbols totalling 33 bytes. Corrected diagnostic subsets are 188 B (coefficient/LUT/MLP/multilayer KAN) and 192 B (DT5). Total static DRAM and all physical observations are unchanged. Original records keep the original 221/225 B metadata; no historical evidence is rewritten.

No board needs another run for the present paper. Rebuilding these maintained source trees would create a new build and new provenance, not recreate an already accepted physical session by declaration. The bootstraps can flash hardware; do not use them for the read-only commands above.

## Interpretation limits

C3 pre- and post-inference lifetime stack watermarks are identical in every accepted run. The 1,504/1,584 B values include task history and establish no model-specific stack advantage. C3 task reservations are part of allocated heap, and row buffers are part of static data: do not add them again or sum columns into a global peak.

HW500 energy is an engineering estimate at the board USB input. Five within-device repeats quantify repeatability, not device-population variation or calibrated uncertainty. No independent meter/clock calibration was performed. The nominal 0.1 s CFN storage grid is not a demonstrated 10 Hz bandwidth; NRG is retained but excluded from the primary integral. Historical 20-flow pilots and the original C3 +7.58% timing result remain separate, with their original scope.

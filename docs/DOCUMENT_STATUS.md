# Paper 1 document and evidence status

Date: 2026-09-08. Overall status: **NOT_HARDWARE_MEASURED**.

The RC3 snapshot and original report are immutable provenance. Current edits
are confined to the working copy. No new dataset, architecture, full CV or
model-weight training cycle is introduced. `adattamento-drift/` remains Paper 2.
No publication, push or flashing is authorized by these documents.

## Active sources

| Item | Current authority | Evidence boundary |
|---|---|---|
| Claims | `docs/CLAIM_EVIDENCE_MAP.csv` | Exact source paths and SHA256; saved author results and current software checks are distinct |
| LUT selection | `results/lut_selection_protocol.json` | Training-only calibration; L1025 is fallback maximum of a fixed candidate grid, not an accuracy optimum |
| LUT test agreement | `results/lut_vs_coeff_postfreeze_test.csv`, `results/lut_postfreeze_test_protocol.json` | 42209/42209 empirical decisions agree after freeze; historically explored test set |
| Training/preprocessing provenance | `artifacts/finalization/train_calibration.json`, `preprocessing_recovery_checks.json` | Model-specific frozen preprocessing reconstructed and compared with saved golden vectors; no weight retraining |
| Common-cohort protocol | `docs/HARDWARE_COMMON_COHORT_IT.md` | Kernel-only measurement boundary and exact new environments |
| Common raw flows | `artifacts/finalization/hardware_cohort.json`, `hardware_flow_ids.csv`, `hardware_raw_flows.csv` | 500 unique IDs, 250+250, same order; per-model transforms |
| Common firmware export | `artifacts/finalization/hardware_cohort_export.json` | Generated header and input/model/kernel provenance; rebuild target toolchains before use |
| Host cohort checks | `artifacts/finalization/host_hardware_cohort_checks/summary.json` | 20/20 host checks on simulated architecture branches; not physical latency/energy |
| XAI | `results/interpretabilita_provenance.json` | Training support for 168834 rows; exact terms; local examples use historical golden IDs 7,140,29 |
| XAI execution logs | `artifacts/finalization/xai_validation/run_summary.json` | 19 passed in the logged run; not a statement about every repository test |
| Physical measurement protocol | `docs/HARDWARE_ENERGY_PROTOCOL_IT.md`, `templates/energy_acquisition.json` | Pending exact instruments and board power/marker configuration |
| New software report generator | `scripts/make_report.py` | Produces `report_KAN_IDS_PAPER1_software_review.pdf`; generation and visual QA require their own logs |

The current LUT uses 20554 B of arrays versus 254 B for coefficients. It has
B=4271 on the complete admissible integer Q12 grid. Of 168834 calibration rows,
34 do not satisfy the sufficient margin condition, although all empirical
training decisions agree. On the test set, four rows are uncertified and none
changes decision. Both full-test F1 values equal 0.9825844106941132.

The condition `abs(logit_coeff) > B` guarantees decision agreement only under
the stated common categorical and integer-arithmetic assumptions. It is not
a guarantee of label correctness or every future input. L513 also has zero
empirical calibration mismatches; L1025 is selected by the declared fallback
because no fixed candidate certifies every calibration decision.

## Historical material and limits

`results/lut_vs_coeff.csv` and `results/lut_vs_coeff_test.csv` retain author
L257/5194B results, whose choice used test-derived margins. The original
`report_KAN-IDS_fase2.pdf`, any `report/report.pdf` in the uploaded package,
`AUDIT*.md` and historical changelogs are not current readiness certificates.
`tools/audit_richieste.py` is a legacy author request-audit tool; until its
LUT/result-generation assumptions are updated it is not a finalization gate.
Do not copy its pass counts, completeness claims or old LUT table into a new
report. Existing author `results/firmware_size.csv` describes RC3 linker sizes,
not modified headers, new common environments, flashing or peak SRAM.

The active README removes ordinary dependent-fold p-values from claims,
separates descriptive CV from single export-fit quality, treats h16/g8 as a
deliberate deployment choice, and does not interpret depth comparisons as
causal evidence. UNSW is not a ceiling; failed significance is not equivalence.
Current XAI figures use training support, true/predicted labels and semantic
categories. Computational terms are not causal attribution; numeric port/DNS
codes do not imply continuous physical relations.

## Remaining inputs and gates

1. Confirm exact Mega and ESP32-C3 board revisions and available host ports.
2. Identify the current/energy instrument, shunt/range, sample rate, bandwidth,
   accuracy and raw export format; simultaneous V/I and two-marker recording
   or an explicitly validated synchronization method are required.
3. Define measured rail, supply voltage, USB/regulator contribution, grounds
   and marker electrical compatibility using the actual instruments.
4. Build the modified source and generated cohort with target toolchains;
   retain versions, flags/LTO, source/header/binary hashes, linker sizes and
   compiler-loop evidence. Historical RC3 binaries cannot serve this gate.
   Once the board/instrument profile and builds are confirmed, freeze the
   hardware-ready sources and binaries before the first physical measurements.
5. After supervisor confirmation, perform the smoke flash and board checks,
   acquire raw latency/energy traces and measure or explicitly scope SRAM.
   Only then fill physical result tables and prepare the measured research release.

Text work may proceed while instrument details are collected. Missing hardware
inputs do not authorize host/Wokwi estimates as physical measurements.

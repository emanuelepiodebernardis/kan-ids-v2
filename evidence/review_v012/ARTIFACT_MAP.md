# Paper 1 v0.12 executable evidence map

This is a post-hoc review of frozen models and saved experiments. No model was
trained, no target threshold/polarity was optimized, and no new hardware run
was performed. Current receipts below are separate from historical RC3 and
v0.11 receipts. Code, model, `results/`, `papers/` and experiment paths are relative to the
repository. Receipt basenames and `overlap/` paths are relative to this
`evidence/review_v012/` directory; **release root** marks external bundled inputs.

| Claim or boundary | Current executable source / receipt | Historical source or input boundary |
|---|---|---|
| Quantized cubic basis sum 196607..196609; signed ceiling and int32 intermediates/accumulators | `scripts/audit_q15_bounds.py`; `q15_bounds.json` | Frozen `mcu_pio/include/kan14*_coeff_int8.h`; actual counterexample q=-4094, t=128 |
| Rich UNSW 57/60 and reduced UNSW 41/60 AUROC below 0.5 | `scripts/analyze_saved_results_v012.py`; `saved_results_unsw_auc.csv`; `saved_results_summary.json` | `results/joint_training_runs_ratio5_cat.csv`, `results/joint_training_runs_ratio5_ridotto_cat.csv`; ten seeds, six models per space |
| Validation 1:5 winner, including all ten per-seed comparisons | `saved_results_ratio_seed_winners.csv`, `saved_results_ratio_means.csv` | `results/joint_ratio_selection_runs.csv`; exact per-seed means, historical four-decimal global rule |
| Exact 44-column raw-row overlap: 5201 / 42209; fixed-model subgroup quality | `scripts/analyze_overlap_subgroups.py`; `overlap/overlap_subgroup_results.json`, `overlap/subgroup_metrics.csv`, `overlap/canonical_test_row_audit.csv.gz` | **Release root** `analysis/evidence/raw/train_test_network.csv`, `analysis/evidence/finalization/*`; hashes in `SOURCE_PROVENANCE.json` and results |
| Signed L=1025 and diagnostic L=513 certificates, fixed-cohort bounds and host equality | Same script; `overlap/subgroup_certificates.csv`, `overlap/overlap_subgroup_results.json` | Current frozen headers plus **release root** `analysis/evidence/headers`, `analysis/evidence/historical/analyze_certificate.py`; legacy B-selection `results/lut_selection_protocol.json` remains historical selection evidence |
| Paired floating versus coefficient predictions; no training or new preprocessing fit | `scripts/recover_frozen_float.py`; `float_recovery_report.json`; `FLOAT_RECOVERY.md` | Current checkpoint plus archived preprocessing arrays; exact Q12/code/label equality is checked, but unavailable original pre-Q12 transformer bit identity is not claimed |
| Native CV task-specific MI, canonical KAN export multiclass MI, DT/MLP export binary MI | `scripts/cv_leakagefree.py`, `kanids/legacy.py`, `scripts/kan14_compile.py`, `scripts/kan14_ml_compile.py`, `scripts/export_tree_c.py`, `scripts/export_mlp_int_c.py` | Corrected `results/protocol_*.json` descriptions identify scope; no saved fit or metric modified |
| CIC common columns are proxy mappings, not proven physically equivalent flow measurements | `kanids/harmonized.py` documentation | Historical numerical mapping unchanged, including `Number`/`Tot size` fallback and flag-state rule |
| Physical energy evidence | `experiments/hardware_energy_20260915/verify_measurements.py` and experiment manifests | Twenty original acquisitions, frozen firmware, supplied photos; historical latency/factorial archives are mapped by **release root** `evidence/ARTIFACT_MAP.md` |
| Current manuscript sources and numerical checks | `papers/ieee_access`, `papers/ru`; their `MANIFEST.json` is Git-source-specific | Standalone release includes compiled PDFs and separate source/PDF manifests |
| Current report generation | `scripts/make_report.py`, `report_provenance.json`, `report_visual_qa.json` | Earlier `evidence/finalization/report_visual_qa.json` applies only to the earlier report |

## Commands

From the repository root, Python 3.12 plus the supplied review dependency list:

```bash
python scripts/verify_review_v012.py
python scripts/audit_q15_bounds.py
python scripts/analyze_saved_results_v012.py
python scripts/recover_frozen_float.py --repository . --evidence ../analysis/evidence --out ../analysis/float_replay
python scripts/analyze_overlap_subgroups.py --evidence ../analysis/evidence --out ../analysis/overlap_replay --floating-predictions ../analysis/float_replay/frozen_singlelayer_float_predictions.npz
python experiments/hardware_energy_20260915/verify_measurements.py
python papers/ieee_access/tools/verify_package.py papers/ieee_access
python papers/ru/tools/verify_package.py papers/ru
python papers/verify_hardware_links.py
python -m pytest -q
python scripts/make_report.py
```

Use fresh output folders for numerical replays. `g++` is required by the
full-edge host C probes; AVR tests additionally require the PlatformIO AVR
toolchain or `AVR_CXX`. The raw CSV and transformed inputs accompany the private
review release and are deliberately outside Git. An applied patch alone does
not confer public redistribution permission or supply those inputs. The exact
archive members and SHA-256 values are retained in `SOURCE_PROVENANCE.json`.
Do not rerun training or substitute a newly fitted preprocessor to recreate a
historical receipt. The overlap definition includes labels and addresses and
does not establish host/time disjointness or a deduplicated-training result.

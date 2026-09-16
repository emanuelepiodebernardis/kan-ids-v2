# Endpoint-pair supplemental software study

This directory integrates the completed Stage1 validation pilot and Stage2 five-seed test. It contains no new training, checkpoint selection or hardware experiment. The original archives are identified by byte count and SHA-256 in `INPUT_ARCHIVES.json`; their CRC checks pass.

`PAIR_STAGE2_SUMMARY.json` preserves all per-seed test confusion counts, metrics, descriptive summaries, paired differences, subgroup summaries, split definitions, selected inputs and model configurations. `STAGE1_VALIDATION_METRICS.json` is the earlier single-seed validation pilot, not another test repetition. The EN/RU LaTeX fragments are manuscript material.

## Verified results

- Stage1: all 214,488 stored validation predictions (four models × 53,622 rows) checked against their stored decisions and confusion counts/rates.
- Stage2: all 761,740 stored test predictions (four models × five seeds × 38,087 rows) checked; probabilities are finite and in [0,1], decisions use threshold 0.5, row labels/types and endpoint groups agree with the source. Full-test, overlap-subgroup and attack-type metrics were independently recomputed. Tie-aware rank-sum AUROC agrees to at most 2.22e-16. These counts describe checks of stored predictions, not independent observations.
- The 60 files named by the fit-freeze receipt match their hashes; the recorded receipt precedes recorded test evaluation. This verifies internal provenance consistency, not independent attestation of an invisible process.
- 718 integrity checks pass: manifests cover 45 Stage1, 111 Stage2 and 48 kit payloads; 24 captured Stage2 source files match the supplied kit; all 211,043 pair assignments were independently reconstructed. Decoded 44-field source tuples have zero cross-partition overlap. Host overlap remains.
- The Stage1 and first Stage2 seed have identical bytes for seven stored model/selection artifacts, including all four models; fitted-preprocessor JSON metadata is identical. The serialized `shared_preprocessor.joblib` bytes differ between those two invocations. No claim of byte identity is made for that file and this audit does not deserialize it. Across the five Stage2 seeds all frozen preprocessor JSON metadata hashes match.

The test has 3,669 normal and 34,418 attack rows. BA (mean ± sample SD): additive KAN 0.894844 ± 0.000220; DT5 0.685096 ± 0; MLP16 0.988852 ± 0.000624; spline GAM 0.970031 ± 0. KAN FPR is 20.594%, although F1 is 0.986953. This is a substantive limitation, not evidence of deployment readiness. KAN outperforms DT5 on this split but underperforms MLP16 and the additive spline baseline.

## Scientific boundaries

The fixed SHA-256 partition groups each unordered source/destination IP pair, without ports. It is pair-disjoint and full-source-row-disjoint, not host-, time- or feature-disjoint. Train/validation/test have 119,334/53,622/38,087 rows and 601/213/190 pair groups. Training and test share 50 hosts. The test lacks backdoor, ransomware and XSS.

The stored transformed-input overlap mask marks 2,889 test rows as matching training inputs (7.585%); the matching subgroup has 46 normal and 2,843 attack rows. The mask is consistent across seeds and with saved reports. The present read-only audit does not reconstruct fitted preprocessing from pickle to rederive this mask. Raw candidate-input overlap was verified by the archived original structural audit, but it is not a substitute for transformed-input equality.

Five seeds describe optimization variation on one fixed split; no confidence interval, significance test, five-independent-dataset claim or pooling of confusion counts is warranted. The KAN has a fixed 250-epoch budget; absence of a convergence warning does not establish optimizer convergence. Protocol settings were frozen locally, not externally preregistered; Stage1 validation and older canonical metrics were known before Stage2. The split, model settings, seeds and threshold were unchanged and no best seed was selected.

These are new floating-point models trained on the TON_IoT source file. They are distinct from the frozen coefficients/LUTs used for integer agreement certificates, latency, energy and RAM. This study does not update those certificates or establish a hardware Pareto point for a newly trained model. Differences from the canonical split cannot be interpreted as causal estimates of leakage bias.

## Read-only replay

Use Python 3.9+ with NumPy. Extract the three archives from `INPUT_ARCHIVES.json`; arguments below refer to each directory containing its `RUN_RECORD.json` (runs) or `KIT_MANIFEST.json` (kit). Set the path placeholders to those extracted directories. Each output must be new and outside its input directory. Neither script loads serialized models, fits a model or touches hardware.

```text
python audit_pair.py --stage1 <stage1_run> --stage2 <stage2_run> --kit <stage2_kit> --out <new_integrity_report.json>
python verify_saved_study.py --run <stage2_run> --csv <stage2_kit>/data/train_test_network.csv --out <new_prediction_report.json>
```

The full source CSV, row-aligned score tables and fitted models stay in the original evidence archives; this compact integration directory alone cannot regenerate predictions or train models. The original kit contains the fixed training implementation if a separately authorized full reproduction is needed.

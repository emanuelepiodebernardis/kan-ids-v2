# Review revision v0.12

The revision preserves the saved training results, fitted model parameters,
integer inference arithmetic, measured firmware snapshots, hardware observations,
and earlier authorship history. No classifier or preprocessor was fitted.

- The Q15 safety argument now uses the implemented quantized cubic bases and
  signed arithmetic-shift ceilings, checking intermediates and layer sums. The
  shared header change is explanatory only; `q15_source_provenance.json` records
  its old/new hashes and the unchanged measured firmware copies.
- Native CV metadata now names the actual task-specific MI target. The canonical
  single-layer and multilayer KAN exports use a separate multiclass-target
  preprocessing path; DecisionTree and MLP exports use binary-target MI. This
  describes the checked-in paths and does not invent missing execution lineage
  for a historical saved fit. No metric CSV was overwritten.
- CIC docstrings describe the existing window/packet-derived proxy mapping and
  preserve its formulas. Presence of common column names is not proof of
  physical flow-measurement equivalence.
- Saved UNSW AUROC and validation ratio comparisons are recomputed directly,
  retaining polarity and using sample SD. Exact-duplicate subgroup and signed
  certificate analyses are post-hoc fixed-model diagnostics. Neither a new
  independent holdout nor duplicate-aware retraining is claimed.
- Original versus reconstructed floating preprocessing identity is stated
  explicitly. Preserved Q12 inputs and category codes match exactly; unavailable
  original floating-transform bit identity is not asserted.

See `evidence/review_v012/ARTIFACT_MAP.md` for exact claims, commands, current
receipts, historical artifacts and the external input boundary. The separately
delivered review report has a current deterministic generator/provenance record;
older PDF receipts retain their historical scope.

# Frozen floating KAN recovery

`frozen_singlelayer_float_predictions.npz` evaluates the unchanged current
`repository/models/kan14_binary_singlelayer.npz` on the 42,209 canonical test
rows. No classifier fitting, preprocessing fitting, threshold selection, or
retraining was performed.

Arrays, in exactly the row order of `evidence/finalization/test_evaluation.npz`:

| Array | Definition |
|---|---|
| `row_ids` | Zero-based original CSV data-row positions |
| `y_true` | Binary labels, normal=0 and attack=1 |
| `float_score` | Original degree-8 Chebyshev KAN logit |
| `float_probability` | The original sigmoid, after clipping logits to [-30,30] |
| `float_pred` | `float_probability >= 0.5`, equivalent here to `float_score >= 0` |

The original CSV test rows were transformed using the preserved quantile arrays,
feature order, log1p mask, vocabulary, and clipping bound. These are floating
inputs **before Q12 rounding**, not dequantized integer inputs. Re-quantization
matched all 422,090 preserved numeric values exactly; all 168,836 categorical
codes, all labels, and the sorted train/test row partitions also matched.

Full-test attack F1 is **0.9831911690918214** and ROC AUC is
**0.9956369368809961**, reproducing the published rounded values of 0.9832 and
0.9956 in `repository/results/kan14_binary_real.csv`. The confusion matrix is
`[[9785, 215], [857, 31352]]` (rows=true, columns=predicted; labels 0,1).

Run from the bundled project directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python analysis/float_recovery/recover_frozen_float.py
```

Defaults are resolved from the script location, independent of the working
directory: evidence in `analysis/evidence`, inference sources in `repository`,
and outputs beside this script. Override with `--evidence`, `--repository`, and
`--out` when needed.

`float_recovery_report.json` records source hashes, runtime versions, verification
results, metrics, and limitations. The model SHA256 is
`344bbd9db0c0c95cad7557f72c8982844662602329148be8284a5f40882b17cc`, matching the
preserved model-immutability evidence.

The preserved preprocessor was itself reconstructed during prior finalization
because the original fitted transform was not saved. Exact Q12 agreement cannot
prove bitwise identity of pre-Q12 values to that unavailable original transform;
the rounded published metric matches provide additional consistency evidence.
Duplicate-subset evaluation is a post hoc sensitivity analysis, not
duplicate-aware retraining or a new untouched test set. Legacy protocol-v1
weights have a different category layout and are not used.

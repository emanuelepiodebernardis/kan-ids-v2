# Review extension, fixed before new predictive runs

Date: 1 November 2026. This is a post-review extension, not a replacement for
the October D campaign. No new outcome was inspected when this protocol was
written. TON D analyses are exploratory because its original results are known.
UNSW-NB15 is a separately collected benchmark; this is a new chronological
evaluation of the same questions, not a claim of field deployment.

## Questions and fixed comparisons

1. Does the head add value beyond a threshold-only controller that can use all
   acquired labels? Compare frozen/CAL256 with FIT192/CAL64, reporting both
   error rates, counts, balanced accuracy, and the seen/unseen strata.
2. Does coupling the threshold update to the fitting gate matter? Compare the
   original frozen/CAL64 with independently eligible frozen/CAL64.
3. Is an intercept correction sufficient? Fit only an offset on FIT192,
   retaining the initial score, and report it with B and its own CAL64 threshold.
   With freely reselected thresholds, a constant offset cannot change ranking
   or the attainable decisions. This is a diagnostic, not a new classifier.
4. Does regularization toward the previous head alter the result? Minimize
   summed logistic loss plus 0.5 times squared distance of the eight weights
   from the previous weights, with an unpenalized intercept. Compare with the
   existing zero-centred L2 refit; same data, gate and CAL64 selection.
5. Does the result extend beyond the additive spline representation? Include
   ordinary logistic regression and River 0.22.0 ARFClassifier (10 trees,
   defaults apart from seed). ARF learns all arriving queried labels once,
   in query order, with an independently selected CAL64 threshold; its label
   and model-state requirements are reported separately from the 256-example
   head-fitting policy. Its CAL64 labels are reused for training, so this
   baseline is a practical equal-acquisition-budget comparator, not a
   disjoint-memory causal attribution experiment. Also report ARF at 0.5 and B.
6. What operating compromises are attainable using only past labels? Select
   thresholds minimizing FNR + lambda FPR, lambda in {0.25, 0.5, 1, 2, 4},
   on the respective recent calibration memory. Lambda=1 is the original BA
   objective. Plot resulting future-stream FPR/recall points; do not construct
   an oracle frontier using evaluation labels or tune lambda on D.

## Common design

- Seeds 42--46; 10,000-row prediction blocks; 1% uniform requests; labels from
  block k arrive after prediction of block k+1 and first affect block k+2.
- Initial transformation from all A, and exactly the same seeded uniform
  400,000 A observations for fitting each model family. ARF processes this
  sample in chronological order; logistic models fit it as a batch.
- B selects each model's initial threshold separately. No model learns B
  labels, and no adaptation state is carried from C into D.
- The extension uses SeedSequence([seed, block, 20261101]) for queries. This
  avoids the original seed+block reuse. All methods receive exactly the same
  requests within a dataset/seed/budget/delay condition.
- A separate TON spline replication keeps the legacy seed+block query rule.
  Original files and reported results remain immutable; any numerical drift
  from the new software/hardware environment must be quantified.
- Keep the original salt, class-quota eviction, FIT192, CAL64, 50-block age
  limit and five observations per class. CAL256 receives ALL acquired labels,
  uses quotas 128/128, the same age limit/minimum, and updates independently.
- Sensitivity: all five seeds of the spline detector at delays 0, 5, 20
  (1 is the main condition), and at budget 0.1% with delay 1, on TON D.
  Nothing is selected as a new primary configuration after seeing results.

## Data provenance and second dataset

TON: use all 23 CSVs from the previously registered Kaggle mirror version 1,
verify each SHA-256 against the existing source manifest, and run the original
normalization and timestamp parser. Reconstruct original A/B/C/D partitions.
Report both file and array digests and check published counts/overlap.

UNSW-NB15: full four raw CSVs from mrwellsdavid/unsw-nb15 mirror version 1,
not the preselected training/testing tables. Record all file hashes. The mirror
has three exact duplicate file-boundary rows (the previous file's last row is
the next file's first row); remove only these three verified duplicates to
recover 2,540,044 records. Use eight complete-flow numeric features:
dur, sbytes, dbytes, sloss, dloss, Spkts, Dpkts, tcprtt. This preserves a compact
numeric contract but is not claimed to be the identical TON feature ontology.
Sort by max(Ltime, Stime + float32(dur)), then original row order. Split valid
rows chronologically 40% A, 20% B, 40% D; no random split or D-based tuning.
Use the A/B inventory for overlap strata. Training starts anew on this dataset.
Record count and duration of each partition, timestamp ties, and per-attack
results. The official dataset describes a separate lab capture, not a live
network validation.

## Reporting and inference

All predictions precede feedback. Preserve per-block confusion counts,
thresholds, within-block AUROC (undefined blocks explicit), label-request
digests, update eligibility, calibration ages, and per-family attack recall.
Report pooled AUROC only for the fixed initial score, alongside macro block
and normal-count-weighted block AUROC. Measure actual elapsed time and model
serialization size; neither is an embedded-device benchmark.

Compute the original four-trace BA interaction, seen/unseen error decomposition,
and precision at hypothetical 1% and 0.1% attack prevalence, explicitly assuming
the observed class-conditional rates transfer. Report annotation volume and
the empirical distribution of elapsed seconds to label arrival/influence.

Seed SDs describe randomization conditional on each stream. Do not present
five seeds as independent datasets or use the nominal sign test as evidence
of significance. Moving-block resampling lengths 5, 20 and 50, 2,000 replicates,
is an optional descriptive sensitivity of *recorded paired errors*, not a
confidence interval for a retrained adaptive process under nonstationarity.
All pre-specified comparisons are retained, including unfavorable outcomes.

## Scope boundaries

No model novelty or superiority claim is assumed. The original four-of-five
joint FN/FP criterion remains visible and is not replaced after failure.
Budget equality concerns acquired labels; effective labels, disjointness,
memory, and update schedules are stated per method. Existing full-stream
structural exposure is disclosed. Public mirrors are hash-identified, without
claiming byte identity with an unavailable publisher checksum.

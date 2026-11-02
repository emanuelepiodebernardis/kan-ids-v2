# Second-review diagnostic extension

Date: 1 November 2026. Parent evidence: `f1f1d05`.

All analyses in this extension are exploratory and conditional on the two
previously inspected captures. Fixing this document before the new runs prevents
undocumented changes within this audit; it does not create an independent test.
The original runner, 55 reports, and model artifacts remain unchanged.

## Recorded-evidence checks

Recompute all lambda operating points, joint FN/FP counts for each learner,
the legacy bridge, and identical-threshold frequencies. Interpolate the existing
paired-seed FPR/recall polylines only within observed support, with no
extrapolation. These are descriptive matched-error comparisons, not a fitted or
deployable policy. Also report interpolation of the mean curve, explicitly
distinguishing it from mean paired-seed interpolation. No lambda is selected as
the new preferred configuration using D.

## Fixed diagnostic runs

Use the original extension queries, delays, initial model states, five seeds
42--46, and full D streams. Preserve predict -> query -> delayed arrival ->
update ordering. The original 55 runs are not overwritten.

1. **Spline and LR:** reproduce the main trajectories and add causal thresholds
   targeting calibration FPR alpha in {0.05, 0.10, 0.20}. Choose the lowest
   observed negative-score threshold with empirical FPR <= alpha, using strict
   score > threshold decisions and whole tie groups. B supplies the initial
   threshold. Frozen uses CAL256 and head uses its disjoint CAL memory.
   Report achieved future FPR and recall; matching requested alpha does not
   imply matching achieved FPR under shift. Also retain lambda in
   {0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4} for the main trajectories.
2. **Frozen-score audit:** verify the attack-label orientation, agreement of
   cached scores with direct model predictions, LR decision_function versus
   coefficients and predict_proba, classes and convergence status. Report
   A/B/D class counts, score distributions and fixed-score AUROC/AP. A is
   descriptive training data, not an out-of-sample estimate. Inversion on D
   is diagnosed but never used to flip a deployed score using D labels.
3. **ARF training-order control:** fit an additional forest on the exact same
   400,000 A sample in its seeded random order instead of chronological order.
   Keep all other parameters fixed. Evaluate its frozen and adaptive traces.
   Retain the original chronological forest as historical protocol evidence.
4. **ARF label-use control:** start from the original chronological model and
   learn only arriving FIT-role labels. Select thresholds exclusively on CAL64
   observations, which are never learned. Report retained tree state and the
   actual learning-label count. This aligns label disjointness, not total RAM
   or the finite FIT192 retention contract.
5. **Spline controller sensitivity:** one factor at a time, both datasets and
   all five seeds. FIT/CAL capacities 96/32 and 384/128 (default 192/64);
   CAL age 10 and 100 blocks (default 50); minimum 10 per class (default 5);
   two additional fixed role salts, `review2-salt-1` and `review2-salt-2`.
   The all-label frozen comparison retains CAL256. No best setting is selected.
6. **PR summaries:** noninterpolated average precision (AP), clearly named,
   computed within each fixed-state prediction block. Report macro and
   class-count-weighted summaries with valid-block coverage; pooled AP only
   for the frozen score. Actual causal operating-point precision and recall
   are calculated from summed confusion counts. Any prevalence-standardized
   precision is hypothetical and assumes transfer of class-conditional rates.

Expected predictive cells: spline 8 conditions x 2 datasets x 5 seeds = 80;
LR main 2 x 5 = 10; ARF chronological, shuffled, and FIT-only 3 x 2 x 5 = 30.
Total: 120, including 30 repeat trajectories used to verify the new audit runner.
ARF FIT-only uses the existing chronological initial state; only the shuffled
condition fits new initial states. If a cell fails, retain the failure and its
reason rather than omitting it from the expected manifest.

## Interpretation and limits

Five seeds are algorithmic randomizations on each single recorded stream.
Report per-seed values and descriptive mean/SD, not independent-stream p-values.
The 4/5 count for LR is an exploratory analogue with a different model/control,
not a pass of the original spline acceptance rule. Weak frozen comparators
cannot establish general learning superiority. Compare ARF also to spline
threshold-only policies at the same acquisition budget and disclosed memory
and label-use contracts. Lambda sensitivity is evidence of operating-policy
sensitivity; on its own it does not identify the cause as sampling noise.

Additional field captures, active acquisition, noisy analyst labels, operational
cost estimation, and event/session identity are outside this case study. Exact
feature overlap is reported as representation recurrence, not memorization or
novel-attack generalization. Permanent DOI deposition requires a real archive
record and author-approved metadata; no identifier will be fabricated.

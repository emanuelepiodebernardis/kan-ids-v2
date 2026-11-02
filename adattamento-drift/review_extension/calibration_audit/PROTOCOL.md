# Calibration eligibility and label-reuse audit

Fixed 2 November 2026 before execution. This is an exploratory diagnostic on
already examined TON and UNSW captures, not independent confirmation.

1. Reuse the ten fitted spline states and original FIT192/ROLE64 score
   trajectories, default salt, seeds 42--46, requests, delay, age and tie rules.
2. Add Head/ALL256 beside Frozen/ALL256. Both thresholds use identical arrived
   observations, independent minimum-per-class gate, cost weights and FPR
   targets. Head fitting still uses only the original FIT192. ALL256 therefore
   reuses some training labels for threshold selection; this is a more
   permissive label-use contract, with no additional label acquisition.
3. Add both ALL256 thresholds on the original head-transaction gate. The
   independent/gated pairs isolate schedule changes while holding the score
   trajectory, calibration population and capacity fixed. The gated ALL256 /
   ROLE256 contrast then holds schedule and capacity fixed, but changes which
   labels can enter calibration.
4. Preserve all earlier policies and require exact per-block agreement with
   the matched-memory reports: thresholds, confusion counts, queries and
   head-state digests. Record all/seen/unseen results, per-class calibration
   counts, and current FIT/ALL feature-digest overlap.
5. Audit normal-vector role allocation for the default salt and the two
   already declared alternatives. Report unique-vector and occurrence-weighted
   CAL fractions, largest-vector mass, top-ten mass, and class/role-conditioned
   FP counts for selected frozen/head policies. These diagnostics assess
   representativeness and concentration; they do not identify all causes of
   the score update's effect or demonstrate a deployed FPR guarantee.
6. Retain nine cost weights and three FPR targets already used. Do not tune,
   choose a new salt, refit initial states or add new streams. Every cell and
   unsupported interpolation is reported. Observation-wise role assignment
   remains a proposed future intervention, not an experimentally tested fix.

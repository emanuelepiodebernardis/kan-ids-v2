# Response to the available reviews

Revision date: 1 November 2026. This is an implementation and evidence map,
not a claim that a journal has accepted the revision. Review 2 was available
in full. Review 1 could not be recovered in full from the previous chat;
its recoverable requests concerned a standard stream baseline and an
independently updated threshold-only baseline. Both are implemented below.

## Major comments

| Comment | Revision and evidence | Interpretation retained |
|---|---|---|
| M1: unequal effective label use | Added independently eligible Frozen/CAL256, using every acquired label with quotas 128/128, and intercept-only FIT192 controls. See `experiment.py`, manuscript tables identified below, and `analysis/paired_differences.csv`. | Equal acquisition does not imply equal memory allocation, label reuse, or model capacity. The original CAL64 contrast is conditional on its information allocation; no model-free percentage of causal attribution is claimed. |
| M2: seen/unseen aggregation | Added the original seen and unseen rate table, their exact FN/FP decomposition, and the same analysis for the new controls on both datasets. | Original seen-vector recall falls while unseen recall rises. Against the stronger new CAL256 control, TON spline recall falls in both strata on average. FP savings are concentrated on seen vectors: 96.7% in the original contrast and 90.5% in the new one. |
| M3: FIT/B and prior shift | Recast FIT/B as a diagnostic operating-point mismatch; explained balanced-memory refitting, zero-centred regularization and the limits of a pure label-shift explanation. Added an intercept-only experiment. | Offset/CAL64 and Frozen/CAL64 have identical recorded confusion counts. A freely chosen threshold absorbs a constant score offset. This does not prove that all observed changes are pure prior shift. |
| M4: GAM versus KAN positioning | Changed title, abstract and keywords; describe an additive logistic spline model and nine-parameter head. Added GAM and neural-additive-model references. | No new KAN architecture or nonlinear online spline learning is claimed. |
| M5: missing baselines and separate data | Executed LR and River 0.22.0 ARFClassifier with ten trees, five seeds each, on TON and a new chronological raw UNSW-NB15 split. All families share the exact A sample and acquired labels. | TON is explicitly post-review exploratory. UNSW is a separate capture with models trained anew, not cross-dataset transfer or a pristine resource in the project's entire history. ARF reuses calibration labels for training and retains an evolving model state. |
| M6: dependence, RNG, interaction and AUC | New queries use `SeedSequence([seed, block, 20261101])`; retained five legacy-query bridges. Added original 2×2 BA interactions, paired moving-block resampling at lengths 5/20/50, fixed-score pooled AUC and normal-weighted block AUC. | Seed SDs are conditional randomization dispersion. The nominal sign-test p-value is deliberately not used as evidence from five independent environments. Block percentiles resample recorded trajectories without refitting; they are not generalization confidence intervals. |
| M7: annotation load and delay | Added queries/hour, empirical query-to-arrival times, delays 0/1/5/20, 0.1% budget, five causal rate-cost policies, and hypothetical prevalence precision. | Label volume is an assumed feedback service, not a feasible single-analyst workload. Block delay is not fixed wall time. Prevalence calculations assume transfer of class-conditional rates. |
| M8: literature and contribution | Expanded direct security-stream, delayed-label, prior-shift, additive-model and dataset-audit literature. Added a five-point reusable evaluation checklist. | Contribution is decomposition and evaluation, not discovery that thresholds affect recall. |
| M9: failed primary criterion | Preserved the original unsuccessful four-of-five criterion and three-of-five observed joint improvements. Reframed the question around incremental benefit and error compromises. | New TON head versus CAL256 has more FN and fewer FP in all five seeds. UNSW reverses the direction: fewer FN and more FP in all five. Neither is joint dominance. |

Tables are identified additionally by their stable LaTeX labels
`tab:tonextension`, `tab:unswextension`, `tab:ablations`, `tab:strata`, and
`tab:sensitivity`; numeric table numbers can change during journal formatting.

## Minor and implementation comments

- Replaced outdated C improvements by 0.0217, 0.0559 and 0.0719 from the final
  twenty-calendar comparison. Added the additive frozen AUC 0.8911 and the
  lower adapted alternatives 0.8341 / 0.8073 / 0.7621.
- Corrected 20.46% to 20.47%, mathematical thousands separators, figure labels,
  error bars and protocol-box layout. Immutable roles precede label arrival.
- Moved the D unseen count out of the chronological partition table; explained
  the newly occurring attack families, exact vector collisions, and the
  inherited compact eight-feature contract.
- Removed the unconfirmed SMARTEST affiliation. Existing author metadata is
  retained for author review. Added English entry-point documentation under
  `ton_temporal/articolo_02/README.md`.
- Explain that the head is fitted anew with shrinkage toward zero; added a
  previous-head-centred L2 alternative. The latter does not uniformly improve
  results and every outcome is retained.
- Consolidated the main limitations, shortened the abstract and replaced the
  misleading small-average-FN phrasing by seed consistency and stratum results.
- Recovered the historical `train_test_network.csv` copy with exactly the
  recorded SHA-256. This closes the missing-copy issue, not the historical
  exposure of D. All 23 TON CSV hashes and 710 original D row/label block
  digests match. The unmodified original A parser matches the new A output.
- Disclose that original Windows NPZ hashes and per-seed predictions are not
  bitwise reproduced. The five legacy-query reruns quantify all discrepancies.
  Do not merge the old and new seed observations into one series.
- Keep complete original and new reports, initial fitted states, array hashes,
  test results, source code and figure-generation code in the revision package.

## Items requiring the human authors

Funding, competing interests, final authorship/CRediT approval, target-journal
format and journal-specific AI disclosure must be confirmed by the authors.
No factual declaration or archival DOI has been invented. The complete package
is ready to deposit; a Git commit is not represented as a permanent archival DOI.
The unavailable full first review cannot honestly be certified as exhaustively
addressed. These are explicitly disclosed boundaries, not missing experiments
from the fixed 55-run protocol.

# Matched calibration-memory audit

Date: 1 November 2026. Exploratory follow-up to the already inspected captures
and earlier diagnostic results. This is not independent confirmation or a
preregistration of the original study.

## Fixed design before execution

Run the spline model on both recorded D streams, seeds 42--46. Reuse the
archived initial states, transformations, request sampler, FIT192 trajectory,
default role salt, delay 1, budget 1%, age limit 50 and class minimum 5.
Do not refit initial models or tune settings using the new outcomes.

Cross two score functions (initial frozen; the existing adaptive head) with
two CAL-role memory capacities (64; 256). Within each capacity, both scores
use exactly the same labeled observations, objective, tie rule and update
times. All four policies update thresholds only at successful transactions
of the original FIT192/CAL64 head; the head trajectory is therefore held
fixed across threshold memories. At 256, this is a controlled threshold-memory
intervention, not a new learner whose fitting gate changes with capacity.

The larger memory accepts only CAL-role observations and remains disjoint
from FIT. It is called CAL-role256 to distinguish it from the earlier
all-label CAL256, which accepts every acquired label. It increases retained
example capacity to 192+256, but does not increase acquisition. No head is
calibrated on FIT labels. Retain the earlier all-label frozen CAL256 as an
independently updated reference and historical B policies as anchors.

Use the fixed nine cost weights 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4 and
normal-label FPR targets 5%, 10%, 20%, with the earlier exact threshold rules.
Report all/seen/unseen confusion counts, per-block threshold and eligibility
records, acquisition hashes, CAL age/count summaries, head-state hashes and
resource contracts. Report mean and sample SD across five paired seeds,
explicitly conditional on each single stream. No significance claims or
independent-stream confidence intervals are planned.

Primary contrasts: head minus frozen at the same CAL-role capacity; capacity
256 minus 64 holding the score trajectory fixed. Show lambda=1 and lambda=2
in the main summary and all weights/targets in machine-readable results.
Report achieved FPR alongside recall for every FPR target. Paired within-
support interpolation at FPR 10%, 13%, 20% is descriptive; no extrapolation.

## Integrity checks

Require exact agreement of every shared per-block confusion matrix, threshold,
head-update status and request hash with the retained second diagnostic run.
Verify class quotas, the CAL64 subset relation to CAL-role256, FIT/CAL digest
disjointness, observation age and delayed causality. Aggregate totals must
equal sums of per-block counts. Record input/model/source SHA-256 digests.
Preserve the prior executed runners unchanged. New runner and protocol must
be committed before full-data execution; tests use only synthetic data.

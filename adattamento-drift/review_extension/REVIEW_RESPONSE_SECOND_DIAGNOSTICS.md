# Second-review response and evidence map

Revision: 1 November 2026. All three current reviews were read in full: two
pasted Russian reviews and the attached English review. The earlier response
is preserved in `REVIEW_RESPONSE_FIRST_EXTENSION.md`. This document records
scientific changes, disagreements, and genuine remaining limits.

## Overall assessment

The criticism of a single lambda=1 headline is justified. The previous 28.36
percentage-point FPR contrast describes two specified update policies, not
uniform score superiority. The new article makes operating-point sensitivity,
initial-score quality, and model-specific error compromises central. Its title
is **Thresholds and Learning in Adaptive Intrusion Detection: A Two-Capture
Case Study**. Historical development and detailed diagnostics are in a separate
Supplement; the main article starts with the current controls.

The positive third review overstates the original ARF counterexample. Its
frozen reference is sometimes degenerate. We retain its useful requests for
PR diagnostics, parameter and salt sensitivity, and disjoint-label ARF, while
reassessing the ARF conclusion with an additional initialization control.

## First current review

| Issue | Action and evidence | Remaining scope |
|---|---|---|
| 1. TON post hoc; UNSW previously used | Abstract, Introduction, Methods and Discussion explicitly call both extensions exploratory on TON and disclose earlier project exposure of UNSW. `review2/PROTOCOL.md` was committed before new diagnostics. | A versioned protocol does not make previously inspected streams independent confirmation. No additional field capture is claimed. |
| 2. Five seeds are not five streams | All main captions and the Supplement identify SD as algorithmic variation conditional on each single capture. No seed-based significance claim is made. | Two captures remain two captures. |
| 3. ARF not equal-resource | Main information-contract table and forest caption distinguish acquisition, learning-label reuse and persistent tree state. Added FIT-only learning with disjoint CAL64. | The new forest still has persistent model memory; it is not a FIT192-equivalent memory baseline. |
| 4. TON LR inversion | Full A/B/D score audit for all seeds; class orientation, coefficient scores, probabilities, solver iterations and cached predictions checked. `review2/analysis/model_audit.csv` and raw `results/audits/`. | Mean A/B/D AUC is approximately 0.883/0.400/0.318. Poor transfer appears already on B. No D-label score flipping is used. The frozen LR is a failure case, not a strong comparator. |
| 5. Macro versus normal-weighted AUC | Main Results explains 0.850 versus 0.557 and its relevance to false-alarm workload; Supplement retains both, coverage and AP. | This weighted average does not imply that every normal-heavy block is near chance or directly determine threshold FPR. |
| 6. Permanent archive | Added a complete versioned delivery package, hash manifest, offline Git bundle and deposit metadata. Availability text explicitly states that no permanent identifier has been registered. | This item remains open until actual archival deposition. A Git SHA, ZIP or reserved DOI is not presented as a published archival record. |
| 7. Historical numbers not bitwise reproduced | Original Windows measurements remain historical evidence. Linux bridge and reproducible extensions are separate throughout. | Initial fits across numerical environments are not asserted identical; shipped fitted states support extension replay. |
| 8. Title too strong | Replaced the title and abstract; conclusions describe two-capture operating policies, not a general gain beyond thresholding. | Generalization requires new captures. |

## Second current review

| Issue | Action and evidence | Judgment |
|---|---|---|
| 3.1. Lambda=1 weak operating point | Main cost table includes lambda=2 for both controllers. Added nine fixed weights, per-seed matched-error interpolation, interpolation of mean curves, and causal FPR-target controllers. `matched_error_interpolation.csv`, `matched_error_dense.csv`, `calibration_transfer.csv`. | Agreed that 28 points is not a curve-wide benefit. The stronger claim that this is solely sampling instability is unsupported: TON calibration FPR itself is 42.67% at lambda=1 versus 8.26% at lambda=2 (weighted by subsequent normal counts). Selection, objective and transfer all matter. |
| 3.2. Degenerate frozen ARF | Explicitly identify constant UNSW seed 43 and inverted rankings; audit all A/B/D initial scores. Added shuffled-A forest trained on the identical sampled rows and FIT-only feedback. Compare forests also to spline threshold policies. | The earlier own-frozen contrast alone cannot establish a generally stronger learner. All initial states and unfavorable outcomes are retained. |
| 3.3. LR 4/5 | State that LR reduces FP in 5/5 and FN in 4/5 against its own CAL256 control; disclose seed 45 (+62,138 FN) and poor frozen transfer. | This is an exploratory analogue with another learner and comparator, not a pass of the original spline acceptance rule. |
| 3.4. Legacy 3/5 becomes 2/5 | Moved the exact FN differences and 2/5 count into the main Stage 1 subsection. `legacy_joint_counts.csv`. | Seed-count success is numerically fragile; no inferential claim rests on the cutoff. |
| 3.5. UNSW threshold adaptation hurts BA | Main Results prominently states Frozen/B 96.29%, CAL64 93.75%, CAL256 95.42%, head 93.22%. | Both default adaptations lose by BA on this stream. Other recall/false-alarm utilities can prefer different policies. |
| 3.6. Normal-heavy ranking | Discussed in the main text and complete ranking summaries. | Agreed, with the weighting qualification above. |
| Coarse lambda grid | Report 238/355 and 201/355 identical-threshold blocks for the cited seed; retain nine-point audit and all per-block thresholds. | More weights do not remove finite-calibration ties. |
| 96.49 versus 96.48 | Correct ARF recall is 96.484968...%; generated tables round to 96.48%. | Old unused extension source files removed from the new LaTeX package. |
| Eligibility header and excessive length | Simplified contract-table headers; separate Supplement contains historical development and detailed resource/AUC/attack/bootstrap tables. | Main paper focuses on the new interpretation. |
| Defensive prose | Rewrote Introduction, Results and Conclusions around observed behavior and supported scope. | Necessary uncertainty remains explicit. |

The review's near-equality at 13% FPR comes from interpolating curves of means.
Within-seed interpolation on the original five weights gives a paired recall
difference of 2.41 +/- 3.11 percentage points; the nine-weight grid gives
1.49 +/- 3.66. These are descriptive summaries of this stream, not evidence
of equivalence or independent-stream superiority. Both procedures are shown,
with incomplete support and absence of extrapolation recorded.

## Attached English review: nine author questions

1. **PR:** causal precision-recall operating points, hypothetical prevalence
   panels, noninterpolated block AP and class-weighted summaries added. AP is
   computed for main and forest diagnostic conditions; pooled AP is limited
   to a fixed initial score. Coverage is retained.
2. **Controller parameters:** all five seeds on both captures for memory
   96/32 and 384/128, CAL age 10 and 100, and CAL eligibility minimum 10.
   FIT still requires both classes, without this CAL minimum or an age limit.
3. **Role salts:** two additional fixed salts, all five seeds and both streams.
   Results show substantial sensitivity; no favorable setting is selected as
   a new default after looking at D.
4. **Look-ahead:** prediction precedes query/arrival/update; tests perturb labels
   and check that earlier predictions/thresholds cannot change. Thirty main
   audit runs must exactly reproduce prior per-block thresholds and counts.
   This replay safeguard is distinguished from exploratory research history.
5. **Prevalence:** actual precision plus explicit 1% and 0.1% hypothetical
   calculations, assuming unchanged class-conditional rates. No fabricated
   benign-majority network experiment is claimed.
6. **Near duplicates:** not silently substituted for exact float32 recurrence.
   A tolerance or session identity needs a justified feature-specific rule;
   the paper explains this limit and avoids novelty/memorization claims.
7. **ARF:** disjoint FIT-only learning versus CAL64 added. Actual learned-label
   counts and persistent state sizes are in reports. This does not equalize
   total memory or erase earlier learning.
8. **Operational lambda:** derive the cost/prevalence mapping and report causal
   FPR-target controllers. Daily analyst capacity is not available in these
   captures, so no fitted SOC workload model or deployed optimal lambda is
   invented. Such a policy needs separate past data and an untouched evaluation.
9. **Release:** runners, protocols, tests, per-seed/per-block reports, models,
   input/source digests, CSVs, figures and both LaTeX documents are supplied.

The suggestions of active acquisition, richer feature spaces, field captures
and specific Bayesian/SRE methods are sensible future studies, not evidence
that the present paired-policy case study requires an unrelated new detector.
The paper connects cost-aligned thresholding to decision costs explicitly and
keeps acquisition fixed to preserve the studied comparison.

## What the new forest controls establish

Shuffled-A forests are strong on UNSW: frozen/B BA is 97.84%, frozen/CAL256
96.94%, and adaptive/CAL256 96.74%. The earlier own-frozen improvement does
not survive this stronger initialization as a BA advantage. On TON the
shuffled forest reaches 89.94% adaptive/CAL256 BA versus 73.67% frozen/CAL256;
at 20% FPR, paired interpolation of the nine-weight CAL64-adaptive versus
CAL256-frozen policies gives +18.24 +/- 13.88 recall points across five
supported pairs. This supports a TON learning contribution under the stated
persistent-state and label-reuse contract. The original positive review's
ARF argument needed these controls and does not generalize to UNSW.
All fixed-B and 0.5 forest policies are retained too; in particular the
shuffled adaptive forest with the fixed B threshold attains 94.37% BA on TON.
The retrospective comparisons do not select a deployment policy using D.

## Diagnostic integrity

`review2/PROTOCOL.md` defines 120 cells, including 30 repeat trajectories.
The first new diagnostic attempt was excluded after a uint8 overflow in the
AP class-validity guard. The fix widens the integer count and has a dedicated
regression test. The retained reports identify the corrected runner hash;
analysis checks AP/AUC valid-block coverage for every condition collecting AP.
The original 55-run extension and historical reports are unchanged. Superseded
attempt identifiers remain in provenance and are not counted as new evidence.

## Genuine remaining author/deposition tasks

A permanent archival record is not yet published. Author approval of study-
specific funding, competing interests, CRediT and submission venue is also
required. The package contains deposit metadata and integration instructions;
these remaining facts are stated plainly rather than marked resolved. Neither
new seeds nor retrospective diagnostics can create an independent capture.

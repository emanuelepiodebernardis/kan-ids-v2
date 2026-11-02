# Internal assessment of the additional critiques

1 November 2026. Author-team working document, not a response letter to journal
referees. Exclude from the journal ZIP. Prior assessments are retained in
REVIEW_RESPONSE_SECOND_DIAGNOSTICS.md and REVIEW_RESPONSE_FIRST_EXTENSION.md.

## New controlled evidence

The unequal CAL256/CAL64 comparison was the most important design criticism.
Ten new spline replays cross frozen/head scores with disjoint CAL-role64 and
CAL-role256. The head trajectory, calibration occurrences within each pair,
objective, tie rule and update times are fixed. CAL-role256 is distinct from
the earlier all-label CAL256. No extra acquisition or FIT/CAL reuse occurs.

At lambda=1 on TON, the same-CAL64 score contrast is -28.15 pp FPR and -1.91 pp
recall; at CAL-role256 it is -28.94 pp and -1.56 pp. Memory mismatch alone does
not explain the original FP reduction. Increasing head CAL capacity adds
1.83 pp BA on TON and 0.59 on UNSW. The UNSW score contrast changes from
-0.54 pp BA at CAL64 to +0.30 at CAL-role256, below historical Frozen/B in both.

At lambda=2, frozen CAL-role256 has TON FPR 32.80%, versus 13.16% for all-label
CAL256. Equal capacity does not imply equal observations or update gates.
Label eligibility and feature-role partition representativeness therefore
require explicit treatment; the experiment does not isolate one mechanism
for this entire difference.

## Point-by-point assessment

| Concern | Assessment and action |
|---|---|
| Abstract opens with 28.36 pp | Valid. It now opens with the evaluation finding and reports the new same-memory contrast. The mixed-package result is conditional in Results. |
| About +11 pp recall at the same FPR | Overstated: achieved FPRs are 18.37% and 19.95%, not identical. Both rates and the 10.97 pp recall difference are explicit. |
| Benefit disappears at low FPR | Too categorical. Dense paired interpolation at 13% gives +1.49 +/- 3.66 pp, neither an equivalence test nor absence throughout the low-FPR region. The older 10% estimate has only four supported pairs and high variation. |
| Target violations | Valid: Head/CAL64 reaches 12.07% and 16.03% at 5% and 10% targets. CAL-role256 still reaches 11.21% and 14.49%. All failures are reported. |
| Understated uncertainty | Configuration sensitivity is larger than the default seed SD. Salt-specific means and SDs are separate; three fixed salts do not estimate a general uncertainty distribution. No seed confidence intervals are claimed. |
| Historical B is the practical winner | Correct for the highest-BA evaluated points. TON's winning shuffled ARF still learns its score. Recommend retaining a historical-threshold control, not choosing B universally from test outcomes. |
| 28.36 pp isolates head learning | Incorrect attribution: this is a policy-package contrast. The new matched-memory pairs supply the controlled score intervention. |
| ARF advantage | Keep equal acquisition but different retained state and label use. Standard ARF reuses CAL for learning; the FIT-only diagnostic remains. |
| Plain KAN headings | Checked: additive spline logistic GAM; KAN appears only as related work, not architecture novelty. |
| Different figure axes | Original operating plots now have an in-panel warning; the new figure uses common axes throughout. |
| Precision at 1% prevalence | A conditional transformation of achieved rates, not a deployment prediction or isolated score comparison. |
| DOI | Still outstanding. Complete archive and updated metadata are prepared; no invented/reserved DOI is called a published record. |
| Funding and interests | GAIA support for De Bernardis and related interests are disclosed, including Frontoni's board role. Unrelated HEART grant identifiers are not assigned to this IDS study. |
| CRediT | Original and extended work are distinguished. Only documented organizational roles are assigned to Frontoni; scientific roles require factual author input. |
| AI disclosure | Research use in Methods includes code, execution, analysis and plotting; manuscript preparation is separately disclosed. AI receives no authorship or CRediT role. |
| Submission files | Clean LaTeX ZIP excludes internal assessments and handover notes. The full integration package retains them for the authors. |

## Venue judgment and remaining author facts

Preferred first target: ACM Digital Threats: Research and Practice, as an
evaluation case study with positive and negative results. The editorial blog
explicitly welcomes negative cybersecurity results. Its practical-threat scope
remains a limitation for two controlled captures; acceptance is not assumed.
JISA is an alternative, but its current scope excludes routine application of
existing ML models to standard datasets. The submission must establish the
value of the evaluation procedure and its security consequences. Computers &
Security is not recommended: the publisher currently states an AI/ML moratorium.

Official sources checked 1 November 2026:

- https://dtrap-blog.acm.org/2019/08/26/negative-results-are-a-good-thing/
- https://www.sei.cmu.edu/blog/acm-digital-threats-research-and-practice/
- https://shop.elsevier.com/journals/journal-of-information-security-and-applications/2214-2126
- https://shop.elsevier.com/journals/computers-and-security/0167-4048
- https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals
- https://www.acm.org/publications/policies/new-acm-policy-on-authorship

ACM's May 2026 policy requires Methods disclosure of research AI use, while no
longer requiring writing-only disclosure. Elsevier separately requests writing
and research-use disclosures. The fuller current statement accommodates both.
The exact backend model snapshot was not retained; reproducibility rests on
the executable artifacts. All authors must verify their contributions and
relationships before submission; no blanket no-conflict statement is invented.

## Verification

Ten complete new reports; 274,090 replay checks; 1,930 source/aggregation checks;
93 targeted tests pass. Shared thresholds, counts and request trajectories
agree exactly with earlier diagnostics. Historical reports and both previously
executed runners remain unchanged. These replays are not independent streams.

# Paper 2: English evidence map

The original paper separates score-head and threshold adaptation under sparse,
delayed labels. The original final campaign is retained in
`validazione_finale_D/`; its protocol is `protocollo_validazione_finale.md`.

The [post-review extension](../../review_extension/README.md) supplies the
revised LaTeX, complete original and new reports, stronger threshold controls,
logistic regression and adaptive-random-forest baselines, an independent
chronological UNSW-NB15 evaluation, sensitivity analyses, and import instructions.

Core implementation names in this directory's parent:

| File | Purpose |
|---|---|
| `replay.py` | Initial models, fixed transformation, memory and head fitting |
| `soglia_bilanciata.py` | Exact balanced-accuracy threshold selection |
| `calibrazione_recente.py` | Immutable roles, recent-label eligibility, atomic update |
| `replay_validazione_finale.py` | Original five-seed final TON replay |
| `split_manifest.json` | Chronology and partition accounting |
| `sovrapposizioni_abcd.json` | Exact vector overlap across intervals |

Treat the original campaign, development C analyses, and post-review extension
as distinct evidence stages. Their numerical environments and query samplers
are explicitly recorded; the extension does not replace the original outcomes.

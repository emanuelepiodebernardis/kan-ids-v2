# Paper 1 manuscripts, v0.12.0

- [English IEEE Access source](ieee_access/README.md), built with pdfLaTeX.
- [Russian author-review version](ru/README_RU.md), built with XeLaTeX.
- [Accepted physical energy evidence](../experiments/hardware_energy_20260915/README.md).

The review now includes signed-certificate and exact-row-overlap diagnostics, stage-specific preprocessing scopes, UNSW AUROC checks and a clearer hardware protocol map. The executable evidence index is [evidence/review_v012/ARTIFACT_MAP.md](../evidence/review_v012/ARTIFACT_MAP.md). No new training or device measurements were performed.

The manuscript sources and input figures are versioned. Compiled manuscript
PDFs, compilation logs and page renders belong to the separately delivered
review release. Their hashes and scientific/visual QA receipts are retained;
the source-export manifests describe only files actually present here.
Figure PDFs are vector input assets and remain tracked.

Run `python papers/verify_hardware_links.py` from the repository root to
check the manuscript evidence against the twenty-acquisition campaign.
Each language's `figures/make_energy_figure.py` reproduces the common figure
from its supplied CSV. The two supplied setup photographs are unchanged;
their meter displays are illustrative setup readings, not the reported active
batch means. All historical tables and protocols are kept separate.

The proposed draft author order is Kuznetsov, De Bernardis, Frontoni, at
Kuznetsov's request. Agreement by other authors is not asserted. Confirm order,
contributions and release permissions jointly before journal submission.
No software commit constitutes coauthor approval of the manuscript.

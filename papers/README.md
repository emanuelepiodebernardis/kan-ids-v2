# Paper 1 manuscripts, v0.15.1

- [English IEEE Access source](ieee_access/README.md), built with pdfLaTeX.
- [Russian author-review version](ru/README_RU.md), built with XeLaTeX.
- [Accepted physical energy evidence](../experiments/hardware_energy_20260915/README.md).

The manuscript presents decision-agreement certificates for a frozen integer
additive KAN, with IoT intrusion detection as the application case. Version
0.15.1 revises the title, abstract and framing in both languages, with small
style and layout fixes. It preserves the numerical results, tables, figures
and historical unfavorable ESP32-C3 observation. This editorial revision adds
no training or device measurements.

Version 0.15.0 integrated the endpoint-pair-disjoint classification study,
HW500 matched-cohort time and whole-board USB energy, and RAM500 observed
memory components. Their evidence and verification are under
[hardware_validation_20260916](../experiments/hardware_validation_20260916/).
Earlier signed-certificate, row-overlap and UNSW diagnostics remain indexed in
[ARTIFACT_MAP.md](../evidence/review_v012/ARTIFACT_MAP.md).

The English text is the collaborative journal manuscript. The Russian text is
an aligned copy for the corresponding author's reading. Both are working
drafts awaiting further author review. The certificate covers coefficient-to-LUT
agreement; it does not certify the floating-to-integer transition. RAM500 does
not establish an exact whole-system memory peak, and C3 speedups remain
specific to the stated placement and timing protocol.

The manuscript sources and input figures are versioned. Compiled manuscript
PDFs, compilation logs and page renders belong to the separately delivered
review release. Their hashes and scientific/visual QA receipts are retained;
the source-export manifests describe only files actually present here.
Figure PDFs are vector input assets and remain tracked.

Run `python papers/verify_editorial_v0151.py` to verify preservation relative
to the supplied v0.15.0 Git history, and `python papers/verify_validation_v015.py`
to check the integrated numerical evidence. Run
`python papers/verify_hardware_links.py` from the repository root to
check the manuscript evidence against the twenty-acquisition campaign.
Each language's `figures/make_energy_figure.py` reproduces the common figure
from its supplied CSV. The two supplied setup photographs are unchanged;
their meter displays are illustrative setup readings, not the reported active
batch means. All historical tables and protocols are kept separate.

The proposed draft author order is Kuznetsov, De Bernardis, Frontoni, at
Kuznetsov's request. Agreement by other authors is not asserted. Confirm order,
contributions and release permissions jointly before journal submission.
No software commit constitutes coauthor approval of the manuscript.

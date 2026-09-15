# KAN-IDS Paper 1 — English manuscript v0.12.0

IEEE Access author-review manuscript, revised 15 September 2026. Build with `python3 scripts/build_paper.py` (pdfLaTeX and BibTeX), which uses a clean temporary directory. For Overleaf select `main.tex` and pdfLaTeX. Publisher class, required fonts and ORCID icon are included.

The paper addresses auditable integer-export agreement and a physical microcontroller case study. The matched ESP32-C3 factorial is the main coefficient/LUT comparison. The original five-model timing experiment, its unfavorable 7.58% C3 LUT slowdown, and the intermediate locality diagnostic remain in an explicit appendix. A four-protocol map separates these experiments from the physical USB-energy pilot. The original setup photographs are preserved byte-for-byte and placed side by side.

This revision adds frozen-model exact-row-overlap sensitivity, subgroup confusion counts and certificates, reaggregated historical UNSW AUROC, exact MI-stage documentation and corrected arithmetic bounds. It introduces no model fits, changed weights, new hardware records, independent grouped holdout, full raw-flow IDS evaluation or peak-RAM measurement. Full original numeric tables are retained; rounded energy presentation values are accompanied by the original full-precision table and CSV.

Author order is Oleksandr Kuznetsov, Emanuele Pio De Bernardis, Emanuele Frontoni, as requested by Oleksandr Kuznetsov. The other authors' agreement is not asserted. All affiliations, ORCIDs and biographies are retained.

The master release map is `evidence/ARTIFACT_MAP.md`, with a repository copy at `evidence/review_v012/ARTIFACT_MAP.md`. It separates prepared-input arithmetic, historical metric aggregation, conditional preprocessing recovery and physical-measurement provenance. The standalone manuscript source package contains the evidence extracts used for its new tables, but is not the complete raw experimental package. Public deposition and DOI assignment are not claimed.

Run `python3 verification/check_content_v012.py` for current numerical, author-order and photo-byte checks. Earlier V010/V011 receipts concern their named historical revisions and do not certify this version. Regenerate the energy plot with `python3 figures/make_energy_figure.py`.

Verify delivered bytes before rebuilding: `python3 tools/verify_package.py .`. A rebuild changes generated files. The paper source contains no private Russian report. Publisher assets retain their notices; manuscript text, author portraits and photographs are not relicensed under the software licence.

# v0.11.0 — 15 September 2026

- Integrated five-model, two-acquisition USB-energy pilots on Mega 2560 and ESP32-C3, separately from all historical short-kernel and placement experiments.
- Added explicit twenty-prepared-input, approximately 120-second active-batch protocol; LED timestamp alignment; prespecified central sixty MCU-second power window; whole-board energy estimator; and limits on interpretation and measurement uncertainty.
- Added time/power/energy and pilot-firmware size tables, an all-board scientific comparison figure, and two unaltered measurement photographs placed side by side.
- Updated the abstract, contribution statement, discussion, conclusion and availability statement; retained unmeasured peak RAM and corrected global claims that energy was unavailable.
- Preserved the existing fourteen numerical tables, fixed model/certificate and ML results, original hardware numbers, references and affiliations.
- Changed the proposed draft author order to Kuznetsov, De Bernardis, Frontoni in front matter, PDF metadata, running headers and biographies. This records the user’s request, not coauthor approval.
- Restored the retained 1-SE threshold 0.99617 and clarified seed-dependent validation splits shared among candidates within each seed.
- Hardened the build script to compile in a clean temporary directory, wait for stable references, and check completed PDF output before copying it back.
- Retained v0.10.0 verification receipts as historical records and added V011 checks and a current byte manifest. No training, model retuning, Git push, email, or public archival deposition was performed by this manuscript revision.

# v0.10.0 — 12 September 2026

Changes from the preserved v0.9.0 manuscript:

- Replaced the obsolete pre-Q12-input limitation with the delivered native cache check: 2,110,430 numerical Q12 values and 844,172 categorical codes agree across 211,043 known rows. Floating-transform bit identity and equivalence on future raw inputs are not claimed.
- Added the paired 42,209-row check: zero decision changes from input rounding; 40 from floating-to-coefficient export (39 correct-to-wrong, one wrong-to-correct); zero further from the selected LUT. The remaining export difference combines spline approximation, parameter quantization and integer arithmetic.
- Identified the 155/219-byte AVR figures as compiler-reported main-function frames in historical probes with separate weights, not current firmware stack or measured peak SRAM.
- Added checkpoint provenance for 420 current joint evaluation records and 42 pooled matrices. Briefly disclosed twelve incomplete matrices in the superseded ratio-50 archive; current per-fit means and validation-only selection are unaffected.
- Updated reproducibility prose and removed stale author-task lists and old build checks from this standalone manuscript package.

All fourteen tables, displayed equations, figures, author details and physical results are preserved. There is no new model fitting, device execution, energy measurement, public release or DOI. The software integration package remains v0.9.1.

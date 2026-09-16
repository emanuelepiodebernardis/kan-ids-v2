# Observed RAM500 components

Both boards accepted 15 fresh-upload runs and 22,500 reference comparisons, with zero mismatches. Positive stack/heap controls followed the saved baseline. `OBSERVED_RAM_TABLE.json` is derived by `verify_ram.py` from retained records and audited per-upload values. These are repeated executions of 500 inputs, not 22,500 distinct observations.

Mega measured v0.14.1: 335 B static SRAM; observed stacks 93/80/95/208/46 B for coefficient KAN/LUT/MLP16/multilayer KAN/DT5. C3 measured v0.14.2: 15,312 B static DRAM; loop-task lifetime stack use 1,504–1,584 B. In every C3 run its watermark was already reached before inference, so model-specific stack differences cannot be inferred. Reserved task stacks already belong to allocated heap. No global exact or universal worst-case Peak RAM is claimed.

The corrected post-link diagnostic-symbol subset does not change firmware, static RAM totals, or physical results. `evidence/CORRECTION.json` preserves the old and corrected values. Four regression tests execute the real hook with binutils fixtures, including all five retained C3 nm listings.

`python verify_ram.py` checks retained records without accessing raw ZIPs. `python verify_ram.py --artifacts-dir <raw-ZIP-directory>` additionally verifies and replays both original archives. The original read-only acceptance scripts in `audit/` retain their historical kit/hash requirements; see each `--help` for complete binary-level replay.

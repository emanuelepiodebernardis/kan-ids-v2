# Maintained source copy (repository v0.15.0)

Read `../README.md` or the parent experiment README for accepted scientific scope. This directory is a source-only integration, not the unmodified historical kit. `SOURCE_ORIGIN.json` identifies the accepted firmware source and original upstream hashes. `UPSTREAM_MANIFEST.json` is preserved for provenance; `KIT_MANIFEST.json` verifies the maintained copy. Historical validation output and binary images remain external. No additional hardware run is required for the current paper.

For host regression, execute `python -m unittest discover -s tests_runner -v` in this directory. The scripts named START or bootstrap are hardware acquisition entry points and are not required for offline analysis.

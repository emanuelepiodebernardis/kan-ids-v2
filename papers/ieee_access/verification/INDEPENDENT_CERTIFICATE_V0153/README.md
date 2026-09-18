# Independent coefficient-to-LUT arithmetic check

Script supplied by Emanuele Pio De Bernardis during review and retained unchanged. Run `python verifica_certificato.py headers` from this directory. Python standard library only; expected result: eight comparisons coincide, exit code 0.

The two headers are copied byte-for-byte from reviewed software commit `b219fb6671442e2e5dfcda5b6ea49a04d57e7960`. Their bytes are unchanged from manuscript baseline `ca92318`. PROVENANCE.json records hashes and scope. RESULTS.txt records the editorial-review execution; its printed input path is specific to that execution.

This script checks finite-domain numerical-edge arithmetic. It does not independently recount the 42,206 certified cohort decisions, check overflow on a target compiler, certify floating-to-integer export, or measure hardware memory. Categorical byte contributions are fixed constants for these headers. It supplements, rather than replaces, the original verification records.

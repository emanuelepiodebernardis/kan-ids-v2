"""Configurazione di pytest per la radice del repository.

Esiste per una ragione sola: `tools/apply_integration.py` salva le copie
originali dei file che sostituisce in `_integration_receipts/<run>/originals/`,
dentro l'albero del repository, e fra quelle copie ci sono file di test. pytest
raccoglie l'originale e la versione corrente come due moduli con lo stesso
nome, e si ferma in raccolta con `import file mismatch`: zero test eseguiti, non
un fallimento. Chi lancia solo alcuni file per nome non se ne accorge.

Le ricevute sono evidenza dell'applicazione e vanno conservate, quindi qui si
escludono dalla raccolta invece di cancellarle. `.gitignore` le tiene fuori
dai commit.
"""
collect_ignore_glob = ["_integration_receipts/*", "**/_integration_receipts/*"]

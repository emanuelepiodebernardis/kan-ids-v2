"""Configurazione di pytest per la radice del repository.

Esiste per due ragioni, e in entrambe il sintomo e' lo stesso: pytest si
ferma in *raccolta*, quindi zero test eseguiti invece di un fallimento
circoscritto. Chi lancia solo alcuni file per nome non se ne accorge.

**Le ricevute dell'integrazione.** `tools/apply_integration.py` salva le
copie originali dei file che sostituisce in
`_integration_receipts/<run>/originals/`, dentro l'albero del repository, e
fra quelle copie ci sono file di test. pytest raccoglie l'originale e la
versione corrente come due moduli con lo stesso nome e si ferma con
`import file mismatch`. Le ricevute sono evidenza dell'applicazione e vanno
conservate, quindi si escludono dalla raccolta invece di cancellarle;
`.gitignore` le tiene fuori dai commit.

**I progetti annidati sotto `experiments/`.** Sono snapshot congelati di
campagne di validazione hardware: ciascuno ha la propria radice, i propri
`requirements.txt` e i propri test, ed e' pensato per essere eseguito dalla
sua cartella. Raccolti tutti insieme dalla radice del repository si
ostacolano a vicenda, per due motivi indipendenti:

1. moduli omonimi in progetti diversi — `hw500/runner/project/tests/
   test_firmware_host.py` e il suo gemello in `ram500/` — che pytest, senza
   pacchetti, vede come lo stesso nome di modulo e rifiuta;
2. import che valgono solo dalla radice del progetto, come
   `from run_stage2 import archive_run` in `paired_software/
   source_snapshot/tests/`.

Rinominare quei file significherebbe modificare evidenza congelata per
comodita' di uno strumento. Si escludono percio' dalla raccolta in radice, e
si eseguono separatamente: `docs/SUITE_DI_TEST_IT.md` elenca i comandi,
uno per progetto, con gli esiti attesi.

L'esclusione non nasconde nulla, perche' quei test da qui non venivano
comunque eseguiti: bloccavano la raccolta e basta.
"""
collect_ignore_glob = [
    "_integration_receipts/*",
    "**/_integration_receipts/*",
    # I progetti annidati hanno ciascuno la propria radice: si eseguono dalla
    # loro cartella, non da qui. Vedi docs/SUITE_DI_TEST_IT.md.
    "experiments/*",
]

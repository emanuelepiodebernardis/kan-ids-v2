# Revisione editoriale e tecnica v0.15.3

Base tecnica verificata: `b219fb6671442e2e5dfcda5b6ea49a04d57e7960`, PR #1 di `emanuelepiodebernardis/kan-ids-v2`.

Le sette correzioni successive a `ca92318` sono state esaminate e le suite rieseguite su Linux/Python 3.12. `LINUX_VERIFICATION.json` identifica conteggi e log; i risultati Windows restano quelli dichiarati da Emanuele. I 148 test/242 subtest delle cinque esecuzioni comprendono ripetizioni: quelli distinti sono 145/172. Non sono stati ripetuti addestramento o misure hardware.

`independent_certificate/` conserva senza modifiche lo script ricevuto da Emanuele Pio De Bernardis, i due header congelati, la provenienza SHA-256 e l'esito di otto confronti. Non è un nuovo conteggio delle decisioni certificate del dataset. I limiti sono descritti nel README locale.

`EXPERIMENTS_TEXT_IO_AUDIT.json` è un inventario statico in sola lettura: 41 chiamate senza encoding esplicito e 29 scritture senza newline; le liste si sovrappongono. Nessuno snapshot sperimentale è stato modificato. Un inventario non prova che tutte quelle chiamate falliscano.

Le modifiche editoriali ai manoscritti recepiscono il confronto fra i protocolli ESP32-C3, distinguono i due certificati, aggiungono i richiami alle tabelle e commentano il ranking DT5 senza attribuirgli una causa non misurata. La disponibilità del commit pubblico è distinta dal deposito archivistico completo. I numeri, le formule e le figure sono preservati; le ricevute delle revisioni storiche mantengono il loro perimetro.

Le ricevute `papers/*/verification/EDITORIAL_V0153_QA.json` descrivono confronto e compilazione. I PDF sono forniti nei pacchetti sorgenti separati; non fanno parte del commit. L'integrazione del codice non costituisce approvazione della sottomissione alla rivista o accordo definitivo dei coautori.

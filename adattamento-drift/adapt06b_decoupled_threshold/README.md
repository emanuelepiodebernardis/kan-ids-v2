# ADAPT-06B — checkpoint A

Questa directory prepara la separazione fra manutenzione della soglia operativa
e promozione dei pesi. La base è il merge della PR #15:
`665356fd38e5bd65972b39288097432f20728fb6`.

## Eseguito nel checkpoint

- protocollo e configurazione versionati;
- modalità GAM/RANDOM separata e disattivata nel percorso legacy;
- controllo appaiato dei cinque bracci;
- test sintetici di causalità, ruoli, soglia, gate e regressione;
- ricalcolo autonomo di budget, digest, riepiloghi, ruoli e stato terminale;
- controprove negative su sovrapposizioni, causalità e candidati pendenti;
- processi distinti per le misure RSS di legacy e variante;
- comandi e stima dei due replay completi;
- controlli statici, log e manifest del codice.

## Soltanto pianificato al checkpoint A

- replay completi TON e UNSW, seed 42;
- qualunque risultato su dati reali;
- matrice a cinque seed, ACTIVE, LR/ARF e ADAPT-07.

Al commit del checkpoint A il pilot non era stato eseguito e richiedeva il
riesame. Il manifest `CHECKPOINT_MANIFEST.json` e `TEST_LOG.txt` documentano
quella consegna pre-pilot.

## Pilot autorizzato e consegnato

Il pilot completo TON/UNSW, GAM/RANDOM, seed 42 è stato successivamente
autorizzato ed eseguito dal commit verificato
`02b82af36ae8151e4ac83d806f434704e4557abd`. I quattro report, i due confronti
appaiati, i manifest, il log, il CSV e la sintesi sono in
`pilot_seed42_02b82af/`. `summarize_pilot.py` ne verifica gli hash, ricalcola i
confronti e rigenera la sintesi. Cinque seed, ACTIVE e ADAPT-07 non sono stati
eseguiti.

## Matrice di robustezza autorizzata

Il riesame del pilot ha autorizzato l'estensione GAM/RANDOM ai seed 42–46,
senza cambiare il metodo. `MATRIX_PROTOCOL.md` e `matrix_config.json` fissano il
perimetro prima dei nuovi run; `preflight_matrix.py` verifica il pilot, i dieci
stati A/B e gli stream; `run_matrix.py` orchestra soltanto le otto coppie nuove
dei seed 43–46; `analyze_matrix.py` combina queste coppie con seed 42 e produce
le 50 righe dei controlli, le statistiche appaiate e le due figure
preregistrate.

In questo commit di preparazione la matrice non è ancora eseguita. Il commit
contenente protocollo, configurazione, launcher e test deve essere registrato
come commit di esecuzione prima di aprire i dati dei nuovi replay.

## Struttura

- `runner.py`: variante decoupled, limitata a GAM/RANDOM;
- `run_pilot.py`: comando appaiato legacy/variante per un dataset;
- `verify_pair.py`: identità, invarianti autonome e tabella dei cinque controlli;
- `summarize_pilot.py`: verifica e sintesi riproducibile della consegna;
- `MATRIX_PROTOCOL.md`, `matrix_config.json`: disegno e artefatti congelati;
- `preflight_matrix.py`: controllo A/B, stream e pilot seed 42;
- `run_matrix.py`: orchestrazione senza sovrascritture dei seed 43–46;
- `analyze_matrix.py`: verifica, tabelle, figure e manifest della matrice;
- `test_checkpoint.py`: controprove sintetiche;
- `test_matrix.py`: limiti, propagazione del seed e aggregazione completa;
- `test_pilot_delivery.py`: controlli sui report, confronti e manifest finali;
- `COST_ESTIMATE.md`: stima pre-run, distinta dai costi effettivi consegnati;
- `CHECKPOINT_MANIFEST.json` e `TEST_LOG.txt`: provenienza e verifiche.

## Tracciabilità della richiesta

| Richiesta | Materiale verificabile |
|---|---|
| Base e data | `config.json`, `CHECKPOINT_MANIFEST.json` |
| Protocollo, causalità e cinque controlli | `PROTOCOL.md` |
| Variante separata e legacy invariato | `runner.py`, `test_checkpoint.py` |
| Soglie, pesi, ruoli, hash, memoria e tempi | JSON prodotti dal runner e `verify_pair.py` |
| Test sintetici e regressione | `test_checkpoint.py`, `TEST_LOG.txt` |
| Due replay completi autorizzati | `pilot_seed42_02b82af/`, `PILOT_COMMANDS.md` |
| Costo previsto | `COST_ESTIMATE.md` |
| Distinzione eseguito/pianificato | questo README e il manifest |
| Pilot autorizzato | `pilot_seed42_02b82af/`, `summarize_pilot.py` |


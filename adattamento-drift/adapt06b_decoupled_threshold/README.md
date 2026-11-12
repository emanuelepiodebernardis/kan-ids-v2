# ADAPT-06B — checkpoint A

Questa directory prepara la separazione fra manutenzione della soglia operativa
e promozione dei pesi. La base è il merge della PR #15:
`665356fd38e5bd65972b39288097432f20728fb6`.

## Eseguito nel checkpoint

- protocollo e configurazione versionati;
- modalità GAM/RANDOM separata e disattivata nel percorso legacy;
- controllo appaiato dei cinque bracci;
- test sintetici di causalità, ruoli, soglia, gate e regressione;
- comandi e stima dei due replay completi;
- controlli statici, log e manifest del codice.

## Soltanto pianificato

- replay completi TON e UNSW, seed 42;
- qualunque risultato su dati reali;
- matrice a cinque seed, ACTIVE, LR/ARF e ADAPT-07.

Il pilot non è stato eseguito e richiede il riesame del checkpoint A. I comandi
sono in `PILOT_COMMANDS.md`; il disegno è in `PROTOCOL.md`; i parametri esatti
sono in `config.json`.

## Struttura

- `runner.py`: variante decoupled, limitata a GAM/RANDOM;
- `run_pilot.py`: futuro comando appaiato legacy/variante per un dataset;
- `verify_pair.py`: identità di richieste, budget e tabella dei cinque controlli;
- `test_checkpoint.py`: controprove sintetiche;
- `COST_ESTIMATE.md`: costo previsto, distinto dalle misure future;
- `CHECKPOINT_MANIFEST.json` e `TEST_LOG.txt`: provenienza e verifiche.

## Tracciabilità della richiesta

| Richiesta | Materiale verificabile |
|---|---|
| Base e data | `config.json`, `CHECKPOINT_MANIFEST.json` |
| Protocollo, causalità e cinque controlli | `PROTOCOL.md` |
| Variante separata e legacy invariato | `runner.py`, `test_checkpoint.py` |
| Soglie, pesi, ruoli, hash, memoria e tempi | JSON prodotti dal runner e `verify_pair.py` |
| Test sintetici e regressione | `test_checkpoint.py`, `TEST_LOG.txt` |
| Due replay completi da autorizzare | `PILOT_COMMANDS.md`, `run_pilot.py` |
| Costo previsto | `COST_ESTIMATE.md` |
| Distinzione eseguito/pianificato | questo README e il manifest |


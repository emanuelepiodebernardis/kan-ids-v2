# ADAPT-06C — diagnosi dei candidati

Questa directory analizza i 28 cicli candidati già registrati da ADAPT-06B.
Non esegue fit, replay, tuning o accessi a nuovi dataset. La base è il merge
della PR #17, `24e25723ffac292b419adf2cff4378bc8d0b2c4a`.

## Sorgenti in sola lettura

- seed 42: `adapt06b_decoupled_threshold/pilot_seed42_02b82af/`;
- seed 43–46: `adapt06b_decoupled_threshold/matrix_seed42_46_ed50e86/new_replays/`.

Lo script legge esclusivamente i dieci `*_decoupled.json`. Il manifest registra
per ogni sorgente percorso, byte e SHA-256; i JSON non vengono copiati o
modificati.

## Prodotti

- `CANDIDATE_DIAGNOSIS.csv`: una riga per ciclo `FREEZE → esito`;
- `CANDIDATE_LOOKS.csv`: look completi e chiusure parziali diagnostiche;
- `CANDIDATE_NOTE.md`: interpretazione circoscritta e proposta minima;
- `MANIFEST.json`: provenienza e hash dei prodotti;
- `TEST_LOG.txt`: comandi ed esiti delle verifiche.

## Riproduzione su Windows

Dalla radice del clone:

```powershell
py -3.13 adattamento-drift/adapt06c_candidate_diagnosis/analyze_candidates.py
py -3.13 -m pytest adattamento-drift/adapt06c_candidate_diagnosis -q
```

La rigenerazione sovrascrive soltanto i quattro prodotti derivati nella
directory ADAPT-06C. Le confusioni e le condizioni decisionali vengono
ricalcolate dai log, non accettate come riepiloghi dichiarativi.

# Griglia dei rapporti misurata sui test set — superata

Questi file sono la valutazione del joint training a **1:10, 1:20, 1:50 e
1:100 sui test set** (TON_test, BoT_test), prodotta quando il rapporto
veniva scelto guardando come i modelli degradavano su quelle stesse
quantita'.

**Non sono risultati correnti e non vanno citati come tali.** Il protocollo
in vigore, richiesto dal Prof. Kuznetsov, e' quello opposto: il rapporto si
sceglie su una validation ritagliata dentro il training
(`scripts/joint_training.py --select-ratio`) e i test set entrano una volta
sola, al rapporto gia' scelto. Tenerli in `results/` accanto ai risultati
veri significherebbe continuare a pubblicare cinque valutazioni sui test
dove ne e' ammessa una.

Restano qui perche' cancellarli nasconderebbe come si e' arrivati alla
scelta, che e' proprio la cosa da rendere tracciabile. Stessa logica di
`results/protocol_v1/`.

## Cosa guardare invece

| domanda | file corrente |
|---|---|
| perche' 1:5 | `results/joint_ratio_selection.csv`, `..._significativita.csv`, `..._scelta.json` |
| quanto e' stabile ogni modello lungo la griglia | `results/joint_ratio_dispersione.csv` |
| i risultati del joint training | `results/joint_training_*_ratio5_cat.csv` |
| la tabella dell'articolo | `results/tabella_finale.csv` |

Tutti e quattro sono misurati sulla validation interna, tranne l'ultimo
gruppo, che e' l'unica valutazione sui test.

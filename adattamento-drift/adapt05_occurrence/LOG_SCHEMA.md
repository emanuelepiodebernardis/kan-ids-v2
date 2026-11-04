# Formato dei rendiconti ADAPT-05

Ogni JSON è UTF-8, schema `adapt05-occurrence-roles-v1`, e registra:

- commit eseguito, SHA-256 ricalcolato dagli array effettivamente letti da
  `D.npz`, contenitore NPZ, modello e sorgenti;
- dataset, split, seed modello/query, nome e bytes del sale;
- capacità, età, minimo per classe, budget, delay e obiettivi di soglia;
- per blocco: righe, soglie prima del feedback, digest dello stato head,
  confusion matrix per ogni politica, digest dei `row_id` richiesti, digest
  delle assegnazioni, conteggi FIT/CAL e provenienza del feedback;
- per modalità: stato applicato/skip, aggiornamento CAL indipendente,
  composizione, età e vettori unici delle memorie;
- riepilogo globale, aggiornamenti, label pendenti, tempo ed ambiente;
- esito della controprova vettoriale contro il rendiconto Paper 2 archiviato:
  hash delle richieste, conteggi/età, soglie, confusion matrix e stati di update;
  il digest dei parametri e il massimo delta cross-platform delle soglie head
  sono registrati separatamente come diagnostici più stretti.

Le confusion matrix sono ordinate `[TN, FP, FN, TP]`. I log contengono digest e
conteggi, non le feature o le label complete. I JSON di valutazione sono output
offline e non fanno parte dello stato accessibile al learner.

Nel controllo `ALL256`, i campi `retained_*` descrivono il buffer fisicamente
conservato e sono confrontati con l'audit storico. I campi `eligible_*`
descrivono invece il sottoinsieme con eta' non superiore a 50 usato per la
ricalibrazione. Le due viste possono differire durante una lunga assenza di una
classe e non sono intercambiabili. I precedenti campi `normal`, `attack` e
`age_max` restano alias retrocompatibili della vista `eligible_*`.

## Copertura cumulativa e memoria online

La copertura offline dei vettori normali usa tutte le richieste dell'intero
stream, incluse quelle terminali ancora pendenti alla chiusura del replay. E'
quindi una diagnostica cumulativa retrospettiva e non uno stato consultato dal
learner. La CAL64 online e' descritta, dopo ogni feedback, da `size`, `normal`,
`attack`, `recent_normal`, `recent_attack`, `age_min` e `age_max`. Gli ultimi
due campi delimitano l'eta' dell'intera vista ammissibile e non sono separati
per classe. Lo snapshot registrato dopo il feedback nel blocco k diventa
utilizzabile dalle predizioni del blocco k+1. I report non serializzano gli
identificatori trattenuti per blocco, quindi il ricambio esatto non e'
ricostruibile dai soli log.

# Modelli del protocollo v1 — NON usare con il codice attuale

Questi pesi sono stati addestrati prima dell'introduzione dello slot UNK nelle
tabelle categoriche. Le loro tabelle hanno 3/9/13/3 righe; quelle del protocollo
v2 ne hanno 4/10/14/4, dove l'indice 0 e' riservato alle categorie mai viste in
training.

Il codice attuale emette indice 0 per UNK e 1..N per le categorie reali. Passando
quegli indici a una tabella v1 ogni lettura e' sfalsata di una riga e l'ultima
categoria cade fuori dai limiti: nessuna eccezione, solo predizioni sbagliate.

Conservati per tracciabilita'. I modelli correnti sono in ../, descritti in
../MANIFEST.json; gli artefatti realmente deployabili sono gli header C in
../../mcu_pio/include/.

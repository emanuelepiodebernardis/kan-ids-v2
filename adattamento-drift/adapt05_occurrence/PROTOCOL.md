# ADAPT-05 — CAL rappresentativa con ruoli per osservazione

Protocollo fissato il 2 novembre 2026, prima dello smoke e della campagna.
Questa è una diagnosi sui flussi TON e UNSW già esaminati; non costituisce una
nuova validazione indipendente. Il precedente D resta quindi dato di sviluppo.

## Domanda e confronto finito

La domanda è se assegnare FIT/CAL alla singola osservazione, anziché al vettore,
migliori la copertura della calibrazione e quanto beneficio aggiuntivo conservi
l'apprendimento degli score. Il cambio di chiave non è presentato come un nuovo
algoritmo.

Si riusano esclusivamente la GAM logistica additiva a spline e i cinque stati
iniziali archiviati del Paper 2. Il fattoriale principale è:

- score `frozen` oppure `head` adattata;
- ruolo storico per vettore oppure ruolo per osservazione.

Ogni cella usa FIT192 e CAL64, quote minime 96/96 e 32/32 con prestito dei posti
liberi, età massima 50 blocchi e almeno 5 esempi recenti per classe. Entro una
modalità di ruolo, frozen e head ricevono le stesse osservazioni CAL e aggiornano
la soglia negli stessi istanti atomici. La capacità non varia con la modalità.

I riferimenti sono Frozen/B, Head/B, Frozen/ALL256 con gate di calibrazione
indipendente e Frozen/CAL64 con gate indipendente dal FIT. Gli effettivi
calendari di aggiornamento delle due modalità di ruolo possono differire e sono
misurati: la differenza osservata non sarà attribuita automaticamente alla sola
copertura.

Le soglie sono preregistrate per `lambda=1`, `lambda=2` e target FPR 5%, 10% e
20%. Si riusano ordinamento e tie-break del Paper 2. Non si selezionano sale,
soglie o politiche guardando risultati futuri.

## Causalità e appaiamento

- Budget richieste: 1% senza rimpiazzo in ogni blocco.
- Dimensione blocco: 10.000 righe; l'ultimo blocco può essere parziale.
- Ritardo: richiesta nel blocco `k`, etichetta disponibile dopo la predizione di
  `k+1`, primo effetto possibile su `k+2`.
- Modello e query seed coincidono e valgono 42–46.
- Tutte le traiettorie appaiate usano gli stessi `row_id` richiesti.
- Il ruolo viene deciso al momento della richiesta, prima della label.
- Non si effettuano richieste aggiuntive in funzione di label nascoste.

## Contratto byte dei ruoli

I tre sali sono bytes ASCII immutabili:

| nome | UTF-8/ASCII | hex |
|---|---|---|
| default | `ADAPT04-C-seed42-v1` | `414441505430342d432d7365656434322d7631` |
| salt1 | `review2-salt-1` | `726576696577322d73616c742d31` |
| salt2 | `review2-salt-2` | `726576696577322d73616c742d32` |

Il modo vettoriale ripete esattamente la regola storica:
`SHA256(sale || 0x00 || vettore_canonico_float64_le)`. Il ruolo è CAL se
l'intero big-endian del digest è divisibile per 4, altrimenti FIT.

Il modo per osservazione usa:

`SHA256("kan-ids-v2/adapt05/occurrence-role/v1\0" || F(dataset) || F(split) || F(row_id_i64_be) || F(sale))`

dove `F(v) = lunghezza_u32_be || v`. La probabilità CAL è ancora 1/4 con la
stessa regola modulo 4. Feature, score e label non entrano nel digest.

`observation_id`, digest canonico del vettore e digest della decisione sono
campi distinti. Il digest di occorrenza non conta i vettori unici. Uno stesso
`observation_id` non può entrare in FIT e CAL; occorrenze diverse dello stesso
vettore possono invece attraversare i due ruoli.

## Memoria e stato accessibile

Lo stato online contiene soltanto FIT192, CAL64, ALL256 di controllo, label in
attesa e parametri/soglie correnti. Non esiste un archivio laterale illimitato
per recuperare esempi espulsi. I log per la valutazione offline sono separati e
non sono consultati dal learner.

Nel controllo vettoriale il learner usa direttamente la memoria storica. Una
vista di audit, anch'essa limitata a 192/64 elementi, aggiunge i tre identificatori
separati e viene confrontata array per array con la memoria del learner; non è
consultata per fit o calibrazione.

La controprova vettoriale richiede identità esatta di query, composizione/età,
stati update e confusion matrix. Se score e soglie vengono ricalcolati su una
piattaforma numerica diversa da quella Linux archiviata, il delta di ogni soglia
e il massimo assoluto vengono registrati e tutte le confusion matrix devono
restare comunque identiche. Questo caso è marcato esplicitamente e non viene
riportato come identità delle soglie o riproduzione numerica byte per byte.

Le esecuzioni Linux ufficiali usano Python 3.12.14, le dipendenze congelate
del pacchetto Paper 2, un solo thread BLAS e un `OPENBLAS_CORETYPE` esplicito.
Il kernel `SkylakeX`, identificato esclusivamente riproducendo il controllo
vettoriale archiviato, e' congelato prima dello smoke ufficiale. E' parte
dell'ambiente registrato, non un fattore sperimentale. Lo smoke numerico
ufficiale richiede quindi un host Linux con AVX-512; sugli altri host restano
eseguibili i test sintetici, ma non si dichiara identita' numerica.

## Matrice completa e prima consegna

La matrice completa comprende 2 dataset × 5 seed × 3 sali = 30 combinazioni.
Ogni combinazione contiene le quattro traiettorie fattoriali; i riferimenti
invarianti non sono repliche aggiuntive. I cinque seed misurano variabilità di
randomizzazione condizionata ai due stream, non cinque catture.

La prima consegna è limitata a:

1. protocollo, matrice, schema dei log e test sintetici;
2. smoke sui primi 30 blocchi di ciascun D, seed 42 e sale `default`;
3. tutte le quattro traiettorie, anche se non raggiungono l'eligibility;
4. verifica causale e contabile, non selezione di impostazioni;
5. stima dei tempi della matrice completa dopo lo smoke.

La campagna completa partirà soltanto dal commit congelato dopo test PASS. Ogni
eventuale cambiamento al contratto sarà documentato prima di una nuova corsa.

## Esiti da conservare

Il rapporto finale conterrà recall, FPR raggiunto, FP, FN e BA; violazioni dei
target, andamento temporale e primi blocchi separati. I confronti a FPR
raggiunto comune saranno limitati al supporto osservato, senza extrapolazione e
distinti dalle politiche realmente eseguibili.

Saranno inoltre riportati copertura dei vettori normali ponderata per frequenza,
tipi frequenti esclusi da CAL, normali disponibili, età e ricambio, update/skip
e costi; visto/non visto A/B e ripetizioni interne al flusso resteranno distinti.
La dispersione tra sali e tra seed sarà separata. Tutti gli esiti saranno
conservati, anche quando l'apprendimento non supera frozen.

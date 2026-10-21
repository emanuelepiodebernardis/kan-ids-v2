# ADAPT-04: pesi o sola soglia con calibrazione recente separata

## Provenienza e domanda

Il branch parte da `main` al merge commit
`d7bcd716dc416359474f12023cf0e5766d3aa78a`, che integra il PR #5 mantenendo
il commit approvato `b5b56c75f2771d32765871f8e7ac15fca69e5ef5`.

La domanda e' se, con l'1% di etichette e al massimo 256 esempi online,
adattare la testa della KAN dia un beneficio rispetto ad adattare soltanto la
soglia quando fit e calibrazione usano dati recenti distinti. Non si assume
una superiorita' a priori: scarsita' di normali, assenza di aggiornamenti o
assenza di beneficio sono esiti validi.

## Perimetro fissato

- sola KAN, C/seed42, 916 blocchi cronologici da 10.000 righe;
- inizializzazione del modello su A, preprocessing e campionamento uniforme
  dell'1% invariati;
- ritardo di un blocco: le etichette osservate nel blocco `j` arrivano dopo le
  predizioni di `j+1` e possono influire non prima di `j+2`;
- rappresentazione additiva B-spline congelata;
- il fit aggiorna soltanto otto guadagni e intercetta con
  `LogisticRegression(max_iter=1000, C=1.0)`;
- soglia iniziale B gia' verificata e letta dal riferimento ADAPT-03;
- B non viene caricato, ricalibrato o posto in cache durante il replay;
- nessuna nuova architettura, regolarizzazione aggiuntiva o ricerca di
  iperparametri;
- D, cinque seed e validazione finale restano fuori dal protocollo.

## Assegnazione immutabile del ruolo

Ogni riga richiesta riceve il ruolo `fit` o `calibration` prima che la sua
etichetta sia disponibile. Il ruolo e' funzione soltanto del vettore originale
degli otto input di C, nell'ordine conservato nel flusso, e non dipende da
etichetta, blocco, stato delle memorie o stato del modello.

La serializzazione canonica e' fissata cosi':

1. il vettore deve contenere esattamente otto valori numerici;
2. ogni valore e' convertito in IEEE-754 `float64` little-endian;
3. `-0.0` e `+0.0` sono entrambi serializzati come `+0.0`;
4. tutti i NaN sono sostituiti dal quiet NaN canonico con bit
   `0x7ff8000000000000`;
5. `+inf` e `-inf` conservano i rispettivi valori IEEE-754;
6. il messaggio e' il salt ASCII `ADAPT04-C-seed42-v1`, seguito da un byte
   nullo e dai 64 byte del vettore canonico;
7. si calcola SHA-256 e si interpreta il digest come intero unsigned
   big-endian.

Digest modulo 4 uguale a zero assegna `calibration`; gli altri tre valori
assegnano `fit`. Vettori identici hanno quindi sempre lo stesso ruolo, anche
dopo l'espulsione. La proporzione 1/4 descrive la regola e non garantisce il
25% delle righe, soprattutto in presenza di duplicati. Il rendiconto riporta
richieste e vettori unici per ruolo.

## Due memorie indipendenti

- `fit192`: capacita' 192, quote per classe 96/96;
- `calibration64`: capacita' 64, quote per classe 32/32.

Entrambe partono vuote. Nessun esempio di A viene inserito. Ciascuna usa la
regola per classe gia' verificata: i posti liberi possono essere occupati
dall'altra classe; al superamento del limite si espelle prima dalla classe
sopra quota, sempre dall'esempio piu' vecchio. Non si trasferiscono posti,
righe o esempi espulsi fra le due memorie. Una riga `calibration` non entra mai
nel fit, neppure in seguito.

L'eta' e' misurata dal blocco di osservazione, non da quello di arrivo
dell'etichetta. Il limite di 50 blocchi riguarda soltanto gli esempi usati per
calibrare le soglie; `fit192` usa tutto il proprio contenuto corrente.

Questa separazione evita il riuso della stessa riga nei due ruoli, ma non
garantisce indipendenza temporale e non elimina le sovrapposizioni storiche di
C con A o B. Non viene presentata come test indipendente.

## Tre politiche appaiate

1. `frozen_cal64`: parametri iniziali congelati; soglia aggiornata su
   `calibration64`.
2. `fit192_b`: testa adattata su `fit192`; soglia B iniziale sempre fissa.
3. `fit192_cal64`: esattamente la stessa traiettoria della testa della politica
   2; soglia aggiornata sullo stesso `calibration64` della politica 1 e sugli
   score ricalcolati con la testa candidata.

Le tre politiche condividono campioni, assegnazioni, buffer, ordine temporale e
calendario. Per blocco esiste un solo score frozen e un solo score adattato;
gli score delle politiche 2 e 3 devono essere identici. Il confronto principale
e' 3 contro 1; 3 contro 2 isola la soglia a pesi identici. I risultati
ADAPT-03 `threshold-only256` e `historical-B` sono solo riferimenti di contesto,
perche' memoria, inizializzazione e calendario sono diversi.

## Transazione di aggiornamento

Dopo l'arrivo previsto delle etichette, un passo e' ammissibile solo se:

- `fit192` contiene entrambe le classi;
- `calibration64`, dopo il filtro eta' `<= 50`, contiene almeno 5 normali e 5
  attacchi.

Sul medesimo passo si calcola un solo fit candidato. Senza mutare lo stato
corrente si rivaluta `calibration64` con la testa iniziale e con la testa
candidata, quindi si scelgono entrambe le soglie con la funzione BA gia'
verificata: candidati distinti, confronto `float64` esatto e mediana inferiore
fra i massimi.

La transazione viene accettata soltanto se testa e due soglie sono tutte
valide. In tal caso entrano insieme in vigore, dal blocco successivo:

- la nuova testa comune alle politiche 2 e 3;
- la nuova soglia della politica 1;
- la nuova soglia della politica 3.

La politica 2 conserva sempre la soglia B. Se manca un requisito o il fit e'
invalido, tutte le politiche mantengono lo stato precedente e viene registrata
una causa fra `fit_classe_mancante`, `cal_normali_recenti_insufficienti`,
`cal_attacchi_recenti_insufficienti` e `fit_invalido`. Non e' ammessa una nuova
testa associata alla vecchia soglia adattiva.

Le decisioni gia' emesse non vengono ricalcolate. Le etichette complete di C
sono usate soltanto dalla valutazione offline, separata dal controllore.

## Esecuzione e criteri di controllo

Il protocollo e la configurazione vengono committati prima del codice. Le
prove mirate devono coprire almeno:

- determinismo del ruolo e duplicati invarianti nel tempo;
- separazione perpetua fra fit e calibration;
- capacita', quote, espulsione e classi mancanti;
- eta' dall'osservazione e causalita' `j+2`;
- atomicita' e cause degli skip;
- ricalcolo degli score calibration con la testa corrente;
- identita' della traiettoria dei pesi nelle politiche 2 e 3;
- metriche non definite lasciate mancanti.

Lo smoke usa i blocchi `0..19`. Se test, completezza, causalita' e invarianti
passano, lo stesso codice esegue direttamente un solo replay completo sui
blocchi `0..915`, senza cambiare salt, quote o criteri in base allo smoke.

## Consegna e interpretazione

Il report principale non supera due pagine; appendici e file macchina restano
separati. Si consegnano CM globale, FN/FP, recall, FPR, balanced accuracy,
AUROC con denominatori, medie e dispersioni per blocco, differenze appaiate,
periodi di peggioramento, soglie, fit, aggiornamenti, skip e cause.

Per le memorie si riportano composizione ed eta' per classe, vettori unici,
normali recenti e ripartizione effettiva fit/calibration. Costi di inferenza,
fit e calibrazione restano distinti. Payload dei buffer, metadati, richieste in
attesa, stato del modello e RSS sono contabilizzati separatamente; i log
offline non sono memoria del controllore.

Rendiconto originale, CSV per blocco, figure, manifest, SHA eseguito, ambiente
e comando vengono conservati. Il rapporto riassume i quattro riferimenti
ADAPT-03 senza rieseguirli. Il giudizio usa insieme attacchi mancati, falsi
allarmi e continuita' temporale: il solo aumento della balanced accuracy non e'
sufficiente se il FPR cresce molto.

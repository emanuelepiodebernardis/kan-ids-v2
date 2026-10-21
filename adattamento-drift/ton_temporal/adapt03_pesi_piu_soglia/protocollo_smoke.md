# ADAPT-03: protocollo dello smoke pesi adattati + soglia

## Domanda e confini

Il diagnostico verifica se il basso richiamo della KAN adattata con memoria per
classe dipenda, almeno in parte, dal mantenimento della soglia scelta su B prima
degli aggiornamenti dei pesi. Il confronto separa la decisione dalla
traiettoria del modello: uno stesso score viene classificato con due soglie.

La base verificata e' il commit ADAPT-03 `d11c59c`. La prima consegna comprende
questo protocollo, le controprove e uno smoke sui primi 20 blocchi di C. Non
autorizza il replay completo, una campagna su cinque semi o l'uso di D.

## Configurazione fissata

- inizializzazione su A e calibrazione iniziale su B;
- replay sui primi 20 blocchi cronologici di C, seed 42;
- sola KAN, memoria per classe di capacita' totale 256, quote 128/128;
- politica di aggiornamento dei pesi `ogni_blocco`;
- blocchi da 10.000 righe, budget uniforme dell'1% e ritardo di un blocco;
- trasformazione, modello, fit della testa e campionamento invariati rispetto
  ad ADAPT-02;
- controllo: soglia iniziale scelta sull'intero B e poi fissa;
- variante: stessa soglia iniziale, poi ricalibrata sull'intero B dopo ogni
  aggiornamento effettivamente applicato ai pesi;
- predizione `score > soglia`;
- regola di scelta: massimo della balanced accuracy su B, uguaglianza esatta
  fra `float64`; fra i pari merito, mediana inferiore dei candidati ordinati,
  indice `(n-1)//2`.

La variante 192/64 non appartiene a questo protocollo. Non si introduce un
limite assoluto al FPR.

## Unica traiettoria dei pesi e causalita'

Il runner mantiene una sola KAN adattiva. Per ogni blocco:

1. calcola una sola volta gli score con i parametri correnti;
2. applica agli stessi score la soglia B fissa e la soglia ricalibrata;
3. richiede le stesse etichette del runner ADAPT-02;
4. a fine blocco incorpora il lotto maturato nella memoria online;
5. applica, se possibile, un solo aggiornamento dei pesi;
6. soltanto dopo un aggiornamento effettivo rivaluta tutto B e sceglie la
   soglia successiva;
7. rende la nuova soglia attiva dal blocco seguente.

Le etichette e gli score di C non entrano mai nella scelta della soglia. La
soglia non modifica richieste, memoria, calendario o parametri. Controllo e
variante devono quindi avere identici digest dei parametri e degli score,
identico calendario degli aggiornamenti e identica AUROC. Una differenza in
una di queste invarianti interrompe la corsa.

## B e la memoria online sono risorse distinte

L'intero B, circa 5,68 milioni di righe, e' un archivio offline aggiuntivo e
non fa parte dei 256 esempi online. B non entra nel fit della testa. Per evitare
di ricostruire le basi spline a ogni aggiornamento, lo smoke calcola una volta
la rappresentazione KAN di tutto B in blocchi da 200.000 righe e la conserva
come matrice `float64`. Ogni ricalibrazione calcola gli score correnti da questa
cache e applica la regola BA invariata all'intero vettore.

Il rendiconto distingue:

- dimensione compressa di `B.npz`;
- righe e byte logici di `X_B` e `y_B` usati dalla calibrazione;
- byte della cache della rappresentazione KAN;
- byte del vettore di score;
- memoria online da 256 esempi;
- picco RSS del processo e metodo di misura;
- tempo di costruzione della cache;
- per ogni ricalibrazione, tempo di score, selezione BA e totale.

Il picco RSS comprende anche gli input del banco di prova e non rappresenta la
memoria di un dispositivo in linea. I byte logici separano invece archivio,
cache e stato online.

## Appaiamento e controprove

Lo smoke richiede le impronte preregistrate di A, B e C e confronta i primi 20
blocchi con il rendiconto ADAPT-02 KAN/per-classe/ogni-blocco, seed 42. Devono
coincidere soglia B, campioni richiesti, digest dei `row_id`, misure del
controllo a soglia fissa e numero di aggiornamenti applicati.

Le prove negative devono rifiutare almeno:

- un input o un parametro preregistrato diverso;
- B vuoto, parziale o con etichette non binarie;
- una regola BA o un tie-break differente;
- l'uso di esempi C o della memoria online per calibrare la soglia;
- due traiettorie dei pesi, score differenti o AUROC differenti fra i bracci;
- una soglia aggiornata nello stesso blocco che ha prodotto il nuovo peso;
- una ricalibrazione senza aggiornamento effettivo dei pesi;
- uno smoke vuoto, incompleto, duplicato o non ordinato rispetto a `0..19`.

## Misure e decisione successiva

Si riportano globalmente e per blocco FN, FP, recall degli attacchi, FPR,
balanced accuracy e AUROC, insieme a soglie, digest, aggiornamenti e andamento
temporale. Recall, FPR e balanced accuracy vanno letti insieme: un aumento del
solo recall non costituisce un esito favorevole.

Lo smoke serve anzitutto a verificare causalita', appaiamento e fattibilita'.
La stima del replay completo usa il costo osservato delle ricalibrazioni e il
numero di aggiornamenti effettivi del riferimento ADAPT-02 completo. Riporta
una stima centrale e una prudenziale, oltre a memoria e cache. I risultati dei
20 blocchi sono diagnostici e non giustificano conclusioni di superiorita'.
Solo dopo questa consegna si chiede conferma per il replay completo su C.

# ADAPT-06B-MATRIX — protocollo congelato prima dei run

## Domanda e interpretazione

La matrice verifica se il segnale osservato su UNSW con seed 42 rimane visibile
al variare del seme iniziale, senza modificare il metodo. Non costituisce una
conferma su catture indipendenti e non introduce una soglia retroattiva di
successo. Tutti i seed, inclusi quelli nulli o sfavorevoli, entrano nelle
statistiche appaiate.

## Perimetro

- stream completi TON e UNSW;
- sola famiglia spline-GAM e acquisizione RANDOM;
- seed 42, 43, 44, 45 e 46;
- cinque controlli invariati: Frozen/B, threshold-only autonoma, GATED legacy,
  GATED decoupled e threshold-only matched;
- seed 42 riutilizzato byte per byte dal pilot verificato; nessun rerun e
  nessuna sovrascrittura;
- seed 43–46: 16 nuovi replay, legacy e decoupled per due dataset, organizzati
  in otto coppie;
- nessun nuovo fit iniziale, tuning, sweep, ACTIVE, LR/ARF, nuovo dataset o
  ADAPT-07.

## Parametri invariati

FIT192, quote 96+96 e minimo 32 per classe; CAL256, quote 128+128, età massima
50 blocchi e minimo 100 normali/100 attacchi; budget di label 1%, ripartito 25%
CONTROL e 75% RANDOM_FIT; ritardo di una unità e primo effetto dopo un ulteriore
blocco. I look VERIFY restano 100+100, 200+200 e 400+400; scadenza 50 blocchi;
margini +2 punti percentuali di recall, +1 punto di FPR e FPR candidato massimo
6%. La soglia usa il target FPR 5% e la regola `score > threshold`.

## Provenienza e arresto preventivo

Prima dei replay il preflight deve verificare:

1. commit di esecuzione e configurazione versionata;
2. integrità del manifest e di tutti i file del pilot seed 42;
3. presenza, SHA-256 e metadata dei dieci stati A/B spline-GAM;
4. corrispondenza dei digest A, B e D con la provenienza archiviata;
5. identità scientifica di ogni array D effettivamente letto;
6. directory di uscita nuova e limiti BLAS a un thread.

Qualunque mancanza o divergenza interrompe l'esecuzione. Legacy e decoupled di
ogni coppia sono avviati in processi distinti e condividono input, richieste
RANDOM e budget; `verify_reports()` rifiuta una coppia che non rispetta queste
invarianti.

## Analisi fissata

Il confronto principale è GATED decoupled meno threshold-only autonoma. Per
dataset e seed sono riportati insieme delta recall, delta FPR, TP, FN e FP,
promozioni dei pesi e variazioni della soglia. Frozen/B, threshold-only matched
e GATED legacy restano controlli diagnostici separati.

Per ciascun dataset si calcolano media, deviazione standard campionaria,
minimo, massimo e seed peggiore delle differenze appaiate, oltre ai conteggi di
segno favorevole per recall, FPR ed entrambe le metriche. Le traiettorie usano
la stessa frontiera temporale del pilot: prima applicazione della soglia nel
blocco successivo alla prima eleggibilità CAL.

Il seed peggiore è definito prima dei run ordinando prima il delta recall in
senso crescente e, a parità, il delta FPR in senso decrescente. Uno zero resta
neutro: è favorevole soltanto un delta recall strettamente positivo o un delta
FPR strettamente negativo.

Le due figure preregistrate sono:

1. piano delta FPR/delta recall per tutti i seed e dataset, con assi a zero;
2. traiettorie cumulative recall/FPR dei due controlli principali per tutti i
   seed, con prima applicazione CAL e promozioni indicate.

Il pacchetto è tecnicamente completo anche con esito nullo o negativo. Le medie
non autorizzano conclusioni di superiorità generale.

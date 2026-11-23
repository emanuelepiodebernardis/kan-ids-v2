# ADAPT-06E — VERIFY 50/100 a budget invariato

## Domanda e ipotesi

Il protocollo verifica se destinare temporaneamente metà delle richieste di
etichetta a VERIFY, quando un candidato è già aperto, consenta decisioni più
tempestive e un beneficio operativo successivo senza aumentare i falsi
allarmi. L'ipotesi riguarda la ripartizione delle stesse etichette, non un
aumento del budget.

## Perimetro congelato

- Stream completi TON e UNSW; seed 42–46; GAM con acquisizione RANDOM.
- Stati iniziali A/B archiviati; FIT192, CAL256; nessun nuovo fit iniziale.
- Budget per blocco pari all'1% delle osservazioni, ritardo di un blocco,
  soglia e predizioni calcolate senza usare etichette future.
- Margini, look VERIFY (100/100, 200/200, 400/400) e scadenza di 50 blocchi
  identici ad ADAPT-06B.
- Nessun tuning, nuovo dataset, ACTIVE o ADAPT-07.

## Assegnazione delle richieste

La selezione RANDOM storica viene eseguita integralmente prima di osservare
le etichette. In assenza di un candidato aperto, la ripartizione resta
CONTROL=25% e RANDOM_FIT=75% del budget `Q`.

Se un candidato è aperto all'inizio del blocco:

1. tutti i CONTROL originali restano CONTROL;
2. `floor(Q/2) - CONTROL` ID vengono prelevati dalla testa dell'ordine
   RANDOM_FIT originale e diventano VERIFY_EXTRA;
3. gli ID RANDOM_FIT rimanenti restano FIT.

Con `Q=100`, la ripartizione è 25 CONTROL, 25 VERIFY_EXTRA e 50 FIT. L'insieme
e l'ordine degli ID richiesti sono identici al riferimento ADAPT-06B.

Ogni VERIFY_EXTRA appartiene al solo ciclo aperto all'inizio del blocco. Le
relative etichette arrivano nel blocco successivo e non entrano mai in FIT o
CAL. Se il ciclo è già chiuso all'arrivo, sono conteggiate come tardive e non
riutilizzate. I CONTROL originali conservano invece la gestione storica
VERIFY/CAL. Una decisione alla fine del blocco influisce soltanto sulle
predizioni dei blocchi successivi.

## Controlli prima dei risultati

I test sintetici devono dimostrare:

- conservazione del budget e identità degli ID richiesti;
- ritardo causale di un blocco;
- esclusione completa di VERIFY_EXTRA da FIT e CAL, anche dopo chiusura;
- attribuzione univoca al ciclo e contabilità dei campioni tardivi;
- gestione dell'ultimo blocco e dei campioni ancora pendenti;
- identità del percorso storico quando la nuova opzione è disattivata.

Prima della variante si riesegue ADAPT-06B su TON e UNSW, seed 42, e si
confrontano ID, decisioni e confusioni operative con gli artefatti archiviati.
Ogni discrepanza interrompe la matrice.

## Confronti e misure

Per dataset e seed si riportano TP, FN, FP, TN, recall e FPR della variante e
le differenze appaiate rispetto a:

1. ADAPT-06B (GATED decoupled storico);
2. pesi congelati con gli stessi eventi e la stessa CAL della variante
   (`threshold_matched`);
3. Frozen/B;
4. threshold-only autonoma storica.

Per ogni promozione si indicano blocco decisionale, primo blocco operativo,
osservazioni residue e variazioni TP/FP fino alla successiva promozione. Gli
esiti VERIFY restano separati dalle metriche operative. Si riportano inoltre
tempo alla decisione, candidati irrisolti, composizione FIT, etichette uniche,
tempo e memoria.

## Regola di sviluppo preregistrata

Per ciascuno stream l'esito è GO soltanto se, negli stessi almeno tre seed su
cinque, la variante aumenta TP senza aumentare FP sia rispetto ad ADAPT-06B
sia rispetto al controllo a pesi congelati, e se in nessun seed supera l'FPR
di ADAPT-06B di oltre un punto percentuale. Tutti gli svantaggi rispetto a
Frozen/B e threshold-only sono mantenuti visibili. Un maggior numero di
decisioni senza beneficio operativo chiude l'ipotesi.


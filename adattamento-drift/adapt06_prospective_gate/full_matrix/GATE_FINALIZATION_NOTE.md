# ADAPT-06 — eleggibilità CAL e diagnosi del gate

## Perimetro

Questa nota usa esclusivamente i dieci report della configurazione primaria GAM ACTIVE della matrice già eseguita: TON e UNSW, seed 42–46. Non introduce fit, replay, tuning, soglie o criteri nuovi. Il commit eseguito resta `2b2ec4d02672be8432f9e462e7011359bf5c66d8`; il commit della consegna è quello che contiene questa nota e deve essere riportato nella PR. L'esito preregistrato resta **NO-GO** per entrambi i dataset.

`PRIMARY_REFERENCE_COMPARISON.csv` riporta per ogni dataset e seed recall e FPR globali di Frozen/B, threshold-only, ALWAYS e GATED. Le promozioni riguardano i candidati di modello: Frozen/B e threshold-only ne hanno zero. Per threshold-only è riportato separatamente il numero di variazioni effettive della soglia, evitando di chiamarle promozioni.

## Quando CAL diventa eleggibile

CAL256 è eleggibile quando la memoria causale, dopo scadenze ed espulsioni, contiene almeno 100 normali e 100 attacchi. Nei seed 42–46 il primo blocco eleggibile è rispettivamente **275, 275, 275, 274, 274** su TON e **17, 18, 19, 17, 18** su UNSW. Il primo congelamento del candidato avviene agli stessi blocchi: **275, 275, 275, 274, 274** su TON e **17, 18, 19, 17, 18** su UNSW. In questi dieci report eleggibilità e congelamento coincidono; concettualmente restano distinti, perché il congelamento richiede anche FIT192 eleggibile e nessun candidato già aperto.

## Perché i candidati non vengono promossi

Su TON i cinque seed producono 18 candidati: nessuna promozione, 10 rifiuti per una frontiera di danno preregistrata e 8 chiusure per supporto insufficiente. Su UNSW i candidati sono 17: 1 promozione, 10 rifiuti per danno, 3 chiusure per supporto insufficiente e 3 chiusure ambigue. I conteggi completi per seed sono in `PRIMARY_GATE_DIAGNOSTICS.csv`.

Le cause possono sovrapporsi nello stesso rifiuto. Su TON la perdita di recall oltre la frontiera compare 7 volte, l'FPR candidato sopra il 6% 3 volte e l'incremento di FPR oltre 1 punto 2 volte. Su UNSW i conteggi sono rispettivamente 6, 2 e 6. Questi sono motivi diagnostici derivati dagli stessi conteggi usati dal gate, non nuovi criteri decisionali.

Un candidato non valutabile non è trattato come dato mancante favorevole: rimane un withholding. Analogamente, un look intermedio che non soddisfa né accettazione né danno resta WAIT e, se lo stream termina, viene chiuso secondo la regola terminale già preregistrata.

## Protezione e miglioramento sono domande diverse

Su TON GATED coincide esattamente con Frozen/B in 5/5 seed e non promuove alcun candidato. Il gate evita i 10 aggiornamenti giudicati dannosi e non promuove gli altri 8 candidati, chiusi per supporto insufficiente; conserva però anche la soglia B iniziale. In media, GATED/Frozen-B ha FPR **12,59%** e recall **20,93%**; threshold-only porta l'FPR a **4,64%** con recall **20,73%**. Il calo dell'FPR non deriva dall'apprendimento dei pesi: è l'effetto descrittivo della ricalibrazione causale della soglia sugli stessi score congelati.

Su UNSW il gate rifiuta 10 candidati che oltrepassano almeno una frontiera di danno e accetta soltanto 1 candidato. I risultati documentano una protezione rispetto ad ALWAYS, non una superiorità stabile rispetto a Frozen/B. Le medie descrittive GATED sono recall **94,70%** e FPR **1,99%**, contro **94,53%** e **1,95%** per Frozen/B; le differenze per seed restano visibili nella tabella e non vengono compresse in un'affermazione di vittoria generale.

## Conclusione operativa

Il gate svolge il compito conservativo per cui è stato progettato: blocca candidati con evidenza prospettica sfavorevole. Il NO-GO mostra però che questa protezione non basta a produrre un beneficio stabile rispetto allo stato congelato e può mantenere una soglia iniziale inadatta al nuovo regime. Prima di ADAPT-07 occorre quindi discutere separatamente come trattare la soglia operativa e come ottenere supporto CAL/VERIFY sufficiente, senza modificare retroattivamente il protocollo di questa campagna.

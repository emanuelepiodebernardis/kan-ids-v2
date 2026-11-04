# Schema temporale eseguibile

Lo schema definisce l'ordine normativo. Una futura implementazione deve
produrre gli stessi eventi a parità di input e seed.

## Stato minimo

```text
always_model, always_threshold
gated_model, gated_threshold
fit192, cal256, pending_labels, deferred_control
shared_query_selector                 # solo ACTIVE
candidate = null | {shared_model, shared_threshold, gated_incumbent_shadow,
                    recalibrated_incumbent_threshold_shadow, freeze_block,
                    digest, verify_counts, next_look}
audit_log
cycle_closed_this_block              # flag transitorio, falso a inizio blocco
```

Le memorie partono vuote sullo stream di sviluppo; modello e soglia iniziali
provengono da A/B. Le metriche offline non appartengono allo stato.

## Regole delle memorie

FIT192 e CAL256 sono FIFO per classe. FIT ha quote 96/96 e nessun limite di
età; CAL ha quote 128/128 ed età massima di 50 blocchi dal blocco di
osservazione. I posti inutilizzati possono essere prestati all'altra classe.
Quando arriva un elemento della classe proprietaria si espelle prima il
prestito più vecchio; se la quota proprietaria è piena si espelle invece il
suo elemento più vecchio. CONTROL in VERIFY e CONTROL osservati entro il freeze
ma arrivati dopo sono trattenuti separatamente in `deferred_control`. Entrano
in CAL soltanto alla chiusura del ciclo, in ordine di arrivo e col blocco di
osservazione originale; se hanno superato age 50 sono registrati e scartati. I
secondi non entrano in VERIFY perché non sono osservazioni future.

## Selettore ACTIVE

Le estrazioni uniformi usano generatori NumPy `PCG64` indipendenti, costruiti
per blocco con `SeedSequence([seed, dataset_code, block_index, role_code])`,
dove `dataset_code` è 0 per TON e 1 per UNSW e `role_code` è 0 per CONTROL, 1
per RANDOM-FIT e 2 per ACTIVE-EXPLORE. Il modello usa un flusso distinto. In
questo modo CONTROL non dipende dall'ordine delle politiche o dal numero di
estrazioni negli altri ruoli.

Per ogni blocco e ruolo si conserva SHA-256 della sequenza selezionata: righe
UTF-8 `dataset,seed,block,role,observation_id\n` nell'ordine di selezione. Il log
conserva anche probability quando definita, rango ACTIVE, score pre-feedback,
observation/arrival/first-effect block. Il manifest registra commit, ambiente,
configurazione e digest di input, stati A/B e contratto delle feature.

Per GAM/LR il selettore ombra parte dallo stato A/B comune ad ALWAYS e GATED.
Al freeze di ogni candidato riceve atomicamente il modello e la soglia comuni
dal blocco successivo, anche se il gate in seguito rifiuta il candidato; non si
aggiorna negli altri blocchi. Per ARF ACTIVE coincide con l'unica ARF operativa,
aggiornata dalle sole FIT arrivate, e usa la soglia B. Dopo EXPLORE uniforme,
EXPLOIT ordina le righe rimaste per distanza assoluta dalla soglia e forma una
shortlist di dieci volte i posti. Sulla shortlist applica farthest-first nelle
feature trasformate rispetto a FIT192 e agli elementi già scelti nel blocco;
`row_id` risolve gli spareggi. Il selettore non legge label correnti.

## Soglie e riferimento threshold-only

I candidati soglia sono gli score float64 distinti, il massimo osservato e il
valore `nextafter(minimo, -inf)`; la predizione è attacco se `score > soglia`.
Fra i candidati con FPR CAL non superiore al target si massimizza il recall,
poi si minimizza l'FPR; a confusioni pari si sceglie la mediana inferiore delle
soglie ordinate. La politica threshold-only viene scelta per famiglia, sullo
sviluppo, soltanto fra Frozen/B e
Frozen/CAL256-target5. È eleggibile con FPR medio non superiore al 6% su
entrambi gli stream e in almeno quattro seed su cinque; fra le eleggibili vince
il recall medio maggiore, poi l'FPR minore. Se nessuna è eleggibile, Frozen/B
resta il riferimento e la selezione fallita viene registrata.

## Ordine per il blocco k

```text
1.  carica lo stato entrante;
2.  emetti le predizioni operative ALWAYS e GATED su tutto il blocco k;
3.  se candidate != null, emetti gli shadow necessari alla verifica appaiata;
4.  calcola Q, CONTROL e FIT senza leggere label del blocco;
5.  campiona CONTROL uniforme con RNG comune;
6.  campiona FIT RANDOM oppure ACTIVE fra gli ID rimasti;
7.  registra ID, probabilità/rango, score e soglie prima del feedback;
8.  accoda le richieste del blocco k con arrival_block = k + 1;
9.  ricevi soltanto le label la cui arrival_block è k;
10. aggiorna FIT192; indirizza CONTROL a CAL, VERIFY o deferred_control secondo
    observation_block, arrival_block e freeze_block;
11. per GAM/LR, se candidate esiste e VERIFY raggiunge un look o la scadenza,
    calcola una sola decisione;
12. se ACCEPT: allinea GATED al candidato da k + 1 e marca il ciclo chiuso;
13. altrimenti, se REJECT/WITHHOLD: conserva GATED, lascia ALWAYS sulla
    promozione già fatta e marca il ciclo chiuso;
14. altrimenti WAIT: conserva entrambi e lascia aperto il ciclo;
15. se il ciclo è chiuso, trasferisci VERIFY e gli altri CONTROL differiti in
    CAL256 in ordine di arrivo, mantenendo observation_block ed eliminando
    quelli oltre age 50; dopo il log della decisione svuota le code trasferite,
    poni candidate = null e cycle_closed_this_block = true;
16. se candidate è null e cycle_closed_this_block è falso, verifica
    l'eligibility FIT/CAL per GAM/LR;
17. se eligible, addestra, calibra e congela un candidato comune; ALWAYS e il
    selettore ACTIVE lo adottano da k + 1, GATED lo valuta soltanto su dati
    successivi; la soglia ricalibrata dell'incumbent resta shadow;
18. nel percorso ARF, ignora 11–17 e aggiorna l'unico stato
    operativo/selettore soltanto con FIT;
19. scrivi digest, contatori, tempi e motivazioni; passa a k + 1.
```

Una label richiesta in `k` non può essere ricevuta al passo 9 né influire sui
passi 10–19 prima della fine di `k+1`; il primo blocco predetto con il suo
effetto è `k+2`. Le richieste terminali rimaste pendenti vengono contate ma non
forzate nel replay.

## Definizioni temporali delle metriche

I tre segmenti cronologici sono i terzi consecutivi dei blocchi, determinati
dal solo indice. Un segmento operativo stabile è un intervallo massimale con
modello e soglia invariati, esclusi i dieci blocchi successivi a una promozione;
richiede almeno dieci blocchi, altrimenti è `INSUFFICIENT`.

Ogni promozione apre una finestra transitoria di dieci blocchi, durante la
quale l'incumbent precedente continua come shadow. Il recupero è il primo
blocco in cui la media mobile su tre blocchi dell'errore totale del promosso
non supera quella dello shadow; se non accade è censurato a `>10`. Le aree
transitorie sono le somme separate delle differenze appaiate di FP e FN. Il
peggior caso è il massimo FPR e FNR di blocco fra quelli con denominatore
definito. Nessuna di queste regole usa le label per scegliere gli intervalli.

## Invarianti per i futuri test

1. Predizioni e query precedono ogni feedback dello stesso blocco.
2. CONTROL è identico fra tutti i confronti di una cella dataset/seed.
3. FIT è identico fra `ALWAYS` e `GATED` dello stesso braccio/famiglia.
4. FIT è disgiunto da CONTROL; per il candidato corrente, CAL e VERIFY sono
   due usi temporalmente disgiunti di CONTROL.
5. I pesi del candidato non usano CONTROL; la sua soglia usa CAL ma non VERIFY.
6. Una decisione modifica insieme modello e soglia, mai uno solo dei due.
7. Un `WAIT` non cambia lo stato operativo.
8. Budget, arrotondamenti e ultimo blocco coincidono con `LABEL_BUDGET.csv`.
9. Gli score/label completi offline non sono accessibili al controller.
10. Ogni decisione conserva blocco, supporto, delte, digest e motivazione.
11. Un nuovo candidato è vietato in entrambi i bracci durante VERIFY.
12. ALWAYS e GATED condividono candidato, freeze, CAL e punti di controllo.
13. La soglia shadow dell'incumbent non diventa operativa durante `WAIT`.
14. L'età CAL deriva dal blocco di osservazione, anche dopo un differimento in
    VERIFY; gli elementi scaduti non vengono ringiovaniti al reinserimento.
15. Nessun reset o taglio di finestra è dedotto retrospettivamente dalle label.
16. La chiusura di un ciclo impedisce di congelare un nuovo candidato nello
    stesso blocco; il controllo di eligibility riprende dal blocco successivo.
17. CONTROL osservati entro il freeze ma arrivati dopo non possono calibrare o
    verificare il candidato corrente.
18. Per GAM/LR il selettore ACTIVE cambia soltanto al freeze comune; per ARF
    cambia soltanto con una label FIT arrivata.

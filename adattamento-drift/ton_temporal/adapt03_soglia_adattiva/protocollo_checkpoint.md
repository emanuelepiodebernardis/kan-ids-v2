# ADAPT-03: protocollo del primo checkpoint

## Domanda e confini

Il checkpoint distingue il beneficio dell'aggiornamento dei pesi da quello
ottenibile lasciando invariati i pesi e aggiornando soltanto la soglia di
decisione. Il flusso C resta materiale di sviluppo gia' esaminato; D non viene
letto. Non si introducono nuove architetture, politiche evidence, budget o
campagne su cinque semi.

Base del lavoro: merge ADAPT-02
`6d945abf5eda35d35e93e151622369e7e84bfdd6`.

## Matrice fissata

- flussi: addestramento A, calibrazione B, replay C;
- seed: 42;
- modelli: LR, MLP e KAN gia' definiti dal runner;
- blocco: 10.000 righe; campionamento uniforme: 1%;
- ritardo: un blocco; le etichette richieste in `j` arrivano a fine `j+1`;
- memoria: 256 esempi, inizializzata dallo stesso campione di A del runner;
- varianti separate: FIFO e per-classe 128/128, mai sommate;
- soglia iniziale: soglia storica scelta su B per ciascun modello;
- smoke: primi 20 blocchi cronologici di C;
- replay diagnostico: C completo, soltanto dopo il PASS dello smoke.

## Controllo di sola soglia

I parametri frozen e gli score non cambiano. Per ogni riga richiesta si
conservano gli score frozen gia' emessi, l'etichetta quando arriva e il blocco
di arrivo. Una soglia aggiornata a fine blocco puo' influenzare soltanto il
blocco successivo: per le etichette richieste in `j`, il primo effetto e' quindi
in `j+2`. Le predizioni gia' emesse non vengono ricalcolate.

Quando arrivano nuove etichette, per ciascun modello si considerano gli esempi
presenti nella memoria con eta' non superiore a 50 blocchi. La soglia e'
aggiornabile soltanto con almeno 5 normali e 5 attacchi ammissibili. Il limite
di 50 blocchi impedisce che la variante per classe stimi la soglia su normali
arbitrariamente vecchi; il minimo di 5 evita di chiamare informativa una
balanced accuracy definita da pochissimi casi, senza rendere la FIFO
indisponibile per costruzione.

Fra le soglie candidate si massimizza la balanced accuracy. Le candidate sono
i valori distinti degli score piu' un valore sotto il minimo. Il massimo usa
uguaglianza esatta fra `float64`; a pari merito si sceglie la mediana inferiore
dei candidati ordinati, indice `(n-1)//2`. La predizione resta
`score > soglia`.

Se manca una classe, non si raggiunge il minimo o tutti gli esempi utili sono
troppo vecchi, non si aggiorna: resta valida l'ultima soglia ammessa, inizialmente
quella di B. Il motivo e il numero di blocchi senza evidenza sono registrati.
Le metriche sulla memoria scelgono la soglia; il risultato si misura soltanto
sui blocchi successivi.

## Invarianti e confronti

Test sintetici e smoke devono dimostrare:

- score frozen, AUROC per blocco, richieste e digest dei `row_id` invariati;
- stesso budget, ritardo, campione iniziale e limite di memoria;
- assenza di effetti retroattivi modificando etichette future;
- comportamento deterministico per classe assente, eta', minimi e pareggi;
- soglia B finche' non esiste un aggiornamento ammissibile;
- FIFO e per-classe trattate come esperimenti distinti.

Il replay completo produce due controlli di sola soglia. Il riferimento frozen
con soglia B e le teste ADAPT-02 `ogni_blocco` dello stesso seed vengono
riutilizzati solo se coincidono file, hash, inizializzazione, soglie, richieste,
digest dei `row_id` e misure frozen per blocco; altrimenti il confronto si
ferma e la differenza viene documentata.

## Misure e decisione

Per modello e variante si consegnano CM globale, FN, FP, recall, FPR e AUROC,
oltre alle medie per blocco con denominatori. Un peggioramento temporale del
recall e' un blocco con delta recall rispetto a frozen minore di zero; per il
FPR e' un blocco con delta maggiore di zero. Si riportano numero, intervalli
contigui e durata massima, senza sostituire le misure globali.

Si registrano traiettoria e aggiornamenti della soglia, blocchi senza evidenza,
composizione ed eta' della memoria, tempo degli aggiornamenti e stato logico
aggiuntivo in byte: score, etichette, eta', soglie e contatori. Non si deduce
alcuna misura MCU.

Il checkpoint propone il passo successivo senza richiedere superiorita': soglia
sola se offre un compromesso comparabile a costo inferiore; ristima vincolata
dei pesi se resta un vantaggio del riaddestramento; una sola modifica motivata
se scarsita' o eta' impediscono stime affidabili.

# ADAPT-07 — conferma su catture indipendenti

## Domanda e perimetro

La memoria aggiornata di ADAPT-06I recupera attacchi senza aumentare i falsi
allarmi anche su catture nuove? Il confronto primario riguarda le due sessioni
IoT-23 identificate nell’[inventario](INVENTORY.md), con seed 42 e feedback
da etichette del publisher. Si congelano prima gli ID delle catture, il
preflight e le regole del confronto. Il codice del replay, l’ambiente e gli
hash degli input si registrano prima di calcolare le prestazioni.

## Preflight e ordine delle osservazioni

Si ammettono soltanto sessioni con provenienza e licenza documentate,
acquisite separatamente e senza uso precedente individuato nel progetto.
Si verificano schema Zeek, identificatori, dizionario delle etichette,
completezza dei log e coerenza del periodo. Si usa il log completo di ogni
sessione. La mappa delle otto feature è nell’inventario. Si applica il
contratto numerico TON di ADAPT-06I, con valori finiti, non negativi e
rappresentabili in float32. Un campo Zeek non valorizzato resta mancante.
Le righe incomplete sono rendicontate per campo e classe e restano fuori
dalla popolazione con vettore osservabile. Si preserva la prevalenza
delle righe ammissibili. Un’etichetta binaria sconosciuta blocca il
preflight. Un UID ripetuto blocca la cattura finché il publisher o la
provenienza non chiariscono l’identità delle osservazioni. Vettori
uguali con UID diversi sono ricorrenze e si conservano.

Si usa il proxy di disponibilità `ts + duration` con risoluzione di
microsecondi, come nel flusso TON. I pari seguono il numero di riga
originale. Il log riporta l’inizio della connessione e la durata, senza
un timestamp separato di emissione del record. L’ordine è quindi quello
del completamento stimato e il ritardo è espresso in blocchi di tale
flusso. Il proxy non certifica la latenza effettiva di consegna di Zeek.

Si riservano in ordine cronologico il primo 40% dei record ammissibili
ad A, il successivo 20% a B e gli ultimi 40% a D. I due tagli, calcolati
per troncamento intero, si spostano alla fine del gruppo con pari tempo
di disponibilità, per tenere separati i periodi. Entrambe le classi
devono essere osservabili in A, B e nel segmento valutato di D. D deve
contenere più di 200.000 record ammissibili per lasciare osservazioni
dopo il riscaldamento. La scarsità di chiavi ricorrenti non è un criterio
di esclusione. Se manca un requisito essenziale in una cattura, ci si
ferma prima del fit e del replay di entrambe. Il registro conserva il
motivo e il supporto osservato per il riesame della scheda.

## Inizializzazione e tre politiche

In ciascuna cattura A stima log1p e standardizzazione con
`Trasformazione`, oltre alla GAM di ADAPT-06I (`ModelloKAN` additiva,
otto basi, regressione logistica C=1, massimo 400.000 esempi, seed 42).
La procedura storica di sottocampionamento uniforme per il fit si
applica solo in A. Se il campione A non contiene entrambe le classi,
si documenta l’impossibilità di inizializzare. Su B si massimizza
balanced accuracy a λ=1, con spareggio della mediana inferiore delle
soglie ottime come in `review_extension/experiment.py`. Si salva stato,
soglia e provenienza prima di valutare D. A/B sono dati etichettati offline.
Il budget online si applica esclusivamente a D.

Si confrontano Frozen/B, memoria fissata a inizio blocco 20 e memoria
FIT192 aggiornata. Nei primi 20 blocchi da 10.000 record di D si costruisce
FIT partendo da vuoto. La valutazione appaiata va dal blocco 20 fino alla
fine, ultimo blocco parziale incluso. Per blocco di n record si estraggono
`floor(n/100)` richieste senza reinserimento. Si usa PCG64 con
`SeedSequence([42, scenario, blocco, 20261127])`, dove scenario è 1 o 48.
Il primo quarto intero delle richieste ha ruolo CONTROL. Il resto entra
in RANDOM_FIT, nello stesso ordine. Il calendario è comune ai tre
bracci e indipendente da etichette, score, predizioni e ricorrenze.
Il confronto è privo di cicli VERIFY. Le label di k arrivano dopo le
predizioni di k+1 e sono utilizzabili da k+2.

Si riusano FIFO per classe con quote 96/96 e prestito, chiave canonica
`vector_digest` sulle otto feature grezze, e regola di almeno tre UID
distinti da almeno due blocchi. Si cambia decisione solo con etichette
unanimi. In caso di conflitto o supporto insufficiente si applica
Frozen/B con `score > soglia`. Il dizionario fissato resta quello del
primo blocco valutato. Pesi e soglia restano comuni e congelati.

## Esito primario e rendicontazione

Per cattura si misurano ΔTP > 0 e ΔFP ≤ 0 nei totali contro ciascun
controllo, sugli stessi record. Si richiedono anche almeno due episodi
favorevoli non contigui e due chiavi qualificanti secondo il
[protocollo 06I](../adapt06i_recurring_feedback/PROTOCOL.md#decisione-per-lestensione).
Un intervallo massimo di blocchi consecutivi favorevoli conta una volta.
Si riportano separatamente gli esiti delle due sessioni, con il criterio
completo esplicito per ciascuna. Si parla di replicazione nelle due
sessioni soltanto se entrambe lo soddisfano. Sono due acquisizioni
dello stesso laboratorio, quindi la conclusione resta circoscritta
alle sessioni studiate. Con due sessioni e un seed per sessione si
riportano conteggi e traiettorie appaiate, senza inferenza su una
popolazione di catture o aggregazione dei record delle due sessioni.

La consegna del replay, se ammesso, include confusioni e tassi con
denominatori, FP per 10.000 normali, ΔTP/ΔFP per blocco, peggior blocco
per ΔFP e perdita di TP, blocchi fino al primo episodio favorevole,
ricorrenze per chiave, etichette e traffico esclusi, tipi visti in A/B
e nuovi in D, memoria e tempi di fit, calibrazione e replay. FPR e
recall sono vuoti in assenza della relativa classe. Un FPR sopra il
livello Frozen/B è segnalato per blocco e nei totali. Le valutazioni
senza variazioni o sfavorevoli sono parte della consegna. L’input
integrale resta esterno e si versionano manifest, comandi, prodotti
e un archivio riproducibile con hash. Eventuali correzioni di bug
conservano anche gli esiti precedenti.

Il primo confronto usa solo feedback originale, budget 1% e ritardo
di un blocco. Le prove su budget diversi, ritardi e rumore restano
successive alla lettura del primario. I costi preventivi sono indicati
nell’inventario.

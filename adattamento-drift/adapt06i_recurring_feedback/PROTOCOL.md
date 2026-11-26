# ADAPT-06I — feedback passato sui vettori ricorrenti

## Perimetro e sorgenti

Pilot sui flussi di sviluppo TON e UNSW, solo seme 42. Blocchi cronologici da
10.000 righe, compreso l’ultimo blocco parziale. Modelli GAM congelati su A,
soglia B archiviata per λ=1, score e confronto stretto `score > soglia` come
in Frozen/B. Si riusano i due report originali ADAPT-06E del seme 42, prodotti
dal commit `28a81d5de70642d6f76d31a79d865103e4eb5bd2`. Archivio originale
SHA-256 `91c2cf5c3fae62b1b6009714ed4a33560e5077e8cad647d87ad1c1cb9fb30eef`.
Verificare gli hash dei report, degli array in ordine e dei modelli e metadati
A/B. Le scelte di questo protocollo sono fissate prima del pilot.

## Chiave e politica

Per ogni riga usare `adapt05_occurrence.roles.vector_digest` sulle otto
feature in ingresso, prima della trasformazione del modello. È il digest
canonico ADAPT-05 del vettore float64 little-endian e del suo dominio. Vale
la normalizzazione ADAPT-05 di zero e NaN. ID, etichetta, score, timestamp
e ruolo restano fuori dalla chiave. L’identità della riga è il `row_id`
archiviato, univoco entro dataset.

A inizio blocco, una chiave è ammissibile se nella FIT192 ci sono almeno
tre ID distinti, da almeno due blocchi di osservazione precedenti, e tutte
le loro etichette coincidono. In quel caso si usa la classe concorde. Per
una chiave con classi diverse, supporto insufficiente o assenza dalla
memoria si usa la decisione Frozen/B. Si conservano i pesi e la soglia.

## Memoria e tempo

Si parte da FIT vuota e si riproduce il log di richieste ADAPT-06E in
ordine. Verificare che gli ID corrispondano alle righe del blocco, che non
si ripetano e che il totale sia `floor(n/100)` per un blocco di n righe.
CONTROL e VERIFY_EXTRA non entrano in FIT. Solo le richieste RANDOM_FIT
entrano, nell’ordine storico. FIT192 usa la FIFO storica per classe con
quote 96/96 e prestito di capacità. ADAPT-06E non impone un limite d’età
a FIT192. L’età 50 si applica al buffer CAL, escluso da questo pilot.

Una richiesta in k arriva dopo la previsione di k+1 ed è disponibile
da inizio k+2. A fine blocco verificare gli ID FIT contro lo snapshot
ADAPT-06E. Lo snapshot di inizio blocco coincide con quello della fine
del precedente. Usare le etichette di valutazione solo per la misura
offline. Le predizioni non modificano le richieste storiche.

## Tre decisioni appaiate

Blocchi 0–19 per il riscaldamento. Dal blocco 20 fino alla fine confrontare
sugli stessi ID: Frozen/B, stessa regola con il dizionario ammissibile
preso a inizio blocco 20 e mantenuto fisso, e stessa regola con FIT aggiornata
causalmente. Controllare che la confusione Frozen/B coincida per blocco
con il riferimento archiviato. Inizio blocco 20 include al massimo le
richieste fino al blocco 18.

Per ogni blocco e stream registrare supporti, TP/FP/FN/TN per braccio,
delta e ID distinti valutati. Separare le righe con chiave ammissibile
dalle decisioni cambiate e, per i cambi, gli attacchi recuperati, gli
attacchi persi, i falsi allarmi corretti e quelli aggiunti. Report per
chiave per tutte le chiavi usate da almeno un braccio di memoria, con
frequenza totale, copertura e cambi. Nel riepilogo indicare anche la
chiave più frequente dell’intera finestra, la principale per copertura
e la principale per cambi, con denominatori espliciti. Un ID contribuisce
una volta ai conteggi di ciascun confronto.

## Decisione per l’estensione

Per ogni stream, estensione proponibile solo se la memoria aggiornata
ha ΔTP > 0 e ΔFP ≤ 0 contro ciascuno dei due controlli nei totali, e
il beneficio riguarda almeno due episodi temporali non contigui e due
chiavi distinte. Un blocco favorevole rispetta la stessa regola contro
entrambi i controlli. Massimi intervalli consecutivi di blocchi favorevoli
formano gli episodi. Una chiave conta per il criterio se, in un blocco
favorevole, contribuisce con ΔTP > 0 e ΔFP ≤ 0 contro entrambi i controlli
sulle proprie righe. Riportare anche gli episodi favorevoli senza chiavi
qualificanti. I semi condividono una cattura. Questo pilot riguarda
solo il seme 42 e ADAPT-07 rimane in backlog.

## Prodotti

Tabella per blocco, tabella per chiave, tabella per episodio, riepilogo per
stream e decisione. Manifest con fonti, ambiente, comandi e SHA dei prodotti.
Test mirati su ritardo, duplicati di ID, conflitto e verifica dei controlli.

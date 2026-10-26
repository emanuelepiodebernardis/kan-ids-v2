# Proposta minima di validazione finale

## Domanda e disegno

Validare su D, una sola volta e dopo il congelamento del protocollo, il
compromesso osservato per KAN fra quattro tracce: `frozen+B`,
`frozen+CAL64`, `FIT192+B` e `FIT192+CAL64`. Il riferimento `frozen+B` separa
il comportamento del modello iniziale con la soglia esterna fissa dagli
effetti della calibrazione recente e dell'aggiornamento dei pesi.

Usare i semi 42--46, un calendario comune per seme, memoria totale 192+64,
quota di etichette 1%, ritardo `j+2`, ruoli immutabili per vettore e la regola
BA con il tie-break gia' fissato. B non viene riletto durante il replay: le
tracce `frozen+B` e `FIT192+B` usano la soglia fissa su B preregistrata prima
dell'avvio. D non interviene in alcuna scelta.

## Inizializzazione esplicita

Per ogni seme, la trasformazione viene stimata soltanto su A. La KAN viene
inizializzata con quel seme e addestrata da zero su A; nessun peso proveniente
da ADAPT-01--04 viene riusato. Le quattro tracce ricevono copie byte-identiche
dei parametri iniziali. `frozen+B` e `frozen+CAL64` mantengono quei pesi per
tutto D; `FIT192+B` e `FIT192+CAL64` condividono una sola traiettoria di pesi
aggiornata su FIT192. Le soglie B sono calcolate prima dell'apertura di D e
restano fisse nelle due tracce `+B`.

## Due aggregazioni sulle stesse predizioni

D intero, 3.549.973 righe valide, e' l'analisi principale. Il
replay scorre tutto D nell'ordine di `feature_ready_time`; nessuna riga viene
rimossa e D non modifica trasformazione, inizializzazione o criteri.

L'analisi supplementare applica alle stesse predizioni una maschera per i
vettori mai comparsi in A, B o C: 2.228.436 righe, di cui
129.982 normali e 2.098.454 attacchi.
Un vettore e' identificato dallo SHA-256 dei 32 byte degli otto `float32`
normalizzati, come in `sovrapposizioni_abcd.json`. La maschera viene applicata
soltanto in fase di aggregazione: non cambia stream, richieste, memoria o
predizioni.

I conteggi del perimetro provengono dall'inventario versionato
`sovrapposizioni_abcd.json`. Questa proposta non apre D e non contiene
risultati su D: D resta riservato fino all'approvazione del protocollo.

## Controlli preregistrati

1. Registrare commit, comando, ambiente e SHA-256 di A, B e D prima dei run.
2. Richiedere blocchi completi e ordinati, digest identici di righe ed etichette
   fra le quattro tracce e digest iniziali uguali entro ciascun seme.
3. Verificare causalita' `j+2`, separazione FIT/CAL, atomicita' degli update e
   identita' di pesi e score entro ciascuna coppia a pesi comuni.
4. Conservare i valori mancanti nei blocchi monoclasse e riportare per ogni
   metrica il denominatore valido; non sostituirli con zero.
5. Conservare raw, confronto per blocco, manifest, hash e tempi effettivi in un
   archivio esterno immutabile.

## Metriche e criteri

Le metriche primarie sono FN, FP, recall e FPR globali, riportate sia su D
intero sia sul sottoinsieme non visto. Balanced accuracy e AUROC per blocco,
con denominatore, sono secondarie. Riportare differenze appaiate per seme,
dispersione, segno in almeno quattro semi su cinque, peggiori periodi
consecutivi, strati update/skip, aggiornamenti disponibili/applicati e costi.

Una variante puo' essere detta superiore solo se non peggiora ne' FN ne' FP e
migliora almeno uno dei due aggregati, con lo stesso segno in almeno quattro
semi su cinque. Segni opposti fra FN e FP definiscono un compromesso. Un aumento
di balanced accuracy o del solo recall non basta; l'AUROC va interpretata
separatamente dalle soglie. I criteri non saranno modificati dopo l'apertura di
D.

## Costo previsto

ADAPT-04 ha richiesto 190,088 s per 916 blocchi: 35,162 s di addestramento
iniziale, 147,160 s di replay e 7,766 s residui. D intero richiede
355 blocchi da 10.000 righe, con l'ultimo parziale. La stima per cinque
semi e' 499,80 s, circa 8,3 minuti seriali, secondo la
formula `5 * (42,928 + 0,160655 * N)`. Il raw atteso e' circa 7,4 MB e lo ZIP
circa 0,9 MB. Il benchmark ha raggiunto circa 1,25 GiB RSS; la somma
conservativa dei massimi osservati del solo stato logico del controllore e'
29.120 byte, non un picco simultaneo e senza overhead Python.

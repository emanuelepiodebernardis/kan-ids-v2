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
restano fisse nelle due tracce `+B`. La soglia B non viene ereditata da una
corsa precedente: viene ricalcolata sull'intero B separatamente per ciascuno
dei cinque modelli iniziali, dopo l'inizializzazione relativa al suo seme.
Addestramento su A e calibrazione su B terminano per tutti e cinque i semi
prima di calcolare l'impronta o caricare gli array di D.

## Buffer iniziali e assegnazione FIT/CAL

All'inizio di D `fit192` e `calibration64` sono vuoti. Nessun esempio di A, B
o C viene inserito nei buffer online: A serve soltanto per trasformazione e
inizializzazione, B soltanto per le soglie fisse e C soltanto come evidenza
precedente. Le quattro tracce condividono richieste, etichette disponibili,
assegnazioni e stato dei due buffer.

Il ruolo e' una funzione immutabile del vettore. Gli otto valori vengono
serializzati come `float64` little-endian, con `-0.0` normalizzato a `+0.0`,
NaN quiet canonico e segno degli infiniti conservato. Il messaggio contiene il
salt ASCII letterale `ADAPT04-C-seed42-v1`, un byte nullo e i 64 byte del
vettore. SHA-256, letto come intero unsigned big-endian, assegna CAL se il
digest modulo 4 vale zero e FIT altrimenti. Il salt storico viene mantenuto
letteralmente per tutti i semi: il suo nome non introduce C o seed42 nel
replay e vettori identici ricevono lo stesso ruolo in ogni traccia e seme.

`fit192` ha capacita' 192 e quote 96/96; `calibration64` ha capacita' 64 e
quote 32/32. I posti liberi possono essere occupati dall'altra classe; quando
la capacita' e' superata si espelle l'esempio piu' vecchio della classe sopra
quota. Non sono ammessi trasferimenti fra buffer. L'eta' parte dal blocco di
osservazione: il limite `<= 50` vale soltanto per CAL64, mentre FIT192 usa tutto
il contenuto corrente.

## Requisiti e transazione di aggiornamento

Dopo l'arrivo causale delle etichette, il passo e' ammissibile soltanto se
FIT192 contiene entrambe le classi e CAL64, dopo il filtro di eta', contiene
almeno 5 normali e 5 attacchi. Si stima un'unica testa candidata con
`LogisticRegression(C=1.0, max_iter=1000)` sugli otto guadagni e
l'intercetta; la rappresentazione B-spline resta congelata.

Senza cambiare lo stato corrente, CAL64 viene rivalutata con testa frozen e
testa candidata. Per ciascuna si sceglie la soglia che massimizza la balanced
accuracy fra gli score `float64` distinti; i confronti sono esatti e, a parita'
di massimo, si usa la mediana inferiore. Testa candidata, soglia frozen+CAL64
e soglia FIT192+CAL64 entrano in vigore insieme dal blocco successivo soltanto
se tutte sono valide. `FIT192+B` riceve la stessa testa candidata e conserva la
soglia B; `frozen+B` non cambia mai. Se manca un requisito o il fit e' invalido,
tutte le tracce mantengono lo stato precedente e registrano una causa di skip.

## Due aggregazioni sulle stesse predizioni

D intero, 3.549.973 righe valide, e' l'analisi principale. Il
replay scorre tutto D nell'ordine di `feature_ready_time`; nessuna riga viene
rimossa e D non modifica trasformazione, inizializzazione o criteri. Sono
richiesti esattamente 355 blocchi da 10.000 righe, con l'ultimo blocco
parziale da 9.973 righe: non e' ammesso
troncarlo, completarlo artificialmente o riusare il numero di blocchi di C.

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

## Confronti preregistrati

Il confronto principale e' `FIT192+CAL64` meno `frozen+CAL64`: stessa
calibrazione recente e stesso calendario, con testa adattata contro frozen.
`FIT192+CAL64` meno `FIT192+B` isola la soglia a pesi identici;
`frozen+CAL64` meno `frozen+B` isola la calibrazione recente a pesi frozen.
`frozen+B` e' anche il riferimento esterno del modello iniziale. Tutti i delta
sono variante meno controllo e sono calcolati sulle stesse righe.

## Controlli preregistrati

1. Registrare commit, comando, ambiente e SHA-256 di A, B, C e D prima dei run;
   lo SHA-256 atteso di D e' un argomento obbligatorio, non un valore dedotto
   dopo la corsa.
2. Richiedere blocchi completi e ordinati, digest identici di righe ed etichette
   fra le quattro tracce e digest iniziali uguali entro ciascun seme.
3. Verificare causalita' `j+2`, separazione FIT/CAL, atomicita' degli update e
   identita' di pesi e score entro ciascuna coppia a pesi comuni.
4. Conservare i valori mancanti nei blocchi monoclasse e riportare per ogni
   metrica il denominatore valido; non sostituirli con zero.
5. Conservare raw, confronto per blocco, manifest, hash e tempi effettivi in un
   archivio esterno immutabile. Ogni aggiornamento applicato conserva
   separatamente i secondi di fit, calibrazione frozen e calibrazione adattata;
   ogni rendiconto conserva inoltre il tempo complessivo del replay del seme.

## Runner predisposto, senza apertura di D

`replay_validazione_finale.py` implementa il disegno come runner distinto da
quello specifico per C. A addestra trasformazione e modello; B ricalibra la
soglia del singolo modello iniziale; C costruisce soltanto l'inventario delle
impronte A/B/C; D alimenta le quattro tracce e le due aggregazioni. La maschera
non-visto non genera un secondo replay: viene applicata alle stesse predizioni
gia' prodotte per D completo. I cinque modelli e le cinque calibrazioni B sono
preparati prima che il runner calcoli lo SHA-256 o carichi D.

Prima dell'apertura di D il contratto si verifica esclusivamente su dati
sintetici:

```powershell
python -m pytest test_replay_validazione_finale.py -q
```

Soltanto dopo l'approvazione del commit del runner e dei test, il comando
ufficiale richiedera' un worktree pulito e una directory di uscita nuova:

```powershell
python replay_validazione_finale.py `
  --iniziale flussi/A.npz --calibrazione flussi/B.npz `
  --sviluppo flussi/C.npz --valutazione flussi/D.npz `
  --sha-d-atteso <sha256-D-approvato> `
  --uscita <nuova-directory-vuota>
```

La preparazione del runner e delle controprove non esegue questo comando e non
apre D.

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

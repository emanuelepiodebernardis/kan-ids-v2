# Protocollo proposto ADAPT-06

Versione di progettazione: 5 novembre 2026, revisione del primo checkpoint. Base Git:
`0cd94dee4ba66ddf64be761acbe0ada93757e40b`. Il protocollo precede qualsiasi
nuovo fit o replay.

## 1. Domanda e ipotesi primaria

La domanda è se convenga promuovere un aggiornamento soltanto dopo una verifica
prospettica, quando l'acquisizione FIT usa esempi informativi ma conserva una
quota di esplorazione, invece di aggiornare sempre con le stesse label.

L'unica ipotesi primaria è:

> A budget, ritardo e memorie FIT/CAL uguali, `ACTIVE+GATED` sulla GAM spline
> aumenta il recall di almeno 2 punti percentuali rispetto ad
> `ACTIVE+ALWAYS`, senza uscire dalla stessa regione di FPR effettivamente
> raggiunto.

La regione primaria ha target FPR 5%. Due punti sono comparabili quando la
differenza assoluta di FPR è al massimo 1 punto percentuale; una traiettoria
promossa deve inoltre avere FPR non superiore al 6%. Il risparmio del 25% di
label a recall/FPR equivalenti è secondario e, col budget iniziale fisso, viene
marcato `NOT_ESTIMATED`; appartiene alla validazione successiva.

Un seed ha supporto se la coppia completa passa provenienza e invarianti e ha
denominatori globali normali/attacchi definiti. L'assenza di promozioni e i
withholding sono esiti validi, non missing. Per dataset, la media di recall usa
tutti i seed supportati, inclusi quelli che falliscono l'FPR. Un seed è
`FPR_PASS` se la differenza globale è al massimo 1 pp e `ACTIVE+GATED` non
supera il 6%; altrimenti è un esito sfavorevole visibile.

Il dataset è `GO` con almeno quattro seed supportati, almeno quattro
`FPR_PASS` sui cinque pianificati e differenza media appaiata di recall almeno
+2 pp. Con almeno quattro seed supportati ma un criterio fallito è `NO-GO`; con
meno di quattro è `INCONCLUSIVE`. L'esito complessivo è `GO` soltanto se
entrambi i dataset sono `GO`, `NO-GO` se almeno uno è `NO-GO`, altrimenti
`INCONCLUSIVE`.

## 2. Perimetro comune

- Stream di sviluppo: TON e UNSW già ricostruiti e verificati nel Paper 2.
- Seed modello/query: 42–46; seed 42 soltanto nel futuro pilot.
- Blocchi: 10.000 righe, con ultimo blocco parziale.
- Budget: `floor(0,01 * n_blocco)` richieste senza rimpiazzo.
- Ritardo: richiesta in `k`, label dopo la predizione di `k+1`, primo impatto
  possibile in `k+2`.
- Etichette: corrette e prive di rumore nel primo sviluppo. Budget diversi,
  ritardi diversi e rumore appartengono alla validazione successiva.
- Inizializzazione: stati A e soglie B archiviati; nessuno stato adattivo viene
  trasferito da una campagna precedente.
- Famiglia primaria: GAM logistica additiva a spline. LR è un controllo di
  attribuzione; ARF correttamente inizializzata è una baseline forte.
- Feature, trasformazioni e stati iniziali coincidono entro famiglia.
- Le metriche complete dello stream sono accessibili soltanto al valutatore
  offline e non influenzano query, soglie o promozioni.

FIT192 usa quote 96/96 senza età; CAL256 quote 128/128 e age 50. Le regole FIFO
sono in `TEMPORAL_SCHEMA.md`. Servono almeno 32 esempi per classe in FIT e 100
in CAL. GAM/LR creano dalla rappresentazione iniziale un candidato condiviso,
senza warm start. ARF parte da A/B, aggiorna i pesi soltanto con FIT e conserva
la soglia B; il riuso di CONTROL sarebbe un confronto separato.

## 3. Contratto delle label

Per un blocco di `n` righe si definiscono:

```text
Q = floor(0.01 * n)
CONTROL = floor(Q / 4)
FIT = Q - CONTROL
EXPLORE = floor(FIT / 5)       # soltanto nel braccio ACTIVE
EXPLOIT = FIT - EXPLORE
```

Il riparto è quindi nominalmente 25% CONTROL e 75% FIT; gli scarti sull'ultimo
blocco derivano soltanto dagli arrotondamenti interi dichiarati sopra.

CONTROL è uniforme, precede FIT e ha `observation_id` comuni a tutti i
confronti. RANDOM estrae FIT uniformemente dalle righe rimaste; ACTIVE usa
EXPLORE uniforme ed EXPLOIT causale sul solo blocco osservato.

Sono eleggibili le righe valide del blocco corrente. CONTROL viene estratto per
primo; tutte le selezioni sono senza rimpiazzo. ADAPT-06 non introduce un
fattore `role_salt`: i flussi RNG indipendenti e indicizzati sono fissati in
`TEMPORAL_SCHEMA.md`, senza ricerca di un sale favorevole.

ACTIVE usa un selettore ombra comune ad `ALWAYS/GATED`, inizializzato A/B e
aggiornato al freeze, con incertezza e farthest-first causali. Per ARF coincide
con l'unico stato operativo FIT-only. I dettagli sono in
`TEMPORAL_SCHEMA.md`.

CONTROL, FIT e seconde annotazioni contano nel budget; queste ultime non sono
previste. Si registrano hash, blocchi di osservazione/arrivo/primo impatto,
probability uniformi e rango ACTIVE; propensity non definite restano tali.

## 4. Calibrazione e verifica prospettica

FIT192 addestra il candidato e CAL256 ne sceglie la soglia al target 5%.
All'apertura di VERIFY si congelano anche modello e soglia realmente operativi
di `GATED`: sono il riferimento primario perché costituiscono lo stato che
verrebbe sostituito. La soglia dello stesso incumbent ricalibrata su CAL resta
un confronto diagnostico e non decide il gate. Modelli, soglie, digest e freeze
vengono fissati insieme.

Soltanto CONTROL osservati dopo il freeze alimentano VERIFY e non possono
addestrare o calibrare il candidato. CONTROL precedenti ma arrivati dopo sono
differiti. Dopo la decisione entrambi i gruppi possono entrare in CAL per il
candidato successivo, mai retroattivamente.

Candidato e calendario sono comuni ad `ALWAYS` e `GATED`. `ALWAYS` promuove
modello e soglia dal blocco successivo al freeze; `GATED` conserva lo snapshot
operativo fino alla decisione. CAL è congelata e non si crea un secondo
candidato durante VERIFY.

VERIFY appaia candidato e incumbent sugli stessi CONTROL. I look sono 100/100,
200/200 e 400/400 normali/attacchi; la scadenza è 50 blocchi. Il target 1%
richiede 500 normali in CAL e VERIFY e, incompatibile con CAL256, viene marcato
`INSUFFICIENT`; 1% e 10% restano secondari. Questi minimi fissano risoluzione
empirica, non intervalli di confidenza o garanzie sotto drift arbitrario.
Insufficienza, violazioni e tempo alla decisione sono sempre riportati.

Regola al primo punto di controllo raggiunto, sempre contro lo snapshot
operativo primario:

- `ACCEPT`: recall candidato − incumbent almeno +2 pp, FPR candidato −
  incumbent non superiore a +1 pp e FPR candidato non superiore al 6%;
- `REJECT_HARM`: FPR candidato oltre il 6%, aumento FPR oltre +1 pp oppure
  differenza di recall non superiore a −2 pp;
- `WAIT`: supporto insufficiente o differenze fra −2 e +2 pp senza violazioni
  FPR. Si passa al punto successivo senza cambiare il candidato;
- `WITHHOLD_INSUFFICIENT` o `WITHHOLD_AMBIGUOUS`: scadenza a 50 blocchi senza
  decisione. L'incumbent resta operativo.

Un `WAIT` a 400/400 chiude come `WITHHOLD_AMBIGUOUS`. Scadenza e fine stream
applicano prima l'ultimo look possibile, poi distinguono insufficienza (nessun
look) e ambiguità. Non si forzano label future: GATED resta invariato e il log
conserva supporto, digest e coda.

È ammesso un solo candidato pendente. Promozione e nuova eligibility possono
influire soltanto da blocchi successivi. Non si usa un drift oracle: un
peggioramento non azzera memorie o finestre. Un futuro detector sarebbe
un'ablazione separata e invaliderebbe, senza ritagliarla, la verifica pendente.

## 5. Confronti e reporting

L'ablazione minima è `RANDOM/ACTIVE × ALWAYS/GATED` per GAM e LR. ARF FIT-only
riceve RANDOM o ACTIVE e aggiorna sempre; un eventuale riuso di CONTROL per
apprendere è un confronto separato, non la baseline causale. I riferimenti sono
Frozen/B e la migliore threshold-only selezionata sullo sviluppo con la regola
deterministica di `TEMPORAL_SCHEMA.md`. Se nessuna variante threshold-only è
eleggibile, Frozen/B resta il riferimento; ADAPT-07 non interviene nella
scelta.

Frozen/B non si aggiorna. Frozen/CAL256-target5 mantiene i pesi A, riceve gli
stessi CONTROL con delay 1 e aggiorna la propria CAL256 a ogni fine blocco. Se
ha almeno 100 esempi per classe ricalibra la soglia, efficace dal blocco
successivo; altrimenti conserva la precedente, inizialmente B. Un ricalcolo
finale senza predizioni successive è registrato ma non contato come effetto.

Per seed si riportano label/coda, memorie, candidati, decisioni, aggiornamenti,
tempi e memoria; FP, FN, recall, FPR, balanced accuracy, recupero, transitorio,
segmenti e peggior caso. AUROC resta diagnostica. Manifest e log conservano
commit, ambiente, configurazione e hash.

L'endpoint primario usa confusioni aggregate sull'intero stream, non medie dei
tassi di blocco. La media include ogni coppia supportata; il campo `FPR_PASS`
mantiene visibili le repliche sfavorevoli. Le medie per blocco sono diagnostiche
e vengono etichettate separatamente.

Segmenti cronologici, stabilità, transitorio, recupero e worst case sono
prefissati in `TEMPORAL_SCHEMA.md`, senza scegliere intervalli dalle label.

Le differenze sono appaiate per dataset/seed; i seed non sono dataset
indipendenti. Non si usa interpolazione retrospettiva. Si conservano tutte le
repliche; TON e UNSW sono solo sviluppo, separati dalla futura conferma.

Il confronto ha equal-label-budget e memorie FIT/CAL identiche, ma non viene
descritto come equal-resource: `GATED` mantiene shadow e contatori VERIFY che
`ALWAYS` non richiederebbe in deployment. Questi byte e tempi sono misurati
separatamente.

## 6. Arresto e passaggio successivo

Questo documento non autorizza esecuzioni. Dopo l'approvazione si implementano
prima invarianti sintetiche su causalità, disgiunzione, budget, congelamento,
decisioni e ultimo blocco parziale; soltanto dopo un PASS si esegue il pilot
seed 42. Il pilot determina tempi reali, non modifica ipotesi o soglie. La
matrice completa richiede una decisione esplicita successiva. Non si avvia un
nuovo manoscritto prima del confronto completo.

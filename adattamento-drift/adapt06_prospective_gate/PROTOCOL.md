# Protocollo proposto ADAPT-06

Versione di progettazione: 4 novembre 2026. Base Git:
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

Il risultato primario è `GO` soltanto se, separatamente su TON e UNSW, almeno
quattro seed su cinque hanno supporto sufficiente, FPR comparabile e differenza
media appaiata di recall almeno +2 punti. Se il supporto è sufficiente ma il
criterio non è raggiunto, l'esito è `NO-GO`. Se meno di quattro seed sono
valutabili, l'esito è `INCONCLUSIVE`: non viene scelto un sottoinsieme
favorevole.

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

FIT192 è una FIFO per classe con quote 96/96 e senza limite di età; CAL256 usa
quote 128/128 ed età massima di 50 blocchi dal blocco di osservazione. Quote,
prestiti, espulsioni e reinserimento post-VERIFY sono normati in
`TEMPORAL_SCHEMA.md`. Un candidato richiede almeno 32 esempi per classe in FIT;
la soglia richiede almeno 100 per classe in CAL, altrimenti non viene creata.

Per GAM e LR il fit candidato è una funzione deterministica della
rappresentazione iniziale fissa, dello snapshot FIT192, del seed e degli
iperparametri congelati, senza warm start operativo. Lo stesso candidato è
quindi condiviso da `ALWAYS` e `GATED`. ARF resta FIT-only, sempre adattiva e
fuori dal gate primario: parte dallo stato A/B, aggiorna i pesi soltanto con
FIT arrivate e conserva la soglia B. Un ARF che ricalibri la soglia con CONTROL
sarebbe un confronto separato.

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

CONTROL è estratto uniformemente fra tutte le righe del blocco, prima di
qualsiasi selezione FIT, senza usare score o label. Gli `observation_id`
CONTROL sono comuni a modelli e politiche. Nel braccio RANDOM, FIT è uniforme
fra le righe rimaste. Nel braccio ACTIVE, EXPLORE è uniforme e EXPLOIT è
selezionato causalmente dal solo blocco già osservato.

Sono eleggibili le righe valide del blocco corrente. CONTROL viene estratto per
primo; tutte le selezioni sono senza rimpiazzo. ADAPT-06 non introduce un
fattore `role_salt`: i flussi RNG indipendenti e indicizzati sono fissati in
`TEMPORAL_SCHEMA.md`, senza ricerca di un sale favorevole.

Per mantenere appaiato `ALWAYS` contro `GATED`, ciascuna famiglia ACTIVE usa un
unico selettore ombra condiviso fra le due politiche. Il selettore viene
inizializzato con A/B e, per GAM/LR, riceve modello e soglia del candidato a
ogni freeze, indipendentemente dalla successiva decisione. Non produce le
predizioni valutate. La selezione combina incertezza e farthest-first secondo
`TEMPORAL_SCHEMA.md`, senza consultare label future. Per ARF ACTIVE il
selettore è l'unica ARF operativa aggiornata con FIT e usa la soglia B.

CONTROL, FIT e seconde annotazioni contano nel budget; queste ultime non sono
previste nel primario. Si registrano hash degli ID, blocchi di osservazione,
arrivo e primo impatto, probability per i campioni uniformi e rango/criterio
ACTIVE. Una propensity non definita resta tale.

## 4. Calibrazione e verifica prospettica

FIT192 addestra il candidato; CAL256 sceglie separatamente le soglie di
candidato e incumbent al target 5%. Modello, soglie, digest e freeze vengono
fissati insieme. Soltanto CONTROL osservati dopo il freeze alimentano VERIFY e
non possono addestrare o calibrare il candidato. CONTROL precedenti ma arrivati
dopo il freeze sono differiti. Dopo la decisione entrambi i gruppi possono
entrare in CAL per il candidato successivo, mai retroattivamente.

Candidato e calendario sono comuni ad `ALWAYS` e `GATED`. `ALWAYS` promuove
modello e soglia dal blocco successivo al freeze; `GATED` conserva lo stato
operativo fino alla decisione. La soglia ricalibrata dell'incumbent resta
shadow. CAL è congelata e non si crea un secondo candidato durante VERIFY.

VERIFY appaia candidato e incumbent sugli stessi CONTROL. I look sono 100/100,
200/200 e 400/400 normali/attacchi; la scadenza è 50 blocchi. Il target 1%
richiede 500 normali in CAL e VERIFY e, incompatibile con CAL256, viene marcato
`INSUFFICIENT`; 1% e 10% restano secondari. Questi minimi fissano risoluzione
empirica, non intervalli di confidenza o garanzie sotto drift arbitrario.
Insufficienza, violazioni e tempo alla decisione sono sempre riportati.

Regola al primo punto di controllo raggiunto:

- `ACCEPT`: recall candidato − incumbent almeno +2 pp, FPR candidato −
  incumbent non superiore a +1 pp e FPR candidato non superiore al 6%;
- `REJECT_HARM`: FPR candidato oltre il 6%, aumento FPR oltre +1 pp oppure
  differenza di recall non superiore a −2 pp;
- `WAIT`: supporto insufficiente o differenze fra −2 e +2 pp senza violazioni
  FPR. Si passa al punto successivo senza cambiare il candidato;
- `WITHHOLD_INSUFFICIENT` o `WITHHOLD_AMBIGUOUS`: scadenza a 50 blocchi senza
  decisione. L'incumbent resta operativo.

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

Per seed si riportano richieste e coda, propensities definite, memorie,
candidati, decisioni, aggiornamenti, tempi, stato persistente e memoria totale.
Le metriche includono FP, FN, recall, FPR raggiunto, balanced accuracy,
recupero, area transitoria, segmenti e peggior caso. AUROC resta diagnostica e
non sostituisce il confronto all'operating point. Manifest e log includono
commit, ambiente, configurazione e hash di input, stati e richieste.

L'endpoint primario usa confusioni aggregate sull'intero stream, non medie dei
tassi di blocco. Una coppia supportata è comparabile se la differenza di FPR
globale è al massimo 1 pp e `ACTIVE+GATED` non supera il 6%; tutte le coppie,
incluse quelle non comparabili, restano visibili. Le medie per blocco sono
diagnostiche e vengono etichettate separatamente.

I segmenti cronologici sono i tre terzi prefissati dello stream. I segmenti
stabili, le finestre transitorie di dieci blocchi, il recupero, le aree FP/FN e
il peggior caso sono definiti in `TEMPORAL_SCHEMA.md` soltanto da indice e stato
operativo, senza cercare drift point o intervalli favorevoli nelle label.

Le differenze sono appaiate entro dataset e seed. I cinque seed descrivono
randomizzazione condizionata agli stessi stream, non cinque dataset
indipendenti. Nessuna interpolazione retrospettiva crea un operating point o
una politica di deployment. Tutte le repliche e gli esiti sfavorevoli vengono
conservati. TON e UNSW sono usati esclusivamente come sviluppo; la
configurazione non viene scelta su una futura conferma indipendente.

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

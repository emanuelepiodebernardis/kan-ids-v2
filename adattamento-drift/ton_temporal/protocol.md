# Protocollo sperimentale — adattamento con etichette scarse e ritardate

**Dati:** TON_IoT, parte «Network», 23 file, 22.339.021 righe, 3.372.475.448 byte.
**Data:** 29 settembre 2026. **Versione:** 1.

Ogni numero di questo documento è misurato sui 23 file completi, non stimato.
Gli identificatori per ritrovarli sono in `split_manifest.json`, sezione
*provenienza*.

---

## 1. Domanda di ricerca

Stabilire **quando** piccoli aggiornamenti permettono a un rilevatore di
intrusioni di riconoscere meglio gli attacchi dopo un cambiamento del traffico,
disponendo di poche etichette e ricevendole in ritardo.

È una domanda distinta dalle due linee vicine, e la distinzione va tenuta nei
risultati: una studia la scelta dei modelli e le conseguenze degli errori,
l'altra la coerenza della pipeline installata. Qui si studia **come e quando
apprendere dai dati nuovi**.

## 2. Che cosa i dati sono davvero

Tre fatti misurati cambiano il disegno rispetto a quanto la sola dimensione dei
file lascerebbe supporre.

**I giorni con dati sono dieci, non ventotto.** La copertura va dal 2 al 29
aprile 2019, ma **dal 5 al 22 aprile non c'è alcuna riga**. I tre giorni
iniziali (2, 3, 4 aprile) contengono in tutto 200.723 righe e **nessun
attacco**; il 95% dei dati sta fra il 23 e il 27 aprile.

**I tipi di attacco arrivano a scala e non ricorrono.** Ogni tipo compare in al
massimo due giorni consecutivi e poi scompare:

| tipo | righe | giorni |
|---|---:|---|
| scanning | 7.140.161 | 23–24 apr |
| dos | 3.375.328 | 24–25 apr |
| ddos | 6.165.008 | 25–26 apr |
| injection | 452.659 | 25 apr |
| password | 1.718.568 | 26–27 apr |
| xss | 2.108.944 | 27 apr |
| backdoor | 508.116 | 28–29 apr |
| ransomware | 72.805 | 28 apr |
| mitm | 1.052 | 29 apr |

Il cambiamento da studiare è quindi **la comparsa di classi nuove**, non una
deriva delle covariate entro tipi che ritornano. Sette tipi su nove non
compaiono mai nell'ultimo giorno osservato.

**La prevalenza è rovesciata rispetto a un impiego reale.** Fra il 23 e il 27
aprile gli attacchi sono il 97–99% delle righe. Il traffico normale è la classe
**di minoranza**, fra l'1,6% e l'8,7% a seconda dell'intervallo. La scheda
impone di conservare la prevalenza osservata senza riequilibri: si conserva, ma
le misure vanno lette di conseguenza (§6).

## 3. L'orologio: `feature_ready_time`

Le otto feature comprendono `duration`, che è nota **solo a flusso terminato**.
Un vettore di feature non può quindi esistere prima della fine del flusso. Si
adotta

```
t_start            = ts
t_end              = ts + duration
feature_ready_time = t_end
```

Questa non è una sottigliezza. Misurato sui 23 file:

- ordinando per `ts` le coppie consecutive fuori ordine sono **21** su 22,3
  milioni: la sorgente è pressoché monotona nell'istante di inizio;
- ordinando per `feature_ready_time` diventano **8.703.109**, cioè il 39% delle
  righe;
- il flusso più lungo dura **93.517 secondi**, poco meno di 26 ore.

Conseguenze operative, entrambe vincolanti:

1. I blocchi da 10.000 **non** si formano leggendo i file in sequenza: le righe
   vanno riordinate per `feature_ready_time`.
2. Gli intervalli si definiscono **sul `feature_ready_time`**, non sul `ts`.
   Assegnare per `ts` e ordinare per `feature_ready_time` renderebbe
   disponibile dentro un intervallo una riga assegnata al precedente, che è
   proprio la violazione di causalità che il protocollo vuole escludere. Alla
   granularità del giorno la differenza è piccola — **3.064 righe** su 22,3
   milioni cambiano giorno — ma la regola resta quella giusta.

`row_id`, `t_start`, `t_end` e `feature_ready_time` sono tenuti **separati
dalle feature**, così non possono entrare nel modello.

**Preprocessing stimato solo su A.** Le otto feature sono conteggi di byte e di
pacchetti con code lunghe: si applica `log1p` e poi una standardizzazione.
Media e deviazione sono stimate **una sola volta su A** e non vengono più
toccate; B, C e D si trasformano con quei parametri. Nessuna statistica dei
flussi successivi entra nella trasformazione, e in particolare **D non
contribuisce in alcun modo** al preprocessing.

## 4. I quattro intervalli

Confini scelti sulla disponibilità dei dati e sulla scala dei tipi, non su una
proprietà favorevole. Sono giorni UTC pieni, semiaperti, e **partizionano tutte
le 22.338.152 righe valide** senza sovrapposizioni né residui.

| | ruolo | giorni UTC | righe valide | normali | esposte | tipi nuovi |
|---|---|---|---:|---:|---:|---|
| **A** | addestramento iniziale | 2–4 e 23 apr | 3.956.308 | 5,64% | 83,6% | scanning |
| **B** | calibrazione | 24 apr | 5.681.198 | 1,57% | 83,7% | dos |
| **C** | sviluppo | 25–26 apr | 9.150.673 | 1,89% | 35,0% | ddos, injection, password |
| **D** | valutazione finale | 27–29 apr | 3.549.973 | 8,72% | 25,0% | backdoor, mitm, ransomware, xss |

**Perché A comprende i tre giorni di aprile iniziali.** Sono l'unico traffico
privo di attacchi disponibile, e senza di essi A avrebbe una quota di normali
ancora più bassa. Il 23 aprile è il primo giorno con attacchi e porta il primo
tipo, `scanning`: A è quindi il minimo periodo su cui un modello iniziale possa
essere addestrato su entrambe le classi.

**Perché D comprende anche il 28 e il 29 aprile.** Il solo 27 aprile avrebbe
un'esposizione del 7,3% invece del 25,0%, cioè un D quasi del tutto non visto.
Non è stato scelto: escludere il 28 e il 29 significherebbe rinunciare a
`backdoor`, `ransomware` e `mitm`, e soprattutto sceglierebbe il confine per
ottenere una proprietà favorevole, che la scheda vieta espressamente. Il numero
peggiore viene dichiarato, non evitato.

**Riaddestramento.** I modelli iniziali si riaddestrano su A. I pesi del lavoro
precedente non si riusano, perché possono aver visto dati successivi: lo si
vede dalla misura di esposizione, che su A vale l'83,6%.

## 5. Controllo delle sovrapposizioni degli otto input con il CSV noto

La misura si descrive per quello che è: un **controllo delle sovrapposizioni
degli otto input con il CSV noto**. Una riga valida è «sovrapposta» quando
l'impronta dei suoi otto valori coincide con quella di una riga del campione
`train_test_network.csv`. Complessivamente **il 54,42%** delle righe valide è
sovrapposto, con un andamento molto disuguale fra i file: dal 94,3% del nono al
4,3% del ventunesimo.

**Questo controllo da solo non dimostra l'indipendenza dello stream D.** Dice
che quei vettori di input non erano già comparsi nel campione noto; non dice
che le righe di D siano indipendenti da quelle di A, B e C, né che appartengano
a sessioni, host o campagne distinte. Il confronto diretto fra i quattro
intervalli è ora misurato, e il suo esito è al §5bis: **negativo**.

La misura è ottenuta ricostruendo l'archivio delle impronte, che non è
pubblicato. Il riscontro è però esatto: il codice di riferimento dichiara che
l'archivio originale contiene **92.330 impronte uniche ricavate da 211.043
righe**, e la ricostruzione — partendo dal campione il cui SHA-256 è quello
atteso — ne produce **92.330 uniche da 211.043 righe**.

Limite da riportare ovunque compaia questo numero: è una ricostruzione
dichiarata, non l'artefatto originale, e il controllo di integrità che lo
strumento di valutazione esterna esegue sul proprio `.npz` non è riproducibile.

L'esposizione **non** misura indipendenza: dice che quei vettori di input erano
già stati visti, non che le righe siano statisticamente indipendenti.

## 5bis. Sovrapposizioni dirette fra A, B, C e D

Il §5 confronta gli otto input con un campione **esterno**. Questa sezione
risponde alla domanda diversa, e necessaria prima di usare D: quanto i quattro
intervalli si somigliano **fra loro**. La misura è prodotta da
`sovrapposizioni_abcd.py`, con le stesse `normalize_raw` e `fingerprint_rows`
del §5, così i due numeri sono confrontabili; l'esito completo è in
`sovrapposizioni_abcd.json`.

**Le righe sono disgiunte, e non per assunzione.** Ogni riga valida cade in un
solo intervallo, i conteggi per intervallo coincidono con quelli dichiarati nel
manifest — righe e normali — la loro somma fa le 22.338.152 righe valide, e
nessuna riga resta fuori. Lo script si ferma se una di queste cose non torna.

**I vettori di feature no.** Le otto feature sono aggregati di flusso, e lo
stesso vettore ricompare continuamente, con intensità molto diversa fra gli
intervalli:

| | ruolo | righe valide | impronte distinte | righe per impronta |
|---|---|---:|---:|---:|
| A | addestramento iniziale | 3.956.308 | 249.930 | 15,8 |
| B | calibrazione | 5.681.198 | 405.643 | 14,0 |
| C | sviluppo | 9.150.673 | 3.849.499 | 2,4 |
| D | valutazione finale | 3.549.973 | 1.972.598 | 1,8 |

A e B sono quasi interamente ripetizione: in A un vettore vale in media quasi
sedici righe. C e D sono molto più vari. Questo da solo spiega perché poche
impronte comuni coprano tante righe.

| coppia | impronte comuni | righe del primo | righe del secondo |
|---|---:|---|---|
| A-B | 50.882 | 3.688.555 (93,23%) | 3.826.246 (67,35%) |
| A-C | 2.662 | 3.065.275 (77,48%) | 1.278.598 (13,97%) |
| A-D | 10.309 | 3.070.375 (77,61%) | 752.213 (21,19%) |
| B-C | 22.774 | 4.413.652 (77,69%) | 2.048.122 (22,38%) |
| B-D | 13.045 | 3.853.601 (67,83%) | 723.418 (20,38%) |
| C-D | 157.720 | 2.280.019 (24,92%) | 903.039 (25,44%) |

**L'esito che conta.** Delle 1.972.598 impronte distinte di D, **170.383
(8,64%) compaiono già in A, B o C**, e coprono **1.321.537 righe, il 37,23% di
D**. La parte che pesa è la classe normale: quelle righe comprendono **179.691
normali, il 58,03% dei normali di D**, più 1.141.846 attacchi.

**Non è l'effetto di un vettore degenerato.** L'impronta più frequente della
sovrapposizione vale 49.390 righe, il 3,74%; le prime dieci il 13,97%; servono
235 impronte per metà delle righe e 54.821 per il 90%. C'è una testa molto
frequente, ma la coda è larga: la sovrapposizione è una proprietà dello spazio
di feature, non di una manciata di righe da scartare.

**Non è nemmeno solo il tipo condiviso.** `password` è l'unico tipo presente in
C e in D, e in D conta 549.383 righe: anche se fossero tutte sovrapposte,
resterebbero almeno 1.141.846 − 549.383 = **592.463 righe di attacco
sovrapposte** di tipi che in D compaiono per la prima volta, cioè il 16,7% di
D.

**Che cosa ne segue, e che cosa resta indecidibile.** D non è indipendente da A,
B e C al livello del vettore di feature: un modello può indovinare più di un
terzo delle righe di D, e più della metà dei suoi normali, su vettori già
incontrati. Non ne segue che le righe siano la stessa sessione, lo stesso host
o la stessa campagna: quella distinzione resta non stabilita, qui come al §5, e
non è ricavabile dalle otto feature.

La conseguenza operativa è fissata al §11, punto 5, come l'ha decisa il
relatore: **D intero è l'analisi principale**, e il sottoinsieme a vettore non
visto — 2.228.436 righe, il 62,77% di D, con 129.982 normali, cioè una
prevalenza di normali del 5,83% invece dell'8,72% — è un'**analisi
supplementare**, calcolata come maschera sulle stesse predizioni del replay
completo, senza togliere righe dallo stream. Il supplemento è il confronto che
guarda alla generalizzazione; il principale resta quello che la scheda
definisce.

## 6. Replay

Per ogni blocco di 10.000 righe in ordine di `feature_ready_time`:

1. il modello **prevede**;
2. si **richiede** l'etichetta per l'1% delle righe — 100 per blocco completo,
   `floor(0,01·n)` per l'ultimo — con gli **stessi `row_id` per tutti i
   metodi**, così il confronto è sugli stessi esempi;
3. si **attende**: le etichette del blocco *k* arrivano alla fine del blocco
   *k+1*, quindi un aggiornamento incide per la prima volta sul blocco *k+2*;
4. il modello si **aggiorna** — coefficienti per la regressione logistica,
   ultimo strato per l'MLP, guadagni e intercetta per il modello additivo.

**I tre modelli, con precisione.** La regressione logistica agisce sulle otto
feature. Dell'MLP si aggiorna il solo ultimo strato, sulle attivazioni nascoste
del modello congelato. Il terzo è una **KAN a singolo strato con edge a
B-spline**: il punteggio è una somma di funzioni univariate cubiche, una per
ciascuna delle otto feature, più un guadagno per feature e un'intercetta.

Va detto con precisione che cosa è e che cosa non è, perché una versione
precedente di questo documento lo descriveva come «non il componente a B-spline
del Paper 1», e quella formulazione era fuorviante. La **forma funzionale è la
stessa** di `BSplineKANBinary` in `src/kan_bspline.py`, che è a sua volta a
singolo strato: in entrambi il punteggio è `z(x) = Σ_j Σ_k coef_jk · N_k(x_j)`.
Non è un'architettura diversa. Le differenze stanno altrove, e sono tre:

| | `BSplineKANBinary` (Paper 1) | il nostro |
|---|---|---|
| basi per feature | 11, nodi fissi su [−3,5; 3,5] | 8, nodi uniformi sul min-max osservato in A |
| coefficienti delle spline | 88, tutti addestrabili per discesa del gradiente | 64, stimati una volta su un sottocampione di 400.000 righe di A e poi **congelati** |
| parametri che l'adattamento muove | non previsto | **otto guadagni e l'intercetta**, nove in tutto, che in quel componente non esistono |

È quindi **reimplementato** in `ton_temporal/replay.py`, non importato, e la
differenza che conta per questo studio non è l'architettura ma il regime di
stima: dopo A le forme delle spline non cambiano più.

Resta vero, e va detto perché il Paper 1 ha anche una linea su quello, che
**non è una KAN a più strati**: né il nostro né `BSplineKANBinary` lo sono,
quindi il confronto con i risultati multi-strato di quella linea non si pone.
Nel resto del documento è chiamato «l'additivo», che è un nome breve per questa
cosa, non una famiglia diversa.

Memoria FIFO di 256 esempi, inizializzata da A. Se la memoria contiene una sola
classe l'aggiornamento si salta e il salto si registra.

Ogni modello aggiornato si confronta con una **copia congelata** della stessa
architettura, sugli stessi esempi. Le predizioni già emesse **non si
ricalcolano mai**. Valutazione progressiva, misure per blocco, `NA` dove una
misura non è definita. Seme 42. Solo CPU in questa fase.

**Quanti blocchi, e quanto chiede il blocco finale.** Ogni intervallo finisce
con un blocco incompleto, e il budget su quel blocco è `floor(0,01·n)`, non
100.

| | righe valide | blocchi completi | righe del blocco finale | etichette chieste sul finale |
|---|---:|---:|---:|---:|
| A | 3.956.308 | 395 | 6.308 | 63 |
| B | 5.681.198 | 568 | 1.198 | 11 |
| C | 9.150.673 | 915 | 673 | 6 |
| D | 3.549.973 | 354 | 9.973 | 99 |

I quattro numeri della colonna centrale vengono da `split_manifest.json` e si
ricavano come `righe_valide mod 10.000`. La versione precedente di questa
tabella era una riga di testo con tre numeri sbagliati — B 1.199, C 677, D
9.968 — e nessuno dei tre cambiava un risultato, ma erano i numeri che servono
a ricalcolare la quota di etichette, quindi vanno giusti; adesso un test li
riconta dal manifest.

**Una conseguenza della prevalenza, misurata e non stimata.** Una prima stima
analitica dava la regola del salto attiva sullo 0,7% dei blocchi di C. **È
sbagliata, e il replay eseguito su tutto C lo mostra: i blocchi saltati sono
393 su 915, il 43,0%.**

L'errore stava nell'usare la prevalenza aggregata. In C i normali sono l'1,89%
delle righe, ma **non sono distribuiti uniformemente**: per blocco la media è
di 189 normali e la **mediana 24**, il 28,6% dei blocchi ne ha meno di dieci, e
il 5% dei blocchi più ricchi contiene il 47,3% di tutti i normali. Con l'1% di
budget, un blocco mediano offre in media 0,24 normali fra i 100 campionati:
quasi mai almeno uno. La memoria di 256 copre circa due blocchi e mezzo di
campioni, quindi resta di una sola classe per lunghi tratti e si sblocca solo
quando arriva un blocco ricco di normali.

La quantità che conta è dunque la prevalenza **per blocco**, non quella
aggregata, e la mediana è otto volte più bassa della media. Il numero dei salti
va riportato accanto ai risultati: è un esito del protocollo, non un incidente.

**La quota di etichette spese, e il suo denominatore.** Su C le etichette
richieste sono **91.506 per seme**, uguali in tutte le politiche: 915 blocchi
completi da 100 più 6 sul blocco finale di 673 righe. La quota è quindi
`91.506 / 9.150.673 = 0,0099999`, cioè l'1% dichiarato a meno
dell'arrotondamento del blocco finale.

Il valore pubblicato in precedenza era `0,009990`, perché il denominatore era
`blocchi × dimensione del blocco`, cioè 9.160.000 righe: 9.327 righe che non
esistono. Lo scarto è al quarto decimale e non cambia nessuna conclusione, ma
il numero non era quello che il protocollo definisce, ed era l'unico punto in
cui un numero del rendiconto non si poteva ricalcolare dai suoi stessi campi.
Adesso il rendiconto registra `righe_scorse` — le righe effettivamente scorse —
insieme alla quota e al nome del denominatore, e un test lo verifica su un
flusso costruito con l'ultimo blocco incompleto.

**L'impronta delle righe campionate.** Il confronto fra due esecuzioni — soglia
a zero contro soglia calibrata, politiche diverse sullo stesso flusso — si
regge sul fatto che le righe etichettate siano le stesse, nello stesso ordine.
Il campo che serviva a dimostrarlo si chiamava `row_id_campionati_sha` ma
conteneva `sum(row_id) % 10**9`: **era una somma, non un digest**, insensibile
all'ordine, con collisioni costruibili a mano — `{10, 20, 30}` e `{11, 19, 30}`
hanno la stessa somma — e ridotta modulo 10⁹.

Dalle corse di questo ciclo ogni blocco registra `row_id_campionati_sha256`: lo
**SHA-256** dei `row_id` campionati nell'ordine in cui il campionatore li ha
restituiti, scritti come interi decimali separati da virgola e codificati in
UTF-8. La serializzazione è dichiarata perché l'impronta sia ricalcolabile da
fuori. Il campo storico resta, rinominato `row_id_campionati_somma_storica`,
per continuità con i rendiconti già pubblicati e col nome che dice che cos'è.
Gli strumenti di confronto usano il digest quando c'è, ripiegano sulla somma
sui rendiconti vecchi, e **dichiarano su quale dei due si sono basati**: un
confronto passato sulla somma vale meno di uno passato sul digest, e non va
riportato come se fosse lo stesso controllo.

**Come leggere le misure.** Poiché gli attacchi sono la maggioranza, l'accuratezza
complessiva è poco informativa. Vanno riportate per blocco le misure sulla
classe normale, il tasso di falsi allarmi e, per i tipi nuovi, il richiamo per
tipo.

**Le due aggregazioni del tasso di falsi allarmi.** Su un flusso in cui i
normali sono concentrati in una minoranza di blocchi le due misure seguenti
divergono, e per due modelli su tre hanno verso opposto. Vanno riportate
**affiancate**, ciascuna col suo nome; riportarne una sola senza dichiararla
rende la conclusione non verificabile.

- **FPR medio per blocco**: media non pesata di `fp/normali` sui blocchi che
  contengono almeno un normale — su C sono 875 su 916. **Ogni blocco pesa uno**,
  quindi la misura dice su quanti *blocchi* il rilevatore peggiora.
- **FPR complessivo**: somma delle matrici di confusione su tutti i blocchi,
  poi `fp/(fp+vn)`. **Ogni normale pesa uno**, quindi la misura dice quanti
  *allarmi* in più arrivano all'operatore, e i blocchi ricchi dominano.

Entrambe si ricavano dai rendiconti per blocco con `riepiloghi_semi.py`, che
verifica da sé la coerenza delle matrici di confusione e la chiusura dei propri
conti.

## 7. Che cosa è riusato e che cosa è nuovo

**Riusato senza modifiche.** Il modulo `research028/vendor/audit_ton_full.py`:
`audit_csv` per lo scorrimento a flusso e la classificazione in stage,
`normalize_raw` per il contratto sui valori, `fingerprint_rows` per le
impronte, `parse_timestamp` e `timestamp_iso` per i tempi. E
`research028/tools/coverage028.py`, eseguito così com'è sull'aggregato
prodotto.

**Ricostruito, perché non pubblicato.** L'archivio delle impronte note, secondo
la procedura che il README stesso prescrive per gli array di input mancanti.

**Nuovo.** L'istante di disponibilità delle feature, che l'audit non calcola;
l'aggregazione oraria; l'assemblaggio delle consegne.

**Non eseguibile, e perché.** `diagnose_ton028.py` chiama `verify_kit()`, che
pretende tutte e 42 le voci di `KIT_MANIFEST.json`: **22 non sono nel
repository pubblico**, e lo strumento si ferma alla prima.

Su `run_audit` va corretta un'affermazione precedente. Avevo scritto che la
ricevuta di acquisizione non era pubblicata: **è pubblicata**, sotto un altro
percorso e un altro nome, in
`research030/training/vendor/source_manifest.json`. Il suo SHA-256 è
`8bc238eb…`, esattamente il valore `EXPECTED_RECEIPT` che l'audit pretende.
Copiandola in `research028/inputs/` col nome atteso, `load_pinned_inputs`
**passa**: verificato, 23 voci nel manifest e 23 nella ricevuta. L'unico
ostacolo residuo di `run_audit` è quindi `load_known`, cioè l'archivio delle
impronte, che resta non pubblicato.

**Un componente che non si trasferisce.** `coverage028.py` gira, ma sui dati
completi produce 129 candidati e **zero** con tutti e quattro i suoi gruppi di
test. La causa è misurata, non supposta: la sua regola fissa il test alla coda
finale, e poiché i tipi non ricorrono, l'ultimo giorno contiene solo `backdoor`
e `mitm`, mentre sette tipi su nove non vi compaiono affatto. I suoi gruppi
presuppongono che i tipi tornino; qui non tornano. Il protocollo di questa
attività non ne ha bisogno, perché è progressivo e non usa un test finale
fisso: lo strumento resta utile per la misura dell'esposizione e per
l'enumerazione dei candidati, non per la selezione della partizione.

## 8. I tre controlli richiesti

**1. SHA-256 dei 23 CSV contro il manifest della fonte.** La ricevuta
`research030/training/vendor/source_manifest.json` contiene, per ciascun file,
il campo `local_sha256` oltre a `bytes` ed `expected_bytes`. Confronto eseguito
su tutti e 23: **23 SHA-256 su 23 coincidono**, e altrettante dimensioni. La
copia locale non è quindi soltanto della stessa taglia: è **byte per byte la
stessa**.

Un limite che la ricevuta stessa dichiara: il campo `hash_authority` vale
`locally_computed_not_publisher_verified`. Quegli SHA-256 sono stati calcolati
localmente, non pubblicati dal fornitore del dataset. La coincidenza stabilisce
che i nostri byte sono gli stessi byte scaricati allora, non che siano la
release canonica del fornitore. La ricevuta registra come validatore l'ETag
della risposta HTTP, e `dataset_version` 1 per tutte le voci.

**2. Ordine interno e ordinamento del replay.** Misurato per file: nella
sorgente le coppie consecutive fuori ordine per `ts` sono **21** su 22,3
milioni, concentrate in otto file su ventitré. Il replay è ordinato per
`feature_ready_time`, cioè quando tutte le feature sono disponibili: per quel
criterio le coppie fuori ordine sono **8.703.109**, e il riordino è quindi
necessario e non facoltativo. Il dettaglio per file è in `diagnostics.json`,
campo `salti_indietro_nel_ts_per_file`; la definizione e le sue conseguenze
sono al §3.

**3. Script e conteggi della ricostruzione.** Pubblicati nel PR:
`impronte_note.py`, il rapporto `impronte_note.json` con tutti i conteggi, e i
conteggi per file in `per_file/`. La ricostruzione è descritta come al §5 —
controllo delle sovrapposizioni degli otto input con il CSV noto — con il suo
limite esplicito sull'indipendenza di D.

## 9. Verifiche già eseguite

- I 23 SHA-256 e le 23 dimensioni coincidono con la ricevuta della fonte, per
  3.372.475.448 byte; il manifest della fonte e la ricevuta hanno entrambi lo
  SHA-256 che l'audit pretende.
- Nessuna rottura della catena temporale fra file consecutivi.
- Nessuna riga con `label` sconosciuta, nessuna con `ts` non valido; 869 righe
  su 22.339.021 violano il contratto sui valori, tutte per un campo non numerico.
- L'aggregato orario, arrotolato a giorno, coincide **esattamente** con
  l'aggregato giornaliero prodotto dallo strumento di riferimento, per tutti e
  23 i file: il codice aggiunto è verificato contro quello riusato.
- La partizione A/B/C/D copre tutte le 22.338.152 righe valide, senza
  sovrapposizioni né residui.

## 10. Primo replay eseguito su C

Eseguito con i parametri della scheda — blocchi di 10.000 in ordine di
`feature_ready_time`, budget 1%, ritardo di un blocco, memoria FIFO di 256
inizializzata da A, seme 42, solo CPU. 916 blocchi, 91.506 etichette spese,
esattamente l'1,00% delle righe. Modelli iniziali riaddestrati su A; ogni
modello adattivo confrontato con una copia congelata della stessa architettura
sugli stessi esempi; nessuna predizione ricalcolata.

Il ritardo è verificato sul risultato e non solo sul codice: al blocco 0 e al
blocco 1 adattivo e congelato danno misure **identiche**, perché le etichette
del blocco 0 arrivano alla fine del blocco 1; al blocco 2 divergono. È
esattamente «un aggiornamento incide per la prima volta sul blocco k+2».

Ripetuto su **cinque semi**, da 42 a 46. Si riporta solo ciò che regge sulle
cinque ripetizioni; le medie sui blocchi escludono i blocchi dove la misura non
era definita.

| modello | AUROC congelato | AUROC adattivo | guadagno | semi con guadagno |
|---|---:|---:|---:|:--:|
| LR | 0,7149 | 0,7739 | **+0,0590** | 5 su 5 |
| MLP | 0,7921 | 0,7664 | −0,0257 | 1 su 5 |
| additivo | 0,8909 | 0,8068 | **−0,0841** | 0 su 5 |

**Il tasso di falsi allarmi dipende dall'aggregazione, e va dichiarata.** Le due
misure sono definite al §6 e danno verso opposto su questi dati. Medie sui
cinque semi:

| modello | ric. normali **medio per blocco** | | FPR **medio per blocco** | FPR **complessivo** | falsi positivi |
|---|---:|---:|---:|---:|---:|
| LR | 0,431 → 0,220 | | **+0,211** in 5/5 | **−0,072** in 5/5 | −12.412 |
| MLP | 0,732 → 0,264 | | **+0,468** in 5/5 | **+0,039** in 5/5 | +6.709 |
| additivo | 0,483 → 0,297 | | **+0,186** in 5/5 | **−0,054** in 5/5 | −9.297 |

Una versione precedente di questa tabella riportava 0,734 → 0,267 per l'MLP e
0,462 → 0,276 per l'additivo: valori non riproducibili, corretti qui con quelli
che `riepiloghi_semi.py` ricava dai rendiconti pubblicati.

Sul **FPR medio per blocco** i falsi allarmi aumentano in tutti e tre i modelli
e in tutti e cinque i semi. Sul **FPR complessivo** scendono per la logistica e
per l'additivo, in cinque semi su cinque, e salgono per il solo MLP. Il
meccanismo è comune ai tre: l'adattamento **sposta falsi positivi dai blocchi
ricchi di normali a quelli poveri**, e cambia solo quale trasferimento sia più
grande. Sul seme 42: la logistica aggiunge 4.064 falsi positivi nei 629 blocchi
con 1–199 normali e ne toglie 12.637 nei 246 con almeno 200; l'MLP aggiunge
9.788 e toglie 4.132; l'additivo aggiunge 4.514 e toglie 9.947. Accuratezza e
richiamo sugli attacchi salgono, ma in un flusso al 98% di attacchi questo dice
solo che il modello si sposta verso la classe di maggioranza.

Sull'AUROC solo la **regressione logistica** guadagna in modo stabile.
L'**additivo peggiora** in cinque semi su cinque, pur essendo la migliore delle
tre quando è congelata. L'**MLP non guadagna**: sul solo seme 42 sembrava
migliorare da 0,798 a 0,833, ma sulle cinque ripetizioni la media è −0,026 e il
guadagno è positivo in un seme solo. Quella prima lettura era un effetto del
seme, ed è la ragione per cui le ripetizioni sono state fatte prima di
riportare il risultato.

**Quando l'adattamento serve davvero: il regime di inversione.** La misura più
netta del pilota non sta nelle medie. Il modello congelato ha AUROC **sotto
0,5** — cioè ordina al contrario — in una minoranza di blocchi, e lì
l'adattamento guadagna molto mentre altrove perde poco ma perde.

La diagnosi è stata fatta **sul seme 42**: 174 blocchi su 875 per la logistica,
154 per l'MLP, **soltanto 10** per l'additivo; guadagno **+0,556** contro
−0,056, **+0,655** contro −0,098, **+0,483** contro −0,096.

**Sui cinque semi** il regime regge, con stabilità diversa fra i modelli:

| modello | invertiti su 875 | guadagno dove invertito | guadagno altrove |
|---|---|---:|---:|
| LR | 174 in tutti e cinque | +0,553 | −0,064 |
| MLP | da 145 a 161 | +0,636 | −0,167 |
| additivo | 9 o 10 | +0,437 | −0,090 |

Il 174 costante non è una coincidenza: la logistica congelata è deterministica,
quindi l'insieme dei blocchi invertiti è una proprietà del modello iniziale e
non del seme, e fra i semi variano soltanto i guadagni. Per l'MLP e per
l'additivo dipende dal seme anche il modello iniziale.

**L'inversione segue il tipo di attacco quasi esattamente.** Contando i blocchi
in cui il tipo compare, fra gli 875 misurabili, per la logistica sul seme 42:
`dos` 122 invertiti su 123, `injection` 46 su 49, `ddos` 6 su 586, `password` 0
su 120. `dos` e `injection` sono i primi tipi a comparire dopo A, che contiene
solo `scanning`. Una versione precedente riportava 580 blocchi di `ddos`: il
conteggio corretto è 586 con questa definizione, 584 contando i soli blocchi in
cui `ddos` è l'unico tipo. L'additivo quasi non si inverte ed è anche il
modello con l'AUROC congelato più alto: non ha quasi nulla da correggere, e
paga solo il costo dell'aggiornamento.

**Questo assorbe la lettura per ricchezza di normali** riportata qui sotto. Sul
seme 42 i blocchi ricchi di normali danno +0,409, ma il **67%** di essi è
invertito — 164 su 246, identico in tutti e cinque i semi, perché l'insieme
dipende dal solo modello congelato. Controllando per l'inversione, i ricchi
**non** invertiti danno **+0,071** sul seme 42 e da +0,062 a +0,077 sui cinque,
e i poveri non invertiti −0,136 sul seme 42 e da −0,136 a −0,220 sui cinque. La
ricchezza di normali coincide in larga parte con l'inversione e non la
sostituisce come spiegazione.

Una versione precedente riportava −0,073 per i poveri non invertiti: era il
guadagno «altrove» di un altro seme, trascritto per errore da un'altra analisi.

**Dove vive il guadagno.** Confrontando il guadagno di AUROC per fascia di
ricchezza di normali del blocco, i due regimi sono opposti:

| normali nel blocco | blocchi | AUROC congelato | AUROC adattivo | guadagno |
|---|---:|---:|---:|---:|
| meno di 10 | 221 | 0,845 | 0,713 | −0,132 |
| da 10 a 49 | 275 | 0,817 | 0,782 | −0,035 |
| da 50 a 199 | 133 | 0,733 | 0,700 | −0,032 |
| almeno 200 | 246 | 0,474 | 0,883 | **+0,409** |

Le quattro fasce sommano agli 875 blocchi misurabili: i 41 blocchi senza
normali non hanno AUROC definita e non entrano in nessuna fascia.

La media aggregata è quindi una **miscela di due regimi**: i piccoli
aggiornamenti aiutano quando il blocco contiene abbastanza traffico normale, e
danneggiano quando non ne contiene. È la risposta più diretta che i dati danno
alla domanda «quando» gli aggiornamenti servono.

**La prevalenza aggregata non predice i salti: conta la distribuzione per
blocco.** Misurata sui tre flussi costruiti:

| flusso | quota normali | media per blocco | mediana | blocchi con meno di 10 | normali nel 5% di blocchi più ricco |
|---|---:|---:|---:|---:|---:|
| A | 5,64% | 563,6 | 28 | 88 su 396 | 85,1% |
| B | 1,57% | 157,0 | **150** | 28 su 569 | 15,2% |
| C | 1,89% | 189,2 | 24 | 262 su 916 | 46,9% |

Il risultato è controintuitivo e va detto: **B ha la quota aggregata più bassa
ma la mediana per blocco più alta**, e i suoi normali sono i più uniformemente
distribuiti dei tre. Ci si deve quindi attendere che la guardia scatti **meno**
in B che in C, nonostante la prevalenza aggregata dica il contrario. A, che ha
la prevalenza aggregata più alta, è il flusso più concentrato: il 5% dei
blocchi più ricchi contiene l'85,1% di tutti i suoi normali.

**Un rimedio provato e scartato.** Tenendo lo stesso budget dell'1% ma
distribuendolo su dieci fasce del punteggio della logistica congelata — solo
informazione disponibile al momento della predizione, non le etichette, e un
punteggio unico per tutti i metodi così gli indici restano condivisi — i salti
**salgono** da 393 a 421 e le misure cambiano entro ±0,016. Il punteggio della
logistica congelata colloca quasi tutto nella regione «attacco», quindi anche
la fascia più bassa è in prevalenza attacchi. Risultato **negativo**: esclude
la correzione meno costosa.

Questi numeri vengono dallo stream di sviluppo C, come previsto: **D non è
stato toccato**.

## 10bis. Il confronto con la soglia scelta su B

Il §10 è il **riferimento senza calibrazione**: le misure di decisione sono
calcolate al punto naturale del punteggio, `punteggio > 0`. Resta, e non viene
toccato. Questa sezione è il confronto successivo, con la soglia scelta su B.

**Come è scelta.** Massimo della balanced accuracy — media del richiamo sui
normali e del richiamo sugli attacchi — sui punteggi del **modello congelato su
B**, intervallo che esiste per questo e che non era ancora stato usato. Una
soglia per modello, **identica e fissa** nella coppia congelato / adattivo. È
l'unica scelta causalmente ammissibile: B precede C, e il modello adattivo su B
non esiste perché B non viene replayato. Che la soglia diventi mal tarata per
il modello adattivo, mentre i suoi punteggi si spostano, non è un difetto del
disegno: è uno dei risultati.

**I candidati.** Poiché la predizione è `punteggio > soglia`, una soglia pari a
un valore osservato classifica quel valore come normale. I candidati sono i
**valori distinti** dei punteggi del modello congelato sull'**intero** B — da
405.062 a 405.580 secondo il modello e il seme — più un valore sotto il minimo
osservato, che corrisponde a predire tutto attacco. Con quelli si ottengono
tutte le coppie (richiamo normali, richiamo attacchi) raggiungibili e
nessun'altra.

**La regola di confronto.** Il massimo della balanced accuracy è individuato per
**uguaglianza esatta** fra float64, non entro una tolleranza. La scelta non è
una formalità e va misurata: in tutti e quindici i casi — tre modelli per
cinque semi — il massimo esatto è raggiunto da **un candidato solo**, e lo
stesso vale con tolleranza 1e-12 e 1e-9; ma con tolleranza **1e-6 pareggiano da
1 a 27 candidati**, e con un confronto così lasco la soglia cambierebbe. Per
questo la regola è dichiarata e non sottintesa.

**La regola di parità.** Fra i candidati a pari merito si prende la **mediana
inferiore**: ordinati i candidati, l'elemento di indice `(n-1)//2` con indici
da zero. Per un numero dispari è l'elemento centrale, per un numero pari il
minore dei due centrali; vale senza casi separati. **Su questi dati non è mai
stata applicata**, perché il massimo esatto è sempre unico. Resta nel codice e
dichiarata.

**La soglia è scelta separatamente per ogni modello e per ogni seme**, e poi
resta identica e fissa nella relativa coppia congelato / adattivo.

Candidati, pareggi alle quattro tolleranze, indice scelto e i tre candidati
immediatamente attorno sono in `soglie_candidati.json`, prodotto da
`documenta_soglie.py`: così la convenzione è rifacibile e verificabile sul
numero, non sulla descrizione.

**Che cosa questa convenzione non afferma.** Che la soglia sia stabile. È
deterministica — dati lo stesso B e lo stesso modello congelato produce sempre
lo stesso numero — e questo è tutto ciò che è stato misurato. In particolare il
fatto che per la regressione logistica risulti identica nei cinque semi **non è
evidenza di robustezza**: segue dal fatto che la logistica congelata è
deterministica, cioè che il modello è lo stesso, non che la scelta resista a un
cambiamento dei dati. La stabilità rispetto a ritagli di B, o a un B diverso,
non è misurata e non va dedotta.

| modello | soglia, semi 42–46 | balanced accuracy su B | alla soglia zero |
|---|---|---:|---:|
| LR | +4,3586 in tutti e cinque | 0,6660 | 0,6482 |
| MLP | da +4,3566 a +6,5286 | da 0,7026 a 0,7098 | ~0,68 |
| additivo | da +6,3424 a +7,2005 | da 0,8163 a 0,8168 | 0,6646 |

La soglia della LR risulta identica nei cinque semi per la stessa ragione per
cui lo sono i suoi 174 blocchi invertiti: il modello congelato è
deterministico, quindi i punteggi su B sono gli stessi. Non è una misura di
stabilità della soglia, come detto sopra.

**Che cosa la soglia non può cambiare, e non cambia.** L'AUROC è una misura di
ordinamento e non dipende dal punto di decisione; gli indici campionati non lo
usano. Lo strumento di confronto lo verifica blocco per blocco e si ferma al
primo scostamento: **27.480 coppie di AUROC confrontate e identiche** — cinque
semi per 916 blocchi per tre modelli per due stati — e indici campionati
identici in tutti i blocchi. Di quelle coppie, 26.250 sono su valori definiti e
1.230 sono blocchi in cui l'AUROC non è definita in nessuna delle due varianti,
perché mancano i normali: il confronto le attraversa, ma non sono misure. Tutto il §10 sull'inversione resta quindi valido parola per parola.

**Che cosa cambia.** Le misure di decisione, molto, e in due modelli su tre il
**verso** dell'effetto dell'adattamento si rovescia. Medie sui cinque semi,
variazione adattivo meno congelato:

| modello | FPR medio per blocco | | FPR complessivo | | falsi positivi | |
|---|---:|---:|---:|---:|---:|---:|
| | a zero | su B | a zero | su B | a zero | su B |
| LR | +0,211 | **−0,055** | −0,072 | −0,302 | −12.412 | −52.307 |
| MLP | +0,468 | +0,224 | +0,039 | **−0,132** | +6.709 | −22.879 |
| additivo | +0,186 | +0,248 | −0,054 | −0,022 | −9.297 | −3.890 |

Ogni verso è unanime sui cinque semi. In grassetto i due segni che si
rovesciano: per la LR sulla media per blocco, per l'MLP sul complessivo.

Il richiamo sui normali, medio per blocco, congelato → adattivo:

| modello | a zero | su B |
|---|---|---|
| LR | 0,431 → 0,220 | 0,533 → **0,588** |
| MLP | 0,732 → 0,264 | 0,858 → 0,634 |
| additivo | 0,483 → 0,297 | 0,960 → 0,712 |

**La lettura, e la correzione che impone.** Il §10 concludeva che
l'adattamento peggiora sempre il richiamo sui normali. Con la soglia scelta su
B quella conclusione **non regge più per la LR**, dove l'adattamento lo
migliora, da 0,533 a 0,588, e dove i falsi positivi scendono di 52.307 invece
di 12.412. Per l'MLP resta vera sulla media per blocco ma si rovescia sul
complessivo. Per il modello additivo regge su entrambe le aggregazioni, ed è
l'unico dei tre.

Una parte di quello che avevamo letto come danno dell'adattamento era quindi
**un effetto del punto di decisione**, non dell'adattamento: i modelli congelati
alla soglia zero sono mal tarati su un flusso al 98% di attacchi — il richiamo
sui normali del modello additivo congelato passa da 0,483 a 0,960 solo
cambiando soglia — e il confronto a soglia zero attribuiva all'aggiornamento
una parte di quella mala taratura. Il regime di inversione, che è una misura di
ordinamento, non è intaccato.

## 10ter. Aggiornare solo su evidenza di inversione, con il controllo negativo

Tutto in questa sezione è misurato con la **soglia scelta su B** del §10bis,
non a zero: il confronto a soglia zero attribuirebbe all'aggiornamento una
parte della mala taratura del modello congelato. Parametri invariati: blocchi
da 10.000, budget dell'1%, memoria FIFO di 256, ritardo di un blocco.

### Il primo passo: il verso è stimabile dalle sole etichette arrivate?

Vanno distinti due stimatori, perché uno dei due è cieco.

**Dalla memoria, in campione: non funziona.** L'AUROC calcolata sui 256 esempi
in memoria non riconosce **mai** un blocco invertito — zero su 38 per la
logistica, zero su 26 per l'MLP, zero su 50 per l'additivo sul seme 42 — e la
correlazione con il valore vero è +0,125, +0,118 e −0,003. La ragione è
strutturale: la memoria è l'insieme su cui l'aggiornamento è stato fatto,
quindi quella stima è in campione e vale quasi sempre circa 1,0.

**Dalle righe campionate, fuori campione: funziona in parte.** L'AUROC calcolata
sulle sole cento righe campionate del blocco, con i punteggi **già emessi**, è
fuori campione rispetto al modello che li ha emessi. Sui cinque semi:

| | disponibile | correlazione a *j* | a *j+2* | richiamo a *j+2* | precisione a *j+2* |
|---|---|---:|---:|---:|---:|
| logistica | 1.904 blocchi su 4.580 (41,6%) | +0,687 | +0,363 | 37,2% | 17,0% |
| MLP | idem | +0,692 | +0,325 | 43,9% | 18,1% |
| additivo | idem | +0,729 | +0,292 | 27,4% | 15,3% |

Tre cose vanno lette insieme. La stima **esiste solo nel 41,6% dei blocchi**,
perché fra le cento righe campionate il blocco mediano non contiene **nessun**
normale — media 1,86, e 2.676 blocchi su 4.580 ne hanno zero. Dove esiste,
segue bene l'AUROC del blocco corrente, +0,69 a +0,73, ma **si degrada a
+0,29 ÷ +0,36 al ritardo che conta**, *j+2*, che è il primo blocco su cui la
decisione incide. E come regola di decisione a *j+2* la sua **precisione è
15–18%**: segnala da cinque a sei blocchi per ogni inversione vera.

Dove la stima non esiste la decisione è **dichiarata non disponibile**, e
tenuta distinta da «nessuna evidenza»: la prima dice che non si sa, la seconda
che si sa e non c'è. Nel rendiconto sono due conteggi separati.

### Il risparmio è di aggiornamenti, non di etichette

Va detto prima dei numeri, perché altrimenti le tabelle si leggono male. Il
budget dell'1% si spende **comunque**: le cento righe per blocco vengono
etichettate indipendentemente da quello che la politica decide di farne. Nelle
corse qui sotto le etichette spese sono **91.506 per seme in tutte e tre le
politiche**, identiche. Quello che cambia è il numero di aggiornamenti
applicati, e quello è il risparmio misurabile. Un risparmio di etichette
richiederebbe un budget variabile, che in questa fase non è in discussione.

### Le tre politiche, e il controllo negativo

Quattro alternative sugli stessi blocchi, con gli stessi `row_id` campionati e
la stessa soglia. Il controllo negativo aggiorna su altrettanti blocchi scelti
a sorte, **modello per modello**, estratti fra i blocchi in cui un
aggiornamento è davvero possibile: estraendoli fra tutti, il 43% delle
estrazioni cadrebbe su un blocco con memoria monoclasse e il controllo ne
applicherebbe quasi la metà, misurando la differenza fra i conteggi invece di
quella fra i criteri.

Medie sui cinque semi; `aggiorn.` è il numero medio di aggiornamenti applicati.

| logistica | AUROC | ric. normali | FPR complessivo | aggiorn. |
|---|---:|---:|---:|---:|
| congelato | 0,7149 | 0,5327 | 0,4872 | 0 |
| ogni blocco | 0,7739 | 0,5876 | 0,1854 | 520 |
| **su evidenza** | **0,7756** | 0,5711 | 0,1944 | **45,2** |
| casuale, pareggiato | 0,7499 | 0,5501 | 0,2467 | 45,2 |

| MLP | AUROC | ric. normali | FPR complessivo | aggiorn. |
|---|---:|---:|---:|---:|
| congelato | 0,7921 | 0,8585 | 0,2828 | 0 |
| ogni blocco | 0,7664 | 0,6341 | 0,1508 | 520 |
| **su evidenza** | **0,7956** | 0,6625 | 0,1844 | **33,8** |
| casuale, pareggiato | 0,7265 | 0,6427 | 0,1625 | 33,8 |

| additivo | AUROC | ric. normali | FPR complessivo | aggiorn. |
|---|---:|---:|---:|---:|
| **congelato** | **0,8909** | **0,9602** | 0,1323 | 0 |
| ogni blocco | 0,8068 | 0,7117 | 0,1098 | 520 |
| su evidenza | 0,8358 | 0,7552 | 0,0952 | 19,4 |
| casuale, pareggiato | 0,7384 | 0,6619 | 0,2122 | 19,4 |

### Che cosa ne segue

**Sull'AUROC la politica su evidenza batte il controllo negativo in media su
tutti e tre i modelli**, di +0,026, +0,069 e +0,097. Il confronto è a parità di
aggiornamenti, quindi il margine non viene dall'averne spesi di meno: viene dal
criterio con cui si sceglie quando spenderli. È la risposta che il controllo
negativo esisteva per dare, e sarebbe stata l'altra. È però un margine **medio**
sull'AUROC, misurato contro **un solo sorteggio** per modello e per seme: se
cada fuori dalla variabilità del sorteggio non è stabilito, ed è quello che il
prossimo esperimento deve misurare.

La formulazione precedente di questo paragrafo diceva «in tutti e tre i
modelli» riferendosi alla media, e andava letta come se il confronto valesse
sempre. Non vale sempre, e il dettaglio per seme è questo.

| evidenza − casuale, AUROC | 42 | 43 | 44 | 45 | 46 | media | semi in cui vince |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistica | +0,029 | +0,039 | **−0,032** | +0,023 | +0,069 | +0,026 | 4 su 5 |
| MLP | +0,046 | +0,037 | **−0,013** | +0,192 | +0,084 | +0,069 | 4 su 5 |
| additivo | +0,084 | +0,131 | +0,049 | +0,117 | +0,107 | +0,097 | 5 su 5 |

| evidenza − casuale, FPR complessivo | 42 | 43 | 44 | 45 | 46 | semi in cui vince |
|---|---:|---:|---:|---:|---:|---:|
| logistica | −0,102 | **+0,076** | −0,024 | −0,098 | −0,114 | 4 su 5 |
| MLP | **+0,065** | **+0,113** | **+0,004** | −0,117 | **+0,044** | 1 su 5 |
| additivo | −0,109 | −0,248 | −0,062 | −0,019 | −0,147 | 5 su 5 |

Sull'FPR complessivo il verso favorevole è il valore più basso. Per l'MLP il
controllo casuale produce **meno falsi allarmi su quattro semi su cinque**, e
anche in aggregato: **0,1625 contro 0,1844**. Il criterio batte il sorteggio
sull'ordinamento, non su ogni misura.

E non lo batte sul richiamo degli attacchi, che è la misura per cui il
rilevatore esiste. Aggregato come l'FPR — ogni attacco pesa uno — il confronto
con il controllo casuale è questo.

| evidenza − casuale, richiamo attacchi | 42 | 43 | 44 | 45 | 46 | media | semi in cui vince |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistica | **−0,004** | +0,020 | +0,040 | +0,022 | **−0,046** | +0,006 | 3 su 5 |
| MLP | +0,045 | **−0,160** | **−0,084** | **−0,176** | +0,025 | −0,070 | 2 su 5 |
| additivo | +0,101 | **−0,217** | **−0,011** | **−0,107** | **−0,033** | −0,053 | 1 su 5 |

In aggregato il richiamo sugli attacchi della politica su evidenza è 0,8199
contro 0,8136 del casuale per la logistica, **0,7088 contro 0,7789** per l'MLP
e
**0,6597 contro 0,7129** per l'additivo. Il quadro complessivo è quindi: il
criterio guadagna sull'**ordinamento**, in media e non su tutti i semi, e su
due modelli su tre **perde attacchi** rispetto al sorteggio. Rispetto al
congelato, invece, il richiamo sugli attacchi migliora su tutti e cinque i semi
per la logistica, +0,099 in media, e per l'MLP, +0,309.

Questi sono risultati **esplorativi sullo stream di sviluppo C**: non è stata
dimostrata nessuna superiorità generale, non è stata calcolata nessuna
significatività statistica, e il controllo casuale è un solo sorteggio per
modello e per seme.

Va dichiarato anche il limite del controllo: **una sola pianificazione casuale
per modello e per seme**. Cinque semi danno cinque confronti, non una stima
della variabilità del sorteggio, quindi il confronto non ha un margine di
errore. Il conteggio dei semi in cui vale è l'unica forma di robustezza
disponibile qui, e per questo è riportato.

**La politica su evidenza fa quanto o meglio dell'aggiornamento continuo
spendendo una frazione degli aggiornamenti**: in media sui cinque semi +0,002
per la logistica con 45,2 anziché 520, +0,029 per l'MLP con 33,8 e +0,029 per
l'additivo con 19,4, cioè da 11,5 a 26,8 volte meno. Anche qui le medie nascondono il dettaglio: sulla logistica
il confronto con `ogni blocco` si rovescia su tre semi su cinque. Per l'MLP la
politica su evidenza è l'unica che batte il congelato in media — 0,7956 contro
0,7921 — ma per seme lo batte su tre su cinque, e sul seme 44 perde di 0,054.
Per la logistica tutte le politiche battono il congelato, su tutti e cinque i
semi.

I numeri per seme, con il conteggio dei semi in cui ciascun confronto vale,
sono in `confronto_politiche.json`, campo `per_seme`: lo strumento li calcola e
conta da sé, e dichiara quando un confronto **non** è uniforme.

**Per il modello additivo nessuna politica batte il congelato.** 0,8909 contro
0,8358 della migliore, e **su tutti e cinque i semi con tutte e tre le
politiche**: è l'unico confronto di questa sezione che vale su ogni seme. Il
risultato del §10 regge: quel modello non ha quasi nulla da correggere, e ogni
aggiornamento è un costo netto. Il regime di inversione lo aveva previsto, e
questa sezione lo conferma per via diversa.

Resta il limite da dichiarare accanto a ogni riga di queste tabelle: il
rivelatore ha precisione 15–18% ed esiste nel 41,6% dei blocchi. **Funziona non
perché sia preciso, ma perché aggiornare raramente danneggia poco** anche
quando si sbaglia. Un rivelatore migliore richiederebbe più normali etichettati
per blocco, cioè il punto aperto 3: una memoria che conservi per classe, o un
budget maggiore sui blocchi poveri di normali.

## 10quater. Il costo di un aggiornamento, misurato

Il risparmio della politica si misura in **numero di aggiornamenti**: in media
45,2 per la logistica, 33,8 per l'MLP e 19,4 per l'additivo, contro 520. Da
quel numero non segue nessun guadagno di latenza né di memoria, e finché il
costo del singolo aggiornamento non era misurato quell'inferenza era implicita.
Il rendiconto ora cronometra, con `time.perf_counter`, il solo tratto che un
sistema reale pagherebbe in più: applicare i parametri correnti, calcolare la
rappresentazione della memoria, rifittare i pochi parametri. Il punteggio del
blocco **non** è compreso, perché si paga in ogni politica, congelato incluso.

Misura dedicata, seme 42, macchina altrimenti inattiva — `ogni blocco` con 522
aggiornamenti applicati e 393 tentativi fermati dalla guardia. I tempi sono in
**millisecondi per aggiornamento**, e la colonna che conta è la mediana: la
media è tirata in alto da pochi valori estremi.

| | ms per aggiornamento, mediana | media | min–max | secondi in aggiornamento | quota del replay |
|---|---:|---:|---:|---:|---:|
| logistica | 5,19 | 9,30 | 2,1–61,2 | 4,9 | 5,6% |
| MLP | 5,27 | 5,89 | 1,9–54,9 | 3,1 | 3,5% |
| additivo | 11,08 | 12,94 | 6,1–55,4 | 6,8 | 7,8% |

L'additivo costa circa il doppio degli altri due per aggiornamento, perché la
rappresentazione a B-spline della memoria va ricalcolata; è il modello che
riceve meno aggiornamenti, quindi i due effetti si compensano in parte.

Il conto complessivo, sulla stessa coppia di corse dedicate: con `ogni blocco`
i tre modelli spendono **16,8 secondi** negli aggiornamenti, contando anche i
tentativi fermati dalla guardia, su un replay di **86,6 secondi**; con la
politica su evidenza, **1,7 secondi** su **61,6**. I due replay differiscono di
25,0 secondi, mentre il cronometro degli aggiornamenti ne spiega 15,0: i
restanti 10,0 secondi non si leggono nel cronometro. Profilando 200 blocchi
delle due politiche, la differenza risulta spiegata quasi per intero dalle due
voci che l'aggiornamento comporta — i rifitting della logistica, 587 chiamate
contro 24, e la rappresentazione a B-spline della memoria, 625 contro 431 —
cioè dallo stesso lavoro che il cronometro misura; il residuo resta **non
attribuito a una riga di codice**, e su due CPU condivise non va attribuito.

**La memoria.** Picco di memoria residente del processo: 1.338 MiB, dominato
dagli array dello stream che il banco di prova tiene in memoria, non
dall'adattamento. Lo stato che l'adattamento muove è la memoria FIFO, 10.240
byte per 256 esempi a otto feature in float32, più i parametri: 8.264 byte per
la logistica, 16.520 per l'MLP, 16.456 per l'additivo, rappresentazione della
memoria inclusa. **Non è la memoria di un sistema in linea** e non va
presentata come tale.

**Che cosa questi numeri non dimostrano.** Non dimostrano un guadagno di
latenza: su questa macchina il risparmio è di una quindicina di secondi su un
replay che ne costa quasi novanta per 9,15 milioni di righe, e il resto è
punteggio, che nessuna politica evita. Non dimostrano un guadagno di memoria:
lo stato dell'adattamento è di qualche kilobyte in tutte le politiche, e la
memoria FIFO ha dimensione fissa per costruzione. La misura viene da una
macchina virtuale con due CPU condivise, e il rapporto fra il massimo e il
minimo dello stesso modello lo dice: 29 volte per la logistica e per l'MLP, 9
per l'additivo, sulle corse dedicate. È rumore di scheduling, non varianza del
calcolo: per questo si riporta la mediana, e i numeri valgono come ordine di
grandezza e come rapporto fra modelli, non come latenza in esercizio. Le venti
corse pubblicate portano gli stessi campi, ma sono state eseguite mentre sulla
stessa macchina girava altro lavoro: là il rapporto fra massimo e minimo arriva
a 178 volte, con massimi fino a 384 ms, e quello è rumore e non calcolo. Per
questo la tabella qui sopra viene dalle due corse dedicate in
`replay_evidenza/tempi/`, e `costi_aggiornamento.py` calcola e scrive quel
rapporto per ciascun modello invece di lasciarlo sottinteso.

I numeri si rifanno con `costi_aggiornamento.py`, che rifiuta i rendiconti
prodotti prima di questa correzione invece di interpretarli.

## 10quinquies. Le figure

Quattordici figure in `figure/`, prodotte da `curve_replay.py` dai soli
rendiconti, tutte sul **seme 42**. `figure/LEGGIMI.md` dichiara per ciascuna che
cosa mostra, da quale file viene — con l'impronta SHA-256 dei tre rendiconti —
e il comando che la rifà. Le cinque forme sono: le curve per blocco, il guadagno
per fascia di ricchezza di normali, gli attacchi non rilevati cumulati, il
confronto fra le quattro politiche, e il recupero dopo l'inizio di
un'inversione.

Tre scelte vanno dichiarate qui, perché cambiano che cosa la figura afferma.

**La media progressiva invece della media mobile.** Nel confronto fra politiche
il riquadro di destra riporta la media dell'AUROC dal primo blocco fino a
quello corrente. Non è una scelta estetica: il valore finale di quella curva
**è** l'AUROC media di quella corsa, cioè il valore del **seme 42** nel campo
`per_seme` di `confronto_politiche.json`, che il §10ter cita per i numeri per
seme; quindi la figura e il testo non possono dire due numeri diversi. Una media mobile avrebbe un valore finale che non corrisponde a
nulla di scritto. Le medie sui cinque semi riportate nella tabella riassuntiva
sono un'altra aggregazione e **non** coincidono con la fine della curva: per
l'MLP su evidenza la corsa del seme 42 chiude a 0,8475 mentre la media sui
cinque semi è 0,7956. Lo stesso vale per i conteggi di aggiornamenti scritti in
didascalia, che sono quelli del seme 42 (522, e 54 per la logistica su
evidenza) e non le medie sui semi (520 e 45,2).

**La copia congelata è confrontata, non assunta.** Le quattro politiche
condividono la stessa copia congelata per costruzione. `curve_replay.py` non lo
dà per buono: confronta blocco per blocco i falsi negativi e le AUROC della
copia congelata di ogni rendiconto e si ferma se divergono, perché in quel caso
la figura metterebbe a confronto corse che non sono confrontabili.

**Il recupero media solo gli episodi che si possono seguire.** Gli episodi di
inversione hanno lunghezze molto diverse — sulla logistica 122, 36, 10 blocchi e
sei episodi di un blocco solo — e mediarli tutti a un dato ritardo mescolerebbe
episodi ancora in corso con episodi finiti da tempo. Si mediano quindi i soli
episodi lunghi almeno cinque blocchi, i singoli episodi restano disegnati sotto
la media, e gli esclusi sono scritti in didascalia. Sull'additivo gli episodi
sono tre, lunghi 7, 2 e 1 blocchi: uno solo arriva al minimo, e la figura **non
viene prodotta**. Il programma si astiene da sé e lo dichiara nel proprio
resoconto: `recupero_kan.png` non è un file dimenticato.

Due proprietà sono presidiate da prove e non dall'occhio. La prima: le
grandezze che i documenti citano dalle figure sono ricalcolate dai rendiconti e
confrontate con quello che la figura disegna. Sono quattro, e vale la pena dire
quali, perché il resto non è coperto: il valore finale della media progressiva,
confrontato con la tabella **per seme** di `confronto_politiche.json` e non con
la media sui cinque semi; la somma dei falsi negativi di ciascuna curva
cumulata; gli episodi di inversione, le loro lunghezze in ordine di tempo e
quali di essi la figura del recupero media; i quattro guadagni per fascia, con
i conteggi dei blocchi. Non sono invece verificati i conteggi di aggiornamenti
scritti in didascalia, che vengono letti dal rendiconto senza essere
ricalcolati. La seconda: nessuna figura si salva se un testo esce dal foglio. Una
didascalia troppo lunga non fa cadere il disegno — viene tagliata dal bordo in
silenzio, e la figura resta pubblicabile con mezza frase. Il controllo misura
dopo l'impaginazione dove finisce davvero ogni testo e rifiuta il salvataggio;
è nato da un difetto vero, trovato guardando una figura già prodotta.

Le figure usano la stessa convenzione numerica dei documenti anche sugli assi —
virgola decimale, punto per le migliaia, meno tipografico — perché un asse che
scrive «0.2» accanto a un testo che scrive «0,2» si legge come due misure.

## 11. Punti aperti

1. ~~Base del ramo.~~ **Deciso:** `adapt-01/ton-temporal-pilot` parte
   dall'attuale `main` (`69618ee`), con trasferimento selettivo dei moduli e dei
   test necessari da `drift-protocollo` (`b3d1bef`), dichiarando la provenienza
   nel PR. I due rami esistenti restano invariati e i 18 commit restano nel ramo
   originale.
2. **Il 28 e 29 aprile in D.** Scelta motivata al §4; se si preferisse un D meno
   esposto occorrerebbe rinunciare a tre tipi di attacco, e la motivazione andrebbe
   messa per iscritto.
3. **Budget e memoria.** I circa 395 salti su 915 blocchi dipendono dalla
   rarità dei normali per blocco. La stratificazione sul punteggio è stata
   provata e **non** rimedia. Restano da valutare una memoria che conservi per
   classe, o un budget maggiore sui blocchi poveri di normali. La scheda fissa i
   parametri, quindi non li cambio: è un punto da decidere insieme.
4. **Righe non valide.** Le 869 righe fuori contratto sono escluse da ogni
   conteggio, come fa lo strumento di riferimento. Vanno contate anche nel
   replay o saltate in silenzio? Qui si propone di saltarle e registrarle.
5. ~~Come riportare la valutazione finale su D, dato il §5bis.~~ **Deciso dal
   relatore, e non come era stato proposto.** La proposta era di riportare la
   valutazione due volte, come due analisi affiancate. La decisione è più
   stretta e il protocollo la recepisce così:

   - **D intero è l'analisi principale.** È quella che la scheda definisce, ed è
     quella che va riportata come risultato della valutazione finale.
   - **Il sottoinsieme a vettore non visto è un'analisi supplementare**, e si
     calcola come **maschera sulle stesse predizioni** del replay completo:
     2.228.436 righe, 129.982 normali, prevalenza di normali 5,83%.
   - **Nessuna riga viene esclusa dallo stream.** Il replay scorre D intero,
     nell'ordine di `feature_ready_time`, e la maschera si applica dopo, in sede
     di aggregazione delle misure. Escludere righe dallo stream cambierebbe la
     sequenza che il modello vede, quindi anche le predizioni, e le due misure
     non sarebbero più confrontabili.
   - **Nessuna scelta di modello o di politica si fa sui risultati di D.** Il
     modello e la politica si fissano prima, su A, B e C; D si legge una volta.

   La differenza fra le due aggregazioni resta un risultato da riportare, ma è
   la differenza fra un'analisi principale e una supplementare sulle stesse
   predizioni, non fra due letture di pari rango.
6. ~~La regola di parità sulla soglia.~~ **Fatto e misurato al §10bis.** La
   soglia è scelta su B col massimo della balanced accuracy, identica e fissa
   nella coppia congelato / adattivo. La regola di parità dichiarata — mediana
   delle candidate, con numero pari il minore dei due valori centrali — **non è
   mai servita**: in tutti e quindici i casi il massimo è raggiunto da una
   candidata sola. Resta nel codice e dichiarata. Quello che resta da decidere è
   un'altra cosa: il §10bis mostra che due dei tre risultati sulle misure di
   decisione del §10 erano effetti del punto di decisione, quindi va concordato
   **quale dei due confronti è quello da riportare nell'articolo**, o se vanno
   riportati entrambi affiancati come le due aggregazioni del FPR.

## 12. Correzioni di questo ciclo

Cinque difetti segnalati nel riesame del 3 ottobre. Qui sono elencati con
quello che è cambiato nel codice, nei rendiconti e nei documenti, perché una
correzione annunciata e non tracciabile vale meno di un difetto dichiarato.

Il codice è stato corretto **prima** di rimisurare, e le venti corse — più la
variante scartata sul campionamento stratificato, che è il ventunesimo
rendiconto — sono state rieseguite una volta sola con il codice corretto. Il
riscontro che conta: le misure per blocco si riproducono **identiche** —
`riepiloghi_semi.json` rigenerato dai nuovi rendiconti non differisce in nessun
campo da quello pubblicato — quindi nessuna conclusione di questo protocollo
cambia per effetto delle correzioni. Cambiano i campi del rendiconto e il modo
in cui i numeri sono formulati.

1. **`row_id_campionati_sha` era una somma, non un digest** (§6). Sostituito da
   `row_id_campionati_sha256`, SHA-256 di una serializzazione dichiarata; il
   campo storico resta col nome `row_id_campionati_somma_storica`. Gli strumenti
   di confronto usano il digest quando c'è e dichiarano su quale dei due si sono
   basati. `test_impronta_campione.py` mostra, su casi costruiti, che cosa la
   somma non vedeva.
2. **Il denominatore della quota di etichette era sbagliato** (§6). Era
   `blocchi × dimensione del blocco`; ora è il conteggio delle righe
   effettivamente scorse, registrato come `righe_scorse`. Su C: 91.506 su
   9.150.673. Nella stessa occasione sono stati ricontati dal manifest i
   conteggi dei blocchi incompleti, di cui tre erano sbagliati.
3. **Il risparmio di aggiornamenti non era un risparmio misurato** (§10quater).
   Ora il costo del singolo aggiornamento è cronometrato per modello, la memoria
   è misurata e l'hardware è letto dal sistema; ed è scritto che il numero di
   aggiornamenti non dimostra un guadagno di latenza né di memoria.
4. **La sintesi era cresciuta oltre le due pagine** che la scheda chiede. Il
   dettaglio è ora dichiarato come appendice tecnica, e `ambiente_e_comandi.md`
   è allineato: i conteggi di test non verificabili su questo ramo sono stati
   sostituiti con quello misurato, e D è descritto come letto e contato, non
   costruito né valutato.
5. **«Batte il controllo in tutti e tre i modelli» era una media presentata come
   un fatto uniforme** (§10ter). Riformulato: media sull'AUROC, conteggio dei
   semi in cui il confronto vale, il seme in cui si rovescia, l'FPR complessivo
   accanto all'AUROC, e la dichiarazione che il controllo casuale è una sola
   pianificazione per modello e per seme.

Resta fuori da questo elenco, perché non è un difetto ma una decisione del
relatore, il protocollo su D del §11 punto 5, che sostituisce la proposta fatta
qui nel ciclo precedente.

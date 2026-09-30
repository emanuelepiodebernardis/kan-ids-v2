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

`row_id`, `t_start`, `t_end` e `feature_ready_time` sono tenuti **separati dalle
feature**, così non possono entrare nel modello.

**Preprocessing stimato solo su A.** Le otto feature sono conteggi di byte e di
pacchetti con code lunghe: si applica `log1p` e poi una standardizzazione. Media
e deviazione sono stimate **una sola volta su A** e non vengono più toccate; B,
C e D si trasformano con quei parametri. Nessuna statistica dei flussi
successivi entra nella trasformazione, e in particolare **D non contribuisce in
alcun modo** al preprocessing.

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
che quei vettori di input non erano già comparsi nel campione noto; non dice che
le righe di D siano indipendenti da quelle di A, B e C, né che appartengano a
sessioni, host o campagne distinte. L'indipendenza di D resta un'assunzione
dichiarata, non un risultato misurato.

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

## 6. Replay

Per ogni blocco di 10.000 righe in ordine di `feature_ready_time`:

1. il modello **prevede**;
2. si **richiede** l'etichetta per l'1% delle righe — 100 per blocco completo,
   `floor(0,01·n)` per l'ultimo — con gli **stessi `row_id` per tutti i
   metodi**, così il confronto è sugli stessi esempi;
3. si **attende**: le etichette del blocco *k* arrivano alla fine del blocco
   *k+1*, quindi un aggiornamento incide per la prima volta sul blocco *k+2*;
4. il modello si **aggiorna** — coefficienti per la regressione logistica,
   ultimo strato per l'MLP, guadagno e intercetta per la KAN additiva.

Memoria FIFO di 256 esempi, inizializzata da A. Se la memoria contiene una sola
classe l'aggiornamento si salta e il salto si registra.

Ogni modello aggiornato si confronta con una **copia congelata** della stessa
architettura, sugli stessi esempi. Le predizioni già emesse **non si ricalcolano
mai**. Valutazione progressiva, misure per blocco, `NA` dove una misura non è
definita. Seme 42. Solo CPU in questa fase.

**Quanti blocchi:** A 395 completi più 6.308 righe; B 568 più 1.199; C 915 più
677; D 354 più 9.968.

**Una conseguenza della prevalenza, misurata e non stimata.** Una prima stima
analitica dava la regola del salto attiva sullo 0,7% dei blocchi di C. **È
sbagliata, e il replay eseguito su tutto C lo mostra: i blocchi saltati sono
393 su 915, il 43,0%.**

L'errore stava nell'usare la prevalenza aggregata. In C i normali sono l'1,89%
delle righe, ma **non sono distribuiti uniformemente**: per blocco la media è di
189 normali e la **mediana 24**, il 28,6% dei blocchi ne ha meno di dieci, e il
5% dei blocchi più ricchi contiene il 47,3% di tutti i normali. Con l'1% di
budget, un blocco mediano offre in media 0,24 normali fra i 100 campionati:
quasi mai almeno uno. La memoria di 256 copre circa due blocchi e mezzo di
campioni, quindi resta di una sola classe per lunghi tratti e si sblocca solo
quando arriva un blocco ricco di normali.

La quantità che conta è dunque la prevalenza **per blocco**, non quella
aggregata, e la mediana è otto volte più bassa della media. Il numero dei salti
va riportato accanto ai risultati: è un esito del protocollo, non un incidente.

**Come leggere le misure.** Poiché gli attacchi sono la maggioranza, l'accuratezza
complessiva è poco informativa. Vanno riportate per blocco le misure sulla
classe normale, il tasso di falsi allarmi e, per i tipi nuovi, il richiamo per
tipo.

## 7. Che cosa è riusato e che cosa è nuovo

**Riusato senza modifiche.** Il modulo `research028/vendor/audit_ton_full.py`:
`audit_csv` per lo scorrimento a flusso e la classificazione in stage,
`normalize_raw` per il contratto sui valori, `fingerprint_rows` per le impronte,
`parse_timestamp` e `timestamp_iso` per i tempi. E `research028/tools/coverage028.py`,
eseguito così com'è sull'aggregato prodotto.

**Ricostruito, perché non pubblicato.** L'archivio delle impronte note, secondo
la procedura che il README stesso prescrive per gli array di input mancanti.

**Nuovo.** L'istante di disponibilità delle feature, che l'audit non calcola;
l'aggregazione oraria; l'assemblaggio delle consegne.

**Non eseguibile, e perché.** `diagnose_ton028.py` chiama `verify_kit()`, che
pretende tutte e 42 le voci di `KIT_MANIFEST.json`: **22 non sono nel
repository pubblico**, e lo strumento si ferma alla prima.

Su `run_audit` va corretta un'affermazione precedente. Avevo scritto che la
ricevuta di acquisizione non era pubblicata: **è pubblicata**, sotto un altro
percorso e un altro nome, in `research030/training/vendor/source_manifest.json`.
Il suo SHA-256 è `8bc238eb…`, esattamente il valore `EXPECTED_RECEIPT` che
l'audit pretende. Copiandola in `research028/inputs/` col nome atteso,
`load_pinned_inputs` **passa**: verificato, 23 voci nel manifest e 23 nella
ricevuta. L'unico ostacolo residuo di `run_audit` è quindi `load_known`, cioè
l'archivio delle impronte, che resta non pubblicato.

**Un componente che non si trasferisce.** `coverage028.py` gira, ma sui dati
completi produce 129 candidati e **zero** con tutti e quattro i suoi gruppi di
test. La causa è misurata, non supposta: la sua regola fissa il test alla coda
finale, e poiché i tipi non ricorrono, l'ultimo giorno contiene solo `backdoor`
e `mitm`, mentre sette tipi su nove non vi compaiono affatto. I suoi gruppi
presuppongono che i tipi tornino; qui non tornano. Il protocollo di questa
attività non ne ha bisogno, perché è progressivo e non usa un test finale fisso:
lo strumento resta utile per la misura dell'esposizione e per l'enumerazione dei
candidati, non per la selezione della partizione.

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
campo `salti_indietro_nel_ts_per_file`; la definizione e le sue conseguenze sono
al §3.

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
| KAN | 0,8909 | 0,8068 | **−0,0841** | 0 su 5 |

| modello | ric. normali congelato | ric. normali adattivo | semi in calo |
|---|---:|---:|:--:|
| LR | 0,431 | 0,220 | 5 su 5 |
| MLP | 0,734 | 0,267 | 5 su 5 |
| KAN | 0,462 | 0,276 | 5 su 5 |

**Come leggerla, senza abbellirla.** Il richiamo sui normali **peggiora in tutti
e tre i modelli e in tutti e cinque i semi**: essendo il complemento del tasso
di falsi allarmi, i falsi allarmi aumentano. Accuratezza e richiamo sugli
attacchi salgono, ma in un flusso al 98% di attacchi questo dice solo che il
modello si sposta verso la classe di maggioranza.

Sull'AUROC solo la **regressione logistica** guadagna in modo stabile. La **KAN
additiva peggiora** in cinque semi su cinque, pur essendo la migliore delle tre
quando è congelata. L'**MLP non guadagna**: sul solo seme 42 sembrava migliorare
da 0,798 a 0,833, ma sulle cinque ripetizioni la media è −0,026 e il guadagno è
positivo in un seme solo. Quella prima lettura era un effetto del seme, ed è la
ragione per cui le ripetizioni sono state fatte prima di riportare il risultato.

**Quando l'adattamento serve davvero: il regime di inversione.** La misura più
netta del pilota non sta nelle medie. Il modello congelato ha AUROC **sotto
0,5** — cioè ordina al contrario — in 174 blocchi su 875 per la logistica, 154
per l'MLP e **soltanto 10** per la KAN additiva. In quei blocchi l'adattamento
guadagna moltissimo; in tutti gli altri perde poco ma perde: la logistica
**+0,556** contro −0,056, l'MLP **+0,655** contro −0,098, la KAN **+0,483**
contro −0,096.

I blocchi invertiti sono quasi tutti quelli di `dos` (122) e `injection` (46),
cioè i tipi comparsi dopo A, che contiene solo `scanning`; quelli di `ddos`
(580) e `password` (120) non si invertono quasi mai. La KAN additiva quasi non
si inverte ed è anche il modello con l'AUROC congelato più alto: non ha quasi
nulla da correggere, e paga solo il costo dell'aggiornamento.

**Questo assorbe la lettura per ricchezza di normali** riportata qui sotto. I
blocchi ricchi di normali danno +0,409, ma il **67%** di essi è invertito;
controllando per l'inversione, i ricchi **non** invertiti danno solo **+0,071**,
e i poveri non invertiti −0,073. La ricchezza di normali coincide in larga parte
con l'inversione e non la sostituisce come spiegazione.

**Dove vive il guadagno.** Confrontando il guadagno di AUROC per fascia di
ricchezza di normali del blocco, i due regimi sono opposti:

| normali nel blocco | blocchi | AUROC congelato | AUROC adattivo | guadagno |
|---|---:|---:|---:|---:|
| meno di 10 | 221 | 0,845 | 0,713 | −0,132 |
| da 10 a 49 | 275 | 0,817 | 0,782 | −0,035 |
| da 50 a 199 | 133 | 0,733 | 0,700 | −0,032 |
| almeno 200 | 246 | 0,474 | 0,883 | **+0,409** |

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
la prevalenza aggregata più alta, è il flusso più concentrato: il 5% dei blocchi
più ricchi contiene l'85,1% di tutti i suoi normali.

**Un rimedio provato e scartato.** Tenendo lo stesso budget dell'1% ma
distribuendolo su dieci fasce del punteggio della logistica congelata — solo
informazione disponibile al momento della predizione, non le etichette, e un
punteggio unico per tutti i metodi così gli indici restano condivisi — i salti
**salgono** da 393 a 421 e le misure cambiano entro ±0,016. Il punteggio della
logistica congelata colloca quasi tutto nella regione «attacco», quindi anche la
fascia più bassa è in prevalenza attacchi. Risultato **negativo**: esclude la
correzione meno costosa.

Questi numeri vengono dallo stream di sviluppo C, come previsto: **D non è
stato toccato**.

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

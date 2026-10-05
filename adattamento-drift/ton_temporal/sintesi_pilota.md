# Sintesi del primo ciclo — risultati, limiti, e il passo successivo

**ADAPT-01, stream di sviluppo C.** 3 ottobre 2026. L'evidenza è nel Draft PR #2.
I riepiloghi di questo documento si rifanno con `riepiloghi_semi.py` sui cinque
rendiconti per blocco pubblicati, e quello script verifica da sé i propri
conti.

## La sintesi, in due pagine

Queste sono le due pagine che la scheda chiede. Tutto quello che viene dopo la
riga orizzontale è **appendice tecnica**: le sezioni numerate da 0 a 7, citate
nel testo come «§0 … §7», con i numeri e il modo in cui sono stati ottenuti.
Nessun risultato sta soltanto in appendice.

**La domanda.** Quando piccoli aggiornamenti permettono a un rilevatore di
riconoscere meglio gli attacchi dopo un cambiamento del traffico, con poche
etichette e in ritardo.

**Il risultato principale, che non è una media sui blocchi.** Il modello
congelato ordina al contrario — AUROC sotto 0,5 — in una minoranza di blocchi,
e
**solo lì** l'adattamento guadagna molto: in media sui cinque semi, +0,553 per
la logistica nei blocchi invertiti contro −0,064 negli altri. L'inversione
segue il tipo di attacco quasi esattamente: 122 blocchi di `dos` su 123 sono
invertiti, 46 di `injection` su 49, 6 di `ddos` su 586, 0 di `password` su 120.
Il modello additivo si inverte in 9 o 10 blocchi su 875 e non ha quasi nulla da
correggere.

**Da quel risultato è nata una regola operativa, ed è stata verificata.**
Aggiornare **solo su evidenza di inversione** fa quanto o meglio
dell'aggiornamento continuo spendendo, in media sui cinque semi, 45,2
aggiornamenti anziché 520 per la logistica, 33,8 per l'MLP e 19,4 per
l'additivo — da 11,5 a 26,8 volte meno — e sull'AUROC **batte il controllo negativo** — una
politica che aggiorna sullo stesso numero di blocchi scelti a sorte — **in
media** su tutti e tre i modelli, di +0,026, +0,069 e +0,097. Il confronto è a
parità di aggiornamenti, quindi il margine non viene dall'averne spesi di meno:
viene dal criterio. È però un margine medio sull'AUROC, misurato contro un solo
sorteggio, e se cada fuori dalla variabilità del sorteggio resta da stabilire.

La formulazione precisa conta, perché una versione precedente diceva «in tutti
e tre i modelli» intendendo la media. Il confronto **non è uniforme sui semi**
— vince su 4 semi su 5 per logistica e MLP, 5 su 5 per l'additivo, e si
rovescia sul seme 44 — e **non si estende alle misure di decisione**: sull'FPR
complessivo il sorteggio fa meglio per l'MLP (0,1625 contro 0,1844) e sul
richiamo degli attacchi fa meglio su due modelli su tre (per l'MLP 0,7789
contro 0,7088). Il controllo casuale, poi, è **un solo sorteggio** per modello
e per seme: non ha un margine di errore. Sono risultati esplorativi su C, senza
nessuna significatività statistica calcolata. Il dettaglio è al §6.

**Quattro cose che vanno dette con il risultato.**

1. **L'aggregazione cambia la conclusione.** Il tasso di falsi allarmi va
   riportato sia come media per blocco sia come somma delle matrici di
   confusione: su questi dati divergono, e per due modelli su tre hanno verso
   opposto (§3).
2. **La soglia cambia la conclusione.** Con la soglia scelta su B invece che a
   zero, due dei tre risultati sulle misure di decisione si rovesciano: una
   parte di quello che avevamo attribuito all'adattamento era mala taratura del
   modello congelato (§3bis). L'AUROC non ne è toccata, ed è verificato su
   27.480 coppie: 26.250 su valori definiti e 1.230 in cui la misura non è
   definita in nessuna delle due varianti.
3. **Il rivelatore di inversione è debole.** Esiste solo nel 41,6% dei blocchi,
   perché fra le cento righe campionate il blocco mediano non contiene nessun
   normale, e come regola di decisione ha precisione 15–18%. Funziona non perché
   sia preciso, ma perché aggiornare raramente danneggia poco anche sbagliando.
4. **D non è indipendente da A, B e C.** Il 37,23% delle sue righe e il 58,03%
   dei suoi normali hanno un vettore di feature già visto. Il protocollo su D è
   stato fissato dal relatore: **D intero è l'analisi principale**, e il
   sottoinsieme a vettore non visto è un'**analisi supplementare**, calcolata
   come maschera sulle stesse predizioni del replay completo, senza togliere
   righe dallo stream e senza scegliere modello o politica sui risultati di D
   (§5).

**Per il modello additivo nessuna politica batte il congelato**, su tutti e
cinque i semi e con ciascuna delle tre politiche: in media sui semi 0,8909
contro 0,8358 della migliore, che è `su evidenza` su quattro semi e `ogni
blocco` sul seme 44. È il risultato più scomodo, ed è solido perché uniforme:
sull'AUROC il congelato vince 15 confronti su 15, tre politiche per cinque semi.
È l'unico modello in cui l'adattamento non aiuta mai. In `figure/politiche_kan.png` si vede come: la media progressiva
del congelato chiude sopra quella di ogni politica e resta stabilmente sopra
dal blocco 213 in poi; prima, in 175 blocchi su 213, almeno una politica sta
davanti, mai tutte e tre insieme. Il vantaggio è sul flusso intero, non in ogni
suo punto. Sul seme 42
la politica casuale perde su quel modello più attacchi del congelato: 3.356.503
contro 3.157.627.

**Le figure.** Quattordici in `figure/`, sul seme 42, prodotte dai soli
rendiconti; `figure/LEGGIMI.md` dichiara per ciascuna la provenienza e il
comando che la rifà, e il §10quinquies del protocollo le tre scelte che
cambiano quello che una figura afferma. Quella che risponde alla domanda della
scheda è `attacchi_non_rilevati_<modello>.png`: falsi negativi cumulati,
congelato contro adattivo, con la differenza accanto. Sul seme 42 la logistica
chiude a 1.377.545 attacchi non rilevati in meno su 8.977.357. Il conteggio
dipende dal punto di decisione, e la didascalia lo dichiara dentro la figura.

**Le correzioni di questo ciclo.** Cinque difetti segnalati nel riesame del 3
ottobre, corretti nel codice e rimisurati con una riesecuzione completa delle
venti corse, più la variante scartata; le misure per blocco si riproducono
identiche, quindi nessuna conclusione cambia per effetto delle correzioni.

1. Il campo `row_id_campionati_sha` era una **somma** modulo 10⁹, non un digest.
   Ora ogni blocco porta `row_id_campionati_sha256`, SHA-256 di una
   serializzazione dichiarata; il campo storico resta col nome che dice che
   cos'è, e gli strumenti dichiarano su quale dei due si sono basati (§0bis).
2. La quota di etichette usava come denominatore `blocchi × dimensione del
   blocco`, cioè 9.160.000 righe invece di 9.150.673: il valore corretto è
   91.506 / 9.150.673, e il rendiconto registra ora le righe effettivamente
   scorse. Nella stessa riga del protocollo tre conteggi di blocco incompleto
   erano sbagliati, e sono stati ricontati dal manifest.
3. Il risparmio di aggiornamenti **non** era un risparmio di latenza né di
   memoria misurato. Ora il costo del singolo aggiornamento è cronometrato per
   modello, la memoria è misurata e l'hardware è letto dal sistema (§4bis).
4. Questa sintesi era cresciuta oltre le due pagine: il dettaglio è ora
   dichiarato come appendice.
5. «Batte il controllo in tutti e tre i modelli» era una **media presentata come
   un fatto uniforme**: la riformulazione, con il seme su cui il confronto si
   rovescia e con FPR e richiamo accanto all'AUROC, è al §6.

---

# Appendice tecnica

Le sezioni che seguono sono il dettaglio della sintesi qui sopra. Sono citate
nel testo come «§0 … §7» e i numeri si rifanno con gli script indicati nel
protocollo.

## 0. Due convenzioni, dichiarate prima dei risultati

**I tre modelli.** Regressione logistica sulle otto feature; MLP di cui si
aggiorna il solo ultimo strato; e una **KAN a singolo strato con edge a
B-spline**, chiamata qui «l'additivo» per brevità: il punteggio è una somma di
funzioni univariate cubiche, una per ciascuna delle otto feature, più un
guadagno per feature e un'intercetta.

Sulla terza una precisazione, perché una versione precedente di questo
documento la descriveva come «non il componente a B-spline del Paper 1» e
quella formulazione era fuorviante. La **forma funzionale è la stessa** di
`BSplineKANBinary` in `src/kan_bspline.py`, che è a sua volta a singolo strato:
non è un'architettura diversa, è **reimplementata** invece di importata. Ciò
che differisce, e che conta per questo studio, è il regime di stima: i 64
coefficienti delle spline sono stimati **una volta sola** su un sottocampione
di 400.000 righe di A e poi congelati, e l'adattamento muove soltanto **otto
guadagni e l'intercetta**, nove parametri, che in quel componente non esistono.
Dopo A le forme delle spline non cambiano più. Il dettaglio è al §6 del
protocollo.

**Le due aggregazioni del tasso di falsi allarmi.** Divergono su questi dati, e
riportarne una sola senza nominarla rende la conclusione non verificabile.

- **FPR medio per blocco**: media non pesata di `fp/normali` sui blocchi che
  contengono almeno un normale. **Ogni blocco pesa uno.**
- **FPR complessivo**: somma delle matrici di confusione su tutti i blocchi,
  poi `fp/(fp+vn)`. **Ogni normale pesa uno**, quindi i blocchi ricchi di
  normali dominano.

**I due punti di decisione.** Le misure di ordinamento — l'AUROC, e con essa
tutto il §1 — non dipendono dalla soglia. Le misure di decisione sì, e sono
riportate in due regimi:

- **soglia a zero**: il punto naturale del punteggio, `punteggio > 0`. È il
  **riferimento senza calibrazione**.
- **soglia scelta su B**: massimo della balanced accuracy sui punteggi del
  modello congelato su B, una per modello, identica e fissa nella coppia
  congelato / adattivo.

Il §3 mostra che la differenza fra i due non è un dettaglio: in due modelli su
tre rovescia il verso dell'effetto dell'adattamento sui falsi allarmi.

## 0bis. Come si verifica che due corse abbiano usato le stesse righe

Tutti i confronti di questo documento — soglia a zero contro soglia calibrata,
politiche diverse sullo stesso flusso — presuppongono che le righe etichettate
siano **le stesse, nello stesso ordine**. Il campo che serviva a dimostrarlo si
chiamava `row_id_campionati_sha` ma conteneva `sum(row_id) % 10**9`: una somma,
non un digest, insensibile all'ordine e con collisioni costruibili a mano
(`{10, 20, 30}` e `{11, 19, 30}` danno 60 entrambe).

Dalle corse di questo ciclo ogni blocco porta `row_id_campionati_sha256`: lo
SHA-256 dei `row_id` campionati nell'ordine restituito dal campionatore,
scritti come interi decimali separati da virgola e codificati in UTF-8. La
serializzazione è dichiarata perché l'impronta sia ricalcolabile da fuori. Il
campo storico resta, rinominato `row_id_campionati_somma_storica`. Gli
strumenti di confronto usano il digest quando c'è, ripiegano sulla somma sui
rendiconti vecchi, e **scrivono nel proprio rendiconto su quale dei due si sono
basati**.

## 1. La domanda, e la risposta che i dati danno

La domanda è **quando** piccoli aggiornamenti permettono a un rilevatore di
riconoscere meglio gli attacchi dopo un cambiamento del traffico, con poche
etichette e in ritardo.

La risposta del pilota non era quella che ci aspettavamo: **gli aggiornamenti
servono quando il modello congelato ha smesso di ordinare nel verso giusto, e
costano poco ma costano in tutti gli altri casi.**

Sul flusso C il modello congelato ha AUROC **inferiore a 0,5** — cioè ordina al
contrario — in una minoranza di blocchi, e lì l'adattamento guadagna molto.
**Sul seme 42**, che è il seme su cui la diagnosi è stata fatta:

- **logistica**: 174 blocchi invertiti su 875 (20%), da 0,319 a 0,875
  (**+0,556**); altrove da 0,813 a 0,757 (−0,056).
- **MLP**: 154 invertiti (18%), da 0,282 a 0,937 (**+0,655**); altrove da 0,909
  a 0,810 (−0,098).
- **additivo**: **10 invertiti (1%)**, da 0,432 a 0,914 (+0,483); altrove da
  0,891 a 0,795 (−0,096).

**Sui cinque semi**, da 42 a 46, i conteggi e i guadagni sono questi — il
regime regge, e quanto sia stabile dipende dal modello:

| | invertiti su 875 | guadagno dove invertito | guadagno altrove |
|---|---|---|---|
| logistica | 174 in tutti e cinque | +0,553 | −0,064 |
| MLP | da 145 a 161 | +0,636 | −0,167 |
| additivo | 9 o 10 | +0,437 | −0,090 |

Il 174 costante non è una coincidenza fra semi: la logistica congelata è
deterministica, quindi **l'insieme dei blocchi invertiti è una proprietà del
modello iniziale e non del seme**; fra i semi variano soltanto i guadagni, che
dipendono dal campionamento. Per l'MLP e per l'additivo dipende dal seme anche
il modello iniziale, e i conteggi si muovono.

La media sui blocchi del guadagno di AUROC, +0,059 per la logistica, è la somma
dei due regimi e presa da sola non dice nulla di utile.

## 2. Perché l'additivo si comporta diversamente

I modelli iniziali sono addestrati su A, che contiene un solo tipo di attacco,
`scanning`. L'inversione non è sparsa: segue il tipo di attacco quasi
esattamente. Contando i blocchi in cui il tipo compare, fra gli 875 misurabili
e per la logistica sul seme 42:

| tipo | blocchi invertiti | su | quota |
|---|---|---|---|
| `dos` | 122 | 123 | 99% |
| `injection` | 46 | 49 | 94% |
| `ddos` | 6 | 586 | 1% |
| `password` | 0 | 120 | 0% |

`dos` e `injection` sono i primi tipi a comparire dopo A; `ddos` e `password`
arrivano più tardi e non rovesciano il verso. Una versione precedente di questo
documento riportava 580 blocchi di `ddos`: il conteggio corretto è 586 con
questa definizione, 584 contando solo i blocchi in cui `ddos` è l'unico tipo.

L'additivo **quasi non si inverte**: dieci blocchi contro i 174 della
logistica. È anche il modello con l'AUROC congelato più alto, 0,891. Non è che
l'adattamento gli faccia male in modo particolare: è che **non ha quasi nulla
da correggere**, e paga soltanto il costo dell'aggiornamento.

Questo suggerisce una lettura che il pilota non può ancora dimostrare, e che va
tenuta distinta dai risultati misurati: la rigidità descritta al §0 — nove
parametri aggiornabili, forme delle spline fissate su A — potrebbe essere la
ragione per cui il verso non si rovescia, e insieme la ragione per cui non si
corregge. È la prima cosa da mettere alla prova.

## 3. Quello che l'adattamento peggiora, e secondo quale aggregazione

Il richiamo sui normali **medio per blocco** cala in tutti e tre i modelli e in
tutti e cinque i semi: da 0,431 a 0,220 per la logistica, da 0,732 a 0,264 per
l'MLP, da 0,483 a 0,297 per l'additivo. Essendo il complemento del tasso di
falsi allarmi, **sul FPR medio per blocco i falsi allarmi aumentano sempre**:
+0,211, +0,468 e +0,186, in cinque semi su cinque.

**Sul FPR complessivo il verso dipende dal modello**, e per due modelli su tre è
l'opposto:

| | FPR medio per blocco | FPR complessivo | falsi positivi |
|---|---|---|---|
| logistica | +0,211, sale in 5/5 | **−0,072, scende in 5/5** | −12.412 |
| MLP | +0,468, sale in 5/5 | +0,039, sale in 5/5 | +6.709 |
| additivo | +0,186, sale in 5/5 | **−0,054, scende in 5/5** | −9.297 |

Una versione precedente di questo documento affermava che i falsi allarmi
aumentano sempre, senza dichiarare l'aggregazione: sul complessivo è vero per
il solo MLP.

**Il meccanismo è lo stesso nei tre modelli**, e spiega la divergenza senza
invocare differenze di architettura. I normali sono concentrati: 246 blocchi su
916 contengono 154.771 dei 173.316 normali di C, e 41 blocchi non ne hanno
nessuno. L'adattamento **sposta falsi positivi dai blocchi ricchi ai blocchi
poveri**, in tutti e tre i casi; cambia solo quale dei due trasferimenti è più
grande (seme 42):

| | blocchi poveri (1–199 normali) | blocchi ricchi (≥200) | netto |
|---|---|---|---|
| logistica | +4.064 | −12.637 | **−8.573** |
| MLP | +9.788 | −4.132 | **+5.656** |
| additivo | +4.514 | −9.947 | **−5.433** |

Il FPR medio per blocco è dominato dalla prima colonna, il complessivo dalla
seconda. Per un rilevatore da mettere in esercizio contano entrambe, e vanno
riportate affiancate accanto a ogni risultato: la prima dice quanti *blocchi*
peggiorano, la seconda quanti *allarmi* in più arrivano all'operatore.

## 3bis. Con la soglia scelta su B, due di quei risultati si rovesciano

Il §3 è calcolato alla soglia zero. Scegliendo la soglia su B col massimo della
balanced accuracy — una per modello, identica e fissa nella coppia congelato /
adattivo — l'AUROC non cambia di un valore: **27.480 coppie verificate
identiche**, cinque semi per 916 blocchi per tre modelli per due stati, e gli
indici campionati identici in tutti i blocchi. Di quelle coppie 26.250 sono su
valori definiti; nelle altre 1.230 l'AUROC non è definita in nessuna delle due
varianti. Il §1 resta quindi valido parola
per parola. Cambiano le misure di decisione:

| | FPR medio per blocco | | FPR complessivo | | falsi positivi | |
|---|---:|---:|---:|---:|---:|---:|
| | a zero | su B | a zero | su B | a zero | su B |
| logistica | +0,211 | **−0,055** | −0,072 | −0,302 | −12.412 | −52.307 |
| MLP | +0,468 | +0,224 | +0,039 | **−0,132** | +6.709 | −22.879 |
| additivo | +0,186 | +0,248 | −0,054 | −0,022 | −9.297 | −3.890 |

Ogni verso è unanime sui cinque semi. Il richiamo sui normali medio per blocco,
congelato → adattivo, passa da 0,431 → 0,220 a **0,533 → 0,588** per la
logistica: con la soglia calibrata l'adattamento lo **migliora** invece di
dimezzarlo.

**Quello che il §3 attribuiva all'adattamento era in parte un effetto del punto
di decisione.** I modelli congelati alla soglia zero sono mal tarati su un
flusso al 98% di attacchi — il richiamo sui normali del modello additivo
congelato passa da 0,483 a 0,960 solo cambiando soglia — e il confronto a
soglia zero attribuiva all'aggiornamento una parte di quella mala taratura. La
conclusione «l'adattamento peggiora sempre il richiamo sui normali» sopravvive
soltanto per il **modello additivo**, su entrambe le aggregazioni, e per l'MLP
sulla sola media per blocco.

La soglia è scelta separatamente per ogni modello e per ogni seme, sull'intero
B, fra i valori distinti dei punteggi del congelato; fra i candidati a pari
merito si prende la **mediana inferiore**. Su questi dati la regola di parità
non è mai stata applicata, perché il massimo esatto è sempre raggiunto da un
candidato solo su circa 405.000 — ma con una tolleranza di confronto di 1e-6
pareggerebbero fino a 27 candidati, quindi la regola non è una formalità. È una
**convenzione deterministica**: che la soglia della logistica risulti identica
nei cinque semi segue dal fatto che il congelato è deterministico, non è una
misura di robustezza, e la stabilità non è misurata.

## 4. Il protocollo ha un collo di bottiglia misurato

Con blocchi da 10.000, budget dell'1% e memoria FIFO di 256, la regola del
salto per memoria monoclasse si attiva su **395 blocchi in media su 915, il
43,2%** (fra 374 e 414 sui cinque semi). Non è un incidente: i normali per
blocco hanno mediana 24 contro una media di 189, e il 5% dei blocchi più ricchi
contiene il 47,3% di tutti i normali, quindi un blocco mediano offre 0,24
normali attesi fra i cento campionati.

Il rimedio più ovvio non funziona: distribuendo lo stesso budget su dieci fasce
del punteggio della logistica congelata, i salti **salgono** da 393 a 421 e le
misure cambiano entro ±0,016, perché quel punteggio colloca quasi tutto nella
regione «attacco».

## 4bis. Quanto costa un aggiornamento, e cosa il risparmio non dimostra

Il confronto fra le politiche si misura in **numero di aggiornamenti**: in
media 45,2 per la logistica, 33,8 per l'MLP e 19,4 per l'additivo, contro 520.
Da quel numero non segue nessun guadagno di latenza né di memoria, e finché il
costo del singolo aggiornamento non era misurato la frase «aggiornare meno
costa meno» era un'inferenza, non un risultato. Adesso il rendiconto cronometra
il solo tratto che un sistema reale pagherebbe in più — applicare i parametri
correnti, calcolare la rappresentazione della memoria, rifittare — e i numeri
sono questi.

I numeri della tabella vengono da **una misura dedicata**: una sola corsa, seme
42, politica `ogni blocco`, **522 aggiornamenti applicati** e 393 tentativi
fermati dalla guardia, con la macchina altrimenti inattiva. Non sono una media
sui cinque semi.

| misura dedicata, seme 42 | ms per aggiornamento (mediana) | secondi in aggiornamento, 522 aggiornamenti | quota del replay |
|---|---:|---:|---:|
| logistica | 5,19 ms | 4,9 s | 5,6% |
| MLP | 5,27 ms | 3,1 s | 3,5% |
| additivo | 11,08 ms | 6,8 s | 7,8% |

Le stesse misure su tutti e venti i rendiconti, dove gli aggiornamenti applicati
sono 520 in media, stanno in `replay_evidenza/costi_aggiornamento.json`; là i
valori massimi sono inquinati dal carico sulla macchina, come detto sotto.

Passando da `ogni blocco` alla politica su evidenza il tempo speso negli
aggiornamenti, contando anche i tentativi fermati dalla guardia, scende da 16,8
a 1,7 secondi. Il replay completo su 9.150.673 righe ne costa 86,6, e il tempo
del replay è dominato dal **punteggio di ogni blocco, che si paga in tutte le
politiche**, congelato compreso.

I due replay differiscono però di 25,0 secondi, più dei 15,0 risparmiati sugli
aggiornamenti: i restanti 10,0 secondi non si leggono nel cronometro.
Profilando 200 blocchi delle due politiche, la differenza risulta spiegata
quasi per intero dalle due voci che l'aggiornamento comporta — i rifitting
della logistica, 587 chiamate contro 24, e la rappresentazione a B-spline della
memoria, 625 contro 431 — cioè dallo stesso lavoro che il cronometro misura. Il
residuo resta comunque
**non attribuito a una riga di codice**: su due CPU condivise non è prudente
attribuirlo, e in ogni caso si tratta di decine di secondi su un replay che ne
costa quasi novanta.

La memoria: il picco di memoria residente del processo è di circa
1.338 MiB, ma è dominato dagli array dello stream che il banco di prova tiene
in memoria; lo stato che l'adattamento muove è di 10.240 byte per la memoria
FIFO e, per modello, 8.264 byte di parametri per la logistica, 16.520
per l'MLP e 16.456 per l'additivo. **Non è la memoria che
userebbe un sistema in linea**, e non va presentata come tale.

La misura viene da una macchina virtuale con un Intel Xeon a 2,80 GHz, due CPU
utilizzabili e 8.031 MiB di RAM. Fra il minimo e il massimo osservati per uno
stesso modello c'è un fattore 29 per la logistica e per l'MLP e 9 per
l'additivo: è rumore di scheduling, non varianza del calcolo, e per questo si
riporta la mediana. I numeri valgono come ordine di grandezza e come rapporto
fra modelli, **non come misura di latenza di un sistema in esercizio**. Vengono
da due corse dedicate, con la macchina altrimenti inattiva
(`replay_evidenza/tempi/`); le venti corse pubblicate portano gli stessi campi,
ma sono state eseguite mentre sulla stessa macchina girava altro lavoro, e là
il rapporto arriva a 178 volte, con massimi fino a 384 ms.

In sintesi: il risparmio di aggiornamenti è reale e misurato in aggiornamenti;
tradotto in secondi è dell'ordine di 15 secondi su 86,6 secondi,
e **il numero di aggiornamenti non dimostra un guadagno di latenza né di
memoria**. Il guadagno della politica, dove c'è, è sull'AUROC.

## 5. Limiti, dichiarati

**Un solo flusso.** Tutto quanto sopra è misurato su C. Lo stream D non è stato
usato per addestrare né per valutare.

**D non è indipendente da A, B e C.** Il controllo diretto delle sovrapposizioni
è stato eseguito e l'esito è negativo: il **37,23%** delle righe di D e il
**58,03%** dei suoi normali hanno un vettore di feature che compare già in A, B
o C. Non è l'effetto di un vettore degenerato — servono 235 impronte per metà
di quelle righe e 54.821 per il 90% — né del solo tipo condiviso `password`:
almeno 592.463 righe di attacco sovrapposte sono di tipi che in D compaiono per
la prima volta. Una coincidenza di vettori non dice che sia la stessa sessione
o lo stesso host, e quella distinzione resta non stabilita. Il protocollo su D
è stato **fissato dal relatore**, e non come era stato proposto qui: **D intero
è l'analisi principale**; il sottoinsieme a vettore non visto — 2.228.436 righe
con 129.982 normali — è un'**analisi supplementare**, calcolata come maschera
sulle stesse predizioni del replay completo; nessuna riga viene esclusa dallo
stream, e nessuna scelta di modello o di politica si fa sui risultati di D. Il
dettaglio è nel protocollo, §5bis e §11 punto 5.

**Quale dei due confronti riportare non è deciso.** Il confronto a soglia zero
e quello con la soglia scelta su B danno conclusioni diverse sulle misure di
decisione per due modelli su tre (§3bis). Qui sono riportati entrambi, come le
due aggregazioni del FPR; quale sia quello da portare nell'articolo è da
concordare, non da scegliere da parte mia.

**L'inversione è misurata, non spiegata.** Sappiamo *quando* il congelato si
inverte — sui tipi `dos` e `injection` — ma non *perché* quella geometria
rovesci il verso, né se dipenda dal tipo o dalle sue caratteristiche di flusso.

**La ricchezza di normali è un indizio, non la causa.** Il guadagno sembrava
legato al numero di normali nel blocco: **sul seme 42** +0,409 nei blocchi con
almeno 200 normali contro −0,132 in quelli con meno di dieci. Sui cinque semi
il primo valore è stabile, da +0,406 a +0,409; il secondo no, da −0,132 a
−0,217.

Controllando per l'inversione la lettura cade: i blocchi ricchi **non**
invertiti danno da +0,062 a +0,077 sui cinque semi. Il **67%** dei blocchi
ricchi è invertito — 164 su 246, identico in tutti e cinque i semi, perché
l'insieme dipende dal solo modello congelato. La ricchezza coincide in larga
parte con l'inversione e non la sostituisce come spiegazione.

**Il campione noto è una ricostruzione.** Il controllo delle sovrapposizioni
degli otto input con il CSV noto usa impronte ricostruite, non l'archivio
originale, che non è pubblicato. Il riscontro è esatto sulla cardinalità —
92.330 uniche da 211.043 righe — ma resta una ricostruzione, e da sola non
dimostra l'indipendenza di D.

**Nessun costo hardware.** Solo CPU, come previsto in questa fase: il replay su
916 blocchi costa 87 secondi e l'addestramento iniziale dei tre modelli 33,
cioè 0,09 secondi per blocco. I valori sono quelli registrati nella corsa di
riferimento, `replay_C_seme42.json`, e variano di qualche secondo fra
esecuzioni: sono un ordine di grandezza, non una misura ripetibile. Il costo
del singolo aggiornamento, la memoria e l'hardware sono al §4bis.

## 6. L'esperimento proposto, eseguito: aggiornare solo su evidenza di inversione

Le quattro politiche sugli stessi blocchi sono in `figure/politiche_lr.png`,
`politiche_mlp.png` e `politiche_kan.png`: a sinistra gli attacchi non rilevati
cumulati, a destra la media progressiva dell'AUROC, il cui valore finale è
l'AUROC media di quella corsa — il **seme 42**, cioè il valore nel campo
`per_seme` di `confronto_politiche.json`, non la media sui cinque semi delle
tabelle di questa sezione.

Era la proposta del primo ciclo. È stata eseguita su C, con la soglia del
§3bis, parametri invariati, cinque semi, e con il controllo negativo.

**Il primo passo, che veniva prima di tutto: il verso è stimabile in esercizio?**
Sì, ma male, e soltanto fuori campione. La stima calcolata sui 256 esempi in
memoria **non riconosce mai** un blocco invertito — zero su 38, zero su 26,
zero su 50 — perché la memoria è l'insieme su cui l'aggiornamento è stato
fatto: è una stima in campione e vale quasi sempre circa 1,0. La stima
calcolata sulle cento righe campionate del blocco, con i punteggi già emessi, è
invece fuori campione e porta segnale: correlazione da +0,69 a +0,73 con
l'AUROC vera del blocco. Ma **esiste solo nel 41,6% dei blocchi**, perché fra
le cento righe campionate il blocco mediano non contiene **nessun** normale; si
degrada a +0,29 ÷ +0,36 al ritardo che conta, *j+2*; e come regola di decisione
ha
**precisione 15–18%**, cioè segnala cinque o sei blocchi per ogni inversione
vera.

**Il risparmio è di aggiornamenti, non di etichette.** Il budget dell'1% si
spende comunque: nelle tre politiche le etichette spese sono 91.506 per seme,
identiche. Cambia il numero di aggiornamenti applicati, e quello è il risparmio
misurabile; un risparmio di etichette richiederebbe un budget variabile, fuori
discussione in questa fase.

**Le quattro alternative**, medie sui cinque semi, AUROC e aggiornamenti
applicati:

| | congelato | ogni blocco | su evidenza | casuale, pareggiato |
|---|---:|---:|---:|---:|
| logistica | 0,7149 | 0,7739 (520) | **0,7756 (45)** | 0,7499 (45) |
| MLP | 0,7921 | 0,7664 (520) | **0,7956 (34)** | 0,7265 (34) |
| additivo | **0,8909** | 0,8068 (520) | 0,8358 (19) | 0,7384 (19) |

**Il confronto con il controllo negativo, formulato come i dati lo
sostengono.** Sull'AUROC la politica su evidenza batte il controllo casuale
**in media** su tutti e tre i modelli — +0,026, +0,069, +0,097 — e **non su
tutti i semi**: vince su quattro semi su cinque per la logistica e per l'MLP,
su cinque su cinque per l'additivo. Dove si rovescia è il **seme 44**:
logistica −0,032, MLP −0,013.

| evidenza − casuale, AUROC | 42 | 43 | 44 | 45 | 46 | media | vince |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistica | +0,029 | +0,039 | **−0,032** | +0,023 | +0,069 | +0,026 | 4/5 |
| MLP | +0,046 | +0,037 | **−0,013** | +0,192 | +0,084 | +0,069 | 4/5 |
| additivo | +0,084 | +0,131 | +0,049 | +0,117 | +0,107 | +0,097 | 5/5 |

**Sulle misure di decisione il confronto non regge allo stesso modo.** Sull'FPR
complessivo, per l'MLP il controllo casuale produce **meno** falsi allarmi su
quattro semi su cinque e anche in aggregato, 0,1625 contro 0,1844; per la
logistica la politica vince su quattro semi su cinque, per l'additivo su
cinque. Sul **richiamo degli attacchi**, che è la misura per cui il rilevatore
esiste, il sorteggio fa meglio su due modelli su tre: in aggregato 0,7088
contro 0,7789 per l'MLP e 0,6597 contro 0,7129 per l'additivo, mentre per la
logistica la politica resta avanti di poco, 0,8199 contro 0,8136.

| evidenza − casuale | AUROC | FPR complessivo | richiamo attacchi |
|---|---:|---:|---:|
| logistica | +0,026 (4/5) | −0,052 (4/5) | +0,006 (3/5) |
| MLP | +0,069 (4/5) | +0,022 (1/5) | −0,070 (2/5) |
| additivo | +0,097 (5/5) | −0,117 (5/5) | −0,053 (1/5) |

Fra parentesi i semi su cui il confronto è favorevole; sull'FPR il verso
favorevole è il valore più basso. Quindi: il criterio batte il sorteggio
**sull'ordinamento**, in media e non su tutti i semi, e su due modelli su tre
**perde attacchi**. Rispetto al congelato, invece, il richiamo sugli attacchi
migliora su tutti e cinque i semi per la logistica, +0,099 in media, e per
l'MLP, +0,309.

Sono risultati **esplorativi su C**: nessuna superiorità generale dimostrata,
nessuna significatività statistica calcolata.

**Il controllo casuale è un solo sorteggio per modello e per seme.** Non è una
distribuzione di pianificazioni casuali, quindi il confronto non ha un margine
di errore: cinque semi danno cinque confronti, non una stima della variabilità
del sorteggio. Dire di più richiederebbe più pianificazioni per seme.

**Batte anche l'aggiornamento continuo spendendo una frazione degli
aggiornamenti**: in media sui cinque semi 45,2 contro 520 per la logistica,
33,8 per l'MLP e 19,4 per l'additivo, cioè da 11,5 a 26,8 volte meno. Anche questo non è uniforme:
sulla logistica il confronto con `ogni blocco` si rovescia su tre semi su
cinque, pur restando in media positivo (+0,002). Per l'MLP la politica su
evidenza è l'unica che batte il congelato **in media** — 0,7956 contro 0,7921 —
ma per seme lo batte su tre su cinque, e sul seme 44 perde di 0,054. Per la
logistica tutte le politiche battono il congelato, e questo sì su tutti e
cinque i semi.

**Per il modello additivo nessuna politica batte il congelato**: 0,8909 contro
0,8358 della migliore, e questo su **tutti e cinque i semi** con tutte e tre le
politiche: sull'AUROC il congelato vince 15 confronti su 15, tre politiche per
cinque semi. È l'unico modello in cui l'adattamento non aiuta mai; confronti
uniformi ce ne sono altri — `confronto_politiche.json` ne marca quindici in
tutto — ma vanno tutti nella direzione opposta. Il §1 lo aveva
previsto — quel modello si inverte in 9 o 10 blocchi su 875 a seconda del
seme, non ha quasi nulla da correggere — e qui la previsione si verifica per una via diversa.

Il limite va detto accanto a ogni riga: il rivelatore ha precisione 15–18% ed
esiste nel 41,6% dei blocchi. **Funziona non perché sia preciso, ma perché
aggiornare raramente danneggia poco anche quando si sbaglia.** Un rivelatore
migliore richiede più normali etichettati per blocco, cioè la questione del
budget al §7.

**Costo**: venti esecuzioni da circa quattro minuti su CPU, meno di un'ora e
mezza in tutto.

## 7. Che cosa resta da concordare

- Quale dei due punti di decisione riportare nell'articolo, dato che per due
  modelli su tre cambiano il verso del risultato (§3bis). La regola di parità
  dichiarata non è mai servita, quindi non c'è nulla da concordare su quella.
- Il budget: i 395 salti dipendono dalla rarità dei normali per blocco, e la
  stratificazione sul punteggio non rimedia. Le alternative sono una memoria che
  conservi per classe o un budget maggiore sui blocchi poveri. La scheda fissa i
  parametri, quindi non li cambio senza indicazione.
- Conferma dei confini A/B/C/D dopo il controllo diretto delle sovrapposizioni.
  Come riportare la valutazione finale su D **non è più da concordare**: è
  deciso, nei termini del §5, e il protocollo lo recepisce in attesa del
  riscontro sulla versione aggiornata.
- Se il passo successivo sia **migliorare il rivelatore** — più normali
  etichettati per blocco, memoria per classe — oppure portare il risultato del
  §6 così com'è: una politica che con quaranta aggiornamenti su 915 blocchi fa
  quanto o meglio di cinquecentoventi e che, in media sui cinque semi, batte il
  controllo casuale sull'AUROC — con il seme in cui si rovescia e l'FPR dell'MLP
  dichiarati accanto al risultato.

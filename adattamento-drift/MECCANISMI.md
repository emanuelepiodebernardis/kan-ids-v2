# Meccanismi e giustificazioni

`RISULTATI.md` dice **cosa** e' stato misurato. Questo documento dice
**perche' quel numero viene fuori** e **perche' la procedura che l'ha
prodotto e' quella giusta.

Serve a due cose. La prima e' che un risultato spiegato da un meccanismo
regge alle domande di un revisore, mentre un risultato constatato regge
solo finche' nessuno chiede perche'. La seconda e' che i punti dove il
meccanismo e' gia' noto in letteratura sono i punti dove il lavoro va
agganciato a quella letteratura: qui sono marcati `[rif.]`, e le citazioni
si innestano dopo senza riscrivere il testo.

Convenzione: **osservazione** e' cio' che i dati dicono, con il numero;
**meccanismo** e' perche' lo dicono; **giustificazione** e' perche' la
procedura che l'ha misurato e' quella corretta.

I numeri senza marcatore vengono dai CSV rigenerati sotto il protocollo
corretto, 10 seed. Quelli marcati **⚠** vengono da script non ancora
rigenerati (`drift_trasferimenti.py` e le sei direzioni di
`drift_senza_etichette.py`), o da una politica non piu' presente negli
script: il meccanismo che spiegano non
cambia, ma la cifra va riconfermata prima di finire in una tabella
dell'articolo.

---

## 1. Il collasso cross-domain non e' un difetto del modello

**Osservazione.** Sei direzioni su sei collassano: da 0,2237 (`ton->unsw`) a
0,7368 (`unsw->bot`) di balanced accuracy, contro 0,8184-0,9931 in-domain.
Il degrado colpisce tutti e sei i modelli, non solo la KAN.

**Meccanismo.** Un modello di intrusion detection addestrato su un dominio
impara, insieme al segnale generale, le regolarita' del banco di prova che
l'ha generato: distribuzione dei servizi, tempi di cattura, mix di attacchi,
convenzioni dello strumento di estrazione. Queste regolarita' sono
perfettamente predittive dentro il dominio e assenti fuori, quindi il
modello che le ha assorbite non ha imparato una cosa sbagliata: ha imparato
una cosa **locale**. Il divario fra prestazione in-domain e cross-dataset e'
il fenomeno atteso, documentato in modo sistematico, non l'eccezione da
spiegare `[rif.]`.

**Perche' va detto cosi'.** Rende non necessario difendersi: il lavoro non
deve giustificare perche' i suoi modelli crollano, deve misurare **quanto**
e proporre cosa fare dopo. Sposta il contributo dal risultato negativo
(«crolla») a quello positivo («si recupera con 24 byte»).

**Giustificazione della procedura.** Per ogni sorgente il modello si
addestra **una volta sola** e si valuta su tutti e tre i domini: cosi' i
degradi sono confrontabili fra loro, perche' hanno lo stesso modello alle
spalle e differiscono solo per il bersaglio. Lo spazio armonizzato usa la
stessa formula di derivazione su tutti i domini, con le esclusioni
motivate una per una e il loro costo misurato invece che assunto. E il
vincolo cross-domain non e' dichiarato ma **imposto da un test**: si fitta
due volte sullo stesso source con target radicalmente diversi e si pretende
che tutto l'appreso sia identico bit per bit; introducendo la violazione
classica il test fallisce.

---

## 2. Il transfer invertito: il modello usa l'informazione col segno sbagliato

**Osservazione.** In `ton->unsw` e `unsw->ton` il ROC-AUC sul target e' 0,27
e 0,26, cioe' **sotto il caso**, con deviazione standard piccola (0,003 e
0,012 su 10 seed). Non e' rumore: e' un ordinamento sistematicamente
rovesciato.

**Meccanismo.** Un ROC-AUC sotto 0,5 non significa assenza di informazione —
significa informazione col segno invertito. Se non ci fosse informazione il
valore starebbe **a** 0,5, non a 0,26. La relazione fra una feature e la
classe si e' capovolta passando di dominio: una durata lunga che nel source
indicava traffico benigno nel target indica un attacco lento, e cosi' via.
Il modello continua a separare le due classi, le assegna solo alle etichette
sbagliate.

**Conseguenza operativa, ed e' il ponte verso la sezione 3.** Nessuna
ricalibrazione della soglia puo' rimediare a un ordinamento invertito,
perche' spostare la soglia non cambia l'ordine. Un metodo non supervisionato
basato sul prior o sull'entropia non puo' rimediare per lo stesso motivo.
Un guadagno per edge **si**, perche' puo' cambiare segno: e' l'unico
intervento fra quelli provati che ha il grado di liberta' giusto per il
difetto osservato.

**Osservazione nuova, dai dati a 10 seed.** Su `ton->bot` tutti e sei i
modelli stanno fra 0,38 e 0,55 di ROC-AUC sul target. Il fenomeno non e'
una proprieta' dell'architettura: e' una proprieta' della coppia di domini.
Va scritto cosi', perche' la lettura opposta — «la nostra architettura
degrada meno» — non e' sostenuta dai dati e sarebbe la formulazione piu'
forte del dato.

---

## 3. Perche' 13 coefficienti bastano

**Osservazione.** In `ton->bot` il modello deployato e' al caso (ROC-AUC
0,52) e i suoi **stessi edge**, solo ripesati con 32 etichette, arrivano a
0,86. Il rifit completo dei 101 parametri non fa meglio: il delta medio a
n=128 sulle cinque direzioni dove la selezione riesce e' **−0,0016**, e dopo
correzione per confronti multipli su 15 confronti ne restano due
significativi, uno per parte.

**Meccanismo.** E' l'istanza, su dati tabellari e in aritmetica intera, di
un fenomeno noto: quando un modello e' addestrato dove una correlazione di
comodo regge, cio' che si degrada fuori dominio non e' la **rappresentazione**
ma lo **strato che la combina**; le feature apprese restano di buona
qualita' e vengono solo pesate male, tanto che riaddestrare il solo strato
finale su un piccolo insieme bilanciato recupera gran parte del divario
`[rif.]`.

La KAN single-layer e' il caso estremo di questa struttura. Essendo
additiva — `z(x) = Σ φ_i(x_i) + Σ Tab_j[c_j]` — ogni edge produce un
contributo scalare indipendente, quindi «lo strato che combina» **e'** un
guadagno per edge piu' un termine noto: 13 numeri. Non e' una scelta di
comodo fra molte possibili, e' l'unico riassunto lineare che l'architettura
ammette.

**Tre predizioni del meccanismo, tutte confermate dai dati.**

1. *Il rifit completo non deve vincere*, perche' non c'e' informazione in
   piu' da recuperare nei coefficienti spline: sono gia' giusti. Confermato,
   pareggio in valore atteso.
2. *L'insieme di riponderazione deve essere bilanciato*, perche' e' lo
   sbilanciamento a produrre i pesi sbagliati. Confermato: il prelievo
   bilanciato e' il miglior selettore in ogni cella misurata, e tutto il
   collo di bottiglia del lavoro e' **trovare** la classe rara, non
   adattarsi una volta trovata.
3. *L'intervento deve poter cambiare segno*, per le due direzioni a
   ordinamento invertito. Confermato: sono fra quelle che recuperano di
   piu' (`ton->unsw` da 0,2237 a 0,7495, `unsw->ton` da 0,2984 a 0,8364).

**Cosa questo autorizza a scrivere, e cosa no.** Autorizza: *l'informazione
sopravvive al drift, la sua combinazione no, e per un modello additivo
ricombinarla costa 13 numeri.* Non autorizza: *la KAN e' piu' accurata delle
alternative.* Il confronto con l'ultimo strato di una MLP non lo sostiene
(sezione 8 di questo documento).

---

## 4. Perche' i metodi non supervisionati falliscono, uno per uno

**Osservazione.** Undici metodi senza etichette provati; uno solo (IM) da'
un guadagno parziale, e in una direzione su sei toglie 8 punti.

Il valore di questa sezione non e' la lista dei fallimenti: e' che **ogni
fallimento e' spiegato dalle ipotesi del metodo**, non dalla sfortuna. Un
metodo che fallisce fuori dalle proprie ipotesi non e' una prova contro il
metodo, ed e' invece una prova a favore dell'affermazione che su questo
problema le etichette servono.

| metodo | ipotesi che assume | perche' qui non vale |
|---|---|---|
| EM sul prior | lo shift e' di **solo prior**: `p(x\|y)` invariante | misurato falso: le marginali per feature non si sovrappongono (minimo 0,085 su `byte_rate`); la stima del prior arriva a 0,27 dove il vero e' 0,998, e in una direzione collassa a 0,000 |
| riallineamento dei quantili | il drift e' una **traslazione** delle marginali, annullabile riallineandole | il drift non e' una traslazione: riallineare peggiora il modello di partenza in entrambe le direzioni |
| TENT (entropia) | il minimo di entropia coincide con la decisione giusta | con prior estremo il minimo di entropia e' il **collasso su una classe**: frazione di positivi predetti 0,283 dove il vero e' 0,998 |
| TENT filtrato | filtrare i campioni ambigui evita il collasso | la correzione standard non basta: il collasso resta |
| IM / SHOT | entropia **piu'** un termine di diversita' che impedisce il collasso | il termine di diversita' presuppone classi bilanciate: e' il migliore dei quattro e l'unico che aiuta piu' spesso di quanto danneggi (4 direzioni su 6), ma in `bot->unsw` toglie 9 punti |
| soglia da prior / mediana / quantile | esiste una soglia che ripara | dove l'ordinamento e' invertito nessuna soglia ripara (sezione 2) |

**Il caso piu' forte e' un'impossibilita' dimostrata, non constatata.**
Per l'innesco a martingala conformal, con esponente ε ∈ (0,1) la soglia
raggiungibile tende a 1/e ≈ 0,368, mentre il p-value medio misurato sotto
deriva si ferma a 0,40. **Nessuna scelta di ε puo' funzionare con quel
segnale di conformita'**: non e' che non abbiamo trovato il valore giusto,
e' che non esiste. Questo trasforma un parametro non tarato in un limite
strutturale del segnale scelto, e indica anche il rimedio — cambiare
segnale, non cambiare ε.

**Un'osservazione che la misura a sei direzioni rende netta.** EM, TENT e
TENT filtrato migliorano in 3 direzioni su 6 e peggiorano nelle altre 3:
sono monete. E il loro contributo massimo cade dove il modello e' piu'
scalibrato — le due direzioni a ordinamento invertito — perche' li'
correggere il prior serve comunque, pur restando 25-30 punti sotto le 32
etichette. Un metodo che aiuta solo dove il modello e' rotto in modo
grossolano non e' un metodo di adattamento: e' una ricalibrazione.

**Cosa autorizza a scrivere.** Che su questo problema **un piccolo budget di
etichette e' necessario**, con undici metodi a sostenerlo e le ragioni di
ciascun fallimento. E' l'argomento che giustifica la scelta di progetto di
chiedere 32 etichette invece di inseguire l'adattamento autonomo: senza
questa sezione sarebbe una rinuncia, con questa sezione e' una conclusione.

---

## 5. Perche' la regola a margine fallisce (e la adattiva no)

**Osservazione, dai dati a 10 seed.** La regola a margine e' **la peggiore
di tutte** in `bot->ton`, sotto il prelievo casuale a ogni budget
(0,6039 / 0,6147 / 0,5517 / 0,5293), e raccoglie 0,9 normali su 32
etichette contro i 6,7 del prelievo casuale. In `ton->bot` e' battuta dalla
regola adattiva a ogni budget e produce un numero solo in 4 seed su 10 a
n=8.

**Meccanismo.** Selezionare per incertezza ottimizza l'**informativita'
locale** del campione, non la sua **rappresentativita'**. Il confine di
decisione e' una fetta sottile della distribuzione, e i punti che ci stanno
sopra non sono un campione di quella distribuzione. Ne segue un effetto in
due regimi opposti, che si legge tutto nella colonna dei normali raccolti:

- **Classe rara lontana dal confine** (`bot->ton`, 23,7% di normali): la
  fetta vicino al confine e' popolata dalla classe abbondante, e il margine
  raccoglie quasi solo quella. Il prelievo casuale, che non ottimizza
  nulla, fa meglio proprio perche' campiona la distribuzione vera.
- **Classe rara concentrata vicino al confine** (`ton->bot`, 0,013% di
  normali): il margine funziona, ma **esaurisce il bacino**: oltre le ~28
  normali disponibili vicino al confine il budget aggiuntivo raccoglie solo
  attacchi, e l'accuratezza smette di salire.

E' il bias di campionamento noto delle strategie a incertezza `[rif.]`, qui
in una forma particolarmente severa perche' i due regimi si presentano
**nello stesso lavoro, sugli stessi due dataset, a seconda della direzione**.

**Perche' la regola adattiva e' la risposta strutturale, non un espediente.**
Il dispositivo non sa in quale dei due regimi si trova, e il primo tentativo
di dedurlo — dalla frazione di positivi predetti — **fallisce in modo
istruttivo**: essendo scalibrato sul target, il modello predice il 44% di
attacchi dove la verita' e' il 99,987%. Quella quantita' e' inutilizzabile
proprio perche' il modello e' scalibrato, che e' il problema che si sta
cercando di risolvere.

Cio' che il dispositivo osserva **senza scalibrazione** sono le etichette
che sta gia' raccogliendo. Da qui la regola: preleva a caso; se le prime 8
ricadono tutte nella stessa classe, sei nel regime raro e passi al margine.
Il campione di sondaggio entra comunque nel training, quindi **il costo
aggiuntivo in etichette e' zero**. E' l'unica delle regole provate che
commuta fra i due regimi, e i dati lo confermano: e' la sola a stare fra le
migliori in entrambe le direzioni.

---

## 6. Perche' la deduplicazione intera sblocca una direzione

**Osservazione.** La catena intera recupera in `unsw->bot`, dove quella in
virgola mobile fallisce del tutto. Il motivo non e' l'aritmetica: e' la
selezione.

**Meccanismo.** Dopo la quantizzazione i flood di BoT-IoT collassano su
pattern identici: 200.477 righe si riducono a 32.118 pattern distinti, il
16%. ⚠ (conteggio dal protocollo precedente, da riconfermare.) Spendere etichette su duplicati esatti e' spreco puro, ma il punto e'
piu' fine di cosi': la ridondanza non e' uniforme sulle due classi, e
rimuoverla **riespone la coda**. Deduplicando sui contributi interi —
uguaglianza esatta, che in aritmetica intera costa nulla — e tenendo la
molteplicita' come peso, la resa passa da 1 a 10 normali su 32 etichette.

**L'osservazione da valorizzare, che il documento non aveva riconosciuto.**
La deduplicazione era nata per risparmiare etichette e si e' rivelata un
criterio di **selezione**. Piu' in generale: dove il modello in virgola
mobile e quello quantizzato differiscono, differiscono sui casi al limite —
cioe' proprio quelli informativi. Due modelli che si discostano sono un
segnale di difficolta' disponibile a costo quasi nullo, e qui e' stato usato
di fatto senza essere nominato `[rif.]`.

**E la misura a sei direzioni conferma il principio per una via
indipendente.** Scegliere le etichette sul punteggio gia' adattato da IM —
cioe' su un secondo modello che differisce dal primo — sblocca `unsw->bot`,
l'unica direzione dove ogni altra regola raccoglie zero normali: 0,8031 con
32 etichette contro 0,7258 non adattato, e +0,3031 sulla regola adattiva,
significativo dopo correzione. Nelle altre cinque direzioni non cambia
niente (tutti i p corretti a 1,00). Un secondo punto di vista sul punteggio
non serve a decidere meglio: serve a **dire dove guardare**, ed e' la stessa
cosa che fa la deduplicazione intera.

**Prova che la quantizzazione non e' il problema.** Con gli **stessi**
campioni, l'adattamento sui contributi interi fa 0,8981 contro 0,8136 del
float ⚠ (misura dal protocollo precedente). Il confronto a campioni identici e' la giustificazione procedurale
che rende l'affermazione «l'aritmetica intera non costa accuratezza»
verificabile invece che asserita: senza fissare i campioni, la differenza
fra selezione e stima resta confusa.

---

## 7. Perche' il buffer e' tutto nel riadattamento continuo

**Osservazione.** Riadattando a ogni batch **senza memoria** — rifacendo i
guadagni da zero sulle 32 etichette del batch corrente — la media e' 0,8134
con oscillazioni fra 0,44 e 0,96: **peggio del modello statico**.
Conservando le ultime 256 etichette e rifittando sull'intero buffer si passa
a 0,9433. ⚠ *(il confronto «senza buffer» viene da una politica poi rimossa dagli
script: non e' oggi rigenerabile, e quel numero va rimisurato o
l'affermazione riformulata senza cifra. Il resto della sezione 7 e' invece
verificato sui CSV a 10 seed e sei direzioni.)*

**Meccanismo.** Stimare 13 coefficienti da 32 osservazioni e' una stima ad
alta varianza; ripeterla da zero a ogni batch fa oscillare il modello
intorno alla soluzione giusta senza mai raggiungerla, e ogni oscillazione
costa accuratezza sul batch successivo. Il buffer non aggiunge informazione
nuova: **riduce la varianza dello stimatore** mediando su piu' batch, ed e'
per questo che il guadagno e' cosi' grande a parita' di etichette spese.

**Cosa autorizza a scrivere, ed e' un'affermazione di progetto piu' che di
accuratezza.** Un dispositivo che adatta senza memoria fa danno. E' una
raccomandazione operativa che vale piu' del numero, perche' la memoria e'
esattamente la risorsa che un microcontrollore non ha — ed e' la ragione
per cui i minimi quadrati ricorsivi, che tengono lo stesso effetto in 728
byte invece che in 12 KB, valgono lo sforzo di essere resi affidabili.

---

## 8. Costo, non accuratezza: cosa il confronto con le baseline autorizza

**Osservazione.** Contro l'ultimo strato di una MLP(16) — 17 parametri,
quanto i nostri 13 — il confronto appaiato per seed, corretto per confronti
multipli, **non produce nessuna differenza significativa**. Le entita' pero'
sono asimmetriche di un ordine di grandezza: dove la KAN vince (`ton->bot`)
vince di +0,20; dove perde (`bot->ton`) perde di 0,02-0,06.

**Meccanismo.** La ragione per cui l'MLP tiene il passo e' la stessa per cui
la KAN funziona: anche lei ha un ultimo strato piccolo, quindi anche lei
ammette un aggiornamento minimo strutturale. Il meccanismo della sezione 3
**non e' esclusivo della KAN** — vale per ogni architettura che separi una
rappresentazione da una combinazione lineare piccola. Cio' che distingue
la KAN non e' l'esistenza dell'aggiornamento, e' la sua **forma**: e' una
tabella di moltiplicatori Q15 che il kernel intero gia' usa, quindi
riscriverla non richiede di toccare il firmware.

Contro gli ensemble ad albero, invece, il vantaggio e' reale e strutturale:
l'aggiornamento minimo di LightGBM e' un peso per albero, 401 parametri, e
in `ton->bot` non recupera.

**Cosa va scritto, quindi.** L'argomento del lavoro e' di **costo**, non di
accuratezza — 24 byte contro 250, nessun riaddestramento sul dispositivo — e
il fatto che non ci sia nemmeno uno svantaggio di accuratezza da compensare
e' esattamente cio' che lo rende difendibile. Rivendicare accuratezza
sarebbe la formulazione piu' forte del dato, e la tabella che la smentisce
sta dentro il lavoro stesso.

---

## 9. Informativita' e adattabilita' sono due cose diverse

**Osservazione.** Ridurre lo spazio armonizzato da 13+2 a 6+2 feature costa
**0,0089 in-domain** ma fino a **0,239 sul risultato adattato**
(`ton->bot` a 32 etichette: 0,8517 nello spazio ricco contro 0,6131 nel
ridotto; a 128 etichette −0,227).

**Meccanismo.** Le sette feature che cadono sono quelle direzionali — le
asimmetrie fra le due direzioni del flusso, i payload medi per direzione.
Una feature serve all'**accuratezza** se separa le classi dentro il dominio;
serve all'**adattamento** se il suo contributo cambia fra i domini in modo
che un guadagno per edge possa correggere. Sono due proprieta' distinte, e
possono non coincidere: una feature che discrimina bene e allo stesso modo
in entrambi i domini e' ottima per l'accuratezza e inutile per
l'adattamento, perche' non c'e' niente da correggere.

**Perche' vale come risultato a se'.** Non era il risultato cercato — la
misura era nata per decidere se accettare il costo di un terzo dominio — e
dice qualcosa che una valutazione in-domain non puo' vedere: **quali feature
tenere se il modello dovra' adattarsi**. Una selezione ottimizzata sulla
sola accuratezza le avrebbe scartate quasi a costo zero.

**Giustificazione della procedura.** Il costo e' stato misurato proiettando
i domini esistenti sullo spazio ridotto a parita' di righe, seed e modelli:
solo meno colonne. Senza quel controllo, il crollo osservato aggiungendo un
dominio nuovo in uno spazio piu' povero sarebbe stato inattribuibile — non
si sarebbe potuto distinguere l'effetto del dominio da quello dello spazio.

---

## 10. Quando una costante e' determinata dal problema e quando no

**Osservazione.** Riselezionate onestamente su dati mai riportati, le due
costanti del lavoro si comportano in modo opposto. Il **ridge** ha un massimo
**interno** alla griglia con curva unimodale (0,75 / 0,82 / **0,92** / 0,90 /
0,84) e batte quattro alternative su sei dopo correzione. Le **iterazioni**
hanno argmax al **bordo** con curva monotona crescente, nessuna differenza
significativa (p corretti tutti a 0,62) e tutte le configurazioni entro un
errore standard.

**Meccanismo.** Sono due tipi diversi di parametro. Il ridge controlla un
fenomeno reale del problema — la quasi-separazione al primo aggiornamento,
quando 13 coefficienti vengono stimati da poche etichette che li separano
perfettamente — quindi esiste un valore giusto, troppo poco diverge e troppo
schiaccia. Le iterazioni non controllano nulla del problema: sono un
**budget di calcolo**, e piu' se ne spendono meglio si converge, fino a un
plateau. Un budget non ha un ottimo interno per costruzione, e cercarne uno
produce esattamente cio' che si osserva: argmax al bordo e nessuna
separazione statistica.

**Cosa ne segue, ed e' una scelta di progetto.** Per il ridge si sceglie il
valore. Per le iterazioni **non si sceglie un numero**: si adotta il passo
adattivo, che si dimezza sui plateau e si ferma sul punto fisso, spendendo
1.800-4.200 iterazioni sulle direzioni facili e fino a 12.800 sulla piu'
difficile. E' la risposta strutturale a un parametro che il problema non
determina, e ha il vantaggio di coprire il rischio di coda — il seed raro
che a 2.000 iterazioni non converge — senza pagare il caso peggiore
dappertutto.

**Perche' questo episodio vale come argomento metodologico.** Le due
costanti erano state scelte con lo stesso metodo sbagliato, guardando i
numeri poi riportati. Rifacendo la scelta correttamente, una regge e una
cade. **Guardando i numeri sbagliati non era possibile sapere in quale dei
due casi ci si trovava**: e' l'argomento piu' forte per correggere un
protocollo invece di difenderlo, ed e' verificabile perche' entrambe le
misure sono nel repository.

---

## 11. Il collo di bottiglia, nominato per quello che e'

Attraversa tutto il lavoro e conviene dirlo una volta con chiarezza invece
che sei volte come eccezione locale: **il limite di questo metodo non e'
l'adattamento, e' la raccolta delle etichette.**

Le prove convergono da cinque sezioni diverse:

- `unsw->bot` fallisce in **10 seed su 10**, e non perche' l'adattamento sia
  incapace: perche' la selezione non raccoglie **mai** un normale. BoT-IoT
  ha 477 normali su 3,67 milioni di righe.
- Le direzioni che riescono lo fanno in 6-10 seed su 10, e i seed persi sono
  quelli in cui la selezione ha trovato una sola classe.
- Il prelievo **bilanciato**, che usa un'informazione che il dispositivo non
  ha, e' il miglior selettore in ogni cella: il divario fra lui e le regole
  applicabili misura esattamente il costo di non sapere dove cercare.
- La regola a margine fallisce in entrambi i regimi per la stessa ragione
  (sezione 5), e la deduplicazione intera funziona per la ragione
  complementare (sezione 6).
- A rapporto di undersampling estremo le affermazioni che cedono cedono
  tutte con lo stesso meccanismo: la selezione trova piu' spesso una sola
  classe.

**E ha due rimedi misurati, non solo una direzione indicata.** Su
`unsw->bot`, dove la regola normale raccoglie zero normali in tutti e 10 i
seed, il k-center ne raccoglie 12,4 con 32 etichette e porta a 0,7222, e IM
come selettore porta a **0,8031** — sopra il modello non adattato, con
p=0,0065. Entrambi sono inutili dove la regola normale funziona: e' il
profilo del ripiego, non del sostituto.

**Perche' e' un buon modo di chiudere il lavoro e non una debolezza.**
Un limite diagnosticato in modo convergente da cinque misure indipendenti,
con il meccanismo identificato e **due rimedi misurati**, e' un risultato.
Un limite scoperto da un revisore non lo e'.

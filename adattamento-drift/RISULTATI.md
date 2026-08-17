# Colmare il gap cross-domain — primi risultati

Due esperimenti, entrambi su 3 seed, entrambi con il vincolo del punto 3
rispettato: il modello deployato è addestrato **solo** sul source, e ogni riga
del target usata per adattarlo è esclusa dalla valutazione.

---

## 1. Diagnosi: il collasso non è una soglia sbagliata

Prima di provare qualunque tecnica, la domanda da risolvere era: **quanto del
crollo è rappresentazione rotta e quanto è solo soglia mal posizionata?** È la
differenza fra una correzione da un byte e una da 250, e decide quali metodi
della letteratura valga la pena testare.

Balanced accuracy sul target (0,50 = caso):

| Direzione | Modello | oggi | soglia oracolo | ROC-AUC target |
|---|---|---|---|---|
| TON→BoT | KAN 1L | 0,5563 | 0,6294 | **0,5451** |
| TON→BoT | LightGBM | 0,4797 | 0,6269 | 0,5414 |
| TON→BoT | DecisionTree | 0,5466 | 0,6073 | 0,4339 |
| BoT→TON | KAN 1L | 0,5989 | **0,8280** | **0,8200** |
| BoT→TON | LightGBM | 0,7171 | 0,8107 | 0,7660 |
| BoT→TON | MLP(16) | 0,7343 | 0,7397 | 0,6053 |

La "soglia oracolo" usa le etichette del target: non è una tecnica, è il
**tetto superiore** di qualunque metodo che si limiti a spostare la soglia.

**Le due direzioni sono problemi diversi, e il report attuale li tratta come
uno solo.**

- **BoT→TON** conserva l'ordinamento: ROC-AUC 0,82 e la sola soglia porterebbe
  da 0,599 a 0,828. È un problema di calibrazione.
- **TON→BoT** ha ROC-AUC 0,54, cioè **al caso**. Anche la soglia perfetta si
  ferma a 0,62. Nessuna ricalibrazione della soglia salva questa direzione.

Nota per il paper: la KAN single-layer ha il **miglior ROC-AUC sul target in
entrambe le direzioni**, pur essendo ultima in-domain. È un secondo argomento,
indipendente da quello già nel report, a favore della struttura additiva.

## 2. Quello che non funziona: l'adattamento senza etichette

Testate tre regole di soglia non supervisionate (allineamento del prior, delle
mediane, dei quantili) e il rifit della mappa quantile sul target non
etichettato — l'analogo non parametrico di CORAL, che sarebbe stato la
soluzione ideale perché a costo zero in etichette:

| Metodo | etichette | TON→BoT | BoT→TON |
|---|---|---|---|
| modello deployato | 0 | 0,5632 | 0,5989 |
| + soglia da prior | 0 | 0,4814 | 0,5525 |
| quantili rifittati sul target | 0 | 0,5081 | 0,5264 |

**Tutte peggiorano il modello.** Il drift qui non è una traslazione delle
marginali che si possa annullare riallineandole: coerente con la
sovrapposizione degli istogrammi già misurata (0,085 su `byte_rate`). Va detto
esplicitamente, perché è il primo tentativo che chiunque farebbe.

## 3. Quello che funziona: 13 coefficienti

> **Nota in avanti**: il punto 2 di questa sezione ("aggiornare poco batte
> riaddestrare tutto") era misurato su due sole direzioni. La sezione 11 lo
> ha gia' indebolito a sei direzioni, e a 10 seed (con test t appaiati per
> seed, non un conteggio) l'affermazione sull'**accuratezza** non regge
> piu': in valore atteso i 13 coefficienti e il rifit completo si
> annullano (delta medio −0,002 su cinque direzioni a n=128). Quello che
> resta, e regge, e' un'affermazione di **costo** — 24 byte contro 250,
> nessun riaddestramento sul dispositivo — non di accuratezza. Questa
> sezione non e' stata riscritta: resta come documento di come si e'
> arrivati alla prima versione del claim, con l'evoluzione successiva
> tracciata nelle sezioni 11 e 16.

La KAN single-layer è additiva — `z(x) = Σ φ_i(x_i) + Σ Tab_j[c_j]` — quindi
ogni edge produce un contributo scalare indipendente. L'intervento minimo che
non tocca gli edge è riscrivere **un guadagno per edge più un termine noto**:

    z'(x) = Σ a_i · φ_i(x_i) + b        →  13 numeri

Balanced accuracy sul target, media su 3 seed, valutata sulle righe **non**
usate per l'adattamento:

### TON→BoT (la direzione che sembrava irrecuperabile)

| Intervento | coeff. | 8 etich. | 32 | 128 | 512 |
|---|---|---|---|---|---|
| solo soglia | 1 | 0,5474 | 0,5418 | 0,5355 | 0,5293 |
| **gain per edge** | **13** | **0,9206** | **0,9251** | 0,9416 | 0,9713 |
| rifit completo | 101 | 0,8467 | 0,9368 | 0,9652 | 0,9844 |

Partenza: 0,5632. **Con 8 etichette e 13 coefficienti si passa da 0,563 a
0,921** (dev.std 0,011 su 3 seed).

### BoT→TON

| Intervento | coeff. | 8 etich. | 32 | 128 | 512 |
|---|---|---|---|---|---|
| solo soglia | 1 | 0,7878 | 0,8095 | 0,8159 | 0,8163 |
| **gain per edge** | **13** | 0,7595 | **0,9001** | 0,9136 | 0,9337 |
| rifit completo | 101 | 0,6272 | 0,8691 | 0,9428 | 0,9666 |

Partenza: 0,5989.

### Tre cose che questi numeri dicono

**1. L'informazione negli edge sopravvive al drift; la loro combinazione no.**
In TON→BoT il modello è al caso (AUC 0,545) eppure i suoi stessi edge, solo
ripesati, arrivano a 0,92. Non è un modello che ha imparato la cosa sbagliata:
è un modello che ha imparato le cose giuste con i pesi sbagliati. Questa è la
spiegazione del "perché degrada" che il professore chiedeva al punto 4, e non
si vedeva dalle matrici di confusione.

**2. Con budget piccolo, aggiornare poco batte riaddestrare tutto.** A 8 e 32
etichette il rifit completo (101 parametri) fa peggio dell'aggiornamento a 13:
0,847 contro 0,921 in TON→BoT, 0,627 contro 0,760 in BoT→TON. Il vantaggio si
inverte solo a 512. È esattamente l'argomento del follow-up — *aggiornare
localmente solo una piccola parte dei coefficienti* — misurato invece che
supposto.

**3. È un'affermazione specifica dell'architettura.** L'intervento a 13
coefficienti esiste perché il modello è additivo e ogni edge è isolabile.
Su un ensemble da 400 alberi o su una MLP non c'è un equivalente da 13 numeri:
la controparte è il rifit completo. Il confronto con le baseline su questo asse
è la prossima cosa da misurare.

---

## 4. Chi sceglie le etichette

L'adattamento sopra usa **etichette bilanciate**, cioè metà per classe: un'
informazione che il dispositivo non ha. Su BoT-IoT le normali sono lo 0,013%,
quindi in 32 flussi presi a caso ce ne sono 0,004 attese. Finché il prelievo è
bilanciato, il risultato resta da laboratorio.

Confrontate sei regole di selezione, tutte applicabili in campo perché
guardano solo il punteggio del modello, mai le etichette. Balanced accuracy
dopo l'aggiornamento, media su 3 seed:

| Regola | TON→BoT (0,013% normali) | | BoT→TON (23,7% normali) | |
|---|---|---|---|---|
| | n=8 | n=32 | n=8 | n=32 |
| casuale | *impossibile* | *impossibile* | 0,6852 | 0,8912 |
| conformal (insiemi anomali) | *impossibile* | *impossibile* | 0,6649 | 0,7049 |
| margine (\|z\| minimo) | **0,9002** | **0,9056** | 0,5717 | 0,7851 |
| metà margine metà casuale | 0,7575 | 0,8448 | 0,7496 | 0,6863 |
| **adattiva** | 0,7990 | **0,8954** | **0,7627** | 0,8465 |
| bilanciato *(non applicabile)* | 0,9191 | 0,9218 | 0,7562 | 0,8684 |

*impossibile* = la regola raccoglie **zero** normali, quindi il dispositivo non
ha due classi e non può aggiornare nulla.

**Cautela da dichiarare, non ancora verificata**: le sezioni 15 e 11 hanno
misurato che "impossibile"/"fallisce" su un campione di 3 seed puo' essere
un artefatto del campione (raro ma non zero con piu' seed) oppure un
risultato genuino, e le due cose vanno distinte rimisurando, non
assumendole. Questa tabella viene da `scripts/drift_sampling.py`, che **non
e' stato rilanciato a 10 seed** in quel lavoro (erano in scope
`drift_int_adapt`, `drift_graduale`/`drift_graduale_int` e
`cross_domain`/`tre_domini`, non `drift_sampling`): il verdetto
*impossibile* per `casuale` e `conformal` su TON→BoT qui sopra e' quindi
ancora un risultato a 3 seed, non riverificato.

**Il campionamento conformal non funziona, ed è istruttivo il perché.** Gli
insiemi di predizione anomali selezionano flussi anomali, ma su BoT-IoT gli
anomali sono *attacchi* anomali: zero normali raccolte, esattamente come il
prelievo casuale. La conformal resta valida come **innesco** del
riaddestramento — non come selettore.

**Funziona la regola più banale**: gli n flussi più vicini al confine di
decisione. Con 8 etichette ne pesca 6,7 normali su 477 presenti in 3,67 M di
righe: un arricchimento di tre ordini di grandezza. Ma è anche la peggiore
nell'altra direzione, dove le normali abbondano e il confine è una fetta non
rappresentativa.

### La regola adattiva

Il dispositivo non sa in quale regime si trova. Il primo tentativo — dedurlo
dalla frazione di positivi predetti — **fallisce**, e vale la pena riportarlo:
su BoT-IoT il modello predice il 44% di attacchi dove la verità è il 99,987%.
Essendo scalibrato sul target non si accorge di essere nel regime estremo.

Quello che osserva davvero sono le etichette che sta già raccogliendo:

> preleva a caso; se le prime 8 ricadono tutte nella stessa classe, passa al
> margine. Il campione di sondaggio entra comunque nel training.

Costo aggiuntivo: **zero etichette**. Con n=32 arriva a 0,8954 (TON→BoT) e
0,8465 (BoT→TON), cioè **entro 2-3 punti dal prelievo bilanciato in entrambe
le direzioni**, senza usarne l'informazione. È la prima versione del metodo
interamente eseguibile su dispositivo.

### Due limiti da dichiarare

- Oltre le ~32 etichette il margine **peggiora** (0,9056 → 0,7805 a 512): il
  bacino di normali vicino al confine si esaurisce a ~21 campioni e il resto
  del budget aggiunge solo attacchi. Il budget va limitato, non massimizzato —
  controintuitivo e da verificare su un terzo dominio.
- La dispersione fra seed della regola adattiva è alta a budget piccolo
  (dev.std 0,086 a n=8, TON→BoT): con 8 etichette l'esito dipende da quali.

---

## 5. È la struttura o è il budget?

Il claim implicito era: 13 coefficienti bastano *perché* la KAN è additiva. Se
anche le baseline recuperassero altrettanto con un aggiornamento altrettanto
piccolo, il merito sarebbe delle etichette e non dell'architettura.

A ogni modello si è dato lo stesso budget, le etichette scelte dalla regola
adattiva usando il **suo** punteggio, e il **suo** aggiornamento minimo
strutturale — non il rifit completo. Il numero di parametri non è arbitrario:
è quanti pezzi additivi indipendenti ha quell'architettura.

| Direzione | Modello | par. | n=8 | n=32 | n=128 | n=512 |
|---|---|---|---|---|---|---|
| **TON→BoT** | **KAN single-layer** | **13** | **0,8318** | **0,9067** | **0,9163** | **0,9300** |
| | MLP(16), ultimo strato | 17 | 0,7857 | 0,6541 | 0,7716 | 0,7281 |
| | LightGBM, un peso per albero | 401 | — | 0,6601 | 0,7021 | 0,6845 |
| | XGBoost, un peso per albero | 301 | 0,5084 | 0,5697 | 0,7187 | 0,6915 |
| | Albero d=5, valori delle foglie | 16 | 0,5263 | 0,5058 | 0,9293 | 0,7654 |
| **BoT→TON** | KAN single-layer | 13 | **0,7249** | **0,8938** | 0,9180 | 0,9374 |
| | MLP(16), ultimo strato | 17 | 0,6725 | 0,8792 | **0,9245** | **0,9411** |
| | LightGBM, un peso per albero | 401 | 0,5424 | 0,8849 | 0,9226 | 0,9385 |
| | XGBoost, un peso per albero | 301 | 0,6753 | 0,8433 | 0,9001 | 0,9186 |
| | Albero d=5, valori delle foglie | 14 | 0,7047 | 0,8273 | 0,8232 | 0,8307 |

La decomposizione per albero è esatta: la somma delle colonne riproduce il
punteggio grezzo a meno di 1e-14 su LightGBM. Si era valutata anche la
decomposizione per feature degli ensemble (contributi tipo SHAP, 14 numeri),
ma costa 1,7 ms per riga e richiede comunque l'intero ensemble a runtime: non
è un aggiornamento da 14 coefficienti riscrivibili, è un ricalcolo.

**La risposta è: dipende dalla direzione, e va detto così.**

Nella direzione difficile (TON→BoT) la KAN vince a ogni budget, con il numero
di parametri più basso: 0,9067 con 32 etichette contro 0,6601 di LightGBM che
ne aggiorna 401. Lì il vantaggio è strutturale.

Nella direzione facile (BoT→TON) tutti convergono: a 512 etichette MLP 0,9411,
LightGBM 0,9385, KAN 0,9374 — differenze dentro il rumore fra seed. Il claim
si riduce a **stessa accuratezza con 30 volte meno coefficienti da riscrivere**,
che per un MCU resta l'argomento decisivo ma è un'affermazione diversa, e più
debole, di quella che verrebbe voglia di scrivere.

Un risultato collaterale utile: per la KAN l'aggiornamento a 13 parametri
**batte il proprio rifit completo** in TON→BoT a ogni budget (0,9300 contro
0,7508 a n=512). Il rifit completo su poche righe del target dimentica il
source; ripesare gli edge lo conserva.

---

## 6. Aggiornamento integer-only: 24 byte, bit-esatto

Il kernel intero già deployato calcola `z = Σ (acc_i · MULT[i]) >> 15`, dove
`MULT` è un moltiplicatore Q15 per edge. **La ripesatura degli edge non
richiede di toccare il firmware**: si riscrive quella tabella. Per 10 edge
numerici e 2 categorici sono 12 int16 = **24 byte**, più il termine noto.

Verifica: 200 golden vector, riferimento Python contro C compilato,
`logit diversi: 0, decisioni diverse: 0`.

| | TON→BoT | BoT→TON |
|---|---|---|
| float, non adattato | 0,5632 | 0,5989 |
| intero, non adattato | 0,4622 | 0,7037 |
| intero + guadagni stimati in float, n=128 | 0,8328 | 0,8609 |
| **intero + guadagni stimati in interi, n=128** | **0,7321** | **0,8602** |

La stima intera dei guadagni — discesa del gradiente con sigmoide a LUT,
pesi di classe interi, passo scelto per shift — **pareggia quella in virgola
mobile in BoT→TON** (0,8602 contro 0,8609) e resta indietro in TON→BoT. Non è
parità piena, ma l'intera catena, stima compresa, gira senza un solo float.

### Due ostacoli che il modello float non ha

**I pareggi.** Il punteggio quantizzato è grossolano: 225 righe condividevano
lo stesso |z| minimo. `argsort` rompe i pari per indice, quindi la selezione
per margine prendeva le prime righe del file invece di un campione. Con
rottura casuale dei pareggi il problema sparisce; senza, l'adattamento non
partiva proprio (zero normali raccolte).

**I duplicati.** Dopo la quantizzazione i flood di BoT-IoT collassano su
pattern identici: 200 477 righe si riducono a **32 118 pattern distinti**
(16%). Spendere etichette su duplicati esatti è spreco puro. Deduplicando sui
contributi interi — uguaglianza esatta, che in interi costa nulla — e tenendo
la molteplicità di ciascun pattern come peso intero, la resa passa da 1 a 10
normali su 32 etichette.

Con gli **stessi** campioni, l'adattamento sui contributi interi fa 0,8981
contro 0,8136 del float: la quantizzazione non danneggia l'adattabilità. Tutta
la perdita era nella selezione.

## 7. Deriva graduale

Lo stream parte dal dominio sorgente e ci mescola una frazione crescente di
flussi target, 20 batch da 20 000 flussi. Quattro politiche.

**Quando si rompe.** Il modello statico degrada in modo regolare: in TON→BoT
scende sotto 0,95 al **10% di contaminazione**, sotto 0,90 al 26%, sotto 0,85
al 42%, fino a 0,51 a contaminazione piena.

| Politica | TON→BoT | BoT→TON | adattamenti | etichette |
|---|---|---|---|---|
| statico | 0,8169 | 0,8467 | 0 | 0 |
| **ogni batch, buffer 256** | **0,9433** | 0,8501 | 19 | 608 |
| su innesco conformal | 0,8326 | 0,8535 | 5 / 16 | 160 / 512 |
| oracolo (etichette bilanciate) | 0,9402 | **0,9256** | 19 | 608 |

**Il buffer è tutto.** Nella prima versione ogni riadattamento rifaceva i
guadagni da zero sulle 32 etichette del batch corrente: media 0,8134, cioè
**peggio del modello statico**, con oscillazioni fra 0,44 e 0,96. Conservando
le ultime 256 etichette e rifittando sull'intero buffer si passa a 0,9433,
sopra l'oracolo. Un dispositivo che adatta senza memoria fa danno.

**L'innesco conformal ha la sensibilità invertita.** Scatta 5 volte su 20 in
TON→BoT, dove l'adattamento vale 12 punti, e 16 volte su 20 in BoT→TON, dove
ne vale meno di uno. Come rilevatore di deriva su questo segnale non funziona:
serve una statistica diversa, e per ora l'affermazione "la conformal fornisce
il segnale di innesco" va tolta dal report.

**Una trappola metodologica.** A contaminazione piena tutte le politiche
calano di colpo (0,9672 → 0,8682). La balanced accuracy su uno stream misto è
ottimista, perché la parte sorgente resta facile: misurare la deriva su una
miscela e riportare un numero solo nasconde il caso peggiore.

## 8. Si può fare a meno delle etichette? Quattro metodi dalla letteratura

Il limite più serio del risultato è che l'aggiornamento richiede etichette del
target: è active learning con un operatore nel ciclo, non adattamento
autonomo. La letteratura sull'adattamento a tempo di test esiste proprio per
questo, e i suoi quattro filoni principali si trasportano direttamente sui
nostri 13 parametri — che *sono* i parametri affini che quei metodi
aggiornano.

Balanced accuracy, media su 3 seed:

| Metodo | etichette | BoT→TON | TON→BoT |
|---|---|---|---|
| non adattato | 0 | 0,5989 | 0,5632 |
| EM sul prior (Saerens/MLLS) | 0 | 0,5000 | 0,5606 |
| TENT (minimizzazione dell'entropia) | 0 | 0,6187 | 0,4400 |
| TENT filtrato (campioni affidabili) | 0 | 0,5262 | 0,5439 |
| **IM / SHOT (entropia − diversità)** | **0** | **0,7613** | **0,6411** |
| 8 etichette | 8 | 0,7249 | 0,7218 |
| **32 etichette** | **32** | **0,8939** | **0,9085** |
| IM come selettore + 32 etichette | 32 | 0,8939 | 0,6620 |
| IM come prior + 32 etichette | 32 | 0,7928 | 0,7124 |

**Uno dei quattro funziona, ed è quello che avevo previsto fallisse.** Il
termine di diversità di IM presuppone classi bilanciate; con BoT-IoT al
99,987% di attacchi mi aspettavo facesse danno. È invece l'unico metodo che
migliora in *entrambe* le direzioni, di 16 e 8 punti. In BoT→TON vale più di
8 etichette (0,7613 contro 0,7249): l'unica cosa gratis che abbiamo trovato in
tutto il lavoro.

**Gli altri tre falliscono, ciascuno a modo suo.** L'EM sul prior stima
0,27 dove il vero è 0,998, e in BoT→TON stima 0,000 collassando su
"tutto normale": lo shift qui non è di solo prior, quindi il metodo è
applicato fuori dalle sue ipotesi. TENT peggiora in TON→BoT (0,4400) — è il
collasso su una classe documentato in letteratura, e la frazione di positivi
predetti lo mostra: 0,24. Filtrare i campioni ambigui, che è la correzione
standard, non basta.

**Combinare i due mondi non aiuta.** Né usare IM per scegliere i campioni né
usarlo come prior della stima supervisionata batte le 32 etichette da sole. Le
due fonti di informazione non si sommano: quando le etichette ci sono,
dominano.

### Cosa si è chiuso con questo

Sono ormai **sette** i metodi senza etichette provati e falliti o quasi:
riallineamento dei quantili sul target, tre regole di soglia (prior, mediana,
quantile), EM sul prior, TENT e TENT filtrato. Uno solo dà un guadagno reale
ma parziale. Non è più una lacuna del nostro lavoro: è un risultato, e
sostiene l'affermazione che su questo problema **un piccolo budget di
etichette è necessario**, non una scorciatoia che non abbiamo saputo evitare.

Per il paper questo cambia l'inquadramento: non "KAN integer-only che si
adatta da sola", ma "KAN integer-only che si adatta con 32 etichette e 24
byte, dove i metodi non supervisionati standard recuperano al più un terzo
del gap".

## 9. Metodologie prese da campi che non c'entrano

I metodi della sezione 8 venivano tutti dalla stessa letteratura, quella
dell'adattamento a tempo di test. Qui invece i nostri problemi sono stati
scomposti e cercati **fuori** da quella letteratura, in campi dove qualcuno li
ha già risolti per motivi suoi.

| Nostro problema | Campo di provenienza | Metodo |
|---|---|---|
| 13 parametri da 8 etichette | biostatistica degli eventi rari | Firth / prior di Jeffreys |
| trovare la classe rara col budget | scoperta di farmaci (*active search*) | k-center greedy / core-set |
| il buffer non entra in 8 KB di SRAM | controllo adattivo | minimi quadrati ricorsivi |
| innesco senza etichette | test sequenziali (Vovk) | martingala conformal |

### Quello che ha funzionato

**La martingala conformal risolve l'innesco rotto.** Invece di guardare una
soglia istantanea, accumula evidenza contro l'ipotesi che i dati restino
scambiabili. In TON→BoT — la direzione dove adattare vale davvero —
passa da **0,8430 a 0,9317**, praticamente pari al riadattamento continuo
(0,9490) e con 32 etichette in meno.

| Politica | TON→BoT | BoT→TON | adattamenti |
|---|---|---|---|
| statico | 0,8184 | 0,8466 | 0 |
| innesco a soglia conformal | 0,8430 | 0,8909 | 6 / 16 |
| **innesco a martingala** | **0,9317** | 0,8744 | 18 |
| riadattamento a ogni batch | 0,9490 | 0,8792 | 19 |
| oracolo | 0,9487 | **0,9257** | 19 |

Due errori sono emersi implementandola, entrambi istruttivi. Sommando 20 000
p-value per batch la deriva negativa sotto l'ipotesi nulla affonda la
statistica così in basso che nessuna deriva la recupera: serve il pavimento a
zero, come nel CUSUM. E con l'esponente ε = 0,5 canonico l'incremento è
positivo solo per p < 0,25, mentre sotto deriva i p-value medi scendono a
0,31 — non abbastanza. Con ε = 0,9 la soglia diventa p < 0,35 e la deriva si
vede.

**Il k-center rompe il tetto sul budget.** Il criterio del margine esauriva il
bacino di normali a ~58 campioni e oltre quel punto peggiorava. Selezionando
per copertura invece che per incertezza:

| Normali raccolte | n=32 | n=128 | n=512 |
|---|---|---|---|
| regola adattiva (margine) | 15,7 | 58,0 | **58,0** ← fermo |
| k-center (copertura) | 15,3 | 44,3 | **125,3** |

In BoT→TON il divario è ancora più netto: 184 contro 113. Il meccanismo che
bloccava il metodo è risolto.

### Quello che non ha funzionato, e va detto

**Il k-center trova più normali ma non produce più accuratezza.** La regola
adattiva resta migliore (0,9300 contro 0,8443 a n=512 in TON→BoT). Un
campione diverso è più informativo sulla geometria ma meno rappresentativo
della distribuzione reale: è lo stesso effetto già visto con la
deduplicazione. Il risultato migliore in assoluto della direzione difficile
resta k-center + Firth a n=512 (0,9639), ma non è stabile fra seed e non lo
scriverei come risultato.

**Firth è incostante.** Aiuta dove la separazione è il problema vero (a n=512
con k-center, +0,12) e danneggia altrove. La separazione completa con 8
campioni e 13 parametri c'è, ma la regolarizzazione L2 la gestisce già
abbastanza.

**I minimi quadrati ricorsivi risolvono la memoria ma non sono affidabili.**
Lo stato è una matrice 13×13 più un vettore: **182 numeri, 728 byte**, contro
i 12 KB del buffer da 256 campioni. Sedici volte meno, e sotto gli 8 KB di
SRAM di un ATmega2560. In TON→BoT tiene (0,9058 contro 0,9490 del buffer); in
BoT→TON crolla a 0,6831. È il trasferimento con più valore potenziale — è
l'unico che sblocca il vincolo hardware — ed è quello da far funzionare prima
di qualunque misura su dispositivo.

> **Nota in avanti**: i due numeri di questo paragrafo (0,9058 e 0,6831)
> descrivono un'implementazione che conteneva **due bug**, trovati e
> corretti in seguito — il fattore di dimenticanza applicato cinque volte
> per aggiornamento invece di una, e un ridge troppo debole contro la
> quasi-separazione al primo batch (sezione 16.2). Con entrambi corretti e
> su 10 seed i valori sono 0,9247 e 0,7434: non è più un crollo, è una
> perdita contenuta che si presenta solo quando BoT-IoT è la sorgente.
> Cambia anche la conclusione qui sopra: la RLS non è "l'unica strada che
> sblocca il vincolo hardware", è la strada **551 volte più economica in
> calcolo e 5,9 volte più piccola in RAM** di qualunque alternativa
> misurata (sezione 17c), bloccata da un problema di accuratezza e non di
> risorse. Questo paragrafo resta come stava per documentare da dove si è
> partiti.

## 10. Terzo dominio: perché UNSW-NB15 e non CIC-IoT-2023

Il professore aveva indicato CIC-IoT-2023 come terzo dataset opzionale.
Controllando le sue 47 feature prima di scaricarlo: `flow_duration`,
`Header_Length`, `Protocol Type`, `Rate`, `Srate`, `Drate`, conteggi dei flag
TCP, indicatori di protocollo e statistiche sulla dimensione dei pacchetti.
**Non ci sono i conteggi direzionali** — niente `src_bytes`/`dst_bytes` né
`src_pkts`/`dst_pkts` — e non c'è lo stato della connessione.

Sette delle nostre 13 feature numeriche dipendono dalla direzione. Lo spazio
armonizzato scenderebbe a 6. Prima di accettare quel prezzo l'abbiamo
misurato, proiettando TON_IoT e BoT-IoT sullo spazio ridotto: stesse righe,
stessi seed, stessi modelli, solo meno colonne.

| | ricco (13+2) | ridotto (6+2) | costo |
|---|---|---|---|
| in-domain sul source | 0,9699 | 0,9614 | −0,009 |
| TON→BoT, non adattato | 0,5632 | 0,6229 | +0,060 |
| **TON→BoT, 32 etichette** | **0,9067** | **0,6210** | **−0,286** |
| TON→BoT, 128 etichette | 0,9163 | 0,6790 | −0,237 |
| BoT→TON, 32 etichette | 0,8938 | 0,8482 | −0,046 |

**La riduzione non costa quasi nulla in-domain e fino a 29 punti sul risultato
adattato.** Sono le feature direzionali — le due asimmetrie, i payload medi
per direzione — a rendere gli edge *adattabili*: senza, il modello resta
altrettanto bravo sul proprio dominio e diventa molto meno correggibile su un
altro. È un risultato che non cercavamo e che vale da solo: dice *quali*
feature contano per l'adattamento, e non sono quelle che contano per
l'accuratezza.

Con CIC-IoT-2023 avremmo misurato un crollo senza poter distinguere l'effetto
del dominio nuovo da quello dello spazio dimezzato. **UNSW-NB15** invece usa
lo stesso strumento di cattura di BoT-IoT (Argus) e ha esattamente `dur`,
`proto`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`: lo spazio ricco resta
intatto e non va inventata nessuna corrispondenza nuova. In più il lavoro di
riferimento su *Electronics* usa la stessa terna, quindi i risultati diventano
confrontabili invece che paralleli.

Codice pronto: `kanids/harmonized.py::build_harmonized_unsw`,
`kanids/datasets.py::load_unsw`, e `scripts/cross_domain.py` che rileva il
terzo dominio se i file ci sono e passa da 4 a 9 esperimenti (6 direzioni
cross più 3 riferimenti in-domain).

## 11. Sei direzioni: cosa sopravvive

UNSW-NB15 caricato e armonizzato senza attriti — tutti i suoi stati grezzi
cadono nella mappa Argus già scritta, 257 673 flussi, 36% normali. Tre domini,
sei direzioni cross più tre riferimenti in-domain. Per ogni sorgente il
modello si addestra **una volta** e si valuta su tutti e tre i domini, quindi
il degrado è confrontabile.

**Rilanciato su 10 seed (42-51), spazio ricco (13+2)**: nessun checkpoint a 3
seed esisteva in questo ambiente per questa combinazione esatta di script e
spazio, quindi questa e' la prima misura completa, non un'estensione. Media
± dev.std:

| Direzione | non adattato | ROC-AUC target | 8 etich. | 32 | 128 |
|---|---|---|---|---|---|
| ton→ton *(in-domain)* | 0,9705±0,0009 | 0,9930±0,0004 | — | — | — |
| bot→bot *(in-domain)* | 0,9931±0,0009 | 0,9992±0,0002 | — | — | — |
| unsw→unsw *(in-domain)* | 0,8184±0,0020 | 0,9285±0,0008 | — | — | — |
| unsw→bot | 0,7368±0,0222 | 0,7689±0,0044 | *fallita (0/10)* | *fallita (0/10)* | *fallita (0/10)* |
| bot→ton | 0,6340±0,0619 | 0,8185±0,0272 | 0,7751 (6/10) | 0,8003 (9/10) | 0,8623 (9/10) |
| ton→bot | 0,5554±0,0084 | 0,5257±0,0156 | 0,7724 (6/10) | 0,8657 (8/10) | 0,9018 (9/10) |
| bot→unsw | 0,4551±0,0091 | 0,4164±0,0184 | 0,6488 (8/10) | 0,7381 (9/10) | 0,7552 (10/10) |
| unsw→ton | 0,2984±0,0335 | **0,2569±0,0117** | 0,6093 (8/10) | 0,7065 (10/10) | 0,8365 (10/10) |
| ton→unsw | 0,2237±0,0074 | **0,2734±0,0026** | 0,6433 (10/10) | 0,7158 (10/10) | 0,7494 (10/10) |

(n/10 = quanti seed su 10 trovano entrambe le classi a quel budget. La
sezione 15 aveva mostrato che "impossibile" nello spazio ridotto e' spesso
"raro" con piu' seed — qui, nello spazio ricco, **la distinzione fra le due
letture regge**: si vede sotto.)

### Cosa e' confermato, non un artefatto

**`unsw→bot` fallisce davvero, non solo nel campione a 3 seed usato in
origine — genuinamente 0 su 10.** A differenza di `ton->bot`,
`cic->bot` e `unsw->bot` **nello spazio ridotto** (sezione 15, dove lo
stesso tipo di direzione riesce in 1-7 casi su 10), qui **nello spazio
ricco** `unsw→bot` non trova mai le due classi in nessuno dei 10 seed. Le
due misure non si estrapolano l'una dall'altra — sono spazi di feature
diversi (6+2 contro 13+2) e producono risultati diversi per la stessa
coppia di domini — ed e' esattamente per questo che andava rimisurato
invece di assunto. Il meccanismo resta quello gia' diagnosticato: BoT-IoT
ha 477 normali su 3,67 M e la regola a margine, partendo da UNSW-NB15 come
sorgente, non li intercetta mai.

**Le altre cinque direzioni cross hanno un tasso di successo per seed che
varia (6-10 su 10), non 3/3 o 0/3 come un campione piccolo lascia credere**,
ma le medie restano vicine a quelle del campione a 3 seed originale (es.
`ton->bot` a 128 etichette: 0,9117 allora, 0,9018 ora) — qui la revisione a
10 seed conferma piu' che corregge.

### Cosa tiene

**Il collasso è generale, e peggiore di quanto sapevamo.** Sei direzioni su
sei, da 0,22 a 0,74 di balanced accuracy contro 0,82–0,99 in-domain. Non è una
peculiarità della coppia TON/BoT.

**L'adattamento a 13 coefficienti recupera in 5 direzioni su 6**, con guadagni
da +0,2 a +0,4. Con 32 etichette e 24 byte. `unsw→bot` resta l'eccezione
strutturale: non c'e' adattamento possibile se non si raccoglie nemmeno
un'etichetta della classe minoritaria.

> **Nota in avanti (sezione 18.4)**: l'entita' del recupero regge su tutta
> la griglia di rapporti provata (1, 3, 20, 50, 100), ma la sua
> **affidabilita'** no. A ratio=1 (sorgente ribilanciata 1:1) `bot→ton`
> trova entrambe le classi solo in 8 seed su 10 e `ton→bot` in 7/10,
> contro 9-10/10 a ogni altro rapporto: la selezione delle etichette sul
> target, non l'adattamento in se', diventa meno affidabile quando la
> sorgente e' ribilanciata all'estremo. E lo stesso rapporto estremo rende
> `unsw→bot` — l'eccezione di questo paragrafo — **non piu' assoluta**:
> vedi la nota alla frase "zero normali" piu' sotto.

**Un fenomeno nuovo, invisibile con due domini: il transfer invertito.**
`unsw→ton` e `ton→unsw` hanno ROC-AUC 0,26 e 0,27 — sotto il caso, cioè
l'ordinamento è *sistematicamente rovesciato*, confermato a 10 seed con
dev.std piccola (0,012 e 0,003). Il modello non ha perso l'informazione, la
usa col segno sbagliato. È la conferma più forte della diagnosi della
sezione 1: nessuna soglia e nessun metodo non supervisionato può rimediare
a un ordinamento invertito, mentre ripesare gli edge — che può cambiare
segno — lo raddrizza.

### Cosa non tiene

**"Aggiornare poco batte riaddestrare tutto" non generalizza — confermato,
e ora con un test invece di un conteggio.** Delta (13 coefficienti − rifit
completo) appaiato per seed, test t a un campione, sulle cinque direzioni
cross dove la selezione riesce (esclusa `unsw→bot`):

| budget | vince in media | vince in modo significativo (p<0,05) | perde in modo significativo |
|---|---|---|---|
| n=8 | 2/5 | 1/5 (`bot→ton`, p=0,029) | 0/5 |
| n=32 | 1/5 | 1/5 (`ton→bot`, p=0,007) | 2/5 (`ton→unsw` p=0,025, `unsw→ton` p=0,001) |
| n=128 | 1/5 | 1/5 (`ton→bot`, p<0,0001) | 4/5 (tutte tranne `ton→bot`) |

**`ton→bot` e' l'unica direzione dove i 13 coefficienti battono il rifit
completo in modo significativo a tutti e tre i budget** (t=+13,9 a n=128).
Nelle altre quattro il rifit completo vince in modo significativo a n=128 —
ma **i conteggi nascondono le entita', e le entita' cambiano la
conclusione**. A n=128 le quattro sconfitte sono piccole (−0,038, −0,017,
−0,020, −0,071; somma −0,146) e la sola vittoria (`ton→bot`, +0,135) vale
quasi altrettanto da sola; il delta medio sulle cinque direzioni e'
**−0,002 — un pareggio**, non uno sbilanciamento verso il rifit completo.
La lettura corretta e': **il rifit completo vince piu' spesso e di poco, i
13 coefficienti vincono raramente e di molto, e in media si annullano** —
non "il rifit completo vince chiaramente tranne in una direzione", che e'
vero contando le direzioni ma suggerisce un vantaggio che in valore atteso
non esiste. Il vantaggio dei 13 coefficienti e' quindi di costo — 24 byte
contro 250 e nessun riaddestramento sul dispositivo — **non di
accuratezza, e nemmeno di svantaggio in accuratezza**: sul valore atteso
sono alla pari.

> **Nota in avanti (sezione 18.4) — questo pareggio non regge dappertutto,
> e va corretto, non solo qualificato.** Misurato anche a ratio 1, 3, 20 e
> 100 (oltre al 50 originale): il pareggio **regge da ratio=3 in su**
> (delta fra +0,0002 e −0,0066, sempre p>0,5) ma **cade nettamente a
> ratio=1**: delta −0,033, t=−6,22, **p=0,0002**, e — a differenza di
> ratio 3-100, dove la media vicino a zero nasce da `ton→bot` fortemente
> positivo contro quattro direzioni leggermente negative — a ratio=1
> **tutte e cinque le direzioni sono negative**, `ton→bot` incluso (che
> scende da +0,135 a −0,005). Non e' un outlier che sposta la media, e'
> un peggioramento diffuso: con la sorgente ribilanciata 1:1 il rifit
> completo usa meglio il budget di etichette in ogni direzione, non solo
> in quattro su cinque. "Pareggio in valore atteso" va riscritto come
> "pareggio da ratio 3 in su; a ratio 1 i 13 coefficienti perdono in modo
> diffuso e significativo".

**Una direzione fallisce del tutto — confermato a 10 seed, non un artefatto
del campione piccolo (vedi sopra).** In `unsw→bot` la regola di selezione
raccoglie **zero normali** a ogni budget, in tutti i 10 seed: BoT-IoT ha 477
normali su 3,67 M e da questa sorgente il margine non li intercetta mai. Il
collo di bottiglia non è l'adattamento, è trovare cosa etichettare —
esattamente il punto già emerso nella sezione 4, qui in forma terminale (la
sezione 4 usa pero' uno script diverso, `drift_sampling.py`, non ancora
rilanciato a 10 seed in questo lavoro: la stessa cautela vale ma non e'
stata verificata).

> **Nota in avanti (sezione 18.4)**: "zero normali in tutti i 10 seed"
> regge a ratio 3, 20, 50 e 100 (0/10 ovunque), ma **non a ratio=1**, dove
> **un seed su dieci** trova abbastanza normali da produrre un numero
> (balanced accuracy 0,76, contro 0,74 non adattato — un recupero modesto
> anche quando riesce). "Fallisce in tutti i seed" va corretto in
> "fallisce in tutti i seed per ogni rapporto da 3 in su; al rapporto piu'
> estremo provato (1:1) fallisce in nove seed su dieci, non dieci su
> dieci" — resta il collo di bottiglia piu' severo del lavoro, ma non e'
> piu' letteralmente assoluto.

**Un tetto che non avevamo visto — confermato, stesso numero.** Con
UNSW-NB15 come target l'adattamento si ferma a 0,75–0,76 (`bot→unsw`
0,7552, `ton→unsw` 0,7494 a 128 etichette), mentre altrove arriva a 0,90.
Non è un limite del metodo: UNSW-NB15 fa 0,8184 anche **in-domain** nel
nostro spazio a sette quantità grezze (invariato dal campione a 3 seed,
0,8188), perché la sua capacità discriminante sta nelle 38 feature che
escludiamo. L'adattamento non può superare il soffitto del dominio di
arrivo, e questo va detto prima delle tabelle.

## 12. Le sezioni 5, 8 e 9 rifatte su sei direzioni

Le tre sezioni misurate su due sole direzioni sono state rieseguite su tutte e
sei. **Una regge, una va indebolita, una cade.**

### 8 rifatta — regge: nessun metodo non supervisionato è affidabile

| Metodo | ton→bot | bot→ton | ton→unsw | unsw→ton | bot→unsw | unsw→bot |
|---|---|---|---|---|---|---|
| non adattato | 0,5632 | 0,5989 | 0,2204 | 0,2950 | 0,4587 | 0,7120 |
| EM sul prior | 0,5606 | 0,5000 | 0,3746 | 0,4926 | 0,5000 | 0,5000 |
| TENT | 0,4400 | 0,6187 | 0,2428 | 0,4565 | 0,4887 | 0,6837 |
| TENT filtrato | 0,5439 | 0,5262 | 0,2406 | 0,4999 | 0,4963 | 0,4821 |
| IM (SHOT) | 0,6411 | 0,7613 | 0,2428 | 0,3546 | 0,3781 | 0,7023 |
| **32 etichette** | **0,9085** | **0,8939** | **0,7231** | **0,7727** | **0,7448** | 0,5000 |

Nessuno dei quattro migliora in più di 4 direzioni su 6, e ognuno ne danneggia
almeno due. IM resta il migliore ma perde la sua aria di vincitore: aiuta in
4/6 e in `bot→unsw` toglie 8 punti. Un dettaglio interessante: nelle due
direzioni a ordinamento invertito i metodi non supervisionati danno il loro
contributo massimo (EM +0,20 su `unsw→ton`), perché lì il modello è
grossolanamente scalibrato e correggere il prior serve. Anche così restano
molto sotto le 32 etichette.

### 5 rifatta — cade: il vantaggio non è architetturale

Su due direzioni la KAN dominava. Su sei, l'**ultimo strato di una MLP(16) —
17 parametri, quanto i nostri 13** — vince più spesso:

| Direzione | vincitore a n=32 | KAN |
|---|---|---|
| ton→bot | **KAN** 0,9067 | 0,9067 |
| bot→ton | **KAN** 0,8938 | 0,8938 |
| ton→unsw | LightGBM 0,7874 | 0,7231 |
| bot→unsw | MLP(16) 0,7605 | 0,7448 |
| unsw→ton | MLP(16) 0,8565 | 0,7727 |
| unsw→bot | MLP(16) 0,7824 | *fallita* |

La KAN vince nelle due direzioni della coppia originale e perde nelle quattro
che coinvolgono UNSW-NB15. Quello che va scritto adesso è: **contro gli
ensemble ad albero il vantaggio è reale** — 13 parametri contro 401, e
LightGBM crolla a 0,6601 in `ton→bot` — **ma contro una piccola MLP non c'è
vantaggio di accuratezza**, perché anche lei ha un aggiornamento minimo da 17
numeri. Resta il vantaggio di deployment: la KAN gira già integer-only su
MCU con 250 byte, e l'aggiornamento è una tabella di moltiplicatori Q15 che il
firmware ha già. Non è un'affermazione sull'accuratezza, ed è sbagliato
presentarla come tale.

### 9 rifatta — da indebolire: il k-center serve, ma non per l'accuratezza

Il k-center raccoglie più normali in **6 direzioni su 6**, e soprattutto
**salva l'unica direzione dove tutto il resto fallisce**: in `unsw→bot` la
regola adattiva raccoglie zero normali a ogni budget, il k-center ne trova 12
con 32 etichette e porta la balanced accuracy a 0,6306 (0,6889 con Firth).

Ma sull'accuratezza, dove la regola adattiva funziona, perde: vince in **1 caso
su 15**, mediana −0,13. Il modo corretto di usarlo è quindi come **ripiego**,
non come sostituto: si campiona con la regola adattiva e si passa al k-center
quando il sondaggio non restituisce entrambe le classi.

Firth vince in 20 casi su 49 con mediana −0,0004. È esattamente una moneta:
va tolto.

## 13. Le sezioni 6 e 7 rifatte su sei direzioni

Sono le due che sostengono le affermazioni sul dispositivo, quindi le più
importanti da non lasciare su due direzioni.

### 6 rifatta — regge, e meglio di prima

Balanced accuracy, tutto in interi, 3 seed:

| | ton→bot | bot→ton | ton→unsw | unsw→ton | bot→unsw | unsw→bot |
|---|---|---|---|---|---|---|
| intero, non adattato | 0,4622 | 0,7037 | 0,2981 | 0,4153 | 0,4369 | 0,4136 |
| + guadagni stimati in float | 0,8328 | 0,8609 | 0,5759 | 0,7607 | 0,6694 | 0,5757 |
| **+ guadagni stimati in interi** | 0,7321 | **0,8602** | **0,6434** | 0,7204 | 0,6200 | **0,6715** |

**Sei direzioni su sei migliorano.** E la stima intera batte quella in virgola
mobile in 2 direzioni su 6, restando entro pochi punti nelle altre tranne
`ton→bot`. L'affermazione "l'aritmetica intera non costa accuratezza" regge.

Il risultato inatteso: **la catena intera funziona in `unsw→bot`, dove quella
float falliva del tutto** (0,4136 → 0,6715). Il motivo non è l'aritmetica ma
la selezione: la pipeline intera deduplica sui contributi — uguaglianza esatta
fra interi — e quella deduplicazione trova normali dove la regola adattiva sui
punteggi float ne trovava zero. Un dettaglio implementativo nato per risparmiare
etichette si è rivelato la cosa che sblocca la direzione più difficile.

### 7 rifatta — il riadattamento regge, l'innesco no

| Politica | direzioni in cui batte lo statico | note |
|---|---|---|
| **ogni batch, buffer 256** | **6 su 6** (+0,03…+0,14) | pari all'oracolo |
| oracolo (etichette bilanciate) | 6 su 6 | tetto |
| martingala conformal | 4 su 6 | non scatta mai in 2 direzioni |
| minimi quadrati ricorsivi | 4 su 6 | crolla in `bot→ton` (−0,16) |
| innesco a soglia conformal | 3 su 6 | non scatta mai in 3 direzioni |

**Il riadattamento continuo con buffer generalizza**: 6 direzioni su 6, e
raggiunge l'oracolo. È l'affermazione più solida di tutto il lavoro.

**L'innesco no.** La martingala migliora sulla soglia conformal (4/6 contro
3/6) ma in `ton→unsw` e `unsw→bot` **non scatta mai**: zero adattamenti in 20
batch, mentre l'adattamento continuo lì guadagna 14 e 9 punti. Rilevare la
deriva resta un problema aperto, e con sei direzioni si vede che è più grave di
quanto sembrasse.

**I minimi quadrati ricorsivi restano inaffidabili**: 4 su 6, con un crollo di
16 punti in `bot→ton`. Confermato che sono la cosa da sistemare prima di
qualunque misura su hardware.

## 14. La terna del professore, misurata

Il dataset caricato è **CICIoMT2024** (Internet of Medical Things), non
CIC-IoT-2023 — lo si riconosce dagli attacchi MQTT e dalla nomenclatura
`*_test.pcap.csv`. Stessa famiglia di estrattore, stesse 45 colonne, quindi le
stesse limitazioni.

### Cosa manca davvero, verificato sui dati

- **`Duration` è il TTL, non la durata del flusso**: mediana 64, massimo 248 —
  valori classici del campo IP, non tempi.
- **`IAT` è corrotto**: mescola timestamp Unix assoluti (169470306 → 14
  settembre 2023) con veri tempi di interarrivo (0,091), nella stessa colonna.
- **`Drate` è identicamente zero** in tutti i file: nessuna informazione
  direzionale.

Senza durata cadono anche `flow_rate` e `byte_rate`. Delle 13 numeriche ne
restano **tre**: `bytes_total`, `pkts_total`, `payload_mean`. E c'è una
differenza più profonda delle colonne mancanti: **quelle righe non sono flussi
bidirezionali ma finestre scorrevoli di pacchetti**. Trasferire fra questo
dataset e gli altri tre non è solo cambiare dominio, è cambiare unità di
osservazione.

### Il confronto, a parità di spazio

Le due terne girate nello stesso spazio a tre feature, stessi modelli, stessi
seed. Medie sulle sole direzioni cross:

| Terna | in-domain | cross non adattato | 32 etich. | 128 etich. |
|---|---|---|---|---|
| **ton+bot+cic** (professore) | 0,9774 | 0,5718 | 0,7307 | **0,7852** |
| ton+bot+unsw | 0,9108 | 0,5168 | 0,7290 | 0,7520 |

**La terna del professore dà numeri leggermente migliori**, e va detto. Ma il
motivo non è che il transfer funzioni meglio: è che **CICIoMT2024 è più
facile**. Fa 0,9923 in-domain con tre sole feature — un dataset in cui tre
numeri bastano per il 99% della balanced accuracy sta misurando la struttura
della cattura, non l'intrusion detection. UNSW-NB15, che nello stesso spazio fa
0,7925, è semplicemente più difficile e più informativo.

### Il confronto che conta

| Configurazione | cross non adattato | 32 etich. | 128 etich. |
|---|---|---|---|
| **ton+bot+unsw, spazio ricco (13+2)** | 0,4789 | **0,7989** | **0,8422** |
| ton+bot+unsw, spazio minimo (3+2) | 0,5168 | 0,7290 | 0,7520 |
| ton+bot+cic, spazio minimo (3+2) | 0,5718 | 0,7307 | 0,7852 |

Lo spazio ricco vale **+0,07 sul risultato adattato** rispetto al minimo, e la
terna del professore non recupera quel divario: 0,7307 contro 0,7989. Il costo
di includere CICIoMT2024 non è il dominio, è lo spazio che impone a *tutti* gli
altri.

### Cosa proporrei

Non è una scelta fra le due terne. È: **UNSW-NB15 come terzo dominio
dell'analisi principale** nello spazio ricco, e **CICIoMT2024 come quarto
dominio** in una sezione a parte, nello spazio minimo, come prova di
robustezza. Così la richiesta del professore è soddisfatta con dei numeri, non
elusa, e l'analisi principale non paga il prezzo di sette feature.

Un dettaglio a favore di CICIoMT2024 che va riconosciuto: con l'8,6% di
benigni **nessuna direzione fallisce**, mentre con BoT-IoT allo 0,013% la
raccolta di etichette si rompe. Come banco di prova della *selezione* è meno
severo, e proprio per questo meno utile a noi.

---

## 15. Il punto 14 rifatto: `test.csv` ha una flow_duration vera

La sezione 14 aveva concluso che il file caricato era CICIoMT2024, senza
durata utilizzabile. Con il file attuale in `kanids-data/test.csv` quella
diagnosi non regge piu': va rifatta, non riusata.

### La verifica, sui dati

`test.csv` ha 1 176 851 righe e 47 colonne, e le etichette sono i 34 nomi di
attacco del vero CIC-IoT-2023 (`DDoS-ICMP_Flood`, `BenignTraffic`,
`Mirai-greeth_flood`, ...), non gli attacchi MQTT di CICIoMT2024: e' un file
diverso da quello che la sezione 14 aveva analizzato.

- **`Duration` e' ancora il TTL**, non una durata: mediana 64,0, 73% dei
  valori esattamente 64,0, massimo 255,0 — la stessa firma (mediana 64,
  clustering sul default) misurata in CICIoMT2024, coerente col fatto che
  entrambi i dataset usano lo stesso estrattore con lo stesso bug di nome.
- **`IAT` e' corrotto allo stesso modo**: 97,9% dei valori sta fra 1e6 e
  1,7e8 (due grappoli, ~83M e ~167M, con la forma di un timestamp assoluto o
  di un contatore di sessione), solo il 2,1% ha valori piccoli e plausibili
  come tempi di interarrivo reali. Inutilizzabile, come in CICIoMT2024.
- **`flow_duration`, invece, e' una colonna a se' e sembra vera**: sul
  traffico benigno mediana 26,1 s (media 39,3 s), sugli attacchi mediana
  0,0 s (54% delle righe esattamente zero, media 4,8 s). La differenza ha
  senso — i flood sono per lo piu' flussi da un pacchetto — ed e' il tipo di
  separazione per classe che un artefatto di estrazione non produce.
  Correlazione con `IAT`: 0,008, cioe' non e' la stessa cosa rietichettata.

**La sezione 14 aveva ragione sul dataset che aveva davanti (CICIoMT2024,
senza durata) e la conclusione che ne aveva tratto va rifatta con questo
file**, che ha una `flow_duration` reale.

### Il bug che nascondeva `build_ridotto_cic`

`kanids/harmonized.py` aveva davvero due costruttori per la famiglia CIC,
come notato: `build_minimo_cic` (3+2, quello usato in sezione 14) e
`build_ridotto_cic` (6+2, usa `flow_duration`), scritto ma mai chiamato. Il
motivo non era una dimenticanza banale: `scripts/cross_domain.py::load_harmonized`
importava

    from kanids.harmonized import build_minimo_cic as build_ridotto_cic

— un alias che faceva chiamare **sempre** `build_minimo_cic` sotto un nome
che sembrava l'altro costruttore, indipendentemente dallo spazio richiesto.
Con questo alias, un run in spazio "ridotto" proiettava TON/BoT/UNSW sulle 6
feature ridotte ma teneva CIC fermo a 3, e le colonne non combaciavano.

Corretto in due punti, entrambi dentro `adattamento-drift/`:

1. `kanids/datasets.py::cic_paths` ora riconosce `test.csv` come file
   singolo del vero CIC-IoT-2023 (cercato prima del glob a shard, che
   altrimenti ingoierebbe anche TON_IoT/BoT-IoT/UNSW-NB15 da
   `kanids-data/`), e `load_cic` binarizza l'etichetta multiclasse in
   stringa (`BenignTraffic` contro il resto) invece di assumerla gia' 0/1.
2. `load_harmonized` prende un parametro `spazio_cic` e sceglie fra
   `build_minimo_cic` e `build_ridotto_cic` di conseguenza, con cache
   parquet separate (`harmonized_cic_minimo.parquet` /
   `harmonized_cic_ridotto.parquet`) per non mischiare le due proiezioni.

### Il confronto, rifatto nello spazio ridotto (6+2)

Stesso protocollo della sezione 14 (stessi modelli, stessi 3 seed, stesso
budget), ma ora **CIC entra nello spazio ridotto con la sua vera durata**
invece di restare forzato al minimo. Medie sulle sole direzioni cross:

| Terna | in-domain (media) | cross non adattato | 32 etich. | 128 etich. |
|---|---|---|---|---|
| ton+bot+**cic** ridotto | 0,9816 | 0,4758 | 0,7708 (4/6) | 0,7754 (5/6) |
| ton+bot+**unsw** ridotto | 0,9183 | 0,4658 | **0,7840** (4/6) | **0,7967** (4/6) |

(n/6 = quante delle sei direzioni cross hanno prodotto un numero: in
`ton+bot+cic`, `cic->bot` fallisce a 8 e 32 etichette — vedi sotto — mentre
in `ton+bot+unsw`, `ton->bot` e `unsw->bot` falliscono a ogni budget, come
gia' misurato nelle sezioni 4/11: la regola a margine non trova normali con
BoT-IoT come target, a prescindere dalla terza terna.)

**Questa tabella e le due sotto sono su 3 seed e la lettura "fallisce/(4/6)"
non e' quella giusta — vedi piu' sotto "Da 3 a 10 seed": con 7 seed in piu'
`ton->bot` produce un numero in 3-5 casi su 10, non 0; e' raro, non
impossibile.** Le tre sottosezioni seguenti restano perche' documentano
come si e' arrivati li', ma i numeri da citare sono quelli della sezione
"Da 3 a 10 seed".

### Perche' quelle due medie non si possono confrontare

La prima stesura di questa sezione leggeva quella tabella come "nello stesso
spazio UNSW-NB15 vince, +0,013 a 32 etichette e +0,021 a 128". **E' una
lettura sbagliata, e il modo in cui e' sbagliata vale piu' del numero.**

Le due medie sono calcolate su **insiemi di direzioni diversi**. In
`ton+bot+cic` cade `cic->bot`; in `ton+bot+unsw` cadono `ton->bot` e
`unsw->bot`. Le direzioni hanno difficolta' molto diverse fra loro, quindi
cambiare quali entrano nella media sposta il risultato piu' del margine
riportato: a 128 etichette la media della terna CIC include `cic->bot`
(0,5146, il numero peggiore dell'intera tabella) mentre quella di UNSW non
include nessuna direzione con BoT-IoT come target. **Il margine di 0,021
misura quale direzione e' stata esclusa, non quale dominio trasferisce
meglio.**

Il confronto corretto appaia le direzioni per **ruolo del terzo dominio** —
`ton->cic` con `ton->unsw`, `bot->cic` con `bot->unsw`, `cic->ton` con
`unsw->ton` — cioe' le tre in cui il terzo dominio compare, e che esistono
in entrambe le terne. Media ± dev.std sui 3 seed, per cella:

| | non adattato | 8 etich. | 32 etich. | 128 etich. |
|---|---|---|---|---|
| `ton->cic` | 0,4480 ± 0,0008 | 0,7216 ± 0,0272 | 0,7446 ± 0,0902 | 0,8631 ± 0,1567 |
| `ton->unsw` | 0,2742 ± 0,0188 | 0,6817 ± 0,0754 | 0,7369 ± 0,0519 | 0,7542 ± 0,0058 |
| `bot->cic` | 0,4357 ± 0,0067 | 0,7676 ± 0,0044 | 0,7843 ± 0,0576 | 0,8594 ± 0,0302 |
| `bot->unsw` | 0,4709 ± 0,0164 | 0,6734 ± 0,0813 | 0,7536 ± 0,0354 | 0,7735 ± 0,0082 |
| `cic->ton` | 0,5672 ± 0,0758 | 0,5407 ± 0,1057 | 0,6948 ± 0,0744 | 0,7586 ± 0,1232 |
| `unsw->ton` | 0,3260 ± 0,0011 | 0,7059 ± 0,0453 | 0,7863 ± 0,0155 | 0,7778 ± 0,0506 |
| **media ton+bot+cic** | **0,4836** | 0,6766 | 0,7412 | **0,8270** |
| **media ton+bot+unsw** | **0,3570** | 0,6870 | 0,7589 | **0,7685** |
| delta (unsw − cic) | **−0,1266** | +0,0104 | +0,0177 | **−0,0586** |
| dev.std del delta **fra i 3 seed** | **0,0207** | 0,0678 | 0,0385 | **0,0295** |
| seed con lo stesso segno | **3/3** | 2/3 | 2/3 | **3/3** |

La dev.std per cella era mancante nella prima stesura perche' il file dei
record per seed della terna CIC era stato sovrascritto (vedi sotto): con
entrambe le terne ora salvate su nomi separati
(`results/tre_domini_runs_ridotto_tonbotcic.csv`,
`results/tre_domini_runs_ridotto_tonbotunsw.csv`), la dispersione e'
misurata su entrambe, non solo su UNSW-NB15.

**Il segno si inverte a 128 etichette.** Quello che la tabella precedente
dava come +0,021 a favore di UNSW-NB15 diventa −0,059 a favore di CIC, una
volta tolto l'effetto del sottoinsieme.

### Qual e' l'unita' statistica: una revisione precedente ha sbagliato denominatore

Una stesura intermedia di questa sezione riportava come rumore di
riferimento la dev.std del delta calcolata su **n=9** (3 coppie × 3 seed
messi insieme) — 0,131, 0,137, 0,081, 0,128 — e concludeva che nessuno dei
quattro margini si distingue da zero. **Quel denominatore e' sbagliato, e
sbagliato in una direzione precisa: gonfia il rumore di 4-6 volte.**

Pooling le tre coppie si somma alla varianza fra seed la varianza **fra
coppie**, che non e' rumore: `ton->cic`, `bot->cic` e `cic->ton` sono
problemi di difficolta' diversa e ci si aspetta che il delta sia diverso fra
loro. La replicazione indipendente qui e' il **seed**, non la coppia. Il
calcolo corretto media prima le tre coppie dentro ciascun seed, poi guarda
come oscilla quel numero fra i 3 seed:

| budget | delta per seed 42 / 43 / 44 | media | dev.std | segni |
|---|---|---|---|---|
| non adattato | −0,111 / −0,150 / −0,119 | −0,1266 | **0,0207** | 3/3 negativi |
| 8 etich. | +0,086 / −0,046 / −0,009 | +0,0104 | 0,0678 | discordi |
| 32 etich. | +0,056 / −0,021 / +0,018 | +0,0177 | 0,0385 | discordi |
| 128 etich. | −0,041 / −0,093 / −0,042 | −0,0586 | **0,0295** | 3/3 negativi |

**Il quadro cambia: due colonne su quattro sopravvivono, non zero.**

- **Non adattato**: −0,127 con dev.std 0,021 e tutti e tre i seed dello
  stesso segno. E' sei volte la dispersione — il risultato piu' solido
  dell'intera sezione.
- **128 etichette**: −0,059 con dev.std 0,030, tutti e tre i seed negativi.
  Due volte la dispersione: non e' una prova con tre seed, ma non e' rumore.
- **8 e 32 etichette**: i seed hanno segni discordi e il delta e' piu'
  piccolo della loro dispersione. Qui **non c'e' niente da leggere**, e su
  questo la revisione precedente aveva ragione.

### Lo schema "target facile, sorgente debole" non regge al test per seed

La stessa revisione notava che, scomponendo le coppie, a 8/32/128 etichette
`ton->cic` e `bot->cic` favoriscono sempre CIC mentre `cic->ton` favorisce
sempre UNSW-NB15, e ne traeva che **CIC-IoT-2023 e' un target facile ma una
sorgente meno trasferibile**. Lo schema c'e' davvero nelle medie. **Sui
singoli seed non c'e'**, ed e' lo stesso errore di lettura di prima, un
livello piu' in fondo: tre budget calcolati sugli stessi tre modelli non
sono tre repliche indipendenti, sono lo stesso esperimento a tre soglie di
budget.

Segno del delta per seed (`+` favorisce UNSW, `−` favorisce CIC):

| coppia | 8 etich. | 32 etich. | 128 etich. |
|---|---|---|---|
| `ton->cic` / `ton->unsw` | + − − | + − − | + − − |
| `bot->cic` / `bot->unsw` | + − − | − − − | − − − |
| `cic->ton` / `unsw->ton` | + + + | + + + | **− − +** |

Delle nove celle solo quattro sono concordi sui tre seed. La coppia
`cic->ton`, che e' l'intera base dell'affermazione "sorgente debole", **si
capovolge a 128 etichette**: due seed su tre favoriscono CIC, e la media
positiva (+0,019) viene quasi tutta dal solo seed 44 (+0,193 contro −0,131 e
−0,005). L'affermazione va ritirata.

**Un'anomalia che invece e' sistematica e vale la pena guardare.** Il seed 42
va contro gli altri due in `ton->cic` a **tutti e tre i budget** (+0,061,
+0,089, +0,071 contro valori negativi degli altri due seed), sempre della
stessa entita'. Non e' dispersione casuale: e' un regime diverso, quasi
certamente nella selezione delle etichette, che si riproduce identico a
ogni budget. Con tre seed non si puo' dire altro, ma e' esattamente il tipo
di cosa che con dieci seed diventerebbe o un artefatto da spiegare o una
bimodalita' da riportare. **Prima di qualunque affermazione sul confronto
fra terne servono piu' seed**: e' l'unica raccomandazione metodologica di
questa sezione.

### Da 3 a 10 seed: cosa regge, cosa si rafforza, cosa va corretto

Le due terne rilanciate con 7 seed in piu' (42-51, checkpoint gia' completi
per 42-44, quindi nessun numero precedente e' stato ricalcolato — solo
esteso). `scripts/tre_domini.py::finalize` scrive gia' un file per terna
(sezione precedente), quindi i checkpoint restano separati e riproducibili.

**"Fallisce"/"impossibile" era un artefatto del campione a 3 seed, non una
proprieta' della direzione.** Con solo 42-44, `ton->bot` non trovava normali
in *nessuno* dei tre run, a qualunque budget — da qui il NaN che la tabella
riassuntiva leggeva come fallimento strutturale. Con 10 seed, `ton->bot`
produce un numero in **3-5 casi su 10** a seconda del budget: raro, non
impossibile. Il quadro completo per seed riusciti su 10, sulle direzioni con
BoT-IoT o CIC-IoT-2023 come bersaglio:

| direzione | 8 etich. | 32 etich. | 128 etich. |
|---|---|---|---|
| `ton->bot` (target BoT-IoT) | 3/10 | 5/10 | 5/10 |
| `unsw->bot` (target BoT-IoT) | 1/10 | 1/10 | 2/10 |
| `cic->bot` (target BoT-IoT) | 2/10 | 3/10 | 7/10 |
| `bot->ton`, `bot->unsw`, `ton->cic`, `ton->unsw`, `cic->ton`, `unsw->ton` | 7-10/10 | 9-10/10 | 9-10/10 |

Il target BoT-IoT (0,013% di normali) resta di gran lunga il caso peggiore
per la raccolta di etichette — `unsw->bot` ancora praticamente introvabile,
1-2 seed su 10 — ma "fallisce sempre" era vero solo per il campione
particolare di 3 seed usato finora, non per la direzione. Le medie
riportate sopra e nelle sezioni 4/11/14, dove compare questa frase, andranno
rilette come "raro" non come "impossibile" quando quelle sezioni verranno
riprese (sezione 11 e' nel blocco successivo di questo lavoro).

**La tabella appaiata, rifatta su 10 seed, media ± dev.std per cella:**

| | non adattato | 8 etich. | 32 etich. | 128 etich. |
|---|---|---|---|---|
| `ton->cic` | 0,4462 ± 0,0054 (n=10) | 0,7339 ± 0,0526 (n=9) | 0,7979 ± 0,0706 (n=10) | 0,9137 ± 0,0836 (n=10) |
| `ton->unsw` | 0,2612 ± 0,0147 (n=10) | 0,6913 ± 0,0592 (n=10) | 0,7200 ± 0,0590 (n=10) | 0,7479 ± 0,0599 (n=10) |
| `bot->cic` | 0,4359 ± 0,0100 (n=10) | 0,7521 ± 0,0955 (n=10) | 0,7904 ± 0,0741 (n=10) | 0,8680 ± 0,0216 (n=10) |
| `bot->unsw` | 0,4706 ± 0,0103 (n=10) | 0,6595 ± 0,0573 (n=9) | 0,7160 ± 0,0510 (n=10) | 0,7515 ± 0,0533 (n=10) |
| `cic->ton` | 0,5644 ± 0,0702 (n=10) | 0,5704 ± 0,0899 (n=8) | 0,6824 ± 0,0913 (n=9) | 0,7432 ± 0,1159 (n=9) |
| `unsw->ton` | 0,3255 ± 0,0040 (n=10) | 0,6906 ± 0,0827 (n=10) | 0,7530 ± 0,0646 (n=10) | 0,7634 ± 0,1013 (n=10) |

I conteggi `n` sotto 10 sono seed dove quella cella specifica non ha trovato
normali (soprattutto `cic->ton`, che non e' un target BoT-IoT ma perde
comunque 1-2 seed per budget — la raccolta di etichette e' piu' fragile in
generale di quanto le sole direzioni "target BoT-IoT" suggerissero).

**Delta per seed (media delle tre coppie dentro ciascun seed, poi media ±
dev.std sui seed disponibili) — stessa unita' di replicazione della sezione
precedente, ma con una scelta metodologica da dichiarare esplicitamente: le
celle mancanti non sono casuali, sono assenti proprio dove la selezione non
trova normali, cioe' dove il problema e' piu' difficile — quindi come le si
tratta introduce un bias sistematico, non solo rumore. Due varianti:**

- **Metodo A — coppie disponibili**: per ogni seed, media delle sole coppie
  che quel seed ha (2 su 3 se una fallisce). Usa piu' dati, ma pesa un seed
  con 2 coppie riuscite quanto uno con 3.
- **Metodo B — solo seed completi**: media solo sui seed dove **tutte e tre**
  le coppie hanno prodotto un numero. Piu' conservativo, scarta interi seed.

**Correzione sul denominatore, trovata rileggendo questa sezione**: la prima
stesura usava "rapporto = |media|/dev.std" come se fosse un test di
significativita'. Non lo e' — per sapere se una **media** su n seed e'
distinguibile da zero il denominatore e' l'errore standard, dev.std/√n, cioe'
il test t a un campione. Con dev.std invece di dev.std/√n il rapporto e'
sottostimato di un fattore √n (qui √10≈3,16), quindi la stesura precedente
**sottostimava** l'evidenza per non adattato e 128 etichette, non la
sopravvalutava. Tabella rifatta con t e p (test t a un campione, H0: media=0):

| budget | A: media±std (n) | A: t | A: p | A: neg/pos | B: media±std (n) | B: t | B: p |
|---|---|---|---|---|---|---|---|
| non adattato | −0,1297±0,0249 (10) | **−16,48** | **<0,0001** | 10/0 | −0,1297±0,0249 (10) | −16,48 | <0,0001 |
| 8 etichette | −0,0065±0,0605 (9) | −0,32 | 0,755 | 5/4 | −0,0089±0,0643 (8) | −0,39 | 0,708 |
| 32 etichette | −0,0334±0,0675 (10) | −1,57 | 0,152 | **7/3** | −0,0197±0,0547 (9) | −1,08 | 0,313 |
| 128 etichette | −0,0863±0,0412 (10) | **−6,62** | **0,0001** | 10/0 | −0,0822±0,0415 (9) | −5,94 | 0,0003 |

(neg/pos = seed con delta negativo/positivo, cioe' a favore di CIC/UNSW.
**Correzione precedente, confermata qui**: una stesura ancora piu' vecchia
riportava "3/10 concordi" a 32 etichette, incoerente con una media negativa
— l'errore era aver riportato il conteggio dei positivi, non dei concordi;
il valore corretto e' 7/10 negativi.)

I due metodi concordano sulla sostanza (stesso segno di t, stessa
significativita' o non-significativita' su ogni colonna) tranne che a 32
etichette, dove passare da A a B (piu' conservativo) rende il test ancora
meno significativo (p da 0,15 a 0,31), non di piu'.

**Le due colonne che sopravvivevano a 3 seed sopravvivono, ed entrambe
erano gia' piu' solide di come le avevo descritte.** A 3 seed: non adattato
t=−10,58, p=0,009 (gia' molto significativo, anche col campione piccolo);
128 etichette t=−3,44, **p=0,075 — sopra la soglia convenzionale di 0,05**,
la stesura originale l'aveva descritto correttamente come "non e' una
prova" nonostante il "rapporto 1,99" fosse gia' un numero fuorviante (quel
2 non e' mai stato un t corretto: a 3 seed t = mean·√3/std, non mean/std).
A 10 seed: non adattato **t=−16,48, p<0,0001**; 128 etichette **t=−6,62,
p=0,0001** — quello che a 3 seed era sotto la soglia di significativita'
convenzionale ora e' saldamente sopra. A 8 e 32 etichette resta rumore puro
in entrambi i campioni (p sempre >0,15) — a 32 etichette il segno del delta
si e' persino invertito rispetto ai 3 seed originali (+0,0177 con 42-44,
−0,0334 con tutti e dieci), coerente con l'assenza di segnale, non contro
di essa.

**Lo schema "target facile / sorgente debole", ritirato sulla base di 3
seed, va reintrodotto — ma il meccanismo e' diverso da come lo si sarebbe
descritto a naso, ed e' quello il punto.** Segno del delta per coppia
(`+` favorisce UNSW, `−` favorisce CIC) e test binomiale bilaterale
sull'ipotesi che il segno sia casuale (p=0,5):

| coppia | 8 etich. | 32 etich. | 128 etich. |
|---|---|---|---|
| `ton->cic` / `ton->unsw` (target) | 3+/6− (n=9), p=0,51 | 1+/9− (n=10), **p=0,021** | 1+/9− (n=10), **p=0,021** |
| `bot->cic` / `bot->unsw` (target) | 3+/6− (n=9), p=0,51 | 2+/8− (n=10), p=0,11 | 0+/10− (n=10), **p=0,002** |
| `cic->ton` / `unsw->ton` (sorgente) | 8+/0− (n=8), **p=0,008** | 8+/1− (n=9), **p=0,039** | 6+/3− (n=9), p=0,51 |

(Correzione: una stesura precedente riportava 1+/8− per la coppia `ton`
target a 8 etichette e 7+/2− per la coppia sorgente a 32 etichette; i valori
corretti, verificati riga per riga sui dati per seed, sono quelli sopra —
rispettivamente 3+/6− e 8+/1−. L'errore era nel conteggio manuale dei segni,
non nei dati.)

**Le due coppie hanno andamenti opposti rispetto al budget, e non e' un
caso.** La coppia **sorgente** e' fortissima a poche etichette (8/8 a n=8,
p=0,008; 8/9 a n=32, p=0,039) e sparisce a 128 (6/9, p=0,51). La coppia
**target** e' il contrario: assente a 8 etichette (p=0,51 su entrambe le
sotto-coppie), parzialmente emersa a 32 (`ton` gia' significativa a p=0,021,
`bot` ancora a p=0,11), piena a 128 (`ton` p=0,021, `bot` p=0,002).

Il meccanismo e' che le due coppie misurano cose diverse a seconda di quanta
informazione l'adattamento ha gia' incorporato. La coppia sorgente confronta
`cic->ton` con `unsw->ton`: **stesso target, sorgente diversa**. Con poche
etichette il punteggio dipende ancora soprattutto da quanto bene il modello
sorgente ordinava gia' il target prima di adattare — e quello e' fissato dal
modello, non dal campione di etichette, quindi il segno e' quasi
deterministico rispetto al seed (8/8, 8/9). Con molte etichette
l'adattamento riscrive i 13 coefficienti abbastanza da sovrascrivere quel
prior: il punteggio finale dipende sempre meno da quale fosse il modello di
partenza e sempre piu' da quanto il target stesso e' apprendibile con quei
dati — il vantaggio della sorgente si diluisce, e a 128 etichette e'
sparito. La coppia target confronta `ton->cic` con `ton->unsw`: **stessa
sorgente, target diverso**. Qui vale il contrario: con poche etichette non
c'e' abbastanza segnale per distinguere quale terzo dominio sia piu'
adattabile (il rumore della stima domina), e serve arrivare a un budget dove
l'adattamento converge vicino al soffitto di ciascun target perche' la
differenza strutturale fra i due domini (CIC piu' facile, sezione
successiva) diventi visibile.

**Il seed 42 in `ton->cic`: artefatto isolato, non bimodalita'.** Con 10
seed la domanda della sezione precedente ha risposta: a 32 e 128 etichette
il seed 42 e' l'**ultimo** dei 9-10 seed disponibili (rango 1, z-score
−1,85 e −2,77 rispetto alla media degli altri), mentre a 8 etichette e'
nella norma (rango 2 su 9, z-score −0,68). Gli altri nove seed sono
strettamente raggruppati; non c'e' un secondo gruppo, quindi non e'
bimodalita': e' un singolo seed la cui selezione delle etichette a budget
medio-alto produce una base di adattamento persistentemente peggiore, che
non recupera aumentando il budget — anzi peggiora in termini relativi (rango
worst a 32 e 128, nella norma a 8). Non e' stato investigato oltre quale
riga specifica selezionata da quel seed causi l'effetto.

### Quello che i numeri dicono davvero

**CIC-IoT-2023 non trasferisce meglio: e' piu' facile.** Il divario grande e
stabile e' quello non adattato, −0,130 (10 seed, t=−16,48, p<0,0001, 10/10
seed concordi) a favore della terna CIC prima di qualunque etichetta — con
`ton->cic` a 0,446 contro `ton->unsw` a 0,261 e `cic->ton` a 0,564 contro
`unsw->ton` a 0,326. A 10 seed si aggiunge un secondo margine solido, a 128
etichette (−0,086, t=−6,62, p=0,0001, 10/10 concordi), che a 3 seed era
sotto la soglia di significativita' (p=0,075). Un dominio da cui e verso cui
si trasferisce gia' bene senza
adattarsi non e' un banco di prova severo per un metodo di adattamento: e'
lo stesso argomento della sezione 14 (CICIoMT2024 a 0,9923 in-domain con tre
feature), che **sopravvive intatto al cambio di file, di spazio e di
numero di seed**, solo con numeri diversi.

**La conclusione della sezione 14 va quindi corretta solo nella parte
diagnostica, non in quella di merito.** Non e' piu' vero che "il costo di
includere CIC e' lo spazio che impone a tutti gli altri": con la vera
`flow_duration` quel costo non c'e' piu'. Ma resta vero che **la terna del
professore da' numeri migliori perche' contiene un dominio piu' facile**, e
resta la ragione strutturale che nessuna correzione allo spazio elimina: le
righe di CIC-IoT-2023 sono finestre di pacchetti scorrevoli, non flussi
bidirezionali come TON_IoT/BoT-IoT/UNSW-NB15. L'unita' di osservazione e'
diversa, non solo la formula delle feature.

**La raccomandazione della sezione 14 non cambia**, ma cambia il motivo per
cui si sostiene: UNSW-NB15 come terzo dominio dell'analisi principale non
perche' dia numeri piu' alti — non li da' — ma perche' e' piu' difficile e
misura la stessa unita' di osservazione.

**Un bug di igiene, corretto.** Fino a questa revisione il file dei record
per seed (`results/tre_domini_runs_ridotto.csv`) conteneva **solo l'ultima
terna lanciata**: `scripts/tre_domini.py::finalize` scriveva sempre sullo
stesso nome per spazio, senza includere la terna, quindi il run CIC veniva
silenziosamente sovrascritto da quello UNSW eseguito dopo, e la dev.std
sopra era assunta comparabile fra le due terne senza poterla misurare su
entrambe. Corretto includendo la terna nel nome
(`tre_domini_runs_<spazio>_<terna>.csv`) e rilanciando le due terne: i
checkpoint (`artifacts/tre_domini_*.jsonl`) erano gia' completi, quindi il
fix non ha richiesto ricalcolare nulla, solo riscrivere l'output con il nome
giusto. La dev.std per cella qui sopra e' ora misurata su entrambe le terne.

**Un fallimento che a 10 seed si rivela raro, non assoluto — e va corretto
rispetto a come l'avevo scritto la prima volta.** Con 3 seed `cic->bot`
sembrava non raccogliere mai abbastanza normali a 8 e 32 etichette. Con 10:
2/10 seed a 8 etichette, 3/10 a 32, **7/10 a 128** — la resa migliora col
budget invece di restare bloccata, e quando trova normali il risultato non
e' il 0,5146 riportato prima (calcolato su una sola cella riuscita in 3
seed) ma oscilla molto, da 0,358 a 0,926 a seconda del seed. BoT-IoT ha 477
normali su 3,67 M: resta il target piu' difficile per la raccolta di
etichette (sezione 11), ma "non raccoglie abbastanza" descriveva il campione
di 3 seed, non la direzione — la stessa correzione che vale per `ton->bot` e
`unsw->bot`, dettagliata sopra in "Da 3 a 10 seed".

## 16. I tre punti aperti — chiusi, in parte

### 16.1 — Il divario della stima intera in TON→BoT: non era regolarizzazione

L'ipotesi di partenza era l'assenza di L2 nell'ottimizzatore intero
(`kanids/int_adapt.py::fit_gains_int`, discesa del gradiente pura, contro
`LogisticRegression(C=1.0)` in virgola mobile). **Misurata e falsificata**:
aggiungere weight decay verso l'identita' (guadagno=Q15, coerente con "se le
etichette non dicono nulla i guadagni restano dove sono") non chiude il
divario — resta a 0,7321..0,7328 anche a `reg_shift` forte — e verso zero
(la scelta di sklearn) **peggiora** BoT→TON di 4 punti a regolarizzazione
significativa, senza aiutare TON→BoT.

La causa vera era piu' semplice: **non convergenza**, non overfitting. Il
passo (`lr_shift`) e' scelto una volta sola dal primo gradiente e tenuto
fisso per tutte le iterazioni; a 2000 iterazioni un seed su tre (43) restava
a meta' strada (bal 0,5609 contro 0,84 degli altri due).

**Le iterazioni si fissano su UNA sola direzione, TON→BoT, e si validano
sulle altre cinque.** E' la direzione scelta perche' e' quella con cui il
problema e' stato posto: il divario integer-vs-float di questa sezione e'
misurato per la prima volta li' (sezione 6), non su una direzione qualunque
scelta a posteriori perche' guadagnava di piu'. Fissare l'iperparametro su
una direzione sola e riportare le altre cinque come controllo, invece di
scegliere guardando due direzioni e validare sulle restanti quattro (fatto
nella stesura precedente), toglie la circolarita': nessuna delle cinque
direzioni di validazione entra nella scelta del numero.

Sweep di `iters` su TON→BoT da sola, n=128, media ± dev.std sui 3 seed:

| iters | 2000 | 4000 | 6000 | 8000 | 12000 |
|---|---|---|---|---|---|
| ton→bot | 0,7321 ± 0,1496 | 0,7855 ± 0,0700 | 0,7979 ± 0,0487 | 0,8023 ± 0,0400 | 0,7979 ± 0,0378 |

Il salto grosso e' fra 2000 e 4000 (+0,053, oltre la dev.std di entrambi i
livelli: e' li' che sta il seed non convergente). Da 4000 in su i valori
sono **indistinguibili fra loro data la dev.std** (6000 e 8000 differiscono
di 0,0044, un decimo della loro stessa dispersione): con tre soli seed non
si puo' scegliere fra 4000, 6000, 8000 o 12000 sul massimo della media. Si
sceglie **6000** come punto tondo chiaramente oltre il ginocchio della
curva (2000, rotto) senza essere il piu' grande provato: la giustificazione
e' "abbastanza per uscire dalla non convergenza", non "il valore ottimo",
che con questo campione non e' stimabile.

**Controllo mancante, colmato**: la sezione confrontava una riga "2000 iter"
a 3 seed con tutto il resto rilanciato a 10 — campioni di dimensione
diversa affiancati come se fossero comparabili. `scripts/drift_int_adapt.py`
ora accetta `--iters` (di default 6000; con un valore diverso scrive su
file separati, cosi' non sovrascrive ne' il checkpoint ne' l'header C
canonico), e la stessa pipeline e' stata rilanciata a `--iters 2000` sui
10 seed 42-51, sulle sei direzioni. Confronto appaiato per seed, test t:

| direzione | media 2000 iter | media 6000 iter | delta (6000−2000) | t | p |
|---|---|---|---|---|---|
| ton→bot (calibrazione) | 0,7300 (n=9) | 0,7543 (n=9) | +0,0244±0,0600 | 1,22 | 0,258 |
| bot→ton | 0,8315 | 0,8174 | −0,0141±0,0382 | −1,17 | 0,272 |
| ton→unsw | 0,5488 | 0,5480 | −0,0008±0,0436 | −0,06 | 0,954 |
| unsw→ton | 0,7454 | 0,7290 | −0,0164±0,0430 | −1,20 | 0,259 |
| bot→unsw | 0,6271 | 0,6322 | +0,0052±0,0399 | 0,41 | 0,691 |
| unsw→bot | 0,6313 (n=5) | 0,6310 (n=5) | −0,0004±0,0449 | −0,02 | 0,987 |

**Nessuna delle sei direzioni si distingue dal rumore, calibrazione
inclusa — un "65% del divario chiuso" non e' piu' citabile come effetto
medio.** Il numero precedente veniva da un confronto fra un campione a 3
seed (2000 iter: 0,7321) e uno a 10 (6000 iter: 0,7979): non solo diversi
per dimensione, ma il primo includeva proprio il seed che genera l'unico
effetto reale, pesato un terzo invece di un nono.

**Ma l'effetto reale non e' scomparso: e' concentrato, non diffuso — e
questo e' quello che la media nasconde.** Guardando i 9 seed uno per uno su
TON→BoT: otto hanno delta fra −0,007 e +0,015 (rumore puro), e **uno,
il seed 43, ha delta +0,183** (0,5609→0,7441) — lo stesso seed che aveva
motivato la diagnosi originale ("un seed su tre restava a meta' strada").
**6000 iterazioni non spostano la media, salvano i seed rari che a 2000
non convergono.** E' una forma di assicurazione contro la coda, non un
guadagno medio — una lettura piu' precisa e piu' difendibile di
"chiude il 65% del divario", che descriveva un effetto medio inesistente.
Un ottimizzatore con passo adattivo (non ancora provato) potrebbe sostituire
questa assicurazione a costo fisso — 6000 iterazioni sempre, anche sugli
otto seed su nove che non ne hanno bisogno — con una che si attiva solo
quando la convergenza e' davvero lenta.

Con `iters=6000` fissato su TON→BoT da sola, le altre cinque direzioni,
n=128 (fuori dalla scelta dell'iperparametro), rilanciate su **10 seed**
(42-51), media±dev.std:

| n=128 | ton→bot (calibrazione) | bot→ton | ton→unsw | unsw→ton | bot→unsw | unsw→bot |
|---|---|---|---|---|---|---|
| interi, 6000 iter, float | 0,8124±0,0948 (n=9) | 0,8203±0,0781 | 0,5547±0,0861 | 0,7640±0,0298 | 0,6277±0,0913 | 0,5740±0,0568 (n=5) |
| interi, 6000 iter, interi | **0,7543±0,1144** (n=9) | 0,8174±0,0764 | 0,5480±0,0565 | 0,7290±0,0397 | 0,6322±0,0940 | 0,6310±0,1064 (n=5) |
| delta (interi − float) | −0,0581 | −0,0029 | −0,0067 | −0,0350 | +0,0045 | +0,0570 |

(`ton->bot` e `unsw->bot` non hanno n=10: la selezione delle etichette non
trova entrambe le classi in 1 seed su 10 per `ton->bot` e in 5 su 10 per
`unsw->bot` — lo stesso fenomeno di "impossibile e' raro, non assoluto"
misurato in sezione 15, qui nello spazio ricco. Il confronto "2000 iter
contro 6000" a 10 seed appaiati e' nel riquadro sopra, non in questa
tabella: qui il confronto e' interi-contro-float, entrambi a 6000
iterazioni.)

**Il bilancio resta 2 su 5 in miglioramento, 3 su 5 in peggioramento, media
+0,0034 — sostanzialmente nullo, la stessa conclusione di prima.** Ma le
direzioni che migliorano non sono le stesse: a 3 seed migliorava BoT→TON e
BoT→UNSW; a 10 seed BoT→TON e' leggermente sotto zero (−0,0029, dentro il
rumore) mentre UNSW→BoT migliora parecchio (+0,057) — sul sottoinsieme di 5
seed dove la selezione riesce per entrambe le catene, quindi con piu'
incertezza degli altri numeri di questa tabella. Anche sulla direzione di
calibrazione il divario interi-contro-float a 10 seed (−0,058) e' piu'
ampio di quello misurato a 3 (−0,035): il divario non e' chiuso quanto
sembrava con un campione piccolo, anche se la tendenza (interi sotto float,
non sopra) e' la stessa. La lettura di fondo resta la stessa: **6000
iterazioni sistemano la non convergenza sulla direzione per cui sono state
scelte, e sulle altre non fanno né bene né male in modo distinguibile** —
confermata, non indebolita, dal passaggio a 10 seed.

Verificato con 200 golden vector rigenerati (`mcu/kan_int_adapt.h`,
28 965 B) e **riverificato bit-esatto**: `mcu/run_int_adapt_check.cpp`
compilato con g++ 13.3.0 a `-O2` restituisce `logit diversi: 0, decisioni
diverse: 0`, 24 byte riscritti per l'adattamento (12 int16). Il passaggio a
6000 iterazioni tocca solo la stima dei guadagni, non il kernel di
inferenza: la catena resta identica al riferimento Python bit per bit.

### 16.2 — I minimi quadrati ricorsivi: due bug, non uno

Il crollo in BoT→TON aveva due cause indipendenti nello stesso file
(`scripts/drift_graduale.py::StatSufficienti`), e solo correggendole
entrambe il metodo regge.

**Bug 1 — il fattore di dimenticanza applicato nel posto sbagliato.** Il
forgetting `lam=0.98` veniva applicato dentro le 5 sotto-iterazioni IRLS
(una volta per rilinearizzazione), non una volta per aggiornamento: la
storia decadeva a `lam^5 ≈ 0,904` per batch invece di `0,98`, componendosi
su 20 batch fino a un fattore ~7 volte piu' piccolo del previsto, mentre il
batch corrente veniva riaccumulato cinque volte su basi scontate in modo
incoerente fra loro. Corretto applicando lo sconto una sola volta
all'ingresso di ogni chiamata.

**Da solo, questo fix ha peggiorato le cose**, ed e' un risultato che vale
la pena riportare per intero: con `ridge=1e-3` (il valore originale) la
correzione dell'ordine del forgetting porta le vittorie su statico da 4/6 a
**2/6**, con TON→BoT che crolla da ~0,91 (valore originale, sezione 9) a
0,7077. La correzione era comunque giusta — il calcolo precedente non era
quello che il commento nel codice descriveva — ma non era la causa
principale del crollo.

**Bug 2 — la causa vera: niente prior contro la quasi-separazione.**
`ridge=1e-3` lascia la prima stima IRLS quasi solo alla verosimiglianza dei
32 campioni del primo batch. Con 13 parametri e classi sbilanciate, la
quasi-separazione (lo stesso problema gia' documentato per Firth in sezione
9) fa divergere i coefficienti al primo aggiornamento, e il forgetting lento
(lam=0,98, tempo di dimezzamento ~34 batch, piu' lungo dei 20 batch della
simulazione) non lascia il tempo di riprendersi.

**Anche `ridge` si fissa su UNA sola direzione, TON→BoT — stessa scelta e
stessa motivazione della sezione 16.1** (e' la direzione su cui il problema
integer-vs-float e' stato posto), **e si valida sulle altre cinque**,
incluso BoT→TON, che nella stesura precedente faceva parte dello sweep
insieme a TON→BoT. Sweep di `ridge` su TON→BoT da sola, media ± dev.std sui
3 seed:

| ridge | 1e-3 (originale) | 1e-2 | **1e-1** | 1,0 | 10,0 |
|---|---|---|---|---|---|
| ton→bot | 0,7077 ± 0,0288 | 0,8915 ± 0,0496 | **0,9235 ± 0,0057** | 0,9173 ± 0,0066 | 0,8627 ± 0,0465 |

(Rigenerabile con `scripts/sweep_iperparametri.py`, che riusa
`drift_graduale.py::run_unit` per intero invece di reimplementare una
versione piu' veloce: i numeri sono per costruzione gli stessi delle
tabelle pubblicate, non solo simili.)

**Non c'e' un massimo netto: c'e' un ginocchio e poi un altopiano largo.**
Il salto sta fra `1e-3` e `1e-2` (+0,18, sei volte la dev.std di `1e-2`);
da li' in poi `1e-2`, `1e-1` e `1,0` sono **indistinguibili fra loro**
(`1e-1` supera `1e-2` di 0,032, meno della dev.std di `1e-2`; supera `1,0`
di 0,0062, comparabile alla loro). Solo `10,0` ricomincia a degradare
(−0,06). Clippare i coefficienti invece del ridge (provato come
alternativa, sempre su TON→BoT: `1e-3` + clip(θ,5) da' 0,7432 ± 0,0173,
`1e-1` + clip(θ,5) da' 0,9208 ± 0,0107, indistinguibile da `1e-1` da solo)
non aggiunge nulla: non era un problema di range dei coefficienti, era di
quanta informazione a priori c'e' prima che i dati la sovrastino.
**`ridge=0,1`** e' il punto scelto, sulla sola TON→BoT — al centro
dell'altopiano, non su un ottimo che i dati non identificano.

Questo cambia cosa serve al ridge adattivo (punto 5 di "Cosa resta da
fare"): non deve **azzeccare** un valore, deve solo **stare dentro
l'altopiano** — due ordini di grandezza di tolleranza, da `1e-2` a `1,0`. Un
criterio grossolano sulla frazione di classe minoritaria del primo batch e'
quindi sufficiente, e non serve calibrarlo con precisione.

Con `ridge=0,1` fissato su TON→BoT da sola, le altre cinque direzioni (fuori
dalla scelta dell'iperparametro), rilanciate su **10 seed** (42-51), media
sui batch poi media±dev.std sui seed — a differenza della sezione 15, qui
tutte e sei le direzioni hanno n=10, nessuna cella mancante:

| | ton→bot (calibrazione) | bot→ton | bot→unsw | ton→unsw | unsw→bot | unsw→ton |
|---|---|---|---|---|---|---|
| statico | 0,8184±0,0087 | 0,8179±0,0472 | 0,7177±0,0079 | 0,5766±0,0020 | 0,7751±0,0117 | 0,5626±0,0156 |
| **stat_13x13 (RLS)** | **0,9247±0,0191** | 0,7434±0,0220 | 0,7099±0,0509 | **0,6970±0,0308** | **0,8188±0,0123** | **0,6939±0,0169** |
| buffer (ogni_batch) | 0,9459±0,0212 | 0,8881±0,0273 | 0,8207±0,0117 | 0,7235±0,0251 | 0,8594±0,0112 | 0,7083±0,0161 |

**Sulle cinque direzioni di validazione il bilancio resta 3 su 5 in
miglioramento (TON→UNSW +0,120, UNSW→TON +0,131, UNSW→BoT +0,044), 2 su 5 in
peggioramento (BoT→TON −0,075, BoT→UNSW −0,008) — stesso schema di prima, e
entrambe le perdite si sono ristrette** (BoT→TON da −0,113 a −0,075, BoT→UNSW
da −0,034 a un −0,008 ormai dentro la dev.std). La direzione di
calibrazione, come per le iterazioni, e' quella che guadagna piu' di tutte
(+0,106) — atteso, dato che l'iperparametro e' scelto per massimizzarla. La
differenza rispetto a 16.1 regge anche a 10 seed: qui la validazione fuori
campione batte davvero lo statico nella maggioranza dei casi (3/5), quindi
`ridge=0,1` non e' solo "sistema TON→BoT e altrove non fa danno", **regge
davvero fuori dal campione di calibrazione — confermato, non indebolito,
passando da 3 a 10 seed**.

**Il pattern nelle due perdite e' specifico, non casuale**: le uniche due
direzioni dove RLS perde contro lo statico (BoT→TON, BoT→UNSW) sono anche le
uniche due dove **BoT-IoT e' la sorgente**. Contando anche TON→BoT (la
calibrazione, che vince), lo schema completo sulle sei e' "vince sempre
tranne quando la sorgente e' BoT-IoT": non e' piu' "crolla in una direzione
su due" (la lettura della sezione 13, basata su due sole direzioni), e' "la
sorgente sbilanciata (0,013% di normali) rende difficile il primo
aggiornamento IRLS a prescindere dal target" — una diagnosi piu' precisa,
che dice dove investigare dopo (probabilmente un ridge adattivo sulla
frazione di classe minoritaria del primo batch, non ancora provato).

> **Nota in avanti (sezione 18.4) — questa affermazione va riscritta, non
> solo qualificata.** "Vince sempre tranne quando la sorgente e' BoT-IoT"
> e' stato misurato al rapporto di undersampling 1:50 (e confermato a
> ratio 20 e 100, dove la perdita di BoT→UNSW diventa anzi significativa:
> p=0,036). **Ma a ratio=3 — il secondo punto della griglia, non il piu'
> estremo — BoT→UNSW cambia segno e VINCE in modo significativo
> (+0,029, p=0,035)**, mentre BoT→TON continua a perdere: le due direzioni
> con BoT-IoT come sorgente smettono di comportarsi allo stesso modo. E
> **a ratio=1 UNSW→BoT — sorgente UNSW-NB15, non BoT-IoT — perde in modo
> significativo** (−0,058, p=0,038), mentre BoT→TON smette di perdere in
> modo distinguibile (p=0,93) e BoT→UNSW vince ancora piu' nettamente
> (+0,059). La sorgente da sola non predice piu' chi perde: il pattern
> "BoT-IoT sorgente ⇒ perde" descriveva correttamente la sola zona di
> griglia 20-100, non una proprieta' del meccanismo. La diagnosi
> (quasi-separazione al primo aggiornamento IRLS con pochi normali nel
> training) resta plausibile come un fattore, ma non spiega da sola perche'
> le due direzioni BoT-sorgente divergano fra loro cambiando ratio — la
> spiegazione completa resta da trovare (vedi "Cosa resta da fare").

### 16.3 — L'innesco a martingala in aritmetica intera: portato, con un bug trovato per strada

Stesso principio della sigmoide a LUT (sezione 6): log e potenza frazionaria
si calcolano **una volta, offline**, per costruire una tabella; a runtime
restano solo confronti fra interi e somme
(`kanids/int_adapt.py::build_martingale_lut`, `martingale_batch_int`,
`martingale_update_int`, nuovo script `scripts/drift_graduale_int.py`).
Punteggio, moltiplicatori ed edge sono la stessa catena bit-fedele della
sezione 6 (`quantize_edges` + `edge_parts_int` + `int_forward`), con
`fit_gains_int(iters=6000)` per l'aggiornamento dei guadagni.

**Il primo tentativo aveva il segno invertito.** La costruzione della LUT
usava `p = (n_cal - rango + 0,5)/(n_cal+1)` invece di
`p = (rango + 0,5)/(n_cal+1)`: le righe piu' anomale (rango basso, poche
calibrazioni con conformita' maggiore o uguale) finivano con un p-value
vicino a 1 invece che vicino a 0. L'effetto era visibile prima ancora di
guardare il codice: su TON→BoT l'incremento di log-martingala per batch
diventava **piu' negativo** man mano che la contaminazione dal target
cresceva (da −79,8 a batch 0 fino a −989 a batch 19) — l'esatto contrario di
quello che una statistica di deriva deve fare. Corretto invertendo la
formula.

**Dopo la correzione, TON→BoT resta comunque il caso piu' debole — ma non
per un bug.** Anche usando il rango esatto (non binnato, per escludere che
fosse un problema di risoluzione della LUT a 256 voci), il p-value medio a
contaminazione piena scende solo fino a 0,40, mentre per qualunque epsilon
in (0,1) la soglia di innesco raggiungibile e' `p* = exp(log(eps)/(1-eps))`,
che per eps→1 tende asintoticamente a 0,368: **nessuna scelta di eps fa
scattare la martingala su questa direzione con questo segnale di
conformita'**, non solo eps=0,9. E' un limite del segnale (|z - mediana
della calibrazione| sull'uscita quantizzata a 8 bit), non dell'
implementazione.

**Rilanciato su 10 seed (42-51): questa e' la misura che il compito 2 ha
chiesto esplicitamente, perche' la sezione 15 (spazio ridotto) aveva
mostrato che "non scatta mai" puo' essere un artefatto del campione a 3
seed. Qui il meccanismo e' diverso — non e' la selezione delle etichette che
fallisce (`adaptive_pick`), e' la statistica di martingala che non supera la
soglia — e va verificato separatamente, nello spazio ricco, non dedotto da
quella misura.** Seed su 10 in cui l'innesco scatta almeno una volta
sui 19 batch adattabili:

| direzione | float, seed che scattano | intero, seed che scattano |
|---|---|---|
| bot→ton | 10/10 | 10/10 |
| bot→unsw | 10/10 | 10/10 |
| ton→bot | 10/10 | 8/10 |
| ton→unsw | **5/10** | **0/10** |
| unsw→bot | **0/10** | **8/10** |
| unsw→ton | 10/10 | 10/10 |

Balanced accuracy media sui batch, poi media±dev.std sui seed:

| | bot→ton | bot→unsw | ton→bot | ton→unsw | unsw→bot | unsw→ton |
|---|---|---|---|---|---|---|
| statico (intero) | 0,8417±0,0209 | 0,7040±0,0082 | 0,7683±0,0338 | 0,5876±0,0338 | 0,5987±0,0081 | 0,6227±0,0108 |
| **martingala intera** | 0,8743±0,0213 | 0,8387±0,0162 | 0,8201±0,0290 | 0,5876±0,0338 | **0,7461±0,0897** | 0,7068±0,0172 |
| martingala float (rif.) | 0,8906±0,0237 | 0,8311±0,0155 | 0,9451±0,0164 | **0,5985±0,0454** | 0,7751±0,0117 | 0,7099±0,0156 |

**`unsw→bot` — la scoperta della sezione 13 regge, e si rafforza.** Il float
non scatta **mai**, in nessuno dei 10 seed (0/190 batch adattabili
possibili) — non e' un artefatto del campione a 3 seed come temuto, e' un
"mai" genuinamente robusto: il meccanismo di innesco (statistica di
martingala su soglia fissa) e' qualitativamente diverso dalla selezione
delle etichette della sezione 15 (che dipende dal sorteggio casuale di poche
righe), quindi non eredita la sua fragilita'. L'intero scatta in **8 seed su
10**, non solo "0, 6 o 12 volte in un campione di 3": la frase "la catena
intera funziona in UNSW→BoT dove quella float falliva del tutto" era gia'
corretta a 3 seed e resta corretta a 10, con un dato piu' solido a
sostegno.

**`ton→unsw` va riscritta: il float non e' "debole", e' bimodale per
seed.** A 3 seed sembrava scattare raramente (3 volte su 19 batch, in un
solo run). A 10 seed la vera struttura e' diversa: **5 seed su 10 non
scattano mai, gli altri 5 scattano da 1 a 14 volte** (seed 50: 14
adattamenti, bal 0,73; seed 42: 3 adattamenti, bal 0,38; seed 43-45-47-49:
zero). Non e' un segnale debole ma costante, e' un segnale che c'e' o non
c'e' a seconda del seed — probabilmente della composizione del batch di
calibrazione, non ancora indagato. L'intero resta a **0/10**, confermato
invariato: qui la quantizzazione toglie un segnale che nel float esiste
davvero circa meta' delle volte, non un residuo trascurabile.

**`ton→bot` e' piu' affidabile di quanto i 3 seed suggerissero, ma resta
inconsistente in qualita'.** L'intero scatta in **8 seed su 10** (prima
sembrava 0,8,10 batch su un campione di soli 3 run, lasciando pensare a
un'affidabilita' bassa: con 10 seed solo 2 non scattano mai). Ma quando
scatta la qualita' oscilla molto — balanced accuracy finale da 0,39 a 0,99
a seconda del seed — mentre il float, che scatta sempre (10/10), converge
in modo molto piu' stretto (0,9451±0,0164). **Il float resta piu' forte
qui**, ma "l'intero e' inaffidabile" era una lettura del campione piccolo:
l'innesco stesso funziona quasi sempre, e' la qualita' dell'adattamento
successivo a restare rumorosa.

**Le tre direzioni gia' solide restano solide**: BoT→TON, BoT→UNSW e
UNSW→TON scattano 10/10 in entrambe le versioni, con balanced accuracy
entro pochi punti dal float in tutti e tre i casi.

**Il quadro non e' "portato con successo" ne' "non funziona": e' portato,
corretto da un bug reale, affidabile (10/10) in 3 direzioni su 6 e
parzialmente affidabile (8/10) in una quarta, con lo stesso pattern
speculare gia' visto per i guadagni interi** — aiuta dove il float fatica di
piu' con gli sbilanciamenti estremi (`unsw→bot`), fatica dove il float ha
un vantaggio strutturale (`ton→bot`) o dove il segnale stesso e'
intermittente anche in float (`ton→unsw`).

**Il risultato piu' forte di questa sezione non e' l'innesco, ed e' passato
inosservato: il riadattamento continuo in aritmetica intera batte lo statico
in 6 direzioni su 6.** Rilanciato a 10 seed:

| | bot→ton | bot→unsw | ton→bot | ton→unsw | unsw→bot | unsw→ton |
|---|---|---|---|---|---|---|
| statico (intero) | 0,8417±0,0209 | 0,7040±0,0082 | 0,7683±0,0338 | 0,5876±0,0338 | 0,5987±0,0081 | 0,6227±0,0108 |
| **ogni batch, intero** | 0,8819±0,0282 | **0,8360±0,0167** | 0,9076±0,0355 | **0,7287±0,0143** | 0,8594±0,0116 | 0,7057±0,0142 |
| ogni batch, float (rif.) | 0,8881±0,0273 | 0,8207±0,0117 | **0,9459±0,0212** | 0,7235±0,0251 | 0,8594±0,0112 | 0,7083±0,0161 |

**Questa e' l'affermazione che adesso non e' piu' discutibile.** Delta
(intero − statico) appaiato per seed, test t a un campione:

| direzione | delta | t | p | segni concordi |
|---|---|---|---|---|
| unsw→bot | +0,2607±0,0090 | **91,55** | <0,000001 | 10/10 |
| bot→unsw | +0,1319±0,0165 | **25,33** | <0,000001 | 10/10 |
| ton→unsw | +0,1411±0,0260 | **17,13** | <0,000001 | 10/10 |
| unsw→ton | +0,0830±0,0163 | **16,11** | <0,000001 | 10/10 |
| ton→bot | +0,1393±0,0351 | **12,57** | 0,000001 | 10/10 |
| bot→ton | +0,0402±0,0387 | **3,29** | 0,009 | 9/10 |

Sei direzioni su sei, |t| da 3,3 a 91,5, guadagni da +0,040 a +0,261, segni
concordi 9-10 su 10: questa era gia' "l'affermazione piu' solida di tutto il
lavoro" nella sezione 13, e ora ha i numeri per esserlo davvero.

> **Nota in avanti (sezione 18.4)**: "sei direzioni su sei, \|t\| da 3,3 a
> 91,5" e' una misura al rapporto di undersampling 1:50, non una proprieta'
> del metodo — e l'estremo inferiore era gia' il punto piu' debole. Da
> ratio=3 a ratio=100 la tabella regge (6/6, t sempre sopra soglia, il
> minimo scende a 2,5 a ratio=100). **A ratio=1 (sorgente ribilanciata
> esattamente 1:1) diventa 4 direzioni su 6**: `unsw→bot` pareggia lo
> statico bit per bit su tutti i dieci seed (il buffer di adattamento non
> accumula mai entrambe le classi, quindi i guadagni restano all'identita'
> anche se il contatore "adattamenti" sale), e `bot→ton` smette di
> distinguersi dal rumore (t=1,49, p=0,17: in 8 seed su 10 il delta e'
> zero per lo stesso motivo, solo 2 aggiornano davvero). Il meccanismo non
> e' nell'adattamento ma nella selezione delle etichette a monte, che con
> la sorgente ribilanciata all'estremo trova piu' spesso una sola classe
> nel target. "Sei su sei" va letto come "sei su sei per ratio fra 3 e
> 100", non come un fatto assoluto.

**Il conteggio "vince in 4, perde in 2" per intero-contro-float, invece,
era sbagliato per lo stesso motivo del paragrafo precedente — media/dev.std
invece di media/errore standard — ma nella direzione opposta: non
sottostimava un'evidenza forte, mascherava due differenze reali.** Delta
(intero − float) appaiato per seed, test t a un campione:

| direzione | delta | t | p | segni concordi |
|---|---|---|---|---|
| **ton→bot** | **−0,0383±0,0323** | **−3,75** | **0,0045** | 10/10 (float vince) |
| **bot→unsw** | **+0,0153±0,0171** | **2,83** | **0,020** | 8/10 (intero vince) |
| bot→ton | −0,0063±0,0451 | −0,44 | 0,671 | 7/10 |
| ton→unsw | +0,0052±0,0228 | +0,72 | 0,490 | 7/10 |
| unsw→ton | −0,0026±0,0143 | −0,57 | 0,583 | 5/5 |
| unsw→bot | −0,0000±0,0105 | −0,00 | 0,997 | 6/4 |

**Due direzioni su sei si distinguono, non zero.** Su TON→BoT il float
mantiene un vantaggio piccolo ma reale (10/10 seed concordi, p=0,0045): e'
la direzione dove il float era gia' piu' forte nella martingala (tabella
sopra), coerente con un pattern piu' ampio, non un caso isolato. Su
BoT→UNSW l'intero ha un guadagno piccolo ma reale (8/10, p=0,020). Le altre
quattro restano indistinguibili dal rumore (p tutti >0,45). La lettura
corretta e' **"costo piccolo ma reale e consistente in TON→BoT, guadagno
piccolo in BoT→UNSW, indistinguibile nelle altre quattro"** — non "nessun
costo misurabile" (quello che la versione con denominatore sbagliato
suggeriva) ne' "vince quasi sempre" (la lettura a 3 seed): la catena intera
ha un piccolo prezzo reale su una direzione e un piccolo vantaggio reale su
un'altra, non e' ne' gratis ne' costosa in modo uniforme.

Questo e' anche l'unico punto del documento dove la correzione del
denominatore cambiava la sostanza in entrambe le direzioni nello stesso
paragrafo: nella tabella "intero contro statico" sopra mascherava quanto
forte fosse gia' il risultato (dev.std sola dava rapporti 1,2-fino a circa
29, sempre "sembra forte" ma senza un numero confrontabile con una soglia);
in questa tabella "intero contro float" mascherava due differenze reali
dietro "tutto rumore". La sezione 15 aveva lo stesso bug sulla tabella dei
delta per seed (sopra in questo documento): la' sottostimava, qui
nascondeva — vale la pena ricontrollare ogni futuro confronto di medie su
seed con il t-test invece del rapporto a occhio.

---

## 17. Passo adattivo, ridge adattivo, e il modello di costo

La sezione 11 ha misurato che, in valore atteso, i 13 coefficienti e il
rifit completo si annullano sull'accuratezza. **Se l'accuratezza e' un
pareggio, l'argomento a favore dei 13 coefficienti diventa il costo**: 24
byte contro 250, nessun riaddestramento sul dispositivo, catena intera
bit-esatta. Questa sezione chiude i due punti aperti (16.1, 16.2) e poi
costruisce quel numero esplicitamente.

### 17a — Passo adattivo in `fit_gains_int`

La sezione 16.1 aveva misurato che, a campione appaiato, 6000 iterazioni
fisse non spostano la media: salvano un solo seed su nove che a 2000
iterazioni non era ancora convergente, e quel seed non era vicino alla
convergenza nemmeno a **20 000** iterazioni con il passo fisso scelto dal
primo gradiente — non un problema di quante iterazioni, ma di un passo
troppo piccolo per la curvatura di quel problema specifico (un edge saturava
al clip mentre gli altri due si muovevano di poche unita' per iterazione).

`kanids/int_adapt.py::fit_gains_int` ha ora `adaptive=True`, due meccanismi:

1. **Passo che si dimezza sui plateau**: ogni 100 iterazioni si confronta
   la perdita pesata corrente con il minimo osservato; se non e' migliorata,
   `lr_shift += 1` (il passo si dimezza). Il passo iniziale resta quello
   scelto dal primo gradiente, ma non e' piu' fisso per tutta la discesa.
2. **Arresto esatto sulla convergenza**: la ricorrenza g,b → g',b' e'
   deterministica (stessi P, y, mult), quindi un aggiornamento che lascia
   g e b **esattamente invariati** e' una prova di punto fisso — non serve
   una tolleranza. Si ferma anche se la perdita non migliora per 5 finestre
   consecutive dopo aver gia' dimezzato il passo (arresto pratico).

Rilanciato su TON→BoT, il seed piu' lento (43): con passo fisso, a 20 000
iterazioni la perdita era ancora in discesa. Con passo adattivo, converge in
12 600 iterazioni a una balanced accuracy **migliore** di quella del passo
fisso a 6000 (0,765 contro 0,744). Sulle sei direzioni, `--adaptive` in
`scripts/drift_int_adapt.py`, n=128, 10 seed, confrontato con 6000 fisso
(appaiato per seed):

| direzione | iterazioni usate (media, min-max) | bal_acc adattivo | bal_acc 6000 fisso | t | p |
|---|---|---|---|---|---|
| ton→bot (n=9) | 7933 (3300-12700) | 0,7583 | 0,7543 | +1,55 | 0,16 |
| bot→ton | 6680 (4200-12800) | 0,8222 | 0,8174 | +0,79 | 0,45 |
| ton→unsw | 4220 (2700-5800) | 0,5501 | 0,5480 | +0,15 | 0,88 |
| unsw→ton | 5160 (1800-8600) | 0,7321 | 0,7290 | +1,37 | 0,20 |
| bot→unsw | 4130 (2600-6300) | 0,6270 | 0,6322 | −0,41 | 0,69 |
| unsw→bot (n=5) | 7540 (2900-11000) | 0,6202 | 0,6310 | −1,02 | 0,37 |

**Nessuna differenza di accuratezza e' distinguibile dal rumore in nessuna
direzione (tutti p>0,15): il passo adattivo e quello fisso a 6000 sono
statisticamente pari.** Quello che cambia e' come il costo si distribuisce.
La media delle sei medie e' 5944 iterazioni — quasi esattamente 6000 — ma
la distribuzione e' larga: le direzioni piu' facili convergono in
1800-2700 iterazioni (un terzo del fisso), le piu' difficili arrivano a
12 700-12 800 (piu' del doppio). **Il passo adattivo non riduce il costo
medio, lo riallinea a quanto serve davvero per seed**: nessun seed resta
sotto-convergente come poteva succedere a 2000 fisse, e nessun seed facile
paga il costo pieno di 6000 quando ne bastano 2000.

### 17b — Ridge adattivo per `StatSufficienti`: due tentativi, nessuno migliora il fisso

La sezione 16.2 aveva diagnosticato il meccanismo: RLS perde contro lo
statico esattamente nelle due direzioni dove BoT-IoT e' la sorgente, perche'
il primo aggiornamento arriva con pochi normali nel batch e la
quasi-separazione destabilizza la stima prima che lo storico si sia
accumulato. Due criteri per un ridge che si adatta, invece di restare fisso
a 0,1, provati su `scripts/drift_graduale.py::StatSufficienti`:

**Tentativo 1 — ridge alto quando il batch corrente ha pochi normali
(`ridge_mode="per_batch"`).** Misurato e **falsificato**: su `bot->ton` (3
seed) peggiora nettamente, 0,667 contro 0,751 del ridge fisso. Il conteggio
di normali per batch oscilla molto (0-9 su 32 osservati nella stessa
direzione, stesso seed) per ragioni indipendenti dal vero rischio — la
composizione del batch cambia con la contaminazione crescente — e alternare
ridge alto/basso su quella base aggiunge instabilita' invece di toglierla.

**Tentativo 2 — ridge alto solo per i primi `warmup_updates` aggiornamenti
(`ridge_mode="warmup"`, default 3).** Colpisce il meccanismo diagnosticato
piu' da vicino: il rischio e' che *l'informazione accumulata* sia ancora
debole, non che un batch qualsiasi sia sbilanciato. Su un campione ridotto
(bot→ton, 3 seed) sembrava promettente (0,78 contro 0,75 del fisso) — **ma
non ha retto a 10 seed**:

| direzione | statico | fisso (0,1) | adattivo (warmup) | adattivo−fisso, t | p |
|---|---|---|---|---|---|
| ton→bot | 0,8184 | 0,9247 | 0,9124 | −1,50 | 0,17 |
| bot→ton | 0,8179 | 0,7434 | 0,7192 | −1,83 | 0,10 |
| ton→unsw | 0,5766 | 0,6970 | 0,7004 | +0,68 | 0,51 |
| unsw→ton | 0,5626 | 0,6939 | 0,6937 | −0,09 | 0,93 |
| bot→unsw | 0,7177 | 0,7099 | 0,6839 | −1,64 | 0,13 |
| unsw→bot | 0,7751 | 0,8188 | 0,8168 | −0,67 | 0,52 |

**Nessuna delle sei direzioni si distingue dal ridge fisso (tutti p>0,10),
e sulle due direzioni che il ridge adattivo doveva sistemare la tendenza va
nella direzione sbagliata**: su `bot→ton` l'adattivo perde contro lo
statico ancora di piu' del fisso (−0,099 contro −0,075), su `bot→unsw`
resta una perdita simile (−0,034 contro −0,008, qui il fisso era gia'
quasi a pareggio). **Il ridge adattivo non migliora il fisso — ne' il primo
tentativo ne' il secondo — e questo va riportato come il tentativo che
resta aperto, non richiuso.** La diagnosi (quasi-separazione al primo
aggiornamento con pochi normali) resta valida; il rimedio provato non basta.
Il ridge fisso a 0,1, scelto sull'altopiano misurato in 16.2, resta la
scelta migliore trovata finora.

### 17c — Il modello di costo

Con l'accuratezza in pareggio (sezione 11) e il ridge adattivo che non
migliora il fisso (17b), l'argomento che regge e' il costo. Operazioni
moltiplicazione-accumulo (MAC) int32, lookup in tabella e byte di RAM di
picco, contati sul codice di riferimento gia' verificato bit-esatto dove
possibile (inferenza, stima dei guadagni, martingala — sezioni 6/16.1/16.3)
o proiettati con la stessa struttura di operazioni dove l'implementazione
resta float (RLS, dichiarato esplicitamente).

**Assunzioni dichiarate**: un MAC = una moltiplicazione fra due interi a 32
bit con accumulo, l'unita' atomica; nessun numero di cicli per MAC e'
assunto — si moltiplica per i cicli-per-operazione del target scelto per
ottenere il tempo, come richiesto. `d=13` segue la convenzione del
documento (12 edge + 1 termine noto); per il conteggio delle operazioni
sugli edge, 10 sono numerici (spline cubica) e 2 categorici (tabella +
shift, senza moltiplicazioni proprie). Le costanti piccole nella
combinazione della spline (3, 6, 4, 1) si assumono compilate a shift-somma,
non moltiplicazioni hardware. Per la RAM, i contributi per-edge (`Ps`
interno a `fit_gains_int`) si assumono int16 dopo la riscalatura a
`scale_bits=12` (il codice garantisce `|Ps|<4096`, dentro il range int16);
gli altri buffer di lavoro int32.

**1. Inferenza per flusso** (kernel gia' deployato, sezione 6, invariato da
questo lavoro):

| componente | formula | valore (d=13) |
|---|---|---|
| spline, per edge numerico | 9 MAC (1 proiezione + 2+1+1 combinazione Hermite + 4 pesatura) | 9 |
| spline, 10 edge numerici | 9 × 10 | 90 MAC |
| edge categorici (2) | 0 MAC (lookup + shift) | 0 |
| aggregazione finale (guadagno × contributo, per edge) | 1 × 12 | 12 MAC |
| **totale per flusso** | | **102 MAC, 0 lookup LUT** |

Nessuna sigmoide in inferenza: la decisione e' `sign(z)`.

**2. Stima dei guadagni** (`fit_gains_int`, per iterazione: n=128, d=12):

    MAC/iter  = 2·n·d + n     (z: n·d, err: n, grad: n·d)
    LUT/iter  = n              (isigmoid, una per riga)

| iters | MAC totali | lookup LUT totali |
|---|---|---|
| 2000 (originale, non convergente) | 6 400 000 | 256 000 |
| 6000 (scelto, sezione 16.1) | 19 200 000 | 768 000 |
| adattivo (17a, media 5944, range 4130-7933) | 19 020 800 (media) | 760 832 (media) |

Il passo adattivo non riduce il costo medio (resta vicino a 6000) ma lo
ridistribuisce: la direzione piu' economica (`bot->unsw`, 4130 iter) costa
13 216 000 MAC, la piu' cara (`ton->bot`, 7933 iter) 25 385 600 — un
fattore 1,9 fra la piu' facile e la piu' difficile, invece di pagare il
costo peggiore su tutte.

RAM di picco: `Ps` (128×12, int16) 3072 B + pesi/molteplicita' (128×4 B)
512 B + etichette (128 B, int8) + guadagni correnti (12×4 B) 48 B + LUT
sigmoide (gia' documentata, 512 B) = 4272 B ≈ **4,17 KB**.

**3. RLS 13×13** (`StatSufficienti.aggiorna` — **proiettato, non misurato**:
l'implementazione resta in virgola mobile, mai portata a interi in questo
lavoro; la formula conta le stesse operazioni che un porting diretto
farebbe, sostituendo `exp()` con una LUT come per la sigmoide):

    MAC/aggiornamento = 5 × (n_batch·d + n_batch·d² + n_batch·d + d³/3)
                       = 5 × (32·13 + 32·13² + 32·13 + ⌊13³/3⌋)
                       = 5 × (416 + 5408 + 416 + 732)
                       = 5 × 6972 = **34 860 MAC** (5 iterazioni IRLS, n_batch=32, d=13)

(**Correzione**: una stesura precedente riportava 174 300 — il fattore 5
delle iterazioni IRLS era stato applicato due volte, una dentro la somma e
una fuori. Il valore corretto e' quello sopra, verificato con un ricalcolo
diretto invece che a mente.)

Dominato dal termine `n_batch·d²` (l'accumulo di `X'WX`), non dalla
soluzione del sistema lineare. RAM: 728 B per `(A, c)` (gia' documentato,
sezione 9/16.2), invariata dal ridge adattivo (stesso stato, diversa sola
scelta scalare del ridge).

**Questo cambia l'argomento, non solo il numero.** Con la cifra corretta,
la RLS costa **34 860 MAC per aggiornamento contro i 19 200 000 della stima
a discesa del gradiente a 6000 iterazioni — 551 volte meno calcolo** — oltre
a essere **5,9 volte piu' piccola in RAM** (728 B contro 4272 B della
stima). **La RLS non e' il ripiego per stare negli 8 KB di SRAM: e' di gran
lunga la piu' economica su entrambe le voci**, calcolo e memoria insieme.
L'unica ragione per cui non e' gia' la scelta di default e' l'accuratezza:
perde contro lo statico esattamente quando BoT-IoT e' la sorgente (sezione
16.2/17b), non per un limite di costo. Risolvere quel problema di
accuratezza vale quindi molto di piu' di quanto sezione 16.2 lasciasse
intendere — non sblocca solo una direzione in piu', sblocca l'opzione piu'
economica dell'intero lavoro.

**4. Martingala conformal intera** (`martingale_batch_int`, per batch di
20 000 flussi):

    MAC/batch  = 20 000   (1 moltiplicazione per indice di bin, per flusso)
    LUT/batch  = 20 000   (1 lookup per incremento, per flusso)

RAM: LUT a 256 voci, 1024 B (int32) — piccola, gia' nello spirito della LUT
sigmoide. **Ma la calibrazione ordinata (`s_cal`, dimensione `n_cal`) resta
residente per confrontare ogni flusso via ricerca binaria**, e `n_cal` non
e' piccolo: con `n_cal=200` (il minimo imposto dal codice) sono 800 B, con
le migliaia di righe che un batch di calibrazione realistico userebbe sono
diversi KB — **lo stesso problema di RAM che i minimi quadrati ricorsivi
erano nati per risolvere, qui non affrontato**. Va dichiarato come limite
aperto, non nascosto: la martingala intera costa poco in MAC ma il suo
stato di calibrazione puo' costare piu' della LUT sigmoide e della RLS
messe insieme.

**Tabella istanziata, n=128 (n_batch=32 per la RLS) — il confronto che
conta:**

| | discesa (2000 iter) | discesa (6000 iter) | discesa (adattivo) | RLS 13×13 (proiettata) | Rifit completo |
|---|---|---|---|---|---|
| Byte riscritti (tabella MULT) | 24 (12×int16) | 24 | 24 | 24 | 250 |
| Parametri | 12 (+1 bias a parte) | 12 | 12 | 12 (+1 bias, nello stato) | 101 |
| MAC (un aggiornamento) | 6 400 000 | 19 200 000 | 19 020 800 (media; 13,2M-25,4M a seconda della direzione) | **34 860** | non modellato — vedi sotto |
| RAM di picco | 4272 B (4,17 KB) | 4272 B | 4272 B | **728 B** | non modellato — vedi sotto |
| MAC inferenza/flusso | 102 | 102 | 102 | 102 | 102 (stesso kernel, cambiano solo i coefficienti) |

(I 24 byte sono la tabella `MULT`, dodici moltiplicatori Q15 — verificato
bit-esatto in C, sezione 6/16.1: `byte riscritti per l'adattamento: 24
(12 int16)`. Il termine noto `b` e' un tredicesimo numero, gestito a parte,
non incluso nella tabella riscritta: e' cosi' che il documento cita "24
byte" in ogni altra sezione, e questa tabella doveva coincidere, non
contraddire.)

**La RLS non e' solo un'alternativa che sta nella RAM: e' l'opzione piu'
economica su entrambe le voci, di un ordine di grandezza o piu'.** 551 volte
meno MAC della discesa a 6000 iterazioni, 5,9 volte meno RAM. Il motivo per
cui non e' il default non e' il costo — e' che perde contro lo statico
esattamente nelle due direzioni dove BoT-IoT e' la sorgente (16.2/17b), un
problema di accuratezza non ancora risolto, non di risorse.

**Sul rifit completo il modello resta qualitativo, dichiarato come tale.**
Il rifit non e' mai stato portato nella catena intera bit-esatta (la
sezione 6 lo esclude esplicitamente dal deployment: "nessun riaddestramento
sul dispositivo"), quindi non c'e' un `fit_*_int` da cui contare MAC e LUT
come per gli altri tre blocchi. Un limite inferiore ragionevole: rifit
significa riapprendere i coefficienti spline (non solo un guadagno per
edge fisso), quindi ogni iterazione deve ricalcolare la base di
interpolazione — la stessa spline a 9 MAC per edge dell'inferenza, non un
singolo prodotto — sui 101 parametri invece di 13. Un limite inferiore
grossolano e' quindi (101/13) × (9/1) ≈ **70 volte** il MAC per iterazione
della stima a 13 coefficienti, escludendo la propagazione del gradiente
attraverso la base stessa (che il gain-fit non deve fare, perche' non
tocca i coefficienti della spline). I 250 byte di parametri sono l'unica
cifra misurata con certezza per il rifit; il resto e' la ragione per cui il
codice del progetto lo esclude dal dispositivo, non un numero da citare
come misura.

**Cosa porta l'argomento, quindi.** Non l'accuratezza (sezione 11: pareggio
in valore atteso). E' che l'aggiornamento a 13 coefficienti e' l'unico dei
due che **esiste** come catena integer-only compatta: 24 byte da riscrivere
(la tabella `MULT`, verificata bit-esatta), un costo in MAC che va da 6,4M
(2000 iter fisse) a 13,2M-25,4M (passo adattivo, a seconda della
direzione) con la discesa del gradiente, o **34 860** con la RLS —
quest'ultima anche la piu' piccola in RAM (728 B, un sesto dei 4,17 KB
della discesa, ben dentro gli 8 KB di SRAM di un ATmega2560) — tutto
quantificabile e limitato. Il rifit completo richiede invece una catena di
riaddestramento che il progetto non ha mai costruito per il dispositivo, e
il motivo per cui non l'ha costruita e' proprio l'ordine di grandezza
stimato sopra (~70×, sui soli MAC di stima, escludendo la propagazione
attraverso la base spline che il gain-fit evita del tutto).

---

## 18. Sensibilita' al rapporto di undersampling (1:50 contro 1:1, 1:3, 1:20 e 1:100)

Ultima voce aperta della lista "Cosa resta aperto" della fase 2 che potesse
ancora incrinare i risultati nuovi: il rapporto era fissato a 1:50 da prima
di questo lavoro, mai testato, ed ereditato da ogni script di
`adattamento-drift/` tramite `--ratio` con default 50.0. Quattordici
sezioni di questo documento poggiano su quell'assunzione. Le sottosezioni
18.1-18.2 misurano prima a 20 e 100; la 18.3 mostra perche' quella griglia
non basta e la 18.4 la estende a 1 e 3, dove due delle cinque affermazioni
verificate smettono di reggere.

### La semantica di `undersample()`, prima dei numeri

`scripts/cross_domain.py::undersample(y, idx, ratio, seed)`:

    counts = bincount(y[idx])
    minority = argmin(counts); majority = 1 - minority
    n_keep = min(counts[majority], ratio * counts[minority])
    tiene tutta la classe minoritaria + n_keep campioni della maggioritaria

Tre cose non ovvie, verificate leggendo il codice invece di assumendole:

1. **Si applica SOLO al training split del dominio SORGENTE.** Mai al
   target, mai alla valutazione, mai alla selezione delle etichette in
   fase di adattamento (`adaptive_pick`, `select_int`): quei passaggi non
   ricevono `ratio`.
2. **Riduce sempre e solo la classe maggioritaria** (qui, in tutti e tre i
   domini, sempre gli attacchi: "normale" e' la classe minoritaria in
   TON_IoT, BoT-IoT e UNSW-NB15), tenendo **tutta** la minoritaria.
3. **E' un NO-OP se il rapporto naturale e' gia' piu' equilibrato di
   `ratio`.** `n_keep = min(...)`: se `ratio * minoritaria >= maggioritaria`,
   `n_keep` e' semplicemente tutta la maggioritaria, `rng.choice` non viene
   nemmeno chiamato, e l'insieme di training e' identico bit per bit a
   qualunque altro `ratio` che soddisfi la stessa disuguaglianza —
   indipendente dal seed, perche' non c'e' estrazione casuale da fare.

**Il rapporto naturale maggioritaria:minoritaria dei tre domini** (calcolato
sui parquet armonizzati, l'intero dataset — lo split stratificato all'80%
usato per il training preserva la proporzione):

| dominio | normali | % normali | maggioritaria:minoritaria |
|---|---|---|---|
| TON_IoT | 50 000 / 211 043 | 23,69% | **3,22 : 1** |
| BoT-IoT | 477 / 3 668 522 | 0,013% | **7 689,8 : 1** |
| UNSW-NB15 | 93 000 / 257 673 | 36,09% | **1,77 : 1** |

**Questa e' la frase che andava scritta prima di qualunque tabella:** lo
stesso numero (`ratio`) significa cose opposte nelle due direzioni, perche'
il rapporto naturale di TON_IoT (3,22) e di UNSW-NB15 (1,77) sta **sotto**
ogni valore di `ratio` testato in questo lavoro (20, 50, 100), mentre quello
di BoT-IoT (7 690) sta **tre ordini di grandezza sopra** anche il piu'
permissivo. Per costruzione, quindi:

- **Quando TON_IoT o UNSW-NB15 sono la sorgente, cambiare `--ratio` fra 20,
  50 e 100 non cambia nulla**: il modello addestrato e' identico bit per
  bit, non solo simile. Non e' un'ipotesi, e' una conseguenza diretta del
  `min()` nel codice, verificata empiricamente su quattro combinazioni
  indipendenti (script/direzione/seed) prima di lanciare qualunque run
  lungo — vedi sotto.
- **Solo quando BoT-IoT e' la sorgente il rapporto vincola davvero**, e lo
  fa sempre, a qualunque `ratio` in questo intervallo: cambia quanti
  attacchi entrano nel training rispetto ai 477 normali disponibili (a
  ratio 20: ~381×20≈7 620 attacchi tenuti; a 50: ~19 050; a 100: ~38 100 —
  contro 3,67 M naturalmente disponibili).

**Conseguenza per lo scopo di questa sezione**: delle sei direzioni cross,
**solo `bot->ton` e `bot->unsw`** (piu' il riferimento in-domain `bot->bot`)
possono differire fra i tre rapporti. `ton->bot`, `ton->unsw`, `unsw->bot`,
`unsw->ton` (e i riferimenti in-domain `ton->ton`, `unsw->unsw`) sono
**garantiti identici per costruzione**, non solo "probabilmente simili": la
sensibilita' al rapporto di undersampling e' interamente una domanda su
come reagisce il training su BoT-IoT, non sulle altre due sorgenti.

**Verifica empirica dell'identita', prima di lanciare i run lunghi** (la
stessa cautela gia' imposta per il suffisso dei checkpoint, sotto): un
singolo seed per script, sorgente non-BoT, confrontato bit per bit contro
il checkpoint a ratio=50 gia' presente:

| script | direzione/sorgente | ratio testato | esito |
|---|---|---|---|
| `tre_domini.py` | src=ton, seed 42 | 20 | identico a 4 decimali su tutte le celle (`ton->bot`, `ton->ton`) |
| `tre_domini.py` | src=unsw, seed 42 | 100 | identico a 4 decimali su tutte le celle (`unsw->ton`, `unsw->bot`, `unsw->unsw`) |
| `drift_int_adapt.py` | `ton->bot`, seed 42 | 20 | identico a 4 decimali su tutte le 8 righe (float/intero × non adatt./guadagni float/interi × 3 budget) |
| `drift_graduale.py` | `ton->bot`, seed 42 | 20 | identico su tutti i 20 batch della politica `statico` (confrontati i primi 5) |
| `drift_graduale_int.py` | `ton->bot`, seed 42 | 20 | identico sull'ultimo batch della politica `statico` (0,471147114711471... a entrambi i rapporti) |

Tutte e cinque le verifiche confermano l'identita' predetta dal codice. Da
qui in avanti, per le direzioni con TON_IoT o UNSW-NB15 come sorgente, i
numeri citati a ratio 20 e 100 **sono** quelli gia' misurati a ratio 50
altrove nel documento — non ricalcolati, perche' ricalcolarli
produrrebbe per costruzione lo stesso numero.

### Il bug evitato: i checkpoint non includevano il rapporto nel nome

Prima di lanciare qualunque run, verificato che `tre_domini.py`,
`drift_int_adapt.py`, `drift_graduale.py` e `drift_graduale_int.py`
scrivessero il checkpoint in `artifacts/` e i CSV in `results/` **senza**
il rapporto nel nome — la stessa trappola gia' scattata due volte in questo
lavoro (run per terna sovrascritti in sezione 15, poi `--iters` con file
separati in sezione 16.1): un rilancio a `--ratio` diverso da 50 avrebbe
sovrascritto silenziosamente i risultati a 10 seed appena prodotti.

Corretto in tutti e quattro gli script con un suffisso `_ratio{valore}`,
**vuoto quando `ratio==50`** (cosi' i file esistenti restano dove sono e
non si ricalcola nulla): `tre_domini_{spazio}_{terna}{suffix}.jsonl`,
`drift_int_adapt{suffix_iters}{suffix_ratio}.jsonl`,
`drift_graduale{suffix}.jsonl`, `drift_graduale_int{suffix}.jsonl`, e i
rispettivi CSV in `results/`. In `drift_int_adapt.py` anche la condizione
che scrive l'header C (`mcu_pio/include/kan_int_adapt.h`) ora richiede
`ratio == 50.0`, non solo `iters == 6000`: un rilancio a rapporto diverso
non deve sovrascrivere l'header canonico del dispositivo.

**Verificato su un run breve prima di procedere**, per tutti e quattro gli
script (un seed, una direzione, `--ratio 20`): i file nuovi
(`tre_domini_ricco_tonbotunsw_ratio20.jsonl`,
`drift_int_adapt_ratio20.jsonl`, `drift_graduale_ratio20.jsonl`,
`drift_graduale_int_ratio20.jsonl`, e i CSV corrispondenti) sono comparsi
accanto ai vecchi con lo stesso contenuto atteso, e i checksum MD5 dei file
esistenti a ratio=50 sono rimasti invariati prima e dopo. Solo a questo
punto sono partiti i run completi.

### 18.1 — `tre_domini.py`, spazio ricco, ton/bot/unsw: la sezione 11

Dato che solo le direzioni con BoT-IoT come sorgente possono cambiare,
rilanciato **solo `--src bot`** (non le altre due, per l'identita' sopra) a
`--ratio 20` e `--ratio 100`, sugli stessi 10 seed (42-51) gia' usati per
ratio=50. ~25 s per seed: il training su BoT-IoT sotto-campionato a questi
rapporti resta piccolo (7 620-38 100 righe contro 3,67 M naturali).

Balanced accuracy, media sui 10 seed (n riusciti/10 dove rilevante):

| | non adattato | 32 etichette | 128 etichette |
|---|---|---|---|
| `bot->ton`, ratio 20 | 0,5900 | 0,8449 (10/10) | 0,8273 (10/10) |
| `bot->ton`, ratio 50 | 0,6340 | 0,8003 (9/10) | 0,8623 (9/10) |
| `bot->ton`, ratio 100 | 0,6496 | 0,8681 (9/10) | 0,8538 (9/10) |
| `bot->unsw`, ratio 20 | 0,4671 | 0,7155 (10/10) | 0,7562 (10/10) |
| `bot->unsw`, ratio 50 | 0,4551 | 0,7381 (9/10) | 0,7552 (10/10) |
| `bot->unsw`, ratio 100 | 0,4645 | 0,7075 (10/10) | 0,7582 (10/10) |
| `bot->bot` *(in-domain)*, ratio 20 | 0,9949 | — | — |
| `bot->bot` *(in-domain)*, ratio 50 | 0,9931±0,0009 | — | — |
| `bot->bot` *(in-domain)*, ratio 100 | 0,9945 | — | — |

**Un effetto reale ma piccolo sul modello non adattato, che l'adattamento
assorbe.** Test t appaiati per seed (SEM, non dev.std) fra ciascun rapporto
e il basale a 50:

| direzione | metodo | delta (20−50) | t | p | delta (100−50) | t | p |
|---|---|---|---|---|---|---|---|
| `bot->ton` | non adattato | −0,0440 | −2,76 | **0,022** | +0,0156 | 0,51 | 0,62 |
| `bot->unsw` | non adattato | +0,0120 | 4,17 | **0,0024** | +0,0094 | 2,83 | **0,020** |
| `bot->ton` | 128 etichette | −0,0171 | −0,57 | 0,59 | −0,0085 | −1,40 | 0,20 |
| `bot->unsw` | 128 etichette | +0,0010 | 0,23 | 0,82 | +0,0030 | 0,36 | 0,72 |

Il rapporto di undersampling **sposta davvero, in modo statisticamente
distinguibile, il modello non adattato** addestrato su BoT-IoT (3 celle su
4 significative a p<0,05) — atteso, perche' cambia quanti attacchi vede il
training. Ma **una volta adattato a 128 etichette la differenza sparisce**
(tutte e 4 le celle p>0,19): l'adattamento a 13 coefficienti ririscrive
abbastanza il modello da assorbire la differenza fra i tre training set.
E' lo stesso fenomeno gia' misurato in sezione 15 per il prior della
sorgente ("con poche etichette conta ancora il modello di partenza, con
molte l'adattamento lo sovrascrive"), qui sul rapporto invece che sul
dominio sorgente.

**Le cinque affermazioni, verificate:**

**1. "L'adattamento a 13 coefficienti recupera in 5 direzioni su 6" —
regge a tutti e tre i rapporti.** `unsw->bot` fallisce sempre (0/10 a ogni
ratio, per costruzione — vedi sotto). Le altre cinque, incluse le due
sensibili al rapporto, recuperano con guadagni dello stesso ordine di
grandezza gia' riportato (+0,2/+0,4): `bot->ton` +0,237/+0,228/+0,204 (20/
50/100), `bot->unsw` +0,289/+0,300/+0,294. Il rapporto sposta l'entita' del
recupero di qualche punto, non il fatto che ci sia.

**4. "`unsw->bot` fallisce, zero normali in tutti i seed" — regge per
costruzione, non solo per misura.** La sorgente e' UNSW-NB15: per
l'identita' sopra, il training e' bit-identico a ogni rapporto, e la
raccolta di normali sul target avviene con `adaptive_pick`/`subsample_target`,
che non ricevono `ratio`. Il fallimento a 0/10 seed non e' una scoperta di
questa sezione, e' una conseguenza necessaria di quella gia' misurata in
sezione 11 a ratio=50: non puo' cambiare a nessun rapporto, e non e' stato
ricalcolato per lo stesso motivo per cui non lo sono state le altre tre
direzioni non-BoT-sorgente.

**5. "13 coefficienti contro rifit completo, pareggio, delta medio −0,002"
— regge a questi tre rapporti.** Qui sotto la prima stesura di questo
blocco, che aggregava un t-test sulla media delle 5 direzioni per seed
senza dichiarare quanti seed avessero tutte e cinque le direzioni
disponibili — lo stesso errore, trovato e corretto dopo, che la sezione
18.4 documenta e risolve con test per-direzione. Il numero non cambia
sostanzialmente (i tre rapporti qui hanno completezza alta, 9-10/10 seed
per direzione, quindi l'aggregato non era gravemente distorto come lo era
poi a ratio=1), ma la tabella corretta — con n dichiarato, per direzione —
e' quella di sezione 18.4, non questa:

| ratio | delta medio (5 direzioni, seed disponibili) | t appaiato per seed (SEM) | p | n metodo B (tutte e 5 disponibili) |
|---|---|---|---|---|
| 20 | −0,0055 | −0,43 | 0,68 | 9/10 |
| 50 (originale) | −0,0050 | −0,58 | 0,58 | 8/10 |
| 100 | −0,0066 | −0,70 | 0,50 | 8/10 |

Pareggio non distinguibile da zero a tutti e tre (|t|<1, p>0,5) sia
nell'aggregato sia direzione per direzione (sezione 18.4): il numero
originale (−0,002) non era un artefatto del rapporto 1:50. **Vedi sezione
18.4 per la tabella per-direzione completa e per cosa succede a ratio=1**,
dove la stessa affermazione cade.

**Verdetto del blocco (a): le cinque — anzi qui tre delle cinque, le altre
due non toccate da questo blocco — affermazioni testabili con
`tre_domini.py` (1, 4, 5) sopravvivono al cambio di rapporto fra 1:20 e
1:100.** L'unico effetto reale trovato e' laterale rispetto alle
affermazioni: il rapporto sposta significativamente il punto di partenza
(modello non adattato) quando BoT-IoT e' sorgente, ma l'adattamento a 13
coefficienti — l'oggetto delle cinque affermazioni — lo rende
indistinguibile.

### Un bug trovato analizzando il blocco (b): il checkpoint di `drift_graduale.py` troncava 6 politiche su 7

Prima di riportare le affermazioni 2 e 3, un problema di integrita' dei
dati emerso mentre si analizzavano i risultati — la stessa categoria di
trappola gia' incontrata due volte in questo lavoro, qui una terza volta.

`scripts/drift_graduale.py` scrive il checkpoint con:

    with ckpt.open("a") as fh:
        for r in rows[-N_BATCH * 6:]:
            fh.write(...)

La costante `6` e' il numero di politiche **prima** che sezione 17b
(COMPITO 3b) aggiungesse `stat_13x13_adaptive` come settima. Da quel
momento ogni chiamata a `run_unit` scrive nel `rows` in memoria
`N_BATCH * 7 = 140` righe (20 batch × 7 politiche), ma il codice ne
persiste su disco solo le ultime `120`: **i primi 2-3 batch di 6 politiche
su 7 (tutte tranne l'ultima aggiunta) vengono persi dal file `.jsonl` a
ogni chiamata**, per ogni seed.

**Non intacca i CSV gia' pubblicati**: `finalize()` lavora sull'oggetto
`rows` in memoria, completo per costruzione entro una singola invocazione
continua dello script — motivo per cui `results/drift_graduale.csv` e le
tabelle delle sezioni 9/13/16.2/17b, generate da run continui su tutti i
seed in un colpo solo, sono corrette (verificato: i valori li' — es.
`bot->ton` statico 0,8179, RLS 0,7434 — coincidono esattamente con quelli
ricalcolati dai file `results/drift_graduale_runs.csv`, che vengono dallo
stesso `rows` completo). **Intacca invece qualunque ricarica del
checkpoint**: un run interrotto e ripreso, o — come successo qui — un run
di verifica breve su un seed seguito da un run "completo" sugli stessi
seed, che ricarica quel seed dal `.jsonl` gia' troncato invece di
ricalcolarlo. E' esattamente cosi' che si è' corrotto il primo tentativo
di questa sezione: un run di prova a un seed su `bot->ton,bot->unsw` per
cronometrare la durata, seguito dal run "completo" sugli stessi 10 seed,
che ha ricaricato il seed 42 gia' troncato (17-18 batch su 20) invece di
ricalcolarlo da zero — scoperto confrontando `results/drift_graduale.csv`
(corretto, generato da un run continuo passato) con una prima
ricostruzione manuale dal `.jsonl` grezzo, che dava numeri diversi e
incompatibili con quelli gia' pubblicati.

**Corretto** sostituendo la costante con `N_BATCH * len(politiche)`
(`scripts/drift_graduale.py`), cosi' il numero si aggiorna da solo se in
futuro si aggiunge un'altra politica. Il run di seed 42 corrotto e' stato
cancellato e ricalcolato da zero; da questo punto in avanti tutte le
tabelle di questa sezione vengono da `results/drift_graduale_runs*.csv` e
`results/drift_graduale_int_runs*.csv` (il `rows` completo scritto da
`finalize()`), mai dal `.jsonl` grezzo ricaricato — la stessa cautela gia'
imposta per il denominatore (sezioni 15/16.3) e per i nomi dei checkpoint
(sopra), qui sulla completezza dei dati invece che sulla loro
sovrascrittura o sul loro rumore.

### 18.2 — `drift_graduale.py` e `drift_graduale_int.py`: le sezioni 16.2 e 16.3

Anche qui, solo `bot->ton` e `bot->unsw` possono cambiare (identita' sopra
per le altre quattro direzioni cross). Rilanciati a ratio 20 e 100, 10
seed (42-51), dati verificati completi (20/20 batch per ogni cella) prima
di calcolare qualunque statistica.

**Affermazione 3 (sezione 16.2): "la RLS perde contro lo statico esattamente
e solo quando BoT-IoT e' la sorgente" — regge, e con un dettaglio nuovo a
ratio 100.** Delta (RLS − statico), media sui 20 batch per seed, t
appaiato (SEM):

| direzione | ratio 20 | ratio 50 (originale) | ratio 100 |
|---|---|---|---|
| `bot->ton` | −0,0979±0,0261, t=−3,75, p=0,0046 | −0,0745±0,0165, t=−4,53, p=0,0014 | −0,0652±0,0231, t=−2,83, p=0,020 |
| `bot->unsw` | −0,0053±0,0150, t=−0,35, p=0,73 | −0,0079±0,0157, t=−0,50, p=0,63 | **−0,0413±0,0167, t=−2,47, p=0,036** |

`bot->ton` perde in modo significativo a tutti e tre i rapporti (mai un
pareggio). `bot->unsw` perde sempre di segno ma non in modo distinguibile
dal rumore a ratio 20 e 50 — **a ratio 100 la perdita diventa
statisticamente significativa** (p=0,036, prima >0,6): il rapporto non
cambia *quali* direzioni perdono (restano esattamente le due con BoT-IoT
sorgente, mai le altre quattro, per costruzione), ma a ratio piu' alto la
perdita di `bot->unsw` smette di essere trascurabile. Coerente con la
diagnosi della sezione 16.2 (quasi-separazione al primo batch quando i
normali di BoT-IoT sono pochi): un ratio piu' alto lascia in training
ancora meno segnale relativo sulla classe minoritaria, e il primo
aggiornamento IRLS ne risente un po' di piu'.

**Affermazione 2 (sezione 16.3): "il riadattamento continuo in interi batte
lo statico in 6 direzioni su 6, con \|t\| appaiati da 3,3 a 91,5" — regge
in tutte le sei direzioni a tutti i rapporti, ma il punto piu' debole
(`bot->ton`, gia' il piu' piccolo a ratio 50) si indebolisce ulteriormente
a ratio 100.** Le altre quattro direzioni sono identiche per costruzione
alla misura originale (t da 12,6 a 91,5, invariate). Le due sensibili:

| direzione | ratio 20 | ratio 50 (originale, il minimo della sez. 16.3) | ratio 100 |
|---|---|---|---|
| `bot->ton` | +0,0557±0,0067, t=8,32, p<0,0001 | +0,0402±0,0122, **t=3,29**, p=0,009 | +0,0390±0,0154, **t=2,53**, p=0,032 |
| `bot->unsw` | +0,1190±0,0061, t=19,43, p<0,0001 | +0,1319±0,0052, t=25,33, p<0,0001 | +0,1264±0,0045, t=27,79, p<0,0001 |

`bot->unsw` resta fortissimo ovunque. `bot->ton` — che era gia' il legame
piu' debole della catena "3,3-91,5" a ratio 50 — **resta significativo a
tutti e tre i rapporti (p sempre <0,05) ma il suo `t` scende ulteriormente
a ratio 100 (2,53, contro 3,29 originale)**: l'intervallo da citare non e'
piu' "3,3-91,5" in modo assoluto, e' "il legame piu' debole (`bot->ton`)
resta sopra la soglia di significativita' ma non ha molto margine, e si
restringe ancora se si campiona a ratio piu' alto". Il verdetto qualitativo
— 6 direzioni su 6, sempre — non cambia; il margine del punto piu' debole
si', e va detto con lo stesso numero.

**Verdetto del blocco (b): entrambe le affermazioni testabili qui (2, 3)
sopravvivono al cambio di rapporto fra 1:20 e 1:100, qualitativamente
identiche.** Due dettagli quantitativi da portare avanti: a ratio 100 la
perdita della RLS in `bot->unsw` diventa significativa (prima non lo era),
e il `t` piu' debole della sezione 16.3 (`bot->ton`) scende da 3,29 a 2,53
pur restando sopra soglia. Nessuna delle due affermazioni si capovolge, ma
`bot->ton` in sezione 16.3 e' il punto della intera sezione 18 piu' vicino
a cedere: a un rapporto ancora piu' alto di 100 non e' garantito che regga.

> **Nota in avanti**: questo verdetto e' corretto per la griglia su cui e'
> stato misurato (20-100), ma quella griglia copriva solo due direzioni su
> sei — le altre quattro erano identiche per costruzione a ratio=50 non
> perche' robuste, ma perche' nessuno dei tre rapporti vincolava le
> sorgenti TON_IoT o UNSW-NB15. La sottosezione 18.3 quantifica il buco di
> copertura; la 18.4 lo colma con `--ratio 1` e `--ratio 3`. Il verdetto
> qui sopra **non regge** su quella griglia estesa — vedi il verdetto
> finale in fondo alla sezione.

### 18.3 — Il problema di copertura: quattro direzioni su sei non erano mai vincolate

La scoperta della sezione 18 iniziale — `undersample()` e' un no-op se il
rapporto naturale e' gia' piu' equilibrato di `ratio` — e' corretta, ma ha
una conseguenza che i blocchi (a) e (b) non affrontavano: con la griglia
20/50/100, **nessuno dei tre valori vincola TON_IoT (naturale 3,22) o
UNSW-NB15 (naturale 1,77)**. Le quattro direzioni con quelle sorgenti
(`ton->bot`, `ton->unsw`, `unsw->ton`, `unsw->bot`) erano quindi
matematicamente garantite identiche a ratio=50 a **ogni** valore provato,
non "risultate stabili" — la manopola semplicemente non le muoveva. Dire
"le conclusioni sono stabili fra 1:20 e 1:100" su quella base e' vero e
vuoto insieme per due terzi delle direzioni.

Per vincolare le altre due sorgenti serve `ratio` **sotto** il loro
rapporto naturale:

| sorgente | rapporto naturale | vincolato da |
|---|---|---|
| UNSW-NB15 | 1,77 | solo `ratio=1` |
| TON_IoT | 3,22 | `ratio=1` o `ratio=3` |
| BoT-IoT | 7 689,8 | qualunque `ratio` in questo lavoro (1, 3, 20, 50, 100) |

Rilanciato quindi a **`--ratio 1`** (vincola tutte e tre le sorgenti, un
solo run copre le sei direzioni cross) e **`--ratio 3`** (vincola TON_IoT
e BoT-IoT, non UNSW-NB15 — punto intermedio), sui soliti 10 seed (42-51),
su `tre_domini.py` (spazio ricco, tutte e tre le sorgenti), `drift_graduale.py`
e `drift_graduale_int.py` (tutte e sei le direzioni a ratio=1; a ratio=3
solo le quattro con TON_IoT o BoT-IoT come sorgente, verificata
l'invarianza di `unsw->ton`/`unsw->bot` con lo stesso spot-check a un
seed gia' usato nei blocchi (a)/(b) — `tre_domini.py --src unsw --ratio 3
--seeds 42` riproduce esattamente i valori di ratio=50), e
`drift_int_adapt.py` (`bot->ton`/`bot->unsw` a ratio 20/100 come richiesto
dal blocco (c) rimasto indietro, piu' tutte e sei le direzioni a ratio 1 e
le quattro rilevanti a ratio 3).

A `ratio=1` il training su BoT-IoT diventa minuscolo: 477 normali × 0,8 ≈
381 nel training split, e `ratio=1` tiene tanti attacchi quanti normali,
quindi **~762 righe** invece delle 3,67 M naturali — un training set circa
4 800 volte piu' piccolo (a `ratio=3`, ~1 524 righe: ancora minuscolo, gli
attacchi tenuti triplicano ma restano lontanissimi dal naturale). Per
TON_IoT e UNSW-NB15 la riduzione e' molto meno drastica ma non nulla:
**TON_IoT** a `ratio=1` tiene 40 000 attacchi invece di ~128 800 naturali
(**−69%**, training totale 80 000 righe contro ~168 800), a `ratio=3` la
riduzione e' piccola (120 000 attacchi, **−7%**, training 160 000 righe).
**UNSW-NB15** a `ratio=1` tiene 74 400 attacchi invece di ~131 700
naturali (**−44%**, training 148 800 righe), mentre a `ratio=3` non si
muove affatto (0%, il rapporto naturale 1,77 e' gia' sotto 3).

Durante l'analisi di questo blocco e' emerso anche un bug di logging
(non di calcolo): per direzioni lente come `unsw->bot`, l'output su file
di `drift_graduale.py` puo' bufferizzare tutte le righe di progresso per
seme e mostrare solo la tabella finale — un run che si ferma per
`--max-seconds` puo' quindi sembrare, guardando il solo file di log,
identico a uno completato. Verificato ogni volta contando i seed
effettivamente presenti nel CSV (`groupby(['exp','seed']).seed.nunique()`)
invece di fidarsi del testo del log; un caso (`drift_graduale.py --ratio
1`, `unsw->bot` fermo a 3/10 seed) e' stato scoperto cosi' e completato
con una ripresa.

### 18.4 — I risultati a ratio 1 e 3: dove le cinque affermazioni reggono e dove no

**Il meccanismo comune ai cedimenti trovati qui**, prima dei numeri: a
`ratio=1` la sorgente e' ribilanciata **esattamente 1:1**, non solo "meno
sbilanciata". Il punteggio del modello sorgente sul target (`z0`) cambia
di conseguenza, e la selezione delle etichette sul target
(`adaptive_pick`) — che sceglie i punti piu' vicini al confine di
decisione *di quel punteggio* — trova sistematicamente **meno spesso
entrambe le classi** nel budget disponibile. Non e' un problema nuovo: e'
lo stesso meccanismo di "impossibile e' spesso raro, non assoluto" delle
sezioni 4/11/15, qui mostrato sensibile al rapporto oltre che al seed.

**Affermazione 1 (sez. 11) — regge in media, ma l'affidabilita' cala a
ratio=1.** Successi su 10 seed a 128 etichette, per direzione, sull'intera
griglia:

| direzione | ratio 1 | ratio 3 | ratio 20 | ratio 50 | ratio 100 |
|---|---|---|---|---|---|
| `unsw->bot` | **1/10** | 0/10 | 0/10 | 0/10 | 0/10 |
| `bot->ton` | **8/10** | 10/10 | 10/10 | 9/10 | 9/10 |
| `ton->bot` | **7/10** | 10/10 | 9/10 | 9/10 | 9/10 |
| `bot->unsw` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| `unsw->ton` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| `ton->unsw` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

Dove l'adattamento riesce, il recupero resta forte anche a ratio=1
(`bot->ton` 0,88 contro 0,58 non adattato, `ton->bot` 0,76 contro 0,48,
`bot->unsw` 0,77 contro 0,40, `unsw->ton` 0,82 contro 0,31, `ton->unsw`
0,74 contro 0,22): l'**entita'** del recupero non e' il problema.
**L'affidabilita' si', e solo a ratio=1**: due direzioni che a ogni altro
rapporto riuscivano 9-10 volte su 10 scendono a 7-8/10. Da ratio=3 in su
(compreso il punto intermedio) la tabella e' piatta.

**Affermazione 4 (sez. 11) — cade a ratio=1, regge altrove.** "Zero
normali in tutti i seed" e' vero a ratio 3, 20, 50, 100 (0/10 ovunque, il
numero originale). **A ratio=1 un seed su dieci trova abbastanza normali
da produrre un numero** (bal_acc 0,76, contro 0,74 non adattato — un
recupero modesto anche quando riesce). Non e' piu' letteralmente vero che
"nessun seed" ci riesce: va corretto a "un seed su dieci, solo al rapporto
piu' estremo provato — 0/10 per ogni rapporto da 3 in su".

**Affermazione 5 (sez. 11) — regge fino a ratio=3, cade nettamente a
ratio=1.** Prima versione di questa sottosezione aggregava un t-test sulla
media delle 5 direzioni per seed: sbagliato senza dichiararlo, per lo
stesso motivo gia' documentato in sezione 15 (metodo A/B) — un seed con 2
direzioni disponibili su 5 pesa quanto uno con 5, e la mancanza di
direzioni non e' casuale, si concentra dove la selezione fallisce, cioe'
nei casi piu' difficili. Corretto: **il risultato primario e' il test
per-direzione**, che ha n vicino a 10 in ogni cella ed e' quindi robusto
da solo; l'aggregato, quando riportato, dichiara esplicitamente n e
metodo.

**Un secondo problema, trovato verificando il primo.** Riverificando
manualmente i file grezzi per capire la copertura, `ton->bot` a
ratio=3 e `unsw->ton`/`ton->bot` a ratio=20/100 mostravano in
`results/tre_domini_runs_ricco_tonbotunsw_ratio{3,20,100}.csv` un solo
seed per la sorgente non misurata a quel rapporto (`unsw` a ratio=3 e
100, `ton` a ratio=20) — un residuo dei singoli run a un seed usati in
sezione 18.3 per **verificare** l'identita' (`tre_domini.py --src unsw
--seeds 42 --ratio 3`, eccetera), scritti per costruzione nello stesso
checkpoint del rapporto corrispondente perche' checkpoint e spazio
coincidevano. Il codice di questa sezione non li usa mai (per quelle
sorgenti legge sempre da ratio=50, come da identita' verificata), quindi
non hanno alterato nessun numero qui sopra — ma chiunque leggesse quei
CSV direttamente, come e' stato fatto per trovare il problema
dell'aggregato, ci avrebbe visto esattamente il tipo di copertura parziale
non dichiarata che questa sottosezione doveva evitare. La stessa cosa era
successa in `drift_graduale_int_ratio20.jsonl` e
`drift_int_adapt_ratio20.jsonl` (`ton->bot` a un solo seed, residuo di un
test di cronometraggio). **Tutti e cinque i file ripuliti**: le righe
della sorgente/direzione non misurata rimosse dai checkpoint, i CSV
rigenerati da capo con `finalize()` sugli stessi dati (nessun ricalcolo,
solo riscrittura senza le righe orfane) — verificato che ogni file
`*_runs*.csv` di questa sezione ora contiene **solo** le direzioni
effettivamente misurate a quel rapporto, 10/10 seed ciascuna.

**Quanti seed sono disponibili per cella** (13 coefficienti E rifit
completo entrambi presenti a n=128 — questo conteggio e' il fenomeno che
sezione 18.4 misura, non un dettaglio contabile: cala dove la selezione
delle etichette fatica, cioe' a ratio=1 per le direzioni con BoT-IoT come
target o sorgente):

| direzione | ratio 1 | ratio 3 | ratio 20 | ratio 50 | ratio 100 |
|---|---|---|---|---|---|
| `bot→ton` | 8/10 | 10/10 | 10/10 | 9/10 | 9/10 |
| `ton→bot` | 7/10 | 10/10 | 9/10 | 9/10 | 9/10 |
| `bot→unsw` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| `unsw→ton` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| `ton→unsw` | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |

**Test t appaiato per seed, PER DIREZIONE, delta (13 coeff. − rifit) a
n=128 — risultato primario:**

| direzione | ratio 1 | ratio 3 | ratio 20 | ratio 50 (orig.) | ratio 100 |
|---|---|---|---|---|---|
| `bot→ton` | n=8, **−0,0546, t=−4,01, p=0,0051** | n=10, −0,0249, t=−1,20, p=0,26 | n=10, −0,0412, t=−1,22, p=0,25 | n=9, −0,0380, t=−2,93, p=0,019 | n=9, −0,0465, t=−2,89, p=0,020 |
| `ton→bot` | n=7, **−0,0053, t=−0,18, p=0,86** | n=10, **+0,1232, t=13,18, p<0,0001** | n=9, +0,1347, t=13,91, p<0,0001 | n=9, +0,1347, t=13,91, p<0,0001 | n=9, +0,1347, t=13,91, p<0,0001 |
| `bot→unsw` | n=10, **−0,0229, t=−3,48, p=0,0070** | n=10, −0,0033, t=−0,32, p=0,76 | n=10, −0,0110, t=−1,60, p=0,14 | n=10, −0,0165, t=−2,92, p=0,017 | n=10, −0,0162, t=−1,70, p=0,12 |
| `unsw→ton` | n=10, **−0,0585, t=−3,36, p=0,0083** | n=10, −0,0710, t=−2,45, p=0,037 | n=10, −0,0710, t=−2,45, p=0,037 | n=10, −0,0710, t=−2,45, p=0,037 | n=10, −0,0710, t=−2,45, p=0,037 |
| `ton→unsw` | n=10, **−0,0237, t=−4,29, p=0,0020** | n=10, −0,0230, t=−3,97, p=0,0032 | n=10, −0,0198, t=−3,20, p=0,011 | n=10, −0,0198, t=−3,20, p=0,011 | n=10, −0,0198, t=−3,20, p=0,011 |

(`unsw→ton` e `ton→unsw` a ratio 20/50/100, e `ton→bot`/`unsw→ton`/
`ton→unsw` a ratio=3, vengono dall'identita' di sezione 18.3 — stesso dato
di ratio=50, non ricalcolato, quindi stesso n.)

**Il risultato che conta e' il conteggio dei segni, non l'aggregato: a
ratio=1 tutte e cinque le direzioni sono negative, e quattro delle cinque
in modo significativo** (`ton→bot`, l'unica non significativa, e' anche
quella con meno seed disponibili, n=7). A ratio 3-100 il quadro e' misto:
`ton→bot` vince sempre in modo netto e significativo, le altre quattro
perdono, ma quasi mai in modo significativo con n~9-10 (l'eccezione e'
`bot→ton` a ratio 50/100, dove perde in modo significativo anche li'). Il
salto qualitativo a ratio=1 e' quindi duplice: **il segno di `ton→bot` si
inverte** (da vittoria netta a perdita non significativa, sui pochi seed
dove la selezione riesce) **e le altre quattro perdite, gia' presenti,
diventano quasi tutte significative**.

**Un aggregato, se serve un solo numero — dichiarato con n e metodo, come
in sezione 15.** Media dei delta di seed dove **tutte e cinque** le
direzioni hanno un valore (metodo B; la colonna "ratio 3/20/100" usa
l'identita' di sezione 18.3 per le direzioni non ricalcolate, quindi il
conteggio di seed completi qui non e' una misura pura per quelle celle):

| ratio | n seed completi (metodo B) | delta medio | t | p |
|---|---|---|---|---|
| **1** | **6/10** | **−0,0341** | **−5,04** | **0,0040** |
| 3 | 10/10 (4 direzioni misurate + 1 da identita') | +0,0002 | 0,02 | 0,98 |
| 20 | 9/10 (2 misurate + 3 da identita') | +0,0002 | 0,01 | 0,99 |
| 50 (originale) | 8/10 | −0,0001 | −0,01 | 0,99 |
| 100 | 8/10 | −0,0014 | −0,15 | 0,88 |

A ratio=1 anche l'aggregato via metodo B, su un n piu' piccolo ma
genuino (6 seed dove **tutte e cinque le direzioni erano effettivamente
misurate**, nessuna identita' coinvolta a questo rapporto), conferma il
segno e la significativita' del conteggio per direzione: −0,034, p=0,004.
**Questa e' la cifra da citare per l'affermazione 5**, non quella con
n=10 apparente della prima stesura, che mescolava seed con copertura
diversa senza dirlo.

**Questa affermazione va corretta in sezione 11**: "pareggio in valore
atteso" e' vero da ratio=3 in su (dove il segno resta misto, `ton→bot`
positivo contro quattro negative perlopiu' non significative) ma falso a
ratio=1, dove **cinque direzioni su cinque perdono, quattro in modo
significativo**, sul sottoinsieme di seed dove la selezione riesce.

**Affermazione 2 (sez. 16.3) — regge da ratio=3 in su, cade a ratio=1, e
il meccanismo del cedimento e' la selezione, non l'adattamento.** Delta
(riadattamento continuo intero − statico), media sui 20 batch, t appaiato
per seed:

| direzione | ratio 1 | ratio 3 | ratio 20 | ratio 50 (originale) | ratio 100 |
|---|---|---|---|---|---|
| `unsw->bot` | **0,0000, t=n/d** | +0,2607, t=91,6 | +0,2607, t=91,6 | +0,2607, t=91,6 | +0,2607, t=91,6 |
| `bot->ton` | +0,1284, t=7,3 | +0,0789, t=10,3 | +0,0557, t=8,3 | +0,0402, t=3,3 | +0,0390, t=2,5 |
| `ton->bot` | **+0,0211, t=1,49, p=0,17** | +0,1554, t=8,9 | +0,1393, t=12,6 | +0,1393, t=12,6 | +0,1393, t=12,6 |
| `bot->unsw` | +0,0941, t=11,1 | +0,1158, t=19,6 | +0,1190, t=19,4 | +0,1319, t=25,3 | +0,1264, t=27,8 |
| `unsw->ton` | +0,1449, t=18,1 | +0,0830, t=16,1 | +0,0830, t=16,1 | +0,0830, t=16,1 | +0,0830, t=16,1 |
| `ton->unsw` | +0,2015, t=11,2 | +0,1381, t=13,6 | +0,1411, t=17,1 | +0,1411, t=17,1 | +0,1411, t=17,1 |

Da ratio=3 a ratio=100 tutte e sei restano significative (t da 2,5 a 91,6,
mai sotto soglia). **A ratio=1 due direzioni su sei smettono di battere lo
statico**: `unsw->bot` **pareggia esattamente** — non un pareggio
approssimato, `bal_acc` identico a bit per bit fra le due politiche su
tutti e dieci i seed — perche' in nessuno dei 20 batch, in nessuno dei 10
seed, il buffer accumula mai entrambe le classi (il contatore
"adattamenti" sale comunque a 19, perche' incrementa prima del controllo
sulle classi: la politica *tenta* ma non aggiorna mai i guadagni). E
`ton->bot` diventa non significativo (t=1,49, p=0,17): in 8 dei 10 seed il
delta e' esattamente zero per lo stesso motivo, e solo in 2 (seed 50, 51)
l'aggiornamento riesce (+0,11, +0,10). **"6 direzioni su 6" va corretto in
sezione 16.3**: vero per ratio ≥ 3, falso a ratio=1, dove diventa 4/6 (le
due che cedono non peggiorano l'adattamento — restano pari allo statico,
non sotto — ma smettono di batterlo in modo distinguibile dal rumore).

**Affermazione 3 (sez. 16.2) — questa e' quella che cede peggio, e gia' a
ratio=3.** Delta (RLS − statico), t appaiato per seed:

| direzione | fonte | ratio 1 | ratio 3 | ratio 20 | ratio 50 (orig.) | ratio 100 |
|---|---|---|---|---|---|---|
| `unsw->bot` | UNSW | **−0,058, p=0,038 (PERDE)** | +0,044, p<0,0001 | +0,044, p<0,0001 | +0,044, p<0,0001 | +0,044, p<0,0001 |
| `bot->ton` | BoT | −0,052, p=0,0007 (perde) | −0,086, p=0,003 (perde) | −0,098, p=0,005 (perde) | −0,075, p=0,001 (perde) | −0,065, p=0,020 (perde) |
| `ton->bot` | TON | −0,002, p=0,93 (n.s.) | +0,112, p<0,0001 | +0,106, p<0,0001 | +0,106, p<0,0001 | +0,106, p<0,0001 |
| `bot->unsw` | BoT | **+0,059, p=0,0006 (VINCE)** | **+0,029, p=0,035 (vince)** | −0,005, p=0,73 (n.s.) | −0,008, p=0,63 (n.s.) | −0,041, p=0,036 (perde) |
| `unsw->ton` | UNSW | +0,098, p=0,0007 | +0,131, p<0,0001 | +0,131, p<0,0001 | +0,131, p<0,0001 | +0,131, p<0,0001 |
| `ton->unsw` | TON | +0,189, p<0,0001 | +0,109, p<0,0001 | +0,120, p<0,0001 | +0,120, p<0,0001 | +0,120, p<0,0001 |

L'affermazione era: perde **esattamente e solo** quando BoT-IoT e' la
sorgente. Tre rotture, non una:

1. **`bot->unsw` cambia segno.** A ratio 20-100 perde o pareggia (coerente
   con l'affermazione). **A ratio 3 e 1 vince in modo significativo** —
   una delle due direzioni che l'affermazione descriveva come perdenti
   smette di esserlo, e non di poco (+0,059 a ratio=1).
2. **`unsw->bot` perde a ratio=1** — una direzione con **UNSW-NB15**, non
   BoT-IoT, come sorgente. Questo da solo falsifica "esattamente e solo
   quando BoT-IoT e' la sorgente": a ratio=1 non e' piu' vero che la
   sorgente determina da sola chi perde.
3. **`ton->bot` smette di vincere in modo significativo a ratio=1**
   (p=0,93) — non perde, ma il "vince sempre tranne quando BoT-IoT e'
   sorgente" perde anche il suo contrappunto piu' netto.

**Questa affermazione va riscritta in sezione 16.2, non solo annotata**:
non regge come descritta a nessun rapporto sotto 20, e il pattern
"BoT-IoT sorgente ⇒ perde" **si rompe gia' a ratio=3**, il secondo punto
piu' vicino a quello originale (50), non solo al punto piu' estremo (1).
Il meccanismo diagnosticato in sezione 16.2 (quasi-separazione al primo
batch quando i normali nel training sono pochi) resta plausibile come
spiegazione parziale, ma non basta da solo: se fosse l'unico fattore
`bot->unsw` non dovrebbe cambiare segno mentre `bot->ton` no, dato che
entrambe condividono la stessa sorgente sotto-campionata.

### 18.5 — Blocco (c): `drift_int_adapt.py`, la catena intera end-to-end

Rimasto indietro rispetto ai blocchi (a) e (b); completato qui insieme
all'estensione della griglia. Non testa una delle cinque affermazioni
elencate — la sua misura centrale e' quella delle sezioni 6/13: **la stima
dei guadagni in interi non costa accuratezza rispetto alla stessa stima in
virgola mobile**. Rilanciato a ratio 20 e 100 su `bot->ton`/`bot->unsw`
(le uniche direzioni misurabili, per l'identita' della sezione 18) e a
ratio 1 e 3 su tutte e sei, per coerenza con i blocchi precedenti.

**La parita' intero-float regge su tutta la griglia.** Delta (guadagni
interi − guadagni float) a n=128, t appaiato per seed:

| direzione | ratio 1 | ratio 3 | ratio 20 | ratio 50 | ratio 100 |
|---|---|---|---|---|---|
| `bot->ton` | +0,0048, p=0,75 | −0,0004, p=0,91 | −0,0183, p=0,17 | −0,0028, p=0,85 | −0,0065, p=0,31 |
| `bot->unsw` | +0,0324, p=0,43 | −0,0099, p=0,64 | +0,0616, p=0,065 | +0,0045, p=0,83 | +0,0359, p=0,14 |

Nessuna cella e' significativa (p sempre >0,06, contro la soglia 0,05):
l'affermazione "l'aritmetica intera non costa" non solo regge, e' quella
di questo lavoro che si allontana di piu' dalla soglia di significativita'
su tutta la griglia — nessun segno di cedimento nemmeno a ratio=1.

**Un dettaglio interessante, in direzione opposta a tutto il resto di
questa sezione.** La selezione per `unsw->bot` in questo script non usa
`adaptive_pick` sui punteggi float ma la deduplicazione sui contributi
interi (sezione 6), che gia' a ratio=50 raccoglie normali dove la
selezione float fallisce sempre (5/10 seed qui, contro 0/10 in
`tre_domini.py`/`drift_graduale.py`). **A ratio=1 questa resa migliora
invece di peggiorare**: 6/10 seed, il numero piu' alto di tutta la
griglia per questa direzione. E' l'opposto del pattern trovato altrove
(selezione piu' fragile a ratio=1): la deduplicazione sui pattern interi
sembra meno sensibile, o sensibile in verso diverso, al ribilanciamento
estremo della sorgente rispetto alla selezione per margine sui punteggi
float. Non investigato oltre — un'osservazione, non una spiegazione.

**Verdetto del blocco (c): l'unica affermazione che testa (parita'
intero-float) e' la piu' robusta misurata in questo lavoro, stabile su
tutta la griglia da ratio=1 a ratio=100.**

### Verdetto finale delle cinque affermazioni, sulla griglia completa {1, 3, 20, 50, 100}

| # | affermazione | dove regge | dove cade | correzione necessaria |
|---|---|---|---|---|
| 1 | 13 coeff. recupera in 5/6 direzioni | tutta la griglia, in **entita'** | **affidabilita'** (n/10) cala a ratio=1 per `bot->ton` (8/10), `ton->bot` (7/10) | qualificare, non ritrattare — sez. 11 |
| 4 | `unsw->bot` fallisce, 0 normali in tutti i seed | ratio ≥ 3 | ratio=1: 1/10 seed riesce | correggere "tutti" → "tutti tranne il rapporto piu' estremo provato" — sez. 11 |
| 5 | 13 coeff. vs rifit, pareggio (delta −0,002) | ratio ≥ 3 (segno misto, `ton→bot` positivo contro quattro negative perlopiu' non sig.) | **ratio=1: 5/5 direzioni negative, 4/5 significative per direzione (n=7-10); metodo B su 6 seed genuinamente completi: −0,034, t=−5,04, p=0,0040** | correggere: "pareggio da ratio 3 in su, non a ratio 1" — sez. 11 |
| 2 | riadattamento continuo intero, 6/6, \|t\| 3,3-91,5 | ratio ≥ 3 (tutte e sei, sempre p<0,003) | **ratio=1: 4/6** (`unsw->bot` pareggia esatto, `ton->bot` n.s.) | correggere l'intervallo di `t` e il "6/6" — sez. 16.3 (fatto sotto) |
| 3 | RLS perde esattamente e solo quando BoT-IoT e' sorgente | **solo ratio ≥ 20** | **ratio=3: `bot->unsw` vince** (rottura del pattern); **ratio=1: anche `unsw->bot` (fonte UNSW) perde**, `ton->bot` non vince piu' | riscrivere, non qualificare — sez. 16.2 (fatto sotto) |

**La sintesi che questa sezione avrebbe scritto fermandosi a 20-100 —
"le conclusioni sono stabili fra 1:20 e 1:100" — era vera su quella
griglia e ingannevole sulla domanda che il compito poneva**, perche' su
quella griglia quattro direzioni su sei non erano mai vincolate (18.3).
Estesa a 1-3, la risposta corretta non e' una sola soglia ma due, perche'
le cinque affermazioni non cedono tutte nello stesso punto:

- **Quattro affermazioni (1, 2, 4, 5) reggono, con qualifiche minori a
  ratio=1, per tutto l'intervallo 3-100** — che copre sia il valore
  storico (50) sia un margine ragionevole in entrambe le direzioni. Il
  loro cedimento e' concentrato **solo** al punto piu' estremo, ratio=1
  (ribilanciamento esatto 1:1), e condivide lo stesso meccanismo: la
  selezione delle etichette sul target trova piu' spesso una sola classe
  quando la sorgente e' ribilanciata all'estremo, non un problema
  dell'adattamento in se'.
- **L'affermazione 3 e' la piu' fragile del lavoro, e cede prima delle
  altre**: regge solo per ratio ≥ 20, si rompe gia' a ratio=3 — il secondo
  punto della griglia, non il piu' estremo — e per un motivo diverso dalle
  altre quattro (`bot->unsw` cambia segno mentre `bot->ton`, la stessa
  sorgente, no: non e' spiegabile solo dalla selezione delle etichette).
  La diagnosi esistente in sezione 16.2 (quasi-separazione) resta un
  fattore plausibile ma dimostrabilmente incompleta.

Nessuna delle due letture sostituisce l'altra: il rapporto storico (50) e'
dentro la zona sicura per tutte e cinque, ma "sicura fra 1:3 e 1:100" —
l'affermazione che ci si aspetterebbe di poter scrivere dopo questo lavoro
— e' vera per quattro affermazioni su cinque e falsa per la quinta.

---

## Cosa resta da fare
1. ~~Terzo dominio~~ — **fatto** con UNSW-NB15. Tutte le sezioni da 5 a 9 sono
   state rieseguite su sei direzioni (sezioni 11, 12 e 13). **Sezione 11
   rilanciata su 10 seed** (nessun checkpoint a 3 seed esisteva in questo
   ambiente, quindi e' la prima misura completa): `unsw→bot` **confermato**
   fallire in tutti i 10 seed, non un artefatto — a differenza dello stesso
   tipo di direzione nello spazio ridotto (sezione 15), dove riesce in 1-7
   casi su 10. I due spazi di feature (6+2 contro 13+2) danno esiti diversi
   per la stessa coppia di domini: non si estrapola l'uno dall'altro. La
   tabella "13 coefficienti contro rifit completo" e' stata rifatta con test
   t appaiati per seed invece di un conteggio grezzo: `ton→bot` e' l'unica
   direzione dove i 13 coefficienti vincono in modo significativo a tutti i
   budget, nelle altre il rifit completo vince (spesso significativamente)
   a budget alto. La sezione 4 (`drift_sampling.py`) ha lo stesso tipo di
   affermazione *impossibile* **non ancora rilanciata a 10 seed** — non era
   nello scope degli script assegnati per questo lavoro, dichiarato come
   limite aperto nella sezione stessa.
2. ~~CIC-IoT-2023 come quarto dominio, nello spazio ridotto~~ — **fatto**
   (sezione 15): `test.csv` ha una `flow_duration` vera, il bug che forzava
   sempre lo spazio minimo per CIC e' corretto, e il confronto con UNSW-NB15
   e' rifatto a parita' di spazio **e a parita' di direzioni**, ora su
   **10 seed** (42-51). Prendendo il **seed** come unita' di replicazione,
   due colonne su quattro sopravvivono e una si rafforza rispetto a 3 seed
   (test t a un campione sul delta per seed; **correzione**: la prima
   stesura usava media/dev.std invece di media/errore standard,
   sottostimando l'evidenza di un fattore √10): **non adattato**
   (−0,130 ± 0,025, t=−16,48, p<0,0001, 10/10 concordi) e **128 etichette**
   (−0,086 ± 0,041, t=−6,62, p=0,0001 — **a 3 seed era t=−3,44, p=0,075,
   sotto soglia**, ora saldamente sopra), entrambe a favore di CIC; a 8 e 32
   etichette il rumore resta piu' grande del segnale (p>0,15 in entrambi i
   casi; a 32 etichette il segno del delta si e' persino invertito passando
   da 3 a 10 seed, coerente con l'assenza di segnale). Lo schema "CIC target
   facile / sorgente debole", **ritirato in una revisione intermedia basata
   su 3 seed, va reintrodotto — non come "sempre" ne' come "non regge
   affatto", ma con un meccanismo preciso e opposto per le due coppie**: la
   coppia sorgente (`cic->ton` contro `unsw->ton`) e' fortissima a poche
   etichette (8/8 a n=8, p=0,008; 8/9 a n=32, p=0,039 con test binomiale) e
   sparisce a 128 (6/9, p=0,51) — perche' a poche etichette conta ancora
   quanto bene la sorgente ordinava gia' il target, a molte l'adattamento
   riscrive quel prior. La coppia target (`ton`/`bot` verso `cic` contro
   `unsw`) fa il contrario: assente a 8 etichette (p=0,51), emerge a 32
   (`ton` p=0,021) e piena a 128 (`ton` p=0,021, `bot` p=0,002) — perche'
   serve budget perche' il soffitto di ciascun target diventi visibile.
   **Il seed 42 e'
   confermato un artefatto isolato**, non una bimodalita': a 32/128
   etichette e' l'ultimo su 10 (z-score −1,85/−2,77), gli altri nove sono
   raggruppati stretti. **"Fallisce"/"impossibile" per `ton->bot`,
   `unsw->bot`, `cic->bot` era un artefatto del campione a 3 seed**: con 10,
   producono un numero in 1-7 casi su 10 a seconda della direzione e del
   budget — raro (specie `unsw->bot`, 1-2/10), non impossibile; la stessa
   correzione si applica dove la frase ricorre altrove nel documento
   (sezioni 4, 11, 14), non ancora aggiornate. Resta il limite strutturale
   che nessuna correzione allo spazio elimina: unita' di osservazione
   diversa (finestre di pacchetti contro flussi bidirezionali).
3. ~~Un segnale di innesco che funzioni~~ — **risolto** dalla martingala
   conformal (sezione 9) — e ~~portarlo in aritmetica intera~~ — **fatto**
   (sezione 16.3), con un bug di segno trovato e corretto per strada, e
   **rilanciato su 10 seed** (misurando quanti seed su 10 fanno scattare
   l'innesco, non solo la balanced accuracy media, perche' la sezione 15
   aveva mostrato che "non scatta mai" puo' essere un artefatto del
   campione). Affidabile (10/10 seed) in 3 direzioni su 6; **`unsw->bot`
   conferma e rafforza la scoperta della sezione 13** (float 0/10 seed,
   genuinamente mai — non un artefatto — contro intero 8/10); **`ton->bot`
   e' piu' affidabile del previsto** (intero 8/10 seed, non "0,8,10 su un
   campione di 3", anche se la qualita' quando scatta resta rumorosa);
   **`ton->unsw` va riletta**: il float non e' debole, e' bimodale (5/10
   seed non scattano mai, gli altri arrivano fino a 14 adattamenti),
   l'intero resta 0/10. Da capire se un segnale di conformita' diverso da
   |z − mediana| fa meglio su `ton->unsw`. Il riadattamento continuo in
   interi resta il risultato piu' solido — batte lo statico in 6/6 anche a
   10 seed, con test t appaiati per seed che vanno da t=3,3 (BoT→TON) a
   t=91,5 (UNSW→BoT), tutti p<0,01: non piu' discutibile. Il conteggio
   "vince in 4, perde in 2" contro il float **non regge a 10 seed, ma non
   perche' sia tutto rumore**: la prima versione di questo controllo usava
   lo stesso denominatore sbagliato (dev.std invece di errore standard) e
   quindi nascondeva l'evidenza invece di sovrastimarla. Con il test t
   corretto, **due direzioni su sei si distinguono davvero**: TON→BoT ha un
   costo piccolo ma reale a favore del float (t=−3,75, p=0,0045, 10/10
   concordi), BoT→UNSW ha un guadagno piccolo ma reale a favore dell'intero
   (t=2,83, p=0,020, 8/10 concordi); le altre quattro restano indistinguibili
   (p sempre >0,45). La lettura corretta e' "costo piccolo e reale in
   TON→BoT, guadagno piccolo e reale in BoT→UNSW, indistinguibile nelle
   altre quattro" — ne' "nessun costo misurabile" ne' "vince quasi sempre".
4. ~~Chiudere il divario della stima intera in TON→BoT~~ — **in parte fatto,
   e ridimensionato da un confronto a campioni appaiati** (sezione 16.1):
   non era regolarizzazione (misurato e falsificato), era non convergenza a
   2000 iterazioni fisse. **Il "chiude il 65% del divario" non e' piu'
   citabile**: veniva da un confronto fra un campione a 3 seed (2000 iter)
   e uno a 10 (6000 iter). Rilanciando 2000 iter sugli stessi 10 seed
   (42-51) e appaiando per seed, **nessuna delle sei direzioni si distingue
   dal rumore, calibrazione inclusa** (TON→BoT: t=1,22, p=0,26). L'effetto
   non e' sparito, e' concentrato: su 9 seed validi, 8 hanno delta fra
   −0,007 e +0,015 (rumore), **1 solo (il seed 43, lo stesso della diagnosi
   originale) ha delta +0,183** — 6000 iterazioni non spostano la media, **
   salvano i seed rari che non convergono a 2000**, un'assicurazione contro
   la coda applicata a costo fisso su tutti i seed anche se solo 1 su 9 ne
   ha bisogno. Sulle cinque direzioni di validazione (confronto
   interi-contro-float, non 2000-contro-6000) il bilancio resta 2 meglio/3
   peggio, sostanzialmente nullo. ~~Un ottimizzatore con passo adattivo~~
   — **fatto** (sezione 17a): passo che si dimezza sui plateau piu' arresto
   esatto sul punto fisso. Qualita' statisticamente pari al fisso a 6000 in
   tutte e sei le direzioni (tutti p>0,15); il costo medio non cala (media
   5944 iterazioni, vicina a 6000) ma si ridistribuisce — 1800-4200 sulle
   direzioni facili, fino a 12 800 sulla piu' difficile, nessun seed
   sotto-convergente come poteva capitare a 2000 fisse.
5. ~~Rendere affidabili i minimi quadrati ricorsivi~~ — **in parte fatto,
   confermato e rafforzato su 10 seed** (sezione 16.2): due bug indipendenti
   (ordine del forgetting, ridge troppo debole contro la quasi-separazione
   al primo batch). `ridge=0,1`, fissato su TON→BoT da sola e validato
   sulle altre cinque, rilanciato su 10 seed: batte ancora lo statico in 3
   delle 5 direzioni di validazione, e le due perdite si sono ristrette
   (BoT→TON da −0,113 a −0,075, BoT→UNSW da −0,034 a −0,008, ormai dentro
   la dev.std). Perde sempre quando **BoT-IoT e' la sorgente** (le uniche
   due direzioni perdenti, confermato) — un pattern specifico, non un
   crollo generico. ~~Un ridge adattivo sulla frazione di classe
   minoritaria~~ — **provato, sezione 17b, e non ha funzionato**: ne' il
   criterio per-batch (falsificato, peggiora nettamente su bot→ton) ne'
   quello a riscaldamento sui primi aggiornamenti (indistinguibile dal
   ridge fisso su tutte e sei le direzioni a 10 seed, e sulle due direzioni
   che doveva sistemare la perdita e' uguale o leggermente peggiore). Il
   ridge fisso a 0,1 resta la scelta migliore trovata; il problema della
   quasi-separazione al primo aggiornamento resta diagnosticato ma non
   risolto. **La motivazione per risolverlo e' piu' forte di quanto sembrasse**:
   il modello di costo (17c, corretto dopo un errore aritmetico di un
   fattore 5) misura che la RLS non e' "l'unica strada per stare negli 8 KB
   di SRAM" — e' **551 volte piu' economica in calcolo e 5,9 volte piu'
   piccola in RAM** della stima a discesa del gradiente, su entrambe le
   voci insieme, non solo sulla memoria. Non e' bloccata dal costo, e' l'unica
   strada gia' bloccata da un problema di accuratezza in due direzioni su
   sei: risolverlo sblocca l'opzione piu' economica dell'intero lavoro, non
   solo un'alternativa in piu'.
6. ~~Verifica bit-esatta dell'header C rigenerato~~ — **fatta**:
   `mcu/run_int_adapt_check.cpp` compilato con g++ 13.3.0 a `-O2` contro il
   `mcu/kan_int_adapt.h` rigenerato a 6000 iterazioni da' `logit diversi: 0,
   decisioni diverse: 0` sui 200 golden vector, 24 byte di aggiornamento.
7. ~~Misure sul dispositivo~~ — **il modello di costo che le sostituisce
   quando l'hardware non c'e' e' fatto** (sezione 17c): MAC int32, lookup
   LUT e RAM di picco contati in funzione di n, d, iters per inferenza,
   stima dei guadagni, RLS (proiettato, non misurato: resta float) e
   martingala intera, con il confronto sulle stesse voci contro il rifit
   completo. **Un limite nuovo, trovato costruendo il modello**: lo stato
   di calibrazione della martingala intera (`n_cal` righe ordinate,
   residenti) puo' costare piu' della LUT sigmoide e della RLS messe
   insieme — lo stesso problema di RAM che la RLS era nata per risolvere,
   qui non affrontato. Restano le **misure vere sul dispositivo** (tempo e
   memoria su Mega 2560, ESP32-C3): il modello di costo dice cosa aspettarsi
   moltiplicando per i cicli-per-operazione del target, non lo sostituisce.
8. ~~Sensibilita' al rapporto di undersampling (1:50)~~ — **fatto** (sezione
   18), l'ultima voce aperta dalla fase 2. Il rapporto e' un no-op quando la
   sorgente e' gia' piu' equilibrata di lui (TON_IoT naturale 3,22:1,
   UNSW-NB15 1,77:1): la prima griglia provata (20/50/100) non vincolava mai
   quelle due sorgenti, quindi quattro direzioni su sei non erano testate
   affatto. Estesa a `--ratio 1` e `--ratio 3` (che vincolano anche TON_IoT e,
   a ratio=1, anche UNSW-NB15) su tutte e sei le direzioni, 10 seed. **Tre
   affermazioni reggono con qualifiche minori** (1: l'affidabilita' della
   selezione cala a ratio=1 per due direzioni; 4: `unsw→bot` fallisce in 9/10
   seed a ratio=1 invece di 10/10; 2: "6/6" diventa "6/6 per ratio ≥ 3, 4/6 a
   ratio=1" — corretto in sezione 16.3). **Due cadono davvero**: 5 (13
   coefficienti contro rifit completo, "pareggio" regge da ratio=3 in su ma a
   ratio=1 perde in 5 direzioni su 5, 4 in modo significativo per direzione
   (n=7-10 seed ciascuna) — corretto in sezione 11, con test per-direzione
   invece di un aggregato su seed a copertura diseguale, come gia' in
   sezione 15) e 3 (RLS "perde esattamente e solo quando BoT-IoT e' sorgente"
   si rompe gia' a ratio=3, dove `bot→unsw` cambia segno e vince, e a
   ratio=1 anche `unsw→bot`, sorgente non-BoT-IoT, perde — corretto in
   sezione 16.2, e' l'affermazione piu' fragile del
   lavoro). Il meccanismo comune ai cedimenti (tranne per l'affermazione 3):
   a ratio=1 la selezione delle etichette sul target trova piu' spesso una
   sola classe, perche' il punteggio del modello sorgente ribilanciato 1:1
   cambia. **Non spiegato**: perche' l'affermazione 3 si rompe gia' a
   ratio=3 e in modo asimmetrico fra le due direzioni BoT-sorgente
   (`bot→unsw` cambia segno, `bot→ton` no) — la diagnosi esistente
   (quasi-separazione IRLS) non basta da sola. Un bug reale trovato per
   strada e corretto: il checkpoint di `drift_graduale.py` troncava 6
   politiche su 7 su disco (mai nei CSV pubblicati, generati da run continui
   in memoria) da quando sezione 17b aveva aggiunto una settima politica
   senza aggiornare la costante di scrittura — sistemato rendendo la
   costante dipendente dal numero di politiche invece che fissa.

---

### File

- `scripts/drift_diagnosi.py` — diagnosi soglia contro rappresentazione
- `scripts/drift_adapt.py` — scala di interventi, dal più economico al rifit
- `scripts/drift_sampling.py` — regole di selezione delle etichette
- `scripts/drift_baselines.py` — aggiornamento minimo strutturale per modello
- `scripts/drift_int_adapt.py` — adattamento integer-only ed export C, ora
  con `--iters` (default 6000, file separati per valore diverso — sezione
  16.1, confronto 2000-contro-6000 a campioni appaiati), `--adaptive`
  (sezione 17a, passo che si dimezza sui plateau + arresto sul punto fisso)
  e `--ratio` incluso nel nome di checkpoint/CSV (vuoto a ratio=50 — sezione
  18); l'header C canonico si scrive solo a `iters=6000, ratio=50`
- `scripts/drift_graduale.py` — deriva progressiva e politiche di
  riadattamento; `StatSufficienti` ora supporta `adaptive_ridge` con
  `ridge_mode="per_batch"` (falsificato) o `"warmup"` (non ha battuto il
  ridge fisso — sezione 17b), politica aggiuntiva `stat_13x13_adaptive`;
  `--ratio` incluso nel nome di checkpoint/CSV (sezione 18); bug corretto
  nella scrittura del checkpoint, che troncava su disco (non nei CSV
  pubblicati, generati da `rows` in memoria) i primi batch di 6 politiche
  su 7 da quando la settima e' stata aggiunta senza aggiornare la costante
  di slicing (sezione 18, trovato analizzando la sensibilita' al rapporto)
- `scripts/drift_senza_etichette.py` — EM sul prior, TENT, TENT filtrato, IM
- `scripts/drift_trasferimenti.py` — Firth e k-center da altri campi
- `scripts/spazio_ridotto.py` — costo della riduzione dello spazio armonizzato
- `scripts/sweep_iperparametri.py` — **nuovo**: rigenera gli sweep di
  calibrazione delle sezioni 16.1 e 16.2 (`iters`, `ridge`) sulla sola
  direzione `ton->bot`. Prima queste tabelle venivano da script ad-hoc mai
  versionati e nessuno poteva rifarle; ora sono riproducibili
  (`results/sweep_iperparametri{,_runs}.csv`)
- `scripts/cross_domain.py` — versione a tre (e quattro) domini, con
  `load_harmonized(spazio_cic=...)` che sceglie fra spazio minimo e ridotto
  per CIC invece di forzare sempre il minimo (bug corretto, sezione 15)
- `scripts/tre_domini.py` — terne di domini, ora passa `spazio_cic` coerente
  con `--spazio`, e `finalize()` scrive `tre_domini_runs_<spazio>_<terna>.csv`
  (bug corretto: il nome non includeva la terna e un secondo run la
  sovrascriveva, sezione 15); `--ratio` incluso nel nome dal 18 in poi
- `scripts/drift_graduale_int.py` — **nuovo**: martingala conformal e
  riadattamento in aritmetica intera (sezione 16.3); `--ratio` incluso nel
  nome di checkpoint/CSV dal 18 in poi
- `kanids/harmonized.py`, `kanids/datasets.py` — con UNSW-NB15 e con
  CIC-IoT-2023 vero (`test.csv`, `flow_duration` reale, etichetta multiclasse
  binarizzata, `build_ridotto_cic` finalmente raggiungibile)
- `results/drift_trasferimenti_normali.csv` — normali raccolte, dove si vede
  che il k-center rompe il tetto sul budget
- `kanids/int_adapt.py` — primitive intere: sigmoide a LUT, stima dei
  guadagni (ora con `iters` configurabile, `reg_shift`/`reg_target`
  provati e scartati — sezione 16.1 — e `adaptive=True`/`return_iters` per
  il passo che si dimezza sui plateau — sezione 17a), e la martingala
  conformal intera (`build_martingale_lut`, `martingale_batch_int`,
  `martingale_update_int`, sezione 16.3)
- `mcu/kan_int_adapt.h` — tabelle e 200 golden vector, rigenerati con i
  guadagni a 6000 iterazioni e riverificati bit-esatti (g++ 13.3.0, `-O2`)
- `mcu/run_int_adapt_check.cpp` — verifica di bit-esattezza (compila con g++)
- `results/drift_{diagnosi,adapt,sampling,baselines,int_adapt,graduale,graduale_int}.csv`
- `results/tre_domini_ridotto_bot_cic_ton.csv`,
  `results/tre_domini_ridotto_bot_ton_unsw.csv` — il confronto della
  sezione 15, a parita' di spazio, ora su 10 seed (42-51)
- `results/tre_domini_runs_ridotto_tonbotcic.csv`,
  `results/tre_domini_runs_ridotto_tonbotunsw.csv` — un record per seed per
  ciascuna delle due terne (nome per terna, bug corretto: sezione 15), da
  cui viene la dev.std della tabella appaiata
- `results/tre_domini_ricco_bot_ton_unsw.csv`,
  `results/tre_domini_runs_ricco_tonbotunsw.csv` — sezione 11, spazio ricco,
  10 seed (42-51), prima misura completa in questo ambiente
- `results/drift_int_adapt_iters2000.csv`,
  `results/drift_int_adapt_iters2000_runs.csv` — confronto 2000-contro-6000
  iterazioni a campioni appaiati, sezione 16.1
- `results/*_runs.csv` — un record per run
- `results/drift_sampling_normali.csv` — normali raccolte da ogni regola,
  la colonna che spiega tutte le altre
- `results/drift_graduale_curva.csv` — accuratezza contro contaminazione
- `results/*_ratio{1,3,20,100}*.csv`, `artifacts/*_ratio{1,3,20,100}*.jsonl`
  — sezione 18, la griglia di sensibilita' al rapporto di undersampling;
  vuoto (nessun suffisso) a ratio=50, che resta il file storico

Gli script sono checkpointati e riprendibili, come gli altri del repository,
e non toccano nulla di esistente fuori da `adattamento-drift/`. Verifica
dell'header C, rilanciata e superata (g++ 13.3.0, `-O2`):

    g++ -O2 -I mcu -o /tmp/check mcu/run_int_adapt_check.cpp && /tmp/check
    # golden vector: 200 / logit diversi: 0 / decisioni diverse: 0
    # byte riscritti per l'adattamento: 24 (12 int16)

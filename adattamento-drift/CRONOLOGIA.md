# Cronologia: le versioni precedenti delle sezioni riscritte

`RISULTATI.md` e' lo **stato corrente**: ogni affermazione compare una volta
sola, nella sua versione finale, con la sua misura. Questo documento
conserva le versioni **superate** delle sezioni riscritte, perche' tre cose
richiedono che la storia resti leggibile:

1. **La sezione «limiti» dell'articolo si scrive da qui.** Un lavoro che
   dichiara quali proprie affermazioni sono cadute, e con quale evidenza, e'
   piu' difendibile di uno che presenta solo quelle sopravvissute.
2. **Le affermazioni cadute dicono qualcosa sul metodo, non solo sul
   risultato.** Tre delle quattro sono cadute per la stessa ragione — un
   campione di 3 seed, o p-value non corretti per confronti multipli — e
   quella ragione e' essa stessa un risultato metodologico.
3. **Chi rilegge deve poter verificare che nulla sia sparito in silenzio.**
   Le versioni qui sotto sono quelle esatte, non riassunte.

## Cosa e' cambiato, in breve

| sezione | affermazione superata | perche' e' caduta | dove sta ora |
|---|---|---|---|
| 1 | «la KAN ha il miglior ROC-AUC sul target in **entrambe** le direzioni» | a 10 seed LightGBM sta davanti in TON→BoT (0,5516 contro 0,5171); il divario non e' significativo, quindi le due non si distinguono | sez. 1 |
| 4 | «funziona la regola piu' banale: gli n flussi piu' vicini al confine» | a 10 seed la regola adattiva la batte a ogni budget in TON→BoT, e in BoT→TON il margine e' la peggiore di tutte, sotto il prelievo casuale | sez. 4 |
| 5 / 12 | «il vantaggio della KAN non e' architetturale» (e, prima ancora, «la KAN domina») | entrambe troppo forti: con Holm nessun confronto contro l'MLP e' significativo, ma le entita' sono asimmetriche di un ordine di grandezza | sez. 5 |
| 11 | «il rifit completo vince in modo significativo in 4 direzioni su 5 a n=128» | quei p-value non erano corretti per la famiglia di 15 confronti; con Holm nessuna delle quattro sconfitte sopravvive | sez. 11 |

Il **pareggio** fra 13 coefficienti e rifit completo, invece, non e' caduto:
il delta medio a n=128 e' passato da −0,002 a −0,0016, e ora ha dietro un
test corretto invece di un conteggio di vittorie.

---

<!-- SEZIONE 1 (protocollo v1) -->
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


<!-- SEZIONE 4 (protocollo v1) -->
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


<!-- SEZIONE 5 (protocollo v1) -->
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


<!-- SEZIONE 11 (protocollo v1) -->
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


<!-- SEZIONE 12 (protocollo v1) -->
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



<!-- SEZIONE 13 (numeri non aggiornati dai CSV) -->
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


<!-- SEZIONE 7 (numeri non aggiornati dai CSV) -->
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


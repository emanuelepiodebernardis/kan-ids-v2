# Sintesi del primo ciclo — risultati, limiti, e il passo successivo

**ADAPT-01, stream di sviluppo C.** 2 ottobre 2026. L'evidenza è nel Draft PR #2.
I riepiloghi di questo documento si rifanno con `riepiloghi_semi.py` sui cinque
rendiconti per blocco pubblicati, e quello script verifica da sé i propri conti.

## In breve

Le due pagine che la scheda chiede sono queste; il resto del documento è il
dettaglio, e i numeri si rifanno con gli script indicati nel protocollo.

**La domanda.** Quando piccoli aggiornamenti permettono a un rilevatore di
riconoscere meglio gli attacchi dopo un cambiamento del traffico, con poche
etichette e in ritardo.

**Il risultato principale, che non è una media.** Il modello congelato ordina al
contrario — AUROC sotto 0,5 — in una minoranza di blocchi, e **solo lì**
l'adattamento guadagna molto: +0,553 per la logistica contro −0,064 altrove.
L'inversione segue il tipo di attacco quasi esattamente: 122 blocchi di `dos` su
123 sono invertiti, 46 di `injection` su 49, 6 di `ddos` su 586, 0 di `password`
su 120. Il modello additivo si inverte in 9 o 10 blocchi su 875 e non ha quasi
nulla da correggere.

**Da quel risultato è nata una regola operativa, ed è stata verificata.**
Aggiornare **solo su evidenza di inversione** fa quanto o meglio
dell'aggiornamento continuo spendendo da un undicesimo a un ventisettesimo degli
aggiornamenti, e **batte il controllo negativo** — una politica che aggiorna
sullo stesso numero di blocchi scelti a sorte — in tutti e tre i modelli, di
+0,026, +0,069 e +0,097 di AUROC. Non è quindi il risparmio a produrre il
risultato, è il criterio. Il dettaglio è al §6.

**Quattro cose che vanno dette con il risultato.**

1. **L'aggregazione cambia la conclusione.** Il tasso di falsi allarmi va
   riportato sia come media per blocco sia come somma delle matrici di
   confusione: su questi dati divergono, e per due modelli su tre hanno verso
   opposto (§3).
2. **La soglia cambia la conclusione.** Con la soglia scelta su B invece che a
   zero, due dei tre risultati sulle misure di decisione si rovesciano: una
   parte di quello che avevamo attribuito all'adattamento era mala taratura del
   modello congelato (§3bis). L'AUROC non ne è toccata, ed è verificato su
   27.480 valori.
3. **Il rivelatore di inversione è debole.** Esiste solo nel 41,6% dei blocchi,
   perché fra le cento righe campionate il blocco mediano non contiene nessun
   normale, e come regola di decisione ha precisione 15–18%. Funziona non perché
   sia preciso, ma perché aggiornare raramente danneggia poco anche sbagliando.
4. **D non è indipendente da A, B e C.** Il 37,23% delle sue righe e il 58,03%
   dei suoi normali hanno un vettore di feature già visto. La valutazione finale
   va riportata due volte, sull'intero stream e sul sottoinsieme a vettore non
   visto (§5).

**Per il modello additivo nessuna politica batte il congelato**: 0,8909 contro
0,8358 della migliore. È il risultato più scomodo e il più solido.

---

## 0. Due convenzioni, dichiarate prima dei risultati

**I tre modelli.** Regressione logistica sulle otto feature; MLP di cui si
aggiorna il solo ultimo strato; e una **KAN a singolo strato con edge a
B-spline**, chiamata qui «l'additivo» per brevità: il punteggio è una somma di
funzioni univariate cubiche, una per ciascuna delle otto feature, più un
guadagno per feature e un'intercetta.

Sulla terza una precisazione, perché una versione precedente di questo documento
la descriveva come «non il componente a B-spline del Paper 1» e quella
formulazione era fuorviante. La **forma funzionale è la stessa** di
`BSplineKANBinary` in `src/kan_bspline.py`, che è a sua volta a singolo strato:
non è un'architettura diversa, è **reimplementata** invece di importata. Ciò che
differisce, e che conta per questo studio, è il regime di stima: i 64
coefficienti delle spline sono stimati **una volta sola** su un sottocampione di
400.000 righe di A e poi congelati, e l'adattamento muove soltanto **otto
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
esattamente. Contando i blocchi in cui il tipo compare, fra gli 875 misurabili e
per la logistica sul seme 42:

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

L'additivo **quasi non si inverte**: dieci blocchi contro i 174 della logistica.
È anche il modello con l'AUROC congelato più alto, 0,891. Non è che
l'adattamento gli faccia male in modo particolare: è che **non ha quasi nulla da
correggere**, e paga soltanto il costo dell'aggiornamento.

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
aumentano sempre, senza dichiarare l'aggregazione: sul complessivo è vero per il
solo MLP.

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
adattivo — l'AUROC non cambia di un valore: **27.480 misure verificate
identiche**, cinque semi per 916 blocchi per tre modelli per due stati, e gli
indici campionati identici in tutti i blocchi. Il §1 resta quindi valido parola
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
congelato passa da 0,483 a 0,960 solo cambiando soglia — e il confronto a soglia
zero attribuiva all'aggiornamento una parte di quella mala taratura. La
conclusione «l'adattamento peggiora sempre il richiamo sui normali» sopravvive
soltanto per il **modello additivo**, su entrambe le aggregazioni, e per l'MLP
sulla sola media per blocco.

La soglia è scelta separatamente per ogni modello e per ogni seme, sull'intero
B, fra i valori distinti dei punteggi del congelato; fra i candidati a pari
merito si prende la **mediana inferiore**. Su questi dati la regola di parità non
è mai stata applicata, perché il massimo esatto è sempre raggiunto da un
candidato solo su circa 405.000 — ma con una tolleranza di confronto di 1e-6
pareggerebbero fino a 27 candidati, quindi la regola non è una formalità. È una
**convenzione deterministica**: che la soglia della logistica risulti identica
nei cinque semi segue dal fatto che il congelato è deterministico, non è una
misura di robustezza, e la stabilità non è misurata.

## 4. Il protocollo ha un collo di bottiglia misurato

Con blocchi da 10.000, budget dell'1% e memoria FIFO di 256, la regola del salto
per memoria monoclasse si attiva su **395 blocchi in media su 915, il 43,2%**
(fra 374 e 414 sui cinque semi). Non è un incidente: i normali per blocco hanno
mediana 24 contro una media di 189, e il 5% dei blocchi più ricchi contiene il
47,3% di tutti i normali, quindi un blocco mediano offre 0,24 normali attesi fra
i cento campionati.

Il rimedio più ovvio non funziona: distribuendo lo stesso budget su dieci fasce
del punteggio della logistica congelata, i salti **salgono** da 393 a 421 e le
misure cambiano entro ±0,016, perché quel punteggio colloca quasi tutto nella
regione «attacco».

## 5. Limiti, dichiarati

**Un solo flusso.** Tutto quanto sopra è misurato su C. Lo stream D non è stato
usato per addestrare né per valutare.

**D non è indipendente da A, B e C.** Il controllo diretto delle sovrapposizioni
è stato eseguito e l'esito è negativo: il **37,23%** delle righe di D e il
**58,03%** dei suoi normali hanno un vettore di feature che compare già in A, B
o C. Non è l'effetto di un vettore degenerato — servono 235 impronte per metà
di quelle righe e 54.821 per il 90% — né del solo tipo condiviso `password`:
almeno 592.463 righe di attacco sovrapposte sono di tipi che in D compaiono per
la prima volta. Una coincidenza di vettori non dice che sia la stessa sessione o
lo stesso host, e quella distinzione resta non stabilita. La proposta è
riportare la valutazione finale due volte, sull'intero D e sul solo
sottoinsieme a vettore non visto, 2.228.436 righe con 129.982 normali. Il
dettaglio è nel protocollo §5bis.

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
almeno 200 normali contro −0,132 in quelli con meno di dieci. Sui cinque semi il
primo valore è stabile, da +0,406 a +0,409; il secondo no, da −0,132 a −0,217.

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
916 blocchi costa 84 secondi e l'addestramento iniziale dei tre modelli 33,
cioè 0,09 secondi per blocco. I valori sono quelli registrati nella corsa di
riferimento, `replay_C_seme42.json`, e variano di qualche secondo fra
esecuzioni: sono un ordine di grandezza, non una misura ripetibile.

## 6. L'esperimento proposto, eseguito: aggiornare solo su evidenza di inversione

Era la proposta del primo ciclo. È stata eseguita su C, con la soglia del §3bis,
parametri invariati, cinque semi, e con il controllo negativo.

**Il primo passo, che veniva prima di tutto: il verso è stimabile in esercizio?**
Sì, ma male, e soltanto fuori campione. La stima calcolata sui 256 esempi in
memoria **non riconosce mai** un blocco invertito — zero su 38, zero su 26, zero
su 50 — perché la memoria è l'insieme su cui l'aggiornamento è stato fatto: è
una stima in campione e vale quasi sempre circa 1,0. La stima calcolata sulle
cento righe campionate del blocco, con i punteggi già emessi, è invece fuori
campione e porta segnale: correlazione da +0,69 a +0,73 con l'AUROC vera del
blocco. Ma **esiste solo nel 41,6% dei blocchi**, perché fra le cento righe
campionate il blocco mediano non contiene **nessun** normale; si degrada a
+0,29 ÷ +0,36 al ritardo che conta, *j+2*; e come regola di decisione ha
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

**La politica su evidenza batte il controllo negativo in tutti e tre i modelli**
— +0,026, +0,069, +0,097 — quindi non è il numero ridotto di aggiornamenti a
produrre il risultato, è il criterio. Ed è la prima risposta che il controllo
negativo esisteva per dare: sarebbe stata l'altra.

**Batte anche l'aggiornamento continuo spendendo da un undicesimo a un
ventisettesimo degli aggiornamenti.** Per l'MLP è l'unica politica che batte il
congelato, 0,7956 contro 0,7921. Per la logistica tutte lo battono.

**Per il modello additivo nessuna politica batte il congelato**: 0,8909 contro
0,8358 della migliore. Il §1 lo aveva previsto — quel modello si inverte dieci
volte su 875, non ha quasi nulla da correggere — e qui la previsione si verifica
per una via diversa.

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
- Conferma dei confini A/B/C/D, dopo il controllo diretto delle sovrapposizioni,
  e come riportare la valutazione finale su D dato che il 37,23% delle sue righe
  ha un vettore già visto (§5).
- Se il passo successivo sia **migliorare il rivelatore** — più normali
  etichettati per blocco, memoria per classe — oppure portare il risultato del
  §6 così com'è: una politica che con quaranta aggiornamenti su 915 blocchi fa
  quanto o meglio di cinquecentoventi, e batte il controllo casuale.

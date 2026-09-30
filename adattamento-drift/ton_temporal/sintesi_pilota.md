# Sintesi del primo ciclo — limiti e prossimo esperimento

**ADAPT-01, stream di sviluppo C.** 30 settembre 2026. Tutti i numeri sono
misurati e ripetuti su cinque semi, da 42 a 46; l'evidenza è nel Draft PR #2.

---

## 1. La domanda, e la risposta che i dati danno

La domanda è **quando** piccoli aggiornamenti permettono a un rilevatore di
riconoscere meglio gli attacchi dopo un cambiamento del traffico, con poche
etichette e in ritardo.

La risposta del pilota è netta e non era quella che ci aspettavamo: **gli
aggiornamenti servono quando il modello congelato ha smesso di ordinare nel
verso giusto, e costano poco ma costano in tutti gli altri casi.**

Sul flusso C, il modello congelato ha AUROC **inferiore a 0,5** — cioè ordina al
contrario — in 174 blocchi su 875 per la regressione logistica. In quei blocchi
l'adattamento guadagna **+0,556**; in tutti gli altri perde **−0,056**. La media
aggregata, +0,059, è la somma di questi due regimi, e presa da sola non dice
nulla di utile.

Vale per tutti e tre i modelli, con una differenza che è essa stessa un
risultato:

- **regressione logistica**: 174 blocchi invertiti su 875 (20%), da 0,319 a
  0,875 (**+0,556**); altrove da 0,813 a 0,757 (−0,056).
- **MLP**: 154 invertiti (18%), da 0,282 a 0,937 (**+0,655**); altrove da 0,909
  a 0,810 (−0,098).
- **KAN additiva**: **10 invertiti (1%)**, da 0,432 a 0,914 (+0,483); altrove da
  0,891 a 0,795 (−0,096).

## 2. Perché la KAN additiva si comporta diversamente

I modelli iniziali sono addestrati su A, che contiene un solo tipo di attacco,
`scanning`. I blocchi in cui il congelato si inverte sono quasi tutti quelli di
`dos` (122) e `injection` (46); i blocchi di `ddos` (580) e `password` (120) non
si invertono quasi mai.

La KAN additiva **quasi non si inverte**: dieci blocchi contro i 174 della
logistica. È anche il modello con l'AUROC congelato più alto, 0,891. Non è che
l'adattamento le faccia male in modo particolare: è che **non ha quasi nulla da
correggere**, e paga soltanto il costo dell'aggiornamento.

Questo suggerisce una lettura che il pilota non può ancora dimostrare: una forma
additiva, fittata su un tipo solo, generalizza a tipi nuovi senza rovesciare il
verso, mentre una forma lineare nello spazio pieno o una testa densa lo
rovesciano. È la prima cosa da mettere alla prova.

## 3. Quello che l'adattamento peggiora, sempre

Il richiamo sui normali cala in **tutti e tre i modelli e in tutti e cinque i
semi**: da 0,431 a 0,220 per la logistica, da 0,732 a 0,264 per l'MLP, da 0,483
a 0,297 per la KAN. Essendo il complemento del tasso di falsi allarmi, **i falsi
allarmi aumentano sempre**. L'accuratezza sale, ma in un flusso in cui gli
attacchi sono il 98% questo dice solo che il modello si sposta verso la
maggioranza.

Per un rilevatore da mettere in esercizio è il numero che conta più
dell'AUROC, e va riportato accanto a ogni risultato.

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
toccato e non deve esserlo finché il protocollo non è fissato.

**La soglia resta a zero.** Il richiamo sui normali e i falsi allarmi sono
calcolati al punto di decisione naturale del punteggio, senza calibrazione.
L'intervallo B esiste per questo e non è ancora stato usato.

**L'inversione è misurata, non spiegata.** Sappiamo *quando* il congelato si
inverte — sui tipi `dos` e `injection` — ma non *perché* quella geometria
rovesci il verso, né se dipenda dal tipo o dalle sue caratteristiche di flusso.

**La ricchezza di normali è un indizio, non la causa.** Il guadagno sembrava
legato al numero di normali nel blocco: +0,409 nei blocchi con almeno 200
normali contro −0,132 in quelli con meno di dieci. Controllando per
l'inversione, però, i blocchi ricchi **non** invertiti danno solo +0,071. Il 67%
dei blocchi ricchi è invertito: la ricchezza di normali coincide in larga parte
con l'inversione, e non la sostituisce come spiegazione.

**Il campione noto è una ricostruzione.** Il controllo delle sovrapposizioni
degli otto input con il CSV noto usa impronte ricostruite, non l'archivio
originale, che non è pubblicato. Il riscontro è esatto sulla cardinalità — 92.330
uniche da 211.043 righe — ma resta una ricostruzione, e da sola non dimostra
l'indipendenza di D.

**Nessun costo hardware.** Solo CPU, come previsto in questa fase: il replay su
916 blocchi costa 86 secondi e l'addestramento iniziale dei tre modelli 27,
cioè 0,09 secondi per blocco.

## 6. Il prossimo esperimento che propongo

**Mettere alla prova l'inversione come criterio di aggiornamento.**

Se gli aggiornamenti servono solo quando il congelato si è invertito, allora una
politica che aggiorni **soltanto quando c'è evidenza di inversione** dovrebbe
tenere il guadagno dove c'è e togliere la perdita altrove — cioè battere sia il
congelato sia l'aggiornamento continuo, e spendere meno etichette.

Il disegno, tutto su C e a parametri invariati:

1. **Un rivelatore di inversione utilizzabile in esercizio.** L'AUROC del blocco
   richiede le etichette di tutto il blocco e non è disponibile. Si stima il
   verso sulle sole etichette del budget — cento per blocco — per esempio con la
   concordanza fra punteggio ed etichetta sui campioni disponibili. Il primo
   passo è misurare quanto quella stima su cento esempi segua l'AUROC vera del
   blocco: se non la segue, l'idea cade subito.
2. **Tre politiche a confronto**, con gli stessi blocchi, gli stessi `row_id` e
   lo stesso seme: congelato; aggiornamento a ogni blocco, che è l'attuale; e
   aggiornamento solo su evidenza di inversione.
3. **Le misure**: AUROC e richiamo sui normali per blocco, più **etichette spese
   e aggiornamenti applicati**, perché il risparmio è metà del risultato.
4. **Il controllo negativo**: una politica che aggiorni sullo stesso numero di
   blocchi ma scelti a caso. Se la politica su evidenza non la batte, il segnale
   è il conteggio degli aggiornamenti e non l'inversione.

**Costo stimato**: quattro politiche per cinque semi su C sono circa venti
esecuzioni da due minuti, cioè meno di un'ora di CPU.

**Che cosa deciderebbe.** Se la politica su evidenza tiene il guadagno e toglie
la perdita, il risultato dell'articolo diventa una regola operativa — *quando*
aggiornare — e non solo un confronto fra architetture. Se non lo fa, resta il
risultato descrittivo del §1, che è comunque solido, e la KAN additiva come
modello che si inverte raramente diventa il punto centrale.

## 7. Che cosa serve dal referente

- Conferma che la soglia resti a zero in questa fase, oppure indicazione di
  usare B per calibrarla prima del pilota su D.
- Parere sul budget: i 395 salti dipendono dalla rarità dei normali per blocco,
  e la stratificazione sul punteggio non rimedia. Le alternative sono una
  memoria che conservi per classe o un budget maggiore sui blocchi poveri. La
  scheda fissa i parametri, quindi non li cambio senza indicazione.
- Conferma dei confini A/B/C/D prima di toccare D.

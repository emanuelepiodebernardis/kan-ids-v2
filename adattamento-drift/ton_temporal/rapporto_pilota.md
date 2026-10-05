# Nota del pilota — ADAPT-01, stream di sviluppo C

4 ottobre 2026. I numeri vengono dai rendiconti pubblicati nel Draft PR #2 e si
rifanno con i comandi di `ambiente_e_comandi.md`. I rinvii come «§3» sono alle
sezioni della **sintesi**, quelli come «§11.3» ai punti aperti del
**protocollo**.

## Osservazioni

**Il guadagno dell'adattamento non è diffuso: vive dove il modello congelato
sbaglia verso.** Il congelato ordina al contrario — AUROC sotto 0,5 — in una
minoranza di blocchi, e solo lì l'adattamento guadagna molto: in media sui
cinque semi +0,553 di AUROC per la logistica nei blocchi invertiti, contro
−0,064 negli altri. L'inversione segue il tipo di attacco quasi esattamente:
122 blocchi di `dos` su 123 sono invertiti, 46 di `injection` su 49, 6 di
`ddos` su 586, 0 di `password` su 120 (§2).

**Da lì è nata una regola operativa, ed è stata eseguita.** Aggiornare solo su
evidenza di inversione fa quanto o meglio dell'aggiornamento continuo spendendo
molti meno aggiornamenti — in media sui cinque semi 45,2 contro 520 per la
logistica, 33,8 per l'MLP e 19,4 per l'additivo — e sull'AUROC batte in media
il controllo negativo, che aggiorna lo stesso numero di volte su blocchi scelti
a sorte, di +0,026, +0,069 e +0,097. Il confronto è a parità di aggiornamenti, quindi il margine
non viene dall'averne spesi di meno; ma è un margine medio sull'AUROC, misurato
contro un solo sorteggio per modello e per seme, e non si sa ancora se cada
fuori dal rumore del sorteggio: è questo che il prossimo esperimento deve
stabilire (§6).

**Lo stesso confronto non è uniforme, e va detto con il risultato.** Vince su 4
semi su 5 per logistica e MLP, su 5 su 5 per l'additivo, e si rovescia sul seme
44. Non si estende alle misure di decisione: sull'FPR complessivo il sorteggio
fa meglio per l'MLP (0,1625 contro 0,1844) e sul richiamo degli attacchi fa
meglio su due modelli su tre (per l'MLP 0,7789 contro 0,7088).

**Per il modello additivo nessuna politica batte il congelato**, su tutti e
cinque i semi e con ciascuna delle tre politiche: è il risultato più scomodo e,
essendo l'unico uniforme, il più solido. Uniforme è la direzione, non i numeri:
0,8909 contro 0,8358 sono medie sui semi, e per seme il congelato va da 0,8857
a 0,8944 mentre `su evidenza`, che è la migliore su quattro semi su cinque, va
da 0,7885 a 0,8814; sul seme 44 la migliore è `ogni blocco`, con 0,7926. Il vantaggio è sul flusso
intero e non in ogni suo punto — fino al blocco 212 almeno una politica sta
davanti, in 175 blocchi su 213, mai tutte e tre insieme — e sul seme 42 la politica casuale arriva a perdere più attacchi del
congelato: 3.356.503 contro 3.157.627 (`figure/politiche_kan.png`).

**Dove l'adattamento guadagna dipende da quanti normali il blocco contiene.**
In tutti e tre i modelli il guadagno medio di AUROC è positivo solo nei 246
blocchi con almeno 200 normali, e negativo nelle tre fasce più povere; questo
vale su tutti e cinque i semi. I valori, **sul seme 42**, sono per la logistica
+0,409 contro −0,132, −0,035 e −0,032 (`figure/guadagno_per_fascia_lr.png`); le
quattro fasce contano 221, 275, 133 e 246 blocchi fra gli 875 in cui l'AUROC è
definita.

**Due scelte di misura cambiano la conclusione, non solo il valore.**
L'aggregazione del tasso di falsi allarmi — media per blocco oppure somma delle
matrici di confusione — dà verso opposto per due modelli su tre (§3). E con la
soglia scelta su B invece che a zero, due dei tre risultati sulle misure di
decisione si rovesciano: una parte di quello che era stato attribuito
all'adattamento era mala taratura del congelato (§3bis). L'AUROC non ne è
toccata, su 27.480 coppie confrontate di cui 26.250 su valori definiti.

**Il costo di un aggiornamento è ora misurato e non stimato.** Mediane di 5,19,
5,27 e 11,08 ms per logistica, MLP e additivo, pari al 5,6%, 3,5% e 7,8% del
tempo di replay (§4bis).

## Limiti

Sono risultati **esplorativi su C**, senza nessuna significatività statistica
calcolata; C è lo stream di sviluppo e serve a formulare ipotesi, non a
confermarle. Il controllo casuale è **un solo sorteggio** per modello e per
seme: non ha un margine di errore, quindi i margini sull'AUROC vanno letti come
indicazioni e non come distanze stabilite. Il rivelatore di inversione è debole:
esiste solo nel 41,6% dei blocchi — fra le cento righe campionate il blocco
mediano non contiene nessun normale — e come regola di decisione ha precisione
15–18%; funziona non perché sia preciso, ma perché aggiornare raramente
danneggia poco anche quando sbaglia. Le misure di tempo vengono da una macchina
virtuale con due CPU condivise: sulle corse dedicate il rapporto fra massimo e
minimo dello stesso modello è 29 volte per la logistica e per l'MLP e 9 per
l'additivo, e sulle venti corse pubblicate, eseguite mentre girava altro
lavoro, arriva a 178. Valgono come ordine di grandezza e come rapporto fra
modelli, non come latenza in esercizio. D non è indipendente da A, B e C: il
37,23% delle sue righe e il 58,03% dei suoi normali hanno un vettore di feature
già visto.

## Che cosa è bloccato

**Lo stream D è riservato e resta intoccato.** Il protocollo di valutazione è
fissato — D intero come analisi principale, il sottoinsieme a vettore non visto
come analisi supplementare sulle stesse predizioni, nessuna riga tolta dallo
stream, nessuna scelta di modello o politica fatta sui suoi risultati — ma D
non viene eseguito finché non arriva l'autorizzazione. È l'unico blocco che
ferma del lavoro già pronto.

Restano in attesa di una decisione, e non fermano l'esecuzione: quale dei due
confronti sulla soglia riportare nell'articolo, o se affiancarli come le due
aggregazioni dell'FPR (§11.6); se cambiare budget o memoria sui blocchi poveri
di normali, dato che la scheda fissa i parametri (§11.3); e se le 869 righe
fuori contratto vadano contate nel replay o saltate e registrate (§11.4).

## Il prossimo esperimento

**Primo, dare un margine di errore al controllo negativo.** Oggi il sorteggio è
uno solo per modello e per seme, ed è la debolezza del risultato centrale:
ripeterlo con venti o trenta sorteggi indipendenti, a parità di aggiornamenti,
darebbe la distribuzione dei risultati casuali e direbbe se +0,026 cade dentro
o fuori. Costa solo tempo di calcolo su C, non etichette nuove, e non tocca D.

**Secondo, attaccare il punto in cui il metodo fallisce.** Il guadagno vive nei
blocchi ricchi di normali e il rivelatore è cieco proprio dove i normali
mancano: le due osservazioni indicano lo stesso posto. La prova naturale è una
memoria che conservi i campioni per classe invece che in ordine di arrivo, a
parità di dimensione, così che anche i blocchi poveri abbiano qualcosa su cui
misurare il verso. La stratificazione sul punteggio è già stata provata e non
rimedia. Questo però cambia un parametro fissato dalla scheda, quindi non lo si
fa senza una decisione.

Se sono d'accordo, si parte dal primo, che è l'unico che rafforza un risultato
già scritto senza cambiare nessun parametro.

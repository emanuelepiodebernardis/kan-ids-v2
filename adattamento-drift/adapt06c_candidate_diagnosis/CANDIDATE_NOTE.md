# ADAPT-06C — qualità dei candidati

## Perimetro e metodo

Questa è una diagnosi retrospettiva sui dati di sviluppo. Legge i dieci JSON
decoupled già verificati, senza nuovi fit, replay o tuning. La chiave di un
ciclo è dataset/seed/blocco di `FREEZE`/digest. Il confronto usa esclusivamente
l'incumbent operativo congelato all'apertura e le stesse osservazioni VERIFY.
Le condizioni sono ricalcolate con rapporti esatti sui conteggi: guadagno di
recall almeno 2 punti percentuali, aumento FPR non oltre 1 punto e FPR candidato
non oltre 6%.

I 28 `FREEZE` producono 1 `ACCEPT`,
15 `REJECT_HARM` e 12 `WITHHOLD`;
i dieci `WAIT` sono look intermedi. Ogni ciclo è chiuso. La tabella dei look
contiene 26 decisioni complete e 12 chiusure parziali
diagnostiche. Per queste ultime confusioni e supporti sono conservati, ma delta
e condizioni decisionali restano non valutati.

## TON e UNSW

Su TON si osservano 8 rifiuti per
margine dannoso, 1 esito ambiguo e
4 chiusure per supporto insufficiente;
non vi sono promozioni. La prima calibrazione eleggibile viene proposta a fine
blocco 274–275 e applicata dal blocco successivo
275–276. Questa attesa iniziale è
precedente ai cicli e non è tempo trascorso in VERIFY. Dopo ogni `FREEZE`, la
durata riportata parte invece dall'apertura del candidato.

Su UNSW si osservano 7 rifiuti,
5 esiti ambigui,
2 casi insufficienti e l'unico
`ACCEPT`. CAL diventa disponibile molto prima che su TON e consente più
osservazioni VERIFY nel prefisso, ma questo non rende automaticamente migliori
i pesi. I log non separano causalmente qualità del FIT, composizione temporale
di VERIFY e drift dello stream: mostrano l'esito congiunto del protocollo.

Nei look completi, 14 confronti aumentano sia TP sia
FP: sono compromessi empirici, non automaticamente esiti favorevoli. Altri
3 aumentano i TP senza aumentare gli FP,
4 non aumentano i TP ma aumentano gli FP e
5 non aumentano né TP né FP. Questa lettura
per quadrante è distinta dall'esito del gate. Fra i 15 candidati rifiutati, le
violazioni simultanee restano tutte visibili: FPR candidato oltre 6% in
6 look, delta FPR oltre 1 punto in
13 e danno di recall almeno 2 punti percentuali in
4. Questi conteggi non sono cause indipendenti:
più condizioni possono fallire nello stesso look.

## Il caso accettato

L'unico candidato promosso è UNSW/seed42, congelato al blocco 18 e deciso al
blocco 37 (indici 0-based). Su 350
normali e 100 attacchi registra 94 TP
e 8 FP, contro 89 TP e
6 FP dell'incumbent: delta recall
+5.000 punti, delta FPR
+0.571 e FPR candidato
2.286%. Il guadagno di recall supera il
margine; entrambi i vincoli FPR restano entro i limiti. È un'accettazione
locale su VERIFY, non la prestazione controfattuale dell'intero stream.

## Ipotesi e decisione operativa

**Ipotesi 1 — preferita.** Il limite dominante è la qualità instabile dei
candidati prodotti dal FIT192: molti look attraversano almeno un margine di
danno e la sola maggiore disponibilità di VERIFY non garantisce una
promozione utile. I log sostengono la diagnosi del gate, ma non distinguono se
l'instabilità provenga da composizione, età o capacità di FIT.

**Ipotesi 2.** In una parte dei cicli la potenza di verifica è il limite:
supporto insufficiente o chiusure ambigue impediscono una decisione anche
senza un danno già dimostrato. I log non consentono di sapere se ulteriori
etichette avrebbero confermato o smentito quei candidati.

Propongo un solo esperimento minimo, subordinato a un protocollo separato:
**confronto shadow FIT192/FIT256 nei tre `FREEZE` già osservati su UNSW/seed42**.
Si riusano richieste, label disponibili, blocchi di congelamento, incumbent e
osservazioni VERIFY; i candidati shadow non vengono promossi e non cambiano la
traiettoria. La metrica primaria è il numero di candidati che superano
congiuntamente i tre criteri sul medesimo look; controllo FIT192 registrato,
secondarie confusioni appaiate e costo. Prima dei risultati va fissato GO
soltanto se FIT256 aggiunge almeno un candidato che passa tutti i margini e non
introduce nuovi attraversamenti di danno nei tre confronti. Servono tre fit
shadow e scoring sugli intervalli già fissati, non due replay completi: i tre
fit/calibrazioni candidati originali hanno richiesto complessivamente
0.130 s e l'intero replay decoupled 46.2
s. Con caricamento, ricostruzione, scoring e controlli si stima quindi un wall
time conservativo di 1–2 minuti sullo stesso dispositivo. Un esito nullo
smentirebbe l'ipotesi che la sola capacità FIT sia sufficiente. Non è ancora
giustificata una griglia di tuning né ADAPT-07.

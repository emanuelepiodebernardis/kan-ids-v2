# ADAPT-06B — soglia operativa autonoma e pesi adattivi

## 1. Domanda e perimetro

ADAPT-06 ha mostrato che il gate può evitare la promozione di pesi dannosi ma,
allo stesso tempo, lasciare operativo un valore di soglia iniziale non più
adeguato. ADAPT-06B verifica una sola variazione metodologica: la soglia del
modello realmente operativo viene mantenuta causalmente anche quando i pesi di
un candidato non vengono accettati. La domanda primaria non è se la nuova
variante superi il vecchio GATED, ma se i pesi adattivi aggiungano un beneficio
rispetto a una politica threshold-only adeguata.

Il checkpoint A è limitato alla testa GAM, acquisizione RANDOM, seed 42 e
stream TON/UNSW. Protocollo, codice e test vengono fissati prima dei dati reali.
Il pilot completo sarà eseguito soltanto dopo il riesame. ACTIVE, LR, ARF,
cinque seed, quote diverse, nuovi dataset e ADAPT-07 restano fuori perimetro.

## 2. Componenti invariati

Restano invariati rispetto ad ADAPT-06: FIT192 con quote 96/96 e minimo 32 per
classe; CAL256 con quote 128/128, età massima 50 blocchi e minimo 100/100;
budget complessivo delle label pari all'1% per blocco, ripartito 25% CONTROL e
75% RANDOM_FIT; ritardo di un blocco; look VERIFY 100/100, 200/200 e 400/400;
scadenza a 50 blocchi; target FPR del 5%; frontiere di accettazione e danno
esistenti. Non sono previsti sweep o tuning.

Le richieste RANDOM sono identiche fra i bracci. Ogni `observation_id` riceve
un solo ruolo al momento della richiesta. FIT, CAL e VERIFY devono essere
disgiunti durante un ciclo attivo. Dopo la chiusura, i CONTROL trattenuti
possono entrare causalmente in CAL senza una nuova richiesta di label; quelli
oltre l'età massima vengono espulsi.

## 3. Ordine causale degli eventi

Nel blocco `k` si predice usando esclusivamente modello e soglia già operativi
all'apertura del blocco. Si selezionano quindi CONTROL e RANDOM_FIT e si
registrano gli hash delle richieste. Le label del blocco `k−1` arrivano soltanto
dopo le predizioni del blocco `k`. Una calibrazione calcolata a fine blocco `k`
diventa efficace al blocco `k+1`: di conseguenza il feedback osservato in `k−1`
non può influire prima della predizione `k+1`, cioè due predizioni dopo il
blocco di osservazione. Equivalentemente, il feedback osservato nel blocco `k`
e ricevuto dopo la predizione `k+1` può influire al più presto sulla predizione
`k+2`.

Quando nessun candidato è aperto e CAL è eleggibile, il modello operativo e il
modello congelato del controllo abbinato vengono valutati sulla medesima CAL.
Le due soglie proposte sono registrate con supporti, digest degli ID e blocco di
efficacia. La soglia operativa può essere aggiornata anche se FIT non è ancora
eleggibile. Se anche FIT è eleggibile, il FREEZE avviene dopo che la soglia
proposta è diventata operativa, non nello stesso blocco in cui è stata stimata.

Al FREEZE si fissano atomicamente quattro elementi: pesi e soglia del candidato,
pesi e soglia dell'incumbent operativo. Durante VERIFY queste due coppie non
cambiano. I CONTROL osservati dopo il FREEZE alimentano soltanto VERIFY; non
possono calibrare le coppie che stanno confrontando. Gli arrivi precedenti al
FREEZE sono differiti. La manutenzione della soglia resta sospesa per tutto il
ciclo.

Con ACCEPT viene promossa la coppia candidato validata, efficace dalle
predizioni successive. Con REJECT o withholding restano i pesi incumbent e la
soglia operativa già ricalibrata: non si torna alla soglia B. Dopo la chiusura,
i CONTROL trattenuti vengono trasferiti a CAL in ordine causale e la
manutenzione riprende dal blocco successivo. WAIT, supporto insufficiente,
ambiguità, scadenza e fine stream seguono le regole terminali di ADAPT-06.

## 4. Cinque controlli

1. **Frozen/B**: pesi A e soglia B fissi.
2. **Threshold-only autonoma**: pesi A, CAL256 CONTROL indipendente e
   ricalibrazione a ogni blocco eleggibile. È il riferimento principale di
   efficacia.
3. **GATED legacy**: percorso ADAPT-06 invariato, usato per quantificare la
   variazione metodologica.
4. **GATED decoupled**: nuova variante; la soglia operativa è mantenuta fuori
   dai cicli VERIFY e la promozione dei pesi resta soggetta al gate.
5. **Threshold-only matched**: pesi A e soglia ricalibrata usando gli stessi
   eventi e la stessa CAL della variante. È un controllo diagnostico dipendente
   dal calendario della variante, non una soluzione autonoma.

I due report legacy/variante devono avere gli stessi hash delle richieste e la
stessa sequenza `(observation_id, ruolo, osservazione, arrivo, primo impatto)`.
Il controllo abbinato condivide CAL con la variante e aggiunge soltanto lo stato
della soglia; la baseline autonoma conserva invece la propria CAL256. Memoria e
tempo di entrambi vengono riportati, senza descrivere buffer o calcolo come
gratuiti.

## 5. Log e criteri del pilot

Per blocco si registrano modello e soglia usati, confusioni, supporti e digest
FIT/CAL/VERIFY, proposte e applicazioni della soglia, blocco di efficacia,
FREEZE, look e chiusura, promozioni dei pesi e variazioni della sola soglia. Il
report conserva hash delle richieste, label uniche, memoria numerica, picco RSS,
tempo di predizione, fit e calibrazione.

Il futuro pilot usa gli stream completi TON e UNSW, GAM/RANDOM, seed 42 e stati
A/B archiviati. Non usa `--max-blocks` e non esegue un nuovo fit iniziale. Per
dataset produce un report legacy, uno decoupled e una verifica appaiata. La
tabella prevista riporta recall, FPR, differenze dalla threshold-only autonoma,
promozioni, variazioni della soglia, primo blocco CAL eleggibile, withholding,
label, tempi e memoria, insieme a intervalli prima/dopo l'eleggibilità.

Il passaggio tecnico successivo richiede: provenienza PASS, richieste e budget
identici, nessuna violazione causale o di ruolo, report completi e nessun errore
terminale. Il singolo seed non può stabilire superiorità scientifica. Qualsiasi
decisione su cinque seed, ACTIVE o ADAPT-07 richiede un protocollo successivo e
non può selezionare soltanto risultati favorevoli.


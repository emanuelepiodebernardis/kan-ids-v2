# ADAPT-04: calibrazione recente separata 192/64

## Esito e perimetro

Il replay C/seed42 usa il commit `b9fd9fbd76d5d01a1ff5674b8f5a67cf5b266b3b` e contiene i blocchi
`0..915`. La validazione e' **PASS**: fit192 e calibration64
restano separati, il totale non supera 256 esempi, le etichette del blocco j
agiscono da j+2 e le politiche 2/3 condividono esattamente pesi e score. B non
e' stato riletto: fornisce soltanto la soglia iniziale gia' verificata.

Sono stati applicati 475 aggiornamenti atomici e
saltati 440; cause: `{"cal_normali_recenti_insufficienti": 440}`.

## Risultati globali

| politica | recall compl. | FPR compl. | BA compl. | AUROC validi | AUROC media blocchi | FN / FP |
|---|---:|---:|---:|---:|---:|---:|
| frozen + CAL64 | 84,670% | 29,444% | 77,613% | 875/916 | 0,884232 | 1.376.198 / 51.032 |
| FIT192 + B | 47,741% | 1,010% | 73,365% | 875/916 | 0,774927 | 4.691.521 / 1.750 |
| FIT192 + CAL64 | 82,071% | 13,364% | 84,354% | 875/916 | 0,774927 | 1.609.533 / 23.162 |

Le metriche globali derivano dalle confusion matrix aggregate. Le medie per
blocco e le dispersioni sono nel JSON/CSV; i valori non definiti nei blocchi
monoclasse restano nulli, non sono sostituiti con zero.

Nel confronto principale, FIT192+CAL64 meno frozen+CAL64, i delta **globali**
sono: recall -2,599 punti percentuali, FPR
-16,080 punti percentuali, BA
+6,741 punti percentuali,
FN +233.335 e FP
-27.870. L'AUROC
media per blocco cambia di
-0,109305.

Le differenze **medie non pesate per blocco** sono invece: recall
-0,026088
(dev. std. 0,095017),
FPR 0,047798 (dev. std.
0,418227) e BA
-0,036441 (dev. std.
0,211707). La variante
ha recall peggiore in 595 blocchi, FPR
peggiore in 458 e BA peggiore in
536; le sequenze contigue massime
durano rispettivamente 433,
32 e
174 blocchi.

A pesi identici, i delta globali CAL64 meno B sono recall
+34,331 punti percentuali, FPR
+12,354 punti percentuali e BA
+10,988 punti percentuali; le corrispondenti medie non
pesate per blocco sono 0,348192,
0,126428 e
0,115914.

L'adattamento dei pesi non mostra quindi una superiorita' generale sulla sola
soglia con gli stessi dati recenti: riduce fortemente il FPR globale e aumenta
la BA globale, ma perde recall e AUROC. Recall, falsi allarmi e continuita'
temporale vanno letti insieme; la sola BA non basta.

## Memoria, ruoli e costi

L'hash ha assegnato 69158 richieste al fit e
22348 alla calibrazione, pari rispettivamente a
44966 e 15094
vettori unici. Il massimo payload logico dei due buffer e'
20.736 byte: 8.448
per feature trasformate ed etichette e 12.288 per blocco
di osservazione, ordine e digest. La coda pendente arriva a
8.216 byte per 200 indici,
ruoli, digest e due chiavi di blocco: riferisce le righe di C gia' caricate e
non ne duplica feature o etichette. I 168
byte dello stato numerico contano le due teste di nove `float64` e tre soglie,
non l'intera KAN congelata o il preprocessing. La somma conservativa dei tre
massimi separati e' 29.120 byte; non e' un picco
simultaneo misurato ed esclude l'overhead degli oggetti Python. L'RSS del banco
di prova, 1.245,4 MiB sia prima sia
dopo il ciclo, include runtime, input A/C, modello, log e temporanei e non e'
memoria ammessa al controllore.

I 440 skip dipendono tutti da normali recenti
insufficienti. L'intervallo contiguo massimo va dal blocco
241 al 663
(423 blocchi): il criterio preregistrato e' rimasto
invariato anche durante questa indisponibilita'.

Il tempo dell'intero comando, misurato da prima degli hash e del caricamento
fino alla costruzione del rendiconto ma prima della serializzazione finale, e'
190,088 s. Comprende
35,162 s di addestramento iniziale KAN
su A, 147,160 s per il ciclo sui 916 blocchi e
7,766 s per hash, caricamento, preparazione e riepilogo.
Nel ciclo, i timer di componente registrano 17,731 s
per i 475 fit, 0,156 s e
0,092 s per le calibrazioni e
0,274 s
per le due inferenze. Sono sottoinsiemi del ciclo e non includono
rappresentazioni, gestione delle memorie, campionamento, controlli o log.

## Riferimenti ADAPT-03 e limiti

Questi quattro riferimenti sono soltanto riassunti dai rendiconti gia'
verificati; non sono stati rieseguiti e hanno memoria, inizializzazione o
calendario diversi.

| riferimento | recall compl. | FPR compl. | AUROC validi |
|---|---:|---:|---:|
| frozen + B | 64,885% | 13,254% | 875/916 |
| sola soglia, memoria 256 | 82,612% | 14,731% | 875/916 |
| pesi adattati + B | 20,608% | 0,474% | 875/916 |
| pesi adattati + soglia su B | 81,082% | 30,141% | 875/916 |

Il risultato riguarda un solo replay C/seed42. Non sono stati eseguiti cinque
semi, ricerca di iperparametri o accessi a D. Le appendici JSON/CSV conservano
dispersioni, periodi completi, soglie, composizione ed eta' della memoria,
skip, costi e digest di appaiamento.

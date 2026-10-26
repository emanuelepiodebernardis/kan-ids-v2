# Sintesi ADAPT-01--04

## Perimetro

Le quattro fasi studiano componenti diverse dell'adattamento temporale su C.
Le righe della tabella sono confronti appaiati entro il gruppo indicato, ma i
gruppi non vanno trattati come un'unica graduatoria: cambiano memoria, soglia,
calendario, disponibilita' dei dati e, in ADAPT-03/04, numero di semi. Tutte le
misure derivano dai rendiconti gia' verificati; D non e' stato usato.

ADAPT-01 confronta congelato, aggiornamento a ogni blocco, politica evidence e
calendario casuale con uguale numero di aggiornamenti sui semi 42--46. Per LR e
MLP l'adattamento riduce il FPR complessivo e aumenta il recall rispetto al
congelato. Per KAN ogni-blocco aumenta il recall ma riduce l'AUROC; evidence
riduce il FPR senza un guadagno sostanziale di recall. Il confronto
evidence-casuale mostra che il momento dell'aggiornamento conta, ma non autorizza
una superiorita' generale della politica.

ADAPT-02 sostituisce FIFO con una memoria per classe a spazio ed etichette
invariati. Su cinque semi la nuova memoria migliora l'AUROC e riduce FP/FPR, ma
aumenta nettamente FN e riduce il recall per tutti i modelli e per entrambe le
politiche. Conservare entrambe le classi non basta dunque a migliorare il
comportamento operativo alla soglia fissata.

ADAPT-03 separa soglia e pesi. Con score congelati, la soglia adattiva cambia
recall e FPR lasciando invariata l'AUROC. Nel caso KAN/per-classe il recall sale
dal 64,89% all'82,61%, mentre il FPR sale dal 13,25% al 14,73%. Quando la soglia
su B viene ricalibrata lungo un'unica traiettoria di pesi, il recall cresce
molto ma anche il FPR: recuperare attacchi non equivale da solo a migliorare il
sistema.

ADAPT-04 usa dati recenti separati, 192 esempi per il fit e 64 per la
calibrazione. A pesi identici, CAL64 aumenta recall e FPR rispetto alla soglia
su B. Nel confronto principale FIT192+CAL64 contro frozen+CAL64, il recall
scende di 2,60 punti percentuali e il FPR di 16,08 punti; la balanced accuracy
globale sale di 6,74 punti, ma l'AUROC media per blocco scende di 0,1093. Anche
questo e' un compromesso, non una prova di superiorita'.

## Evidenza temporale

L'analisi dei periodi associa ogni decisione agli errori del blocco successivo.
La riduzione dei falsi allarmi persiste spesso sia dopo update sia negli skip;
l'aumento degli attacchi mancati della memoria per classe e del disegno 192/64
non e' confinato a una sola delle due condizioni. Gli strati non sono
randomizzati e non misurano l'effetto causale dell'update: servono a localizzare
il compromesso e a evitare che gli aggregati nascondano periodi critici.

## Conclusione

L'adattamento puo' produrre un beneficio concreto, ma pesi, soglia, memoria e
calendario spostano in modo diverso FP e FN. La conclusione sostenuta dai dati
e' quindi condizionale: occorre valutare insieme recall, FPR, AUROC, balanced
accuracy, andamento temporale, disponibilita' degli update e costo. La tabella
`tabella_evidenze.csv` conserva i gruppi entro cui i confronti sono diretti;
confronti fra gruppi restano descrittivi.

# ADAPT-03: protocollo del replay completo pesi adattati + soglia

## Provenienza e perimetro

Il replay completo e' autorizzato dopo la verifica dello smoke e parte dal
merge di ADAPT-03 in `main`, commit
`9550decf06156664839affd8db5566fe2fbc8bd0`, che incorpora la versione
approvata `d11c59c`. Il branch separato conserva senza riscriverli i commit e
gli artefatti dello smoke.

Si esegue una sola corsa completa sui 916 blocchi cronologici di C, seed 42.
Restano fuori dal protocollo una campagna su cinque semi, la memoria 192/64 e
qualunque uso di D.

## Configurazione invariata

- inizializzazione su A e calibrazione iniziale sull'intero B;
- sola KAN, memoria per classe di capacita' totale 256, quote 128/128;
- politica di aggiornamento dei pesi `ogni_blocco`;
- blocchi da 10.000 righe, budget uniforme dell'1% e ritardo di un blocco;
- trasformazione, modello, fit della testa, campionamento e calendario
  invariati rispetto ad ADAPT-02;
- un'unica traiettoria dei pesi e un unico vettore di score per blocco;
- controllo con la soglia iniziale scelta su B e poi mantenuta fissa;
- variante con la stessa soglia iniziale, ricalibrata sull'intero B dopo ogni
  aggiornamento effettivamente applicato ai pesi;
- regola BA e tie-break identici allo smoke;
- ogni nuova soglia e' attiva soltanto dal blocco successivo.

B resta un archivio offline aggiuntivo rispetto ai 256 esempi della memoria
online. Etichette, score o altre informazioni di C non intervengono mai nella
scelta della soglia.

## Completezza e appaiamento

Il runner deve produrre esattamente i blocchi `0..915`, senza omissioni,
duplicati o riordinamenti. Prima del confronto richiede che anche il
riferimento ADAPT-02 contenga tutti i 916 blocchi. Per ciascun blocco devono
coincidere campioni richiesti, digest dei `row_id`, traiettoria KAN e misure del
controllo a soglia B fissa. I due bracci devono condividere parametri, score,
richieste, memoria, aggiornamenti e AUROC.

## Metriche non definite

Le metriche per blocco restano `null` quando manca il loro denominatore. Non
vengono mai sostituite con zero. Nel riferimento completo sono presenti 916
blocchi con attacchi, 875 con normali e 875 con entrambe le classi; i 41
restanti contengono soltanto attacchi. I denominatori attesi sono quindi:

- recall degli attacchi: 916/916;
- FPR: 875/916;
- balanced accuracy: 875/916;
- AUROC: 875/916.

Le metriche globali derivano dalle confusion matrix aggregate. Medie per
blocco, dispersioni e confronti appaiati escludono soltanto i valori non
definiti e riportano sempre il numero di blocchi validi sul totale. Nel CSV i
valori non definiti restano celle vuote; nelle figure producono interruzioni,
non zeri.

## Memoria e costi

Il valore di 24.904 byte osservato nello smoke non rappresenta tutto lo stato
online. E' la somma di alcuni array impiegati nel fit: feature ed etichette
della memoria, rappresentazione KAN della memoria e parametri correnti della
testa KAN. Il replay lo registra come `array_fit_online_massimo`, con totale e
componenti. Sono esclusi oggetti del modello e della trasformazione, richieste
in attesa, buffer del blocco, overhead dell'interprete e archivio/cache B.

Si conservano separatamente dimensioni di B e della sua cache, picco RSS,
tempo di costruzione della cache, tempi di fit, tempi di ogni ricalibrazione e
tempo effettivo totale.

## Esecuzione e consegna

Codice, protocollo e controprove vengono committati prima della corsa. Il
runner richiede un worktree pulito e registra lo SHA effettivamente eseguito,
gli hash di A, B e C, l'ambiente e il comando completo. L'uscita usa una nuova
directory e non riutilizza JSON precedenti.

La consegna comprende il rendiconto originale, un confronto per blocco, un
riepilogo leggero, il rapporto, le figure temporali e un manifest con SHA-256.
Il rendiconto completo viene conservato in un archivio esterno compresso; in
Git restano soltanto i materiali leggeri riproducibili.

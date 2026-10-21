# ADAPT-03: rapporto dello smoke pesi adattati + soglia

## Esito tecnico

Lo smoke usa il commit `f9efbe4b1312a82805a532dadd45b8eb11778c1b` e contiene esattamente
i blocchi `0..19` di C/seed42. La validazione e' **PASS**: richieste, traiettoria
KAN e controllo a soglia B coincidono con ADAPT-02; i due bracci condividono
parametri, score e AUROC. Sono stati applicati 19
aggiornamenti dei pesi e altrettante ricalibrazioni, sempre attive dal blocco
successivo.

## Risultati dei 20 blocchi

| metodo | recall compl. | FPR compl. | balanced accuracy | AUROC media |
|---|---:|---:|---:|---:|
| pesi + soglia B fissa | 86,142% | 0,032% | 93,055% | 0,987694 |
| stessi pesi + soglia ricalibrata | 97,897% | 2,024% | 97,937% | 0,987694 |

Questi valori sono diagnostici. Non definiscono una superiorita' e un aumento
del solo recall non e' un criterio sufficiente.

La variante ha recall maggiore in 17/20 blocchi,
FPR maggiore in 14/20 e balanced accuracy maggiore
in 13/20; in
4/20 la balanced accuracy e' minore e
in 3/20 le decisioni coincidono. Il CSV e la figura
conservano l'andamento di ogni blocco e la soglia effettivamente applicata.

## Costo e memoria

B contiene 5.681.198 righe ed e' un archivio aggiuntivo rispetto ai
256 esempi online. `X_B` e `y_B` occupano rispettivamente
181.798.336 e 5.681.198 byte; la cache `float64`
della rappresentazione KAN occupa
363.596.672 byte. Il valore massimo di
24.904 byte descrive soltanto gli array del fit:
feature ed etichette della memoria, rappresentazione KAN della memoria e parametri correnti della testa KAN. Esclude oggetti del modello e della trasformazione, richieste in attesa, buffer del blocco, overhead Python e archivio/cache B; non rappresenta quindi
l'intero stato online. Il picco RSS finale del banco di prova e'
2103,3 MiB,
misurato con
`GetProcessMemoryInfo`; comprende A, B, C e gli array temporanei.

Il rendiconto originale e' conservato fuori Git in `smoke_C_seed42_f9efbe4.zip`
(11.267 byte), SHA-256 `acdeb051288f4b4e8774358020af6f6f1cda90204471894bd0962291596f14c1`. Il manifest
versionato registra anche dimensione e SHA-256 del JSON interno. L'archivio e' scaricabile da `https://drive.google.com/file/d/1RU2QNkTEDIEDJRQW3vJz82kzPlHQn3Ab/view?usp=drive_link`.

Le 19 ricalibrazioni costano in mediana
0,706 s, al 90o percentile
0,773 s e al massimo 0,794 s. In
mediana, il calcolo degli score di B costa 0,039 s e la
selezione BA 0,668 s.
La cache viene costruita una volta in
64,359 s.

## Stima del replay completo

La proiezione usa 916 blocchi e 915 aggiornamenti, come il rendiconto completo
ADAPT-02 KAN/per-classe/ogni-blocco. La stima centrale e'
13,2 minuti; usando il 90o percentile
e' 14,2 minuti; moltiplicando per il massimo
osservato e' 14,5 minuti. Sono
proiezioni dello stesso
dispositivo, non tempi MCU.

Il replay completo non e' stato avviato. D non e' stato letto, non e' stata
eseguita una campagna su cinque semi e la variante 192/64 resta fuori dal
protocollo.

## Verifica

Passano 320 test nella cartella `ton_temporal/`. Dalla radice del repository
passano 954 test, uno e' saltato e passano 5 sottotest. I controlli AVR sono
stati eseguiti fuori dal sandbox Windows dopo che il sandbox aveva restituito
l'errore di rilocazione DLL 623; la seconda esecuzione diretta passa.

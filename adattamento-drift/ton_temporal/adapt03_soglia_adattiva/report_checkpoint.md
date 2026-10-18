# ADAPT-03: rapporto del primo checkpoint

## Provenienza

Il protocollo e' stato fissato nel commit `852f91c`. Il replay diagnostico e'
stato eseguito dal commit pulito
`90597f03413fa5884c6ad7e2895140695e3e14fe`; l'analisi e' stata eseguita dallo
stesso commit pulito, dopo il PASS dello smoke rafforzato.

Sono stati eseguiti due rendiconti completi su C, seed 42: uno FIFO e uno con
memoria per classe. Entrambi usano 256 esempi, campionamento uniforme all'1%,
ritardo di un blocco e soglie iniziali calibrate su B. D non e' stato letto.

I rendiconti completi e lo smoke sono conservati fuori Git nell'archivio
`checkpoint_C_seed42_90597f0.zip`, 1.088.717 byte, SHA-256
`88d5c4df0e637e6c0c0b61f4bae5ceb9e0c3ee99d21f3b54b3df03a77f4add91`.
Il file `manifest_checkpoint.json` registra dimensione e impronta dei cinque
JSON contenuti, oltre al commit effettivamente eseguito.

Lo smoke ha dato PASS su esattamente i blocchi `0..19` per entrambe le memorie.
Il verificatore rifiuta SHA-256 o parametri preregistrati diversi e sequenze
vuote, incomplete, duplicate o fuori ordine. La verifica completa ha poi dato
PASS su 1.832 coppie di blocchi contro i rendiconti ADAPT-02: parametri, soglie
B, richieste, digest dei `row_id` e misure frozen coincidono. Gli score frozen
coincidono inoltre fra le due nuove corse. L'AUROC della sola soglia e' identica
al frozen in ogni blocco.

I nuovi rendiconti registrano gli SHA-256 di A, B e C, coincidenti con quelli
documentati per l'ambiente ADAPT-02. I singoli JSON ADAPT-02 non ripetono pero'
queste impronte: il validatore appaia direttamente nomi degli input, parametri,
soglie, richieste, `row_id` e misure frozen. L'ambiente registrato
nelle due famiglie di report coincide ed e' documentato separatamente. Questa
distinzione evita di attribuire al confronto storico un controllo di hash che
quei JSON non consentono di eseguire da soli.

La verifica finale conta 299 test superati nella cartella `ton_temporal/` e,
dalla radice del repository, 933 test superati, uno saltato e 5 sottotest
superati. Le 20 controprove aggiunte coprono gli errori del validatore; i 7 test
dell'analisi controllano anche gli artefatti consegnati.

## Risultati globali

AUROC e' la media sui 875 blocchi validi dei 916 totali. Recall e FPR sono tassi
complessivi, ottenuti sommando prima le confusion matrix. `Pesi` indica la testa
ADAPT-02 aggiornata a ogni blocco con la stessa memoria e seed.

| memoria | modello | metodo | recall | FPR compl. | AUROC | FN | FP |
|---|---|---|---:|---:|---:|---:|---:|
| FIFO | LR | frozen B | 0,7213 | 48,722% | 0,7149 | 2.501.546 | 84.443 |
| FIFO | LR | sola soglia | 0,8172 | 39,270% | 0,7149 | 1.640.819 | 68.061 |
| FIFO | LR | pesi | 0,8748 | 19,817% | 0,7807 | 1.123.998 | 34.346 |
| FIFO | MLP | frozen B | 0,4050 | 29,118% | 0,8028 | 5.341.319 | 50.467 |
| FIFO | MLP | sola soglia | 0,9707 | 46,478% | 0,8028 | 262.674 | 80.554 |
| FIFO | MLP | pesi | 0,9215 | 18,911% | 0,8386 | 704.311 | 32.776 |
| FIFO | KAN | frozen B | 0,6489 | 13,254% | 0,8842 | 3.152.390 | 22.971 |
| FIFO | KAN | sola soglia | 0,7820 | 19,869% | 0,8842 | 1.956.645 | 34.437 |
| FIFO | KAN | pesi | 0,7711 | 13,327% | 0,7966 | 2.055.069 | 23.098 |
| per classe | LR | frozen B | 0,7213 | 48,722% | 0,7149 | 2.501.546 | 84.443 |
| per classe | LR | sola soglia | 0,8368 | 34,296% | 0,7149 | 1.464.837 | 59.440 |
| per classe | LR | pesi | 0,2208 | 2,246% | 0,8317 | 6.994.940 | 3.893 |
| per classe | MLP | frozen B | 0,4050 | 29,118% | 0,8028 | 5.341.319 | 50.467 |
| per classe | MLP | sola soglia | 0,9587 | 40,902% | 0,8028 | 370.886 | 70.889 |
| per classe | MLP | pesi | 0,3857 | 1,960% | 0,9369 | 5.514.425 | 3.397 |
| per classe | KAN | frozen B | 0,6489 | 13,254% | 0,8842 | 3.152.390 | 22.971 |
| per classe | KAN | sola soglia | 0,8261 | 14,731% | 0,8842 | 1.560.971 | 25.532 |
| per classe | KAN | pesi | 0,2061 | 0,474% | 0,8890 | 7.127.271 | 822 |

## Disponibilita' e memoria

Su 915 occasioni in cui erano arrivate nuove etichette, la FIFO ha ammesso 253
aggiornamenti di soglia e ne ha rifiutati 662 per evidenza insufficiente. La
memoria per classe ne ha ammessi 704 e rifiutati 211. I conteggi sono uguali per
i tre modelli, perche' disponibilita', minimi ed eta' dipendono dalla memoria e
non dagli score.

| memoria | normali medi | attacchi medi | eta mediana normali, media | eta mediana attacchi, media |
|---|---:|---:|---:|---:|
| FIFO | 4,72 | 251,28 | 2,82 | 2,99 |
| per classe | 125,38 | 130,62 | 142,81 | 2,02 |

La FIFO contiene in media meno dei cinque normali preregistrati. La memoria per
classe conserva molti normali, ma una parte rilevante supera il limite di eta'
di 50 blocchi: il bilanciamento aumenta la disponibilita' senza eliminarne il
problema.

L'eta' decorre dal blocco di osservazione della richiesta. La memoria descritta
nella tabella contiene soltanto esempi la cui etichetta e' gia' arrivata. Le
richieste ancora in attesa sono registrate separatamente, con blocco di
osservazione, numero di esempi e byte di score e indici; non consumano la
capacita' 256 della memoria etichettata.

## Non uniformita' temporale

Il miglioramento globale non vale in ogni tratto. Rispetto al frozen, la sola
soglia ha recall peggiore in 196/7/373 blocchi FIFO per LR/MLP/KAN e in
341/1/97 blocchi per-classe. Il FPR e' peggiore in 185/822/360 blocchi FIFO e
224/616/349 blocchi per-classe. Le sequenze peggiori arrivano a 193 blocchi
consecutivi per il recall e 194 per il FPR, secondo modello e memoria.

Gli intervalli completi con timestamp sono in `analisi_checkpoint.json`; il CSV
conserva le metriche e i delta per ogni blocco, modello e memoria.

## Costi

Lo stato logico aggiuntivo massimo e' 11.816 byte per entrambe le memorie:
8.616 byte di score, etichette, eta', soglie e contatori della memoria
etichettata, piu' 2.400 byte di score e 800 byte di indici delle richieste in
attesa. Non include l'overhead degli oggetti Python e non e' una misura MCU.

L'ottimizzazione di una soglia costa in media al massimo 0,23 ms: per modello
il totale resta entro 0,045 s con FIFO e 0,159 s con memoria per classe. Il
replay successivo all'inizializzazione ha richiesto 56,43 s con FIFO e 67,65 s
con memoria per classe;
i rendiconti ADAPT-02 dei pesi riportano 316,79 s per FIFO e 219,14 s per
per-classe. I tempi sono misure del banco di prova sul dispositivo, non una
stima di latenza MCU.

## Lettura e decisione

La soglia adattiva spiega una parte importante del comportamento, ma non
sostituisce in modo generale il riaddestramento.

- Con FIFO/LR, i pesi ottengono insieme recall maggiore, FPR minore e AUROC
  maggiore della sola soglia.
- Con FIFO/MLP, la soglia massimizza il recall ma porta il FPR al 46,5%; i pesi
  mantengono recall elevato con FPR molto inferiore.
- Con FIFO/KAN, la soglia conserva l'AUROC frozen e supera di poco il recall dei
  pesi, pagando circa 6,5 punti percentuali di FPR.
- Con memoria per classe, la soglia evita il crollo del recall osservato con i
  pesi, ma rinuncia quasi interamente alla loro riduzione dei falsi allarmi.

Non emerge un controllo dominante. Il checkpoint e' un **no-go** per una
campagna a cinque semi della sola soglia come sostituto generale. Il passo
piu' diretto e' applicare lo stesso controllo causale di soglia agli score dei
pesi adattati, a memoria e calendario identici. Il forte calo di recall dei pesi
per-classe alla soglia B e' compatibile con un disallineamento di calibrazione,
ma questo checkpoint non ne isola ancora la causa e non dimostra che basti
ricalibrare la soglia.

C resta flusso di sviluppo. D resta riservato fino al congelamento di un
protocollo successivo.

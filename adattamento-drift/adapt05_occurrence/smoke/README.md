# Smoke prefissato ADAPT-05

Questo smoke e' un controllo tecnico preregistrato, non un risultato
scientifico. Copre i primi 30 blocchi di `D` per TON e UNSW, con seed 42 e
sale `default`, dopo il PASS dei test sintetici.

## Provenienza

- merge di PR #10 usato come base: `7e930d51d24bb81e3d46ac41664db95ce0956093`;
- commit eseguito: `b435c4c6badec92f56cca4880105446e110cb475`;
- esecuzione Linux: <https://github.com/emanuelepiodebernardis/kan-ids-v2/actions/runs/41528836967>;
- ambiente numerico: Python 3.12.14, NumPy 2.3.5, un thread BLAS;
- suite eseguita prima dello smoke: 112 test, tutti superati (16 ADAPT-05 e
  96 test scientifici del pacchetto Paper 2);
- gli array ricostruiti di entrambi i dataset hanno superato il confronto
  SHA-256 per ogni campo nominato.

Il workflow completo, inclusi installazione, test, download, ricostruzione,
smoke e caricamento degli artefatti, ha richiesto 4 min 14 s.

## Esito

| Dataset | Blocchi | Righe | Richieste | FIT/CAL vettoriale | FIT/CAL per osservazione | Aggiornamenti principali vettoriale/per osservazione | Tempo replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| TON | 30 | 300.000 | 3.000 | 2.258 / 742 | 2.223 / 777 | 20 / 17 | 1,543 s |
| UNSW | 30 | 300.000 | 3.000 | 2.324 / 676 | 2.239 / 761 | 28 / 29 | 1,782 s |

Entrambi i rendiconti contengono esattamente 30 blocchi ordinati. Il modo
vettoriale coincide con la traiettoria archiviata: digest dei parametri della
testa 30/30, soglie frozen 150/150, soglie head 150/150, matrici di
confusione 300/300, hash delle richieste 30/30, conteggi ALL256 58/58, stati
di aggiornamento 29/29 e controlli CAL64 su conteggi ed eta' 116/116. La
massima differenza assoluta fra soglie e' zero per entrambi i dataset.

I 100 esempi richiesti nell'ultimo blocco restano correttamente pendenti:
saranno disponibili dopo la fine dello smoke e non possono influenzarne le
predizioni.

## Stima della matrice completa

Il flusso TON comprende 355 blocchi e UNSW 102. Scalando i soli tempi di
replay misurati sulle 15 coppie seed/sale per dataset si ottengono circa
364,8 s (6,1 min). Per pianificare includiamo avvio dei 30 processi, controlli,
scrittura dei report e preparazione comune: 10--15 min su un runner Linux
equivalente. Analisi, figure, manifest e verifica finale richiedono fino a
una giornata lavorativa dopo l'approvazione del checkpoint.

La matrice completa non e' stata avviata in questa consegna: protocollo,
codice e smoke vengono prima sottoposti alla verifica prevista.


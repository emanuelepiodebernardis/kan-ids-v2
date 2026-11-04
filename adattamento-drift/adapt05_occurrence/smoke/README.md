# Smoke prefissato ADAPT-05

Questo smoke e' un controllo tecnico preregistrato, non un risultato
scientifico. Copre i primi 30 blocchi di `D` per TON e UNSW, con seed 42 e
sale `default`, dopo il PASS dei test sintetici.

## Provenienza

- merge di PR #10 usato come base: `7e930d51d24bb81e3d46ac41664db95ce0956093`;
- commit eseguito: `e6c939c6860f39d63c6677de85c1626c61e4e668`;
- esecuzione Linux: <https://github.com/emanuelepiodebernardis/kan-ids-v2/actions/runs/41643036662>;
- ambiente numerico: Python 3.12.14, NumPy 2.3.5, un thread BLAS;
- suite eseguita prima dello smoke: 119 test, tutti superati (23 ADAPT-05 e
  96 test scientifici del pacchetto Paper 2);
- la stessa suite di 119 test e' stata superata anche localmente su Windows;
- struttura, shape, dtype, conteggi e unicita' dei `row_id` sono stati
  verificati sugli array effettivamente letti da `D.npz`;
- gli SHA-256 ricalcolati per tutti gli otto array di entrambi i dataset
  coincidono con la provenienza archiviata.

Il workflow completo, inclusi installazione, test, download, ricostruzione,
smoke e caricamento degli artefatti, ha richiesto 4 min 16 s; il job ha
richiesto 4 min 11 s.

## Esito

| Dataset | Blocchi | Righe | Richieste | FIT/CAL vettoriale | FIT/CAL per osservazione | Aggiornamenti principali vettoriale/per osservazione | Tempo replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| TON | 30 | 300.000 | 3.000 | 2.258 / 742 | 2.223 / 777 | 20 / 17 | 1,370 s |
| UNSW | 30 | 300.000 | 3.000 | 2.324 / 676 | 2.239 / 761 | 28 / 29 | 1,611 s |

Entrambi i rendiconti contengono esattamente 30 blocchi ordinati. Il controllo
strutturale vettoriale coincide con la traiettoria archiviata: hash delle
richieste 30/30, conteggi ALL256 58/58, stati di aggiornamento 29/29 e
controlli CAL64 su conteggi ed eta' 116/116. Anche l'audit numerico separato
e' esatto: digest dei parametri della testa 30/30, soglie frozen 150/150,
soglie head 150/150, matrici di confusione 300/300, differenza L1 complessiva
zero e massima differenza assoluta fra soglie pari a zero per entrambi i
dataset.

I test locali servono a verificare il codice su Windows. Lo smoke numerico
ufficiale e' stato eseguito anche su Linux per confrontarlo con le traiettorie
Paper 2 archiviate in quell'ambiente; piattaforma, processore e runtime BLAS
sono registrati in ciascun JSON.

I 100 esempi richiesti nell'ultimo blocco restano correttamente pendenti:
saranno disponibili dopo la fine dello smoke e non possono influenzarne le
predizioni.

## Stima della matrice completa

Il flusso TON comprende 355 blocchi e UNSW 102. Scalando i soli tempi di
replay misurati sulle 15 coppie seed/sale per dataset si ottengono circa
325,3 s (5,4 min). Per pianificare includiamo avvio dei 30 processi, controlli,
scrittura dei report e preparazione comune: 10--15 min su un runner Linux
equivalente. Analisi, figure, manifest e verifica finale richiedono fino a
una giornata lavorativa dopo l'approvazione del checkpoint.

La matrice completa non e' stata avviata in questa consegna: protocollo,
codice e smoke vengono prima sottoposti alla verifica prevista.


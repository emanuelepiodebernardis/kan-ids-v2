# ADAPT-03: replay completo pesi adattati + soglia

## Esito tecnico

Il replay usa il commit `106ee1ccce2f116b7fb9b296df0a403c789f40ab` e contiene esattamente i
blocchi `0..915` di C/seed42. La validazione e' **PASS**: richieste, traiettoria
KAN e controllo a soglia B coincidono con ADAPT-02 su tutti i 916 blocchi. I
due bracci condividono pesi, score, memoria, calendario e AUROC; cambia
soltanto la soglia applicata alle predizioni successive.

Sono stati applicati 915 aggiornamenti dei pesi e
915 ricalibrazioni complete su B.

## Metriche globali

| metodo | recall compl. | FPR compl. | balanced accuracy | AUROC media per blocco |
|---|---:|---:|---:|---:|
| pesi + soglia B fissa | 20,608% | 0,474% | 60,067% | 0,889025 |
| stessi pesi + soglia ricalibrata | 81,082% | 30,141% | 75,471% | 0,889025 |

Le metriche globali derivano dalle confusion matrix aggregate, non dalla media
dei blocchi.

| metodo | VN | FP | FN | VP |
|---|---:|---:|---:|---:|
| pesi + soglia B fissa | 172.494 | 822 | 7.127.271 | 1.850.086 |
| stessi pesi + soglia ricalibrata | 121.077 | 52.239 | 1.698.308 | 7.279.049 |

## Metriche per blocco e denominatori

I 41 blocchi senza esempi normali conservano FPR, balanced accuracy e AUROC
come valori non definiti. Non sono sostituiti con zero.

| metrica | blocchi validi | media controllo | media variante | differenza media |
|---|---:|---:|---:|---:|
| recall attacchi | 916/916 | 0,208057 | 0,811328 | 0,603271 |
| FPR | 875/916 | 0,003549 | 0,419550 | 0,416002 |
| balanced accuracy | 875/916 | 0,606513 | 0,696460 | 0,089947 |
| AUROC | 875/916 | 0,889025 | 0,889025 | 0,000000 |

Le differenze sono sempre `variante - controllo` e usano solo coppie definite.

| metrica | coppie valide | variante maggiore | variante minore | uguali | dev. std. differenza |
|---|---:|---:|---:|---:|---:|
| recall attacchi | 916 | 882 | 16 | 18 | 0,413909 |
| FPR | 875 | 618 | 7 | 250 | 0,453326 |
| balanced accuracy | 875 | 531 | 297 | 47 | 0,160188 |
| AUROC | 875 | 0 | 0 | 875 | 0,000000 |

Il CSV conserva confusion matrix, soglie, digest e costi per ciascun blocco.
Le celle delle metriche non definite sono vuote; le figure mostrano lacune
nelle stesse posizioni.

## Costi effettivi e memoria

Il replay completo ha richiesto 38,294 minuti
complessivi, di cui 34,347 per il ciclo sui
blocchi. Le ricalibrazioni costano complessivamente
1785,007 secondi, con mediana
1,630986 s e 90o percentile
3,493953 s; il massimo osservato
e' 11,109660 s. I 915 fit
online richiedono complessivamente
72,867 secondi. La costruzione iniziale
della cache di B richiede
193,729 secondi.

B contiene 5.681.198 righe ed e' un archivio aggiuntivo
rispetto ai 256 esempi online. La cache della rappresentazione KAN occupa
363.596.672 byte.

Il valore massimo di 24.904 byte descrive soltanto
gli array impiegati nel fit: feature ed etichette della memoria, rappresentazione KAN della memoria e parametri correnti della testa KAN. Esclude
oggetti del modello e della trasformazione, richieste in attesa, buffer del blocco, overhead Python e archivio/cache B. Non rappresenta quindi l'intero stato online. Il picco
RSS finale del banco di prova e' 2101,2 MiB e
comprende anche input, cache e array temporanei.

## Artefatto completo

Il rendiconto originale e' conservato fuori Git in `replay_completo_C_seed42_106ee1c.zip`
(381.478 byte), SHA-256 `b084c4bfb0955698965aacb16e8a6a2ff78fe8f22e31a18ff6a12fd8b1e01d38`. Il manifest
versionato registra anche dimensione e SHA-256 del JSON interno. Il
collegamento scaricabile viene aggiunto dopo il caricamento senza modificare
l'archivio.

## Limiti

Questa e' una singola diagnosi appaiata C/seed42. Non e' stata eseguita una
campagna su cinque semi, la variante 192/64 non e' stata provata e D e' rimasto
riservato. Recall, FPR e balanced accuracy vanno interpretati insieme: un
aumento del solo recall non dimostra superiorita' generale.

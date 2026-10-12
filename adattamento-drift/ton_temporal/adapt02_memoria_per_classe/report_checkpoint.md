# ADAPT-02: report del primo checkpoint

## Ambito

Questo checkpoint non avvia la campagna estesa. Verifica solo che il confronto
fra FIFO e memoria per classe sia implementato in modo causale, testabile e
appaiato su un breve intervallo di C.

Base: `1679f65697ccdf1e92710ebfcbb940ce36233e7e`.

Branch: `adapt-02/memoria-per-classe`.

D non e' stato usato.

## Implementazione

`replay.py` accetta ora `--memoria-strategia`, con default `fifo`.

La FIFO resta la strategia storica. La strategia `per_classe` usa la stessa
capacita' massima, 256 esempi, con quote 128/128 per Normal/Attack. Le quote
sono una garanzia minima quando entrambe le classi sono disponibili: se una
classe ha meno di 128 esempi, i posti liberi possono essere occupati dall'altra
classe. Quando il limite viene superato si espelle dagli esempi piu' vecchi
della classe sopra quota.

La memoria per classe riceve lo stesso insieme iniziale da A della FIFO. Non
ricampiona A, non duplica esempi rari e non recupera campioni eliminati.

Ogni blocco con etichette arrivate registra:

- strategia di memoria;
- normali e attacchi in memoria;
- eta' del buffer complessiva e per classe;
- skip e aggiornamenti gia' presenti nel rendiconto.

## Test

Nuova suite: `test_memoria_per_classe.py`.

Controlla:

- FIFO esplicita uguale al default sui campi riproducibili;
- limite di memoria;
- quote e posti liberi;
- espulsione degli esempi piu' vecchi per classe;
- classe assente ancora monoclasse;
- nessun recupero dei campioni eliminati;
- richieste di etichette e misure frozen identiche fra FIFO e per-classe.

Esito:

```text
6 passed, 4 warnings
```

Le warning sono quelle gia' note di `MLPClassifier` sui piccoli flussi
sintetici di test.

## Smoke su C, seed 42

Comando:

```bash
cd <repo>/adattamento-drift/ton_temporal
python smoke_memoria_per_classe.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --uscita-dir adapt02_memoria_per_classe/smoke_seed42 --seme 42 --max-blocchi 24
```

Output:

- `smoke_seed42/smoke_C_seme42_fifo_ogni_blocco.json`
- `smoke_seed42/smoke_C_seme42_fifo_evidenza_inversione.json`
- `smoke_seed42/smoke_C_seme42_per_classe_ogni_blocco.json`
- `smoke_seed42/smoke_C_seme42_per_classe_evidenza_inversione.json`
- `smoke_seed42/riepilogo_smoke.json`

Lo script ha verificato che i digest dei `row_id` campionati e le misure
frozen per blocco coincidano nelle quattro corse.

Parametri:

- seme 42;
- 24 blocchi di C;
- 240.000 righe scorse;
- 2.400 etichette richieste;
- quota effettiva 1%.

Composizione media della memoria:

| memoria | normali medi | attacchi medi |
|---|---:|---:|
| FIFO | 8,2 | 247,8 |
| per classe | 46,6 | 209,4 |

Aggiornamenti applicati e salti:

| memoria | politica | LR | MLP | KAN | salti monoclasse |
|---|---|---:|---:|---:|---:|
| FIFO | ogni blocco | 23 | 23 | 23 | 0 |
| FIFO | evidence | 2 | 2 | 0 | 0 |
| per classe | ogni blocco | 23 | 23 | 23 | 0 |
| per classe | evidence | 1 | 1 | 0 | 0 |

Il fatto che nei primi 24 blocchi non compaiano salti monoclasse rende lo smoke
utile per verificare funzionamento e appaiamento, non per concludere sul
vantaggio della memoria.

Metriche medie sui blocchi definiti:

| memoria | politica | modello | AUROC adattivo | recall attacchi | FPR medio blocco |
|---|---|---|---:|---:|---:|
| FIFO | ogni blocco | LR | 0,8916 | 0,9619 | 0,1362 |
| FIFO | ogni blocco | MLP | 0,9146 | 0,9753 | 0,1270 |
| FIFO | ogni blocco | KAN | 0,9638 | 0,9601 | 0,0386 |
| FIFO | evidence | LR | 0,8336 | 0,9679 | 0,1801 |
| FIFO | evidence | MLP | 0,7928 | 0,9780 | 0,2333 |
| FIFO | evidence | KAN | 0,9684 | 0,9616 | 0,0008 |
| per classe | ogni blocco | LR | 0,9235 | 0,9561 | 0,0832 |
| per classe | ogni blocco | MLP | 0,9287 | 0,9577 | 0,0803 |
| per classe | ogni blocco | KAN | 0,9878 | 0,8769 | 0,0004 |
| per classe | evidence | LR | 0,8681 | 0,9566 | 0,2127 |
| per classe | evidence | MLP | 0,8173 | 0,9781 | 0,2759 |
| per classe | evidence | KAN | 0,9684 | 0,9616 | 0,0008 |

Questi numeri non sono un risultato scientifico: il tratto e' breve, sta
all'inizio di C e non contiene la situazione che in ADAPT-01 produceva molti
skip. Servono solo a controllare che il meccanismo produca rendiconti
interpretabili.

## Tempi osservati

Tempo totale dello smoke: 1.753,57 secondi, circa 29,2 minuti.

Tempi per corsa:

| file | totale s | iniziale s | replay s |
|---|---:|---:|---:|
| FIFO ogni blocco | 576,63 | 160,46 | 17,13 |
| FIFO evidence | 493,92 | 139,84 | 8,20 |
| per classe ogni blocco | 381,69 | 140,05 | 7,69 |
| per classe evidence | 299,59 | 113,60 | 2,66 |

Il costo e' dominato da addestramento iniziale e soglie su B, ripetuti per ogni
corsa. Il replay dei 24 blocchi pesa poco.

Stima prudente per la campagna da 20 replay sullo stesso PC: 3-4 ore. La stima
include la ripetizione conservativa di addestramento e calibrazione per ogni
corsa. Eventuali riusi delle soglie o dei modelli iniziali vanno introdotti
solo se prima vengono resi verificabili e appaiati.

## Esito del checkpoint

Il checkpoint e' tecnicamente fattibile:

- la FIFO resta retrocompatibile;
- la memoria per classe rispetta i vincoli causali;
- lo smoke produce quattro rendiconti appaiati;
- composizione ed eta' del buffer sono registrate;
- D resta fuori.

La campagna estesa non e' stata eseguita e richiede autorizzazione separata.

# ADAPT-02: report del primo checkpoint

## Ambito

Questo checkpoint non avvia la campagna estesa. Verifica solo che il confronto
fra FIFO e memoria per classe sia implementato in modo causale, testabile e
appaiato su un breve intervallo di C.

Base: `1679f65697ccdf1e92710ebfcbb940ce36233e7e`.

Branch: `adapt-02/memoria-per-classe`.

Codice eseguito nello smoke aggiornato:
`a30898523f4573efaacfda45f5c1cc77a7382992`.

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

Correzione del 13/10/2026: `MemoriaReplay.aggiungi` conserva il dtype
originale delle feature. Il refactoring precedente usava
`np.asarray(x, dtype=float)`, che portava le righe trasformate da `float32` a
`float64` e cambiava anche la FIFO di default. Ora la memoria copia
`np.asarray(x)` senza conversione forzata.

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
- dtype originale delle feature conservato in memoria;
- default confrontato con la FIFO storica del commit base;
- richieste di etichette e misure frozen identiche fra FIFO e per-classe.

Esito:

```text
8 passed, 6 warnings
```

Le warning sono quelle gia' note di `MLPClassifier` sui piccoli flussi
sintetici di test.

Suite completa della cartella `ton_temporal`:

```text
259 passed, 15 warnings in 73.67s
```

Anche qui le warning sono quelle gia' note di `MLPClassifier` nei test
sintetici.

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
- `smoke_seed42/fifo_storica/smoke_C_seme42_fifo_storica_ogni_blocco.json`
- `smoke_seed42/fifo_storica/smoke_C_seme42_fifo_storica_evidenza_inversione.json`
- `smoke_seed42/riepilogo_smoke.json`

Lo script ha verificato che i digest dei `row_id` campionati e le misure
frozen per blocco coincidano nelle quattro corse.

In piu' ha rieseguito la FIFO storica dal commit
`1679f65697ccdf1e92710ebfcbb940ce36233e7e` e l'ha confrontata con la FIFO
corrente prodotta dal codice `a30898523f4573efaacfda45f5c1cc77a7382992`.
Per entrambe le politiche, `ogni_blocco` ed `evidenza_inversione`, il confronto
riporta:

- stesso numero di blocchi;
- riepilogo identico;
- campi per blocco identici su digest dei campioni, etichette richieste,
  conteggi della memoria, misure frozen, misure adattive, stime, decisioni e
  salti.

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
| FIFO | ogni blocco | MLP | 0,9144 | 0,9753 | 0,1270 |
| FIFO | ogni blocco | KAN | 0,9638 | 0,9606 | 0,0388 |
| FIFO | evidence | LR | 0,8336 | 0,9679 | 0,1801 |
| FIFO | evidence | MLP | 0,7928 | 0,9780 | 0,2333 |
| FIFO | evidence | KAN | 0,9684 | 0,9616 | 0,0008 |
| per classe | ogni blocco | LR | 0,9235 | 0,9561 | 0,0832 |
| per classe | ogni blocco | MLP | 0,9287 | 0,9577 | 0,0803 |
| per classe | ogni blocco | KAN | 0,9878 | 0,8770 | 0,0004 |
| per classe | evidence | LR | 0,8681 | 0,9566 | 0,2127 |
| per classe | evidence | MLP | 0,8173 | 0,9782 | 0,2759 |
| per classe | evidence | KAN | 0,9684 | 0,9616 | 0,0008 |

Questi numeri non sono un risultato scientifico: il tratto e' breve, sta
all'inizio di C e non contiene la situazione che in ADAPT-01 produceva molti
skip. Servono solo a controllare che il meccanismo produca rendiconti
interpretabili.

## Tempi osservati

Tempo totale del comando completo: 1.240,34 secondi, circa 20,7 minuti.
Questo include le quattro corse correnti e le due FIFO storiche usate solo per
il confronto di retrocompatibilita'.

Tempi per le quattro corse correnti:

| file | totale s | iniziale s | replay s |
|---|---:|---:|---:|
| FIFO ogni blocco | 257,47 | 91,86 | 5,73 |
| FIFO evidence | 221,12 | 73,70 | 1,82 |
| per classe ogni blocco | 187,01 | 60,09 | 3,22 |
| per classe evidence | 179,39 | 59,25 | 2,67 |

Il costo e' dominato da addestramento iniziale e soglie su B, ripetuti per ogni
corsa. Il replay dei 24 blocchi pesa poco.

Le quattro corse correnti sommano circa 845 secondi. Proiettando linearmente su
20 replay si ottengono circa 70 minuti; una stima prudente sullo stesso PC e'
2-3 ore, includendo variabilita' del carico, scrittura dei rendiconti e
verifiche. Eventuali riusi delle soglie o dei modelli iniziali vanno introdotti
solo se prima vengono resi verificabili e appaiati.

## Esito del checkpoint

Il checkpoint e' tecnicamente fattibile:

- la FIFO resta retrocompatibile;
- la FIFO corrente coincide con la FIFO storica sui campi confrontati;
- la memoria per classe rispetta i vincoli causali;
- lo smoke produce quattro rendiconti appaiati;
- composizione ed eta' del buffer sono registrate;
- D resta fuori.

La campagna estesa non e' stata eseguita e richiede autorizzazione separata.

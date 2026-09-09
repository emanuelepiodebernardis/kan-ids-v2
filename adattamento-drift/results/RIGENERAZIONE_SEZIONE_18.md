# Sezione 18 sotto il protocollo corretto: cosa e' stato rifatto, e cosa e' cambiato

> **Fatto.** 9 stage, 32m48s, 10 seed (42-51). I numeri della sezione 18 di
> `RISULTATI.md` vengono ora da questi run; i precedenti stanno in
> `artifacts/protocollo_v1/` e `results/protocollo_v1/`.

La sezione 18 (sensibilita' al rapporto di undersampling) e' l'ultima area
del documento ferma al protocollo v1. La domanda da chiudere prima di
lanciare calcolo non era «come si rifa'», era **quanto costa e quanto se ne
puo' evitare**: la griglia nominale e' cinque rapporti x sei direzioni x
quattro script x 10 seed, e rifarla per intero sarebbe stato il run piu'
caro dell'intero lavoro. Non serve. Il conto sotto lo mostra, e la
riduzione non e' una scorciatoia: e' una conseguenza del codice.

## Cosa NON va rifatto, e perche' non e' un'approssimazione

**1. I due script prequenziali non sono toccati dal cambio di protocollo.**
`drift_graduale.py` e `drift_graduale_int.py` valutano ogni batch *prima* di
addestrarci sopra (test-then-train): non c'e' un complemento da partizionare,
non importano `dividi_target`, e infatti non compaiono fra i sette stage
della rigenerazione principale. I loro risultati a ratio 1, 3, 20 e 100
restano validi come sono. **La sottosezione 18.2 non va rifatta** — ed e' la
meta' piu' cara della griglia, otto checkpoint da 1 200 a 8 400 righe.
Un test (`tests/test_rigenera.py`) fallisce se uno dei due comincia a usare
la partizione, cosi' l'esenzione non sopravvive alla sua premessa.

**2. `undersample()` e' un no-op quando il rapporto naturale e' gia' piu'
equilibrato di `ratio`.** Tiene tutta la classe minoritaria e taglia la
maggioritaria a `min(maggioritaria, ratio * minoritaria)`: se
`ratio * minoritaria >= maggioritaria` non viene nemmeno estratto un
campione casuale, e il training set e' identico **bit per bit**, non solo
statisticamente simile. Con i rapporti naturali gia' misurati — TON_IoT
3,22:1, UNSW-NB15 1,77:1, BoT-IoT 7 690:1 — il rapporto vincola solo qui:

| ratio | TON sorgente | UNSW sorgente | BoT sorgente | direzioni da calcolare |
|---|---|---|---|---|
| 1 | vincola | vincola | vincola | **6** (tutte) |
| 3 | vincola (3 < 3,22) | no-op | vincola | **4** |
| 20 | no-op | no-op | vincola | **2** |
| 50 | no-op | no-op | vincola | *basale, gia' fatto* |
| 100 | no-op | no-op | vincola | **2** |

Le celle «no-op» non vengono ricalcolate perche' ricalcolarle produrrebbe
per costruzione lo stesso numero gia' presente a ratio 50 — l'argomento e'
lo stesso della sezione 18, dove era stato verificato empiricamente su
cinque combinazioni script/direzione/seed prima di fidarsene.

Restano quindi **due script per quattro rapporti**, su 14 combinazioni
direzione-rapporto invece delle 48 nominali.

## Il costo, misurato e non stimato a occhio

Base: la rigenerazione principale del 9 settembre, sulla stessa macchina,
con gli stessi 10 seed.

| stage misurato | copertura | durata |
|---|---|---|
| `int_adapt` | 6 direzioni x 10 seed | **13 min** |
| `tre_domini_ricco` | 3 sorgenti x 10 seed | **6 min** |

Riscalando per copertura (e per eccesso: a rapporti bassi il training set e'
piu' piccolo di quello a 50, quindi ogni cella costa meno del suo basale):

| stage | copertura | stima |
|---|---|---|
| `s18_tre_domini_r1` | 3 sorgenti | ~6 min |
| `s18_tre_domini_r3_ton` + `_r3_bot` | 2 sorgenti | ~4 min |
| `s18_tre_domini_r20` | 1 sorgente | ~2 min |
| `s18_tre_domini_r100` | 1 sorgente | ~2 min |
| `s18_int_adapt_r1` | 6 direzioni | ~13 min |
| `s18_int_adapt_r3` | 4 direzioni | ~9 min |
| `s18_int_adapt_r20` | 2 direzioni | ~4 min |
| `s18_int_adapt_r100` | 2 direzioni | ~4 min |
| **totale** | | **~45 min** |

Quarantacinque minuti, non una notte: la sezione si rifa'.

## Il costo vero, a consuntivo

| stage | stima | misurato |
|---|---|---|
| `s18_tre_domini_r1` | ~6 min | 5m14s |
| `s18_tre_domini_r3_ton` + `_r3_bot` | ~4 min | 2m23s + 1m41s |
| `s18_tre_domini_r20` | ~2 min | 1m45s |
| `s18_tre_domini_r100` | ~2 min | 2m11s |
| `s18_int_adapt_r1` | ~13 min | 8m57s |
| `s18_int_adapt_r3` | ~9 min | 6m19s |
| `s18_int_adapt_r20` | ~4 min | 1m37s |
| `s18_int_adapt_r100` | ~4 min | 2m37s |
| **totale** | **~45 min** | **32m48s** |

La stima era alta del 37%, nella direzione giusta: riscalare la durata di
uno stage per il numero di direzioni ignora che a rapporti bassi il
training set e' piu' piccolo di quello a 50, quindi ogni cella costa meno
del suo basale.

## Cosa e' cambiato nei risultati

**Niente di sostanziale, ed e' il risultato che serviva.** Le medie si
spostano nella terza-quarta cifra decimale, nessun verdetto cambia segno,
nessuna significativita' attraversa la soglia. L'aggregato
dell'affermazione 5 a ratio 1 passa da −0,0341 (p=0,0040) a −0,0331
(p=0,0043); i conteggi di seed riusciti dell'affermazione 1 sono
identici cella per cella. La correzione del protocollo non toglieva
contaminazione in questa sezione — il complemento delle righe selezionate
non era contaminato nemmeno prima — ma restringeva il test set di circa il
15%, e su decine di migliaia di righe di valutazione questo sposta una
media molto meno della sua dispersione fra seed.

**Due cose sono emerse rifacendo i conti con un unico script**
(`scripts/analisi_sezione18.py`) invece che raccogliendoli a mano:

1. **Un conteggio sbagliato in 18.1**: la colonna "32 etichette" dichiarava
   10/10 seed per `bot->ton` a ratio 20 e 9/10 a ratio 100, quando erano 8
   in entrambi i casi. Le medie erano giuste — erano calcolate sui seed
   giusti — ma il numero fra parentesi veniva da un'altra cella.
2. **Due celle significative nella parita' intero-float** (18.5), che la
   stesura precedente non poteva vedere perche' riportava solo le due
   direzioni con BoT-IoT come sorgente: `ton->bot` a ratio 3 (−0,153,
   p=0,007, che sopravvive a Holm) e `unsw->ton` (−0,035, p=0,043,
   ereditata da ratio 50 e non significativa dopo Holm). L'affermazione
   "l'aritmetica intera non costa" e' stata riformulata di conseguenza,
   qui e in sezione 16.1.

## Come si lancia

Dalla cartella `adattamento-drift/`:

    python rigenera.py --sezione18

Non serve `--dati`: i quattro CSV stanno in `kanids-data/` accanto al repo e
`rigenera.py` ora li cerca li' da solo. `--dati` resta per usare un'altra
cartella, e viene ricordato in `.ultima_cartella_dati`. Gli stage sono
checkpointati come tutti gli altri: Ctrl-C e rilancio riprendono da dove si
erano fermati.

## Igiene, prima di lanciare

**I checkpoint e i CSV vecchi sono stati spostati**, non lasciati sul
posto: `artifacts/protocollo_v1/` e `results/protocollo_v1/` ora contengono
gli otto `*_ratio*.jsonl` di `tre_domini` e `drift_int_adapt` e i loro CSV.
Senza questo passaggio gli script li avrebbero trovati «gia' fatti» e non
avrebbero ricalcolato nulla, riportando i numeri v1 in un documento che li
dichiara nuovi — e' il modo esatto in cui un risultato vecchio era gia'
rientrato una volta in questo lavoro.

**Il guardiano che lo controlla e' stato corretto.** `checkpoint_obsoleto()`
in `rigenera.py` cercava i checkpoint con una glob e **saltava ogni file con
`ratio` nel nome**, cioe' esattamente questi otto: la protezione non si
sarebbe attivata proprio dove serviva. Ora ogni stage dichiara i propri
checkpoint per nome e il guardiano li verifica uno per uno; un test lo
blocca (`test_un_checkpoint_piu_vecchio_del_protocollo_viene_segnalato`,
scritto apposta su un nome con `ratio` dentro).

**L'header C non e' a rischio**: la condizione che scrive
`mcu_pio/include/kan_int_adapt.h` richiede `ratio == 50.0` oltre a
direzione, seed e `iters` canonici, quindi nessuno di questi run lo tocca.

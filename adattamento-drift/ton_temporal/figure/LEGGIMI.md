# Le figure del pilota

Quattordici PNG, prodotti da `curve_replay.py` dai soli rendiconti del replay.
Nessuna figura rilegge i dati grezzi, e nessuna contiene righe del dataset:
leggono i file JSON di `replay_evidenza/`, che portano misure aggregate per
blocco.

Tutte le figure pubblicate qui vengono dal **seme 42**. Gli altri quattro semi
restano nei rendiconti e nelle tabelle; disegnarli tutti darebbe settanta
immagini senza aggiungere un'affermazione che le tabelle non facciano già, e
dove un risultato dipende dal seme — e dipende: il §10ter elenca i casi in cui
si rovescia — è la tabella per seme a dirlo, non una figura.

## Da quali file

| rendiconto | SHA-256 |
|---|---|
| `replay_evidenza/replay_C_calibrato_seme42.json` | `e6197fbd55654e8bcf99ab56a5f9a06dec12c2e4d44c9e901ea118d66e22d101` |
| `replay_evidenza/replay_C_evidenza_seme42.json` | `abb5041f566fc9601249285dd7abea440064b44b1a66be28dede50870be0773f` |
| `replay_evidenza/replay_C_casuale_seme42.json` | `789fa9221a18f6639f43d35132920dde53fc387ee299b3e0342ac8925c979d19` |

Il primo è la politica «ogni blocco» ed è anche il rendiconto da cui vengono le
figure a due serie, perché la copia congelata è la stessa in tutte le politiche
per costruzione. `curve_replay.py` non lo assume: confronta blocco per blocco i
falsi negativi e le AUROC della copia congelata di ogni rendiconto e si ferma se
divergono.

## Come si rifanno

Un comando per modello, dalla cartella `ton_temporal`. Produce tutte le figure
di quel modello nella cartella indicata.

```
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello lr --uscita figure
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello mlp --uscita figure
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello kan --uscita figure
```

I PNG non sono riproducibili byte per byte fra versioni diverse di matplotlib:
cambia la compressione, non il contenuto. Quello che `test_figure.py` verifica
non è il byte, è che i numeri disegnati siano quelli dei rendiconti.

## Che cosa mostra ciascuna

**`curve_per_blocco_<modello>.png`** — AUROC e richiamo sui normali, blocco per
blocco, copia congelata contro adattiva. Traccia sottile il valore del blocco,
linea spessa la media mobile su 25 blocchi. Serve a vedere *quando* le due
copie si separano, non quanto.

**`attacchi_non_rilevati_<modello>.png`** — i falsi negativi sommati blocco per
blocco, e a destra la differenza fra adattivo e congelato. È la figura che la
scheda chiede: la misura per cui il rilevatore esiste. Sul seme 42 la logistica
chiude a 1.377.545 attacchi non rilevati in meno su 8.977.357 attacchi.
**Dipende dal punto di decisione** — la soglia scelta su B col massimo della
balanced accuracy — e la didascalia lo dichiara, perché con la soglia a zero i
conteggi sono altri.

**`guadagno_per_fascia_<modello>.png`** — il guadagno medio di AUROC per fascia
di ricchezza di normali del blocco. Mostra che il guadagno non è diffuso: in
tutti e tre i modelli è positivo **solo** nei 246 blocchi con almeno 200
normali, e nelle tre fasce più povere l'adattamento peggiora. Le quattro fasce
contano 221, 275, 133 e 246 blocchi — 875 in tutto, cioè i blocchi in cui
l'AUROC è definita per entrambe le copie; i blocchi del flusso sono 916, e nei
41 restanti manca la misura, quindi la figura li esclude invece di contarli a
zero. Le fasce sono le stesse per i tre modelli, perché dipendono dal blocco e
non dal modello. Sul seme 42 il guadagno della fascia
ricca è +0,409 per la logistica, +0,438 per l'MLP e +0,045 per l'additivo:
sull'additivo quel guadagno non compensa le perdite delle altre tre fasce, ed è
coerente col fatto che lì la copia congelata resti davanti.

**`politiche_<modello>.png`** — le quattro politiche sugli stessi blocchi e
sulle stesse righe etichettate: congelato, ogni blocco, su evidenza, casuale.
A destra la media progressiva dell'AUROC, non una media mobile, perché il suo
valore finale **è** l'AUROC media di quella corsa, cioè il valore del seme 42
nel campo `per_seme` di `replay_evidenza/confronto_politiche.json`: figura e
testo non possono divergere. Non è la media sui cinque semi delle tabelle
riassuntive, che è un'altra aggregazione e dà altri numeri.

Su `kan` la media progressiva della copia congelata chiude sopra quella di ogni
politica e resta stabilmente sopra **dal blocco 213 in poi**; nei primi 213
blocchi almeno una politica sta davanti in 175 di essi, mai tutte e tre
insieme. Blocco per blocco il congelato è comunque sotto in 401, 298 e 403
blocchi su 875 rispetto a «ogni blocco», «su evidenza» e «casuale». Il
vantaggio è quindi sul flusso intero, non in ogni suo punto. La politica
casuale, su quel modello, perde più attacchi del congelato (3.356.503 contro
3.157.627): è il risultato scomodo, ed è disegnato.

**`recupero_<modello>.png`** — l'AUROC delle due copie allineata all'inizio
degli episodi di inversione del congelato, da cinque blocchi prima a venti
dopo. Nessuna soglia di recupero scelta dopo aver visto il risultato: solo le
curve. Si mediano i soli episodi lunghi almeno cinque blocchi, e i singoli
episodi restano disegnati sottili sotto la media; gli episodi esclusi e la loro
lunghezza sono scritti in didascalia.

### Una figura che manca, e perché

**`recupero_kan.png` non esiste.** Sul modello additivo gli episodi di
inversione del congelato sono tre, lunghi 7, 2 e 1 blocchi: uno solo arriva al
minimo di cinque. La media di un episodio è quell'episodio, e presentarla come
una curva media direbbe più di quello che il dato contiene. Il programma si
astiene da sé e lo scrive nel proprio resoconto; non è un file dimenticato.

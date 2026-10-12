# ADAPT-02: memoria per classe, checkpoint iniziale

## Stato

Base di partenza: `main` dopo il merge di ADAPT-01.

SHA base: `1679f65697ccdf1e92710ebfcbb940ce36233e7e`.

Branch: `adapt-02/memoria-per-classe`.

D resta riservato: nessun training, scoring, confronto o scelta di
configurazione usa D in questo checkpoint.

## Domanda

ADAPT-01 ha mostrato che la FIFO da 256 esempi diventa spesso monoclasse.
Quando questo accade, il tentativo di aggiornamento viene saltato. ADAPT-02
verifica se una memoria che conserva esempi recenti di entrambe le classi,
senza aumentare memoria o budget di etichette, riduce gli skip e cambia il
compromesso fra attacchi mancati e falsi allarmi.

Ipotesi da verificare: la memoria per classe puo' ridurre gli aggiornamenti
saltati e recuperare richiamo sugli attacchi. Non e' un risultato atteso
obbligatorio: puo' anche peggiorare il FPR o trattenere normali troppo vecchi,
ostacolando l'adattamento.

## Memorie confrontate

Le due memorie hanno la stessa capacita' massima: 256 esempi.

`fifo`: memoria storica di ADAPT-01. Gli esempi arrivati entrano in coda, e
quando il limite viene superato escono i piu' vecchi, senza guardare la classe.
Questa resta il default del runner e deve riprodurre la baseline.

`per_classe`: memoria binaria Normal/Attack con quote 128/128. Le quote sono
garanzie minime quando entrambe le classi sono disponibili, non duplicazioni.
Se una classe ha meno di 128 esempi, i posti liberi possono essere occupati
dall'altra classe. Quando il limite totale viene superato, si espelle prima la
classe sopra quota, sempre dagli esempi piu' vecchi di quella classe. In caso
di classe assente, la memoria resta monoclasse e la guardia continua a saltare
l'aggiornamento.

Entrambe le strategie ricevono lo stesso insieme iniziale di 256 esempi da A,
campionato con lo stesso seme. La memoria per classe non ricampiona A e non ha
un archivio nascosto da cui recuperare esempi eliminati.

## Causalita'

La memoria viene aggiornata solo con le etichette effettivamente arrivate dopo
il ritardo dichiarato. Un esempio eliminato non puo' rientrare. Il campionatore
delle richieste resta quello di ADAPT-01: stesso budget 1%, stessi `row_id`
richiesti, nessuna etichetta aggiuntiva per bilanciare il buffer.

La stima evidence resta fuori campione: usa le predizioni gia' emesse e le
etichette appena arrivate del blocco campionato, non il buffer su cui si
addestra.

Preprocessing, feature, soglie scelte su B, pesi iniziali, blocchi da 10.000,
ritardo di un blocco e stream C restano quelli del pilot.

## Confronto fissato

Checkpoint: smoke breve su C, seme 42, intervallo prefissato con
`--max-blocchi 24`.

Matrice:

| memoria | politica |
|---|---|
| fifo | ogni blocco |
| fifo | evidence |
| per_classe | ogni blocco |
| per_classe | evidence |

La campagna successiva, se autorizzata, usera' la stessa matrice sui cinque
semi 42-46. Poiche' il runner produce LR, MLP e KAN/additivo nella stessa
corsa, la matrice completa richiede 20 replay: due memorie x due politiche x
cinque semi.

Non si aggiungono architetture, detector, budget o soglie.

## Metriche

Per modello, seme, memoria e politica:

- AUROC sui blocchi definiti, con denominatore;
- confusion matrix aggregata;
- falsi negativi e falsi positivi;
- richiamo degli attacchi;
- FPR complessivo e FPR medio per blocco;
- aggiornamenti tentati, applicati e saltati;
- composizione della memoria per blocco;
- eta' del buffer per classe;
- costo misurato degli aggiornamenti.

Un miglioramento di AUROC da solo non basta. Meno attacchi mancati con molti
piu' falsi allarmi non viene formulato come superiorita' generale.

## Test del checkpoint

I test sintetici richiesti sono in `test_memoria_per_classe.py`:

- la FIFO esplicita riproduce il default;
- la memoria per classe rispetta il limite di memoria;
- i posti liberi della classe assente possono essere usati dall'altra classe;
- l'espulsione conserva gli esempi piu' recenti per classe;
- un campione eliminato non viene recuperato;
- se una classe e' assente, la memoria resta monoclasse e la guardia scatta;
- richieste di etichette e misure frozen restano appaiate fra le memorie.

## Smoke

Lo smoke e' prodotto da `smoke_memoria_per_classe.py`.

Comando previsto dalla radice del clone:

```bash
cd <repo>/adattamento-drift/ton_temporal
python smoke_memoria_per_classe.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --uscita-dir adapt02_memoria_per_classe/smoke_seed42 --seme 42 --max-blocchi 24
```

Lo script scrive quattro rendiconti brevi e `riepilogo_smoke.json`, e controlla
che nelle quattro corse coincidano i digest dei `row_id` campionati e le misure
del congelato per blocco.

## Criterio per passare alla campagna

La campagna sui 20 replay si avvia solo dopo revisione del checkpoint. Prima
devono essere veri questi punti:

- il default FIFO riproduce la baseline;
- i test causali passano;
- lo smoke produce quattro rendiconti appaiati;
- il riepilogo registra composizione ed eta' della memoria;
- il costo dello smoke consente una stima credibile dei 20 replay;
- nessun artefatto ADAPT-01 viene modificato o sovrascritto.

## Riferimenti pertinenti

Rolnick et al., "Experience Replay for Continual Learning", NeurIPS 2019:
esperienza passata conservata in buffer per ridurre forgetting in stream non
stazionari; utile come riferimento generale, ma non tratta il vincolo causale
specifico del nostro replay IDS.
https://papers.neurips.cc/paper/8327-experience-replay-for-continual-learning

Chrysakis e Moens, "Online Continual Learning from Imbalanced Data", ICML 2020:
propone class-balancing reservoir sampling per stream sbilanciati; rilevante
per il problema della popolazione della memoria, ma usa un'impostazione di
continual learning diversa dal nostro aggiornamento ritardato a budget fisso.
https://proceedings.mlr.press/v119/chrysakis20a.html

Bang et al., "Rainbow Memory: Continual Learning with a Memory of Diverse
Samples", CVPR 2021: mostra che la scelta dei campioni in memoria puo' essere
centrale quanto il modello; rilevante per motivare il confronto, non come
baseline diretta.
https://openaccess.thecvf.com/content/CVPR2021/html/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.html

Yoon et al., "Online Coreset Selection for Rehearsal-based Continual Learning",
ICLR 2022: studia la selezione online dei campioni per rehearsal; utile per
inquadrare il tema della memoria limitata, ma piu' complesso della variante
deterministica richiesta qui.
https://openreview.net/forum?id=f9D-5WNG4Nv

Chaudhry et al., "On Tiny Episodic Memories in Continual Learning", arXiv 2019:
mostra che anche memorie episodiche piccole possono influire molto sul
continual learning; rilevante per il vincolo 256 esempi, senza risolvere il
caso specifico Normal/Attack con etichette ritardate.
https://arxiv.org/abs/1902.10486

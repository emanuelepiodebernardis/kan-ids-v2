# Changelog

## Seconda revisione: verso v2.1-rc (agosto 2026)

Cinque richieste del Prof. Kuznetsov per rendere il lavoro pubblicabile
(IEEE Access). Due di esse ritirano affermazioni scritte nella voce
precedente di questo changelog, che resta com'e' come cronologia.

**1. Il rapporto del joint training non e' piu' scelto sui test set.**
La voce di agosto diceva che 1:5 era stato scelto "sui dati (paired t-test
su 10 seed)". Quei dati erano TON_test e BoT_test: i cinque rapporti erano
stati confrontati sulla stessa quantita' poi riportata come risultato, e i
test set consumati cinque volte invece di una. Ora `joint_training.py`
ritaglia una validation dentro il training di ciascun dominio
(`inner_split`, 20% stratificato) e la scelta avviene li'
(`--select-ratio`); i test entrano una volta sola, al rapporto gia' scelto,
che `--ratio` rilegge da `results/joint_ratio_selection_scelta.json`.

Il rapporto scelto resta **1:5** — la conclusione non cambia, cambia il
modo in cui e' stata raggiunta. Su validation, mediando 6 modelli x 2
domini x 10 seed:

| rapporto | bal. accuracy | vs 1:5 | p |
|---|---|---|---|
| **1:5** | **0,9732 ± 0,0228** | — | — |
| 1:10 | 0,9702 ± 0,0243 | −0,0030 | 2,4·10⁻⁵ |
| 1:20 | 0,9657 ± 0,0319 | −0,0075 | 2,6·10⁻⁴ |
| 1:50 | 0,9592 ± 0,0430 | −0,0140 | 3,2·10⁻⁷ |
| 1:100 | 0,9523 ± 0,0563 | −0,0209 | 9,6·10⁻⁸ |

Monotona, con lo stesso argmax in 10 seed su 10 presi singolarmente. Tutte
le differenze sono significative al 5%, ma quella verso il vicino piu'
prossimo e' piccola: 0,0030 contro i 0,0209 verso 1:100. **1:5 e' il bordo
della griglia** {5, 10, 20, 50, 100} e la curva e' monotona decrescente:
non e' escluso che un rapporto piu' basso faccia meglio, e nessun
esperimento qui lo verifica. `results/joint_ratio_significativita.csv` e'
l'artefatto dietro la tabella. Tre test impediscono ai test set di
rientrare nella scelta: cambiare completamente le righe destinate al test
non sposta di un indice ne' la validation ne' i training bilanciati dei
cinque rapporti candidati.

**Il rapporto non era l'unico.** Ricontrollando il resto del progetto contro
questa stessa richiesta e' venuto fuori che anche `hidden = 16` e
`degree = 8` venivano da ablation misurate sull'held-out: 0,9784 contro
0,9778 per la larghezza, 0,9409 contro 0,9374 per il grado, tutti F1 sullo
stesso 20% poi riportato come risultato. Stesso difetto, un piano sotto.

`scripts/select_architettura.py` rifa' la scelta dove va fatta — validation
ritagliata dentro il training, cinque seed, test calcolato e mai letto — con
una regola scritta nel docstring **prima** che i numeri esistessero: la 1-SE
di Breiman, cioe' fra le configurazioni entro un errore standard dalla
migliore si prende la piu' piccola. Il risultato **non conferma**
l'architettura in uso:

| | hidden | grado | bal. acc. validation | parametri |
|---|---|---|---|---|
| scelta dalla regola | 32 | 6 | **0,99631** | 2.592 |
| stessa media, piu' grande | 32 | 8 | 0,99631 | 3.296 |
| deployata | 16 | 8 | 0,99602 | 1.648 |

Le prime due sono indistinguibili alla sesta cifra (0,996308 entrambe, p
appaiato 0,999) e la regola prende la piu' piccola: e' il suo mestiere. La
16/8 manca la soglia 1-SE (0,99617) per 0,00015. Due fatti sullo scarto,
entrambi negli artefatti e nessuno dei due usato come criterio: la
differenza dalla configurazione scelta e' 2,8·10⁻⁴, e il t appaiato **non
le separa** (p = 0,083 su cinque seed, pur con la piu' grande avanti in 5
su 5). Il criterio era ed e' la regola 1-SE, che la 16/8 la esclude; il
t-test si riporta perche' tacerlo sarebbe lo stesso peccato al contrario.

**Il progetto tiene la 16/8, e questo non e' un risultato della selezione**:
e' il vincolo di dimensione di un microcontrollore, dichiarato come tale.
2,8·10⁻⁴ di balanced accuracy non valgono il 57% di parametri in piu' su un
dispositivo dove il modello divide 256 KB di Flash con il firmware. La
selezione resta agli atti come misura del **prezzo** di quell'eredita', che
era il motivo di eseguirla. Nessuna riga del repository dice che
l'architettura e' stata selezionata su validation, e un test fallisce se la
selezione e cio' che si deploya divergono senza che il README lo dichiari.

Due limiti, dichiarati per la stessa ragione di quelli del rapporto: il
grado 8 della single-layer **e'** confermato, ma sta sul bordo della griglia
{4, 6, 8} con curva monotona crescente (0,9704 → 0,9736 → 0,9773); e h = 32
e' la larghezza piu' grande provata. La regola ha scelto il bordo, che e'
esattamente la situazione in cui a rispondere e' la griglia e non i dati.

**2. Artifact allineati, e una regola di conteggio invece di tre.**
Il problema non era una cifra sbagliata ma una regola duplicata:
`export_e2e_int_c.py` riscriveva a mano il conteggio dei byte e sbagliava
due termini su tre — la LUT del logaritmo contata come `int16` mentre
l'header la dichiara `int32_t[256]`, e gli affini a 100 B invece di 120.
Da li' 822 B invece di 1.334, propagati in `results/e2e_int_export.csv`,
poi in `models/MANIFEST.json`, poi nel PDF e nell'audit; lo stesso era
successo al ramo a 10 classi (13.922 invece di 22.264). Ora i due
esportatori **leggono** i byte dall'header appena scritto con
`c_footprint.scan()`, la stessa funzione che conta tutti gli altri modelli,
e `tests/test_coerenza_artifact.py` fallisce se un artefatto si disallinea
o se ricompare una formula scritta a mano.

Corretti anche: `main_dt5.cpp` dichiarava 141 B contro 250 e **concludeva
che l'albero domina la KAN** (sono 285 contro 254: nessuno dei due domina);
`main_coeff.cpp` e `platformio.ini` dicevano 246 B; il macro-F1 multiclasse
0,9409 era un valore del protocollo v1 rimasto in cinque file, e ora
l'esportatore lo misura e lo scrive lui. La tabella di Pareto del README
aveva la colonna TON→BoT ferma a 3 seed, con due celle (MLP 0,4703,
XGBoost 0,5597) che **non corrispondevano ad alcun artefatto** in
`results/`, ne' a 3 ne' a 10 seed.

Un ultimo disallineamento, trovato rigenerando gli artefatti da una
macchina diversa: nessuno script dichiarava l'encoding nell'I/O testuale.
`Path.write_text()` senza `encoding=` usa la codifica **locale**, cosi'
`export_models.py` rigenerato su Windows scriveva nel MANIFEST `UNK <97>
categoria` e `clip <B1>3.5` — l'em-dash e il ±, ridotti a un byte cp1252.
Un artefatto versionato il cui contenuto dipende da chi lo rigenera e' lo
stesso difetto delle cifre disallineate, in un'altra forma. Ottanta
chiamate in trentuno file ora dichiarano `encoding="utf-8"` in lettura e
in scrittura, e `tests/test_encoding.py` fallisce se ne ricompare una
implicita o se un artefatto non-ASCII smette di essere UTF-8. Il caso
peggiore non era il MANIFEST: `results/tabella_finale_meta.json` contiene
`TON→BoT`, e la freccia in cp1252 non esiste — li' lo script non corrompe
il file, si ferma con `UnicodeEncodeError`.

Sotto c'era la stessa cosa due volte ancora. `str(Path)` usa il separatore
del sistema, quindi il MANIFEST elencava `mcu_pio\include\kan_e2e_int.h`
invece di `mcu_pio/include/kan_e2e_int.h`; e la dimensione degli header era
letta con `st_size`, che su un checkout Windows conta anche i CR —
`kan14_coeff_infer.h` risultava di 2.238 B invece di 2.184, esattamente un
byte per riga. Ora i percorsi passano da `as_posix()`, la dimensione si
misura sul contenuto normalizzato, le scritture fissano `newline="\n"` e un
`.gitattributes` impone LF nel repository indipendentemente da
`core.autocrlf`. Un test per ciascuna delle quattro regole, piu' uno che
guarda l'indice di git e non il disco.

E la stessa radice una quarta volta, dal lato dell'output. Su Windows
`sys.stdout` verso una console usa l'API Unicode e stampa qualunque
carattere; verso una pipe o un file ricade su cp1252. Cosi'
`python tools/audit_richieste.py` funzionava e
`python tools/audit_richieste.py > audit.txt` moriva con
UnicodeEncodeError sulla freccia di "TON→BoT", dopo aver gia' stampato
quaranta righe di "[ok]" — cioe' si rompeva esattamente nei due modi che
si usano per **conservare** il risultato. Ora `kanids/console.py` decide
UTF-8 una volta per tutte (senza toccare stream gia' a posto, perche'
importare `kanids` da un notebook non deve cambiare l'output altrui),
`kanids/__init__` lo applica a chi importa il pacchetto, l'audit lo chiede
per conto suo prima della prima `print`, e `reproduce.py` passa
`PYTHONIOENCODING` ai sottoprocessi che non importano `kanids`. Fra i test
ce n'e' uno che verifica che il difetto sia ancora riproducibile: se un
giorno passasse, gli altri tre non proverebbero piu' niente.

Un ultimo giro sui sette artifact che il relatore elenca ha trovato cinque
residui che i test di allineamento non coprivano, perche' guardavano i file
giusti ma non tutti. La **prima tabella del README** portava ancora
`F1 = 0.9837` del protocollo v1, mentre trenta righe sotto lo stesso file
scrive «0.9835 vs 0.9837 previously reported»; nessun test copriva quella
tabella, e adesso uno la confronta con i CSV della cross-validation.
«Tutti e sei i check host bit-esatti» era sopravvissuto in due punti alla
correzione che aveva sistemato tutti gli altri — sono cinque, il sesto
confronta contro le etichette vere e fa 39/40. `conformal_ids.py` dichiarava
il modello deployato «230 B», un numero che non corrisponde ne' ai 254 di
`footprint.csv` ne' ai 190 dei dieci edge che lo script costruisce, ed era
gia' dentro un CSV pubblicato. `kanids/config.py` giustificava ancora k=10
con un massimo della curva feature_curve che il README ha ritirato.

E soprattutto: `e2e_int_pipeline.py` era la **terza copia** della formula
dei byte, con gli stessi due termini sbagliati corretti altrove — dava
~842 B, cioe' precisamente il valore che il README dichiara superato — ed
era dentro `reproduce.py --stage integer`. Chi eseguiva la riproduzione
documentata si riscriveva il numero sbagliato accanto a quello giusto. Il
test che vietava le formule a mano elencava due file per nome: e' cosi' che
la terza copia era sopravvissuta. Adesso ne esiste uno che guarda tutti gli
script, e la prima volta che l'ho eseguito ne ha trovata una **quarta** in
`coeff_int_inference.py` — quella pero' conta un oggetto diverso, i soli
edge numerici senza tabelle categoriche, ed e' dichiarata come eccezione
con la sua ragione, col nome della colonna che lo dice.

**3. Formulazioni scientifiche ridimensionate.** Dodici affermazioni, tutte
verificate contro i CSV. Le principali:

- *"UNSW 0,8184 e' un ceiling"* — e' un riferimento in-domain di **un**
  modello con **una** soglia, e la stessa run ha ROC-AUC 0,9285: una
  quantita' che una soglia puo' spostare non e' un limite superiore. Viene
  inoltre dal progetto compagno, sotto un altro protocollo.
- *"la KAN single-layer generalizza meglio"* — ha la media piu' alta su
  TON→BoT, ma il t-test appaiato contro XGBoost da' p = 0,62 e XGBoost
  vince in 6 seed su 10. Su TON→BoT c'e' un gruppo di testa di tre modelli
  reciprocamente indistinguibili (p ≥ 0,14) significativamente sopra gli
  altri tre (p ≤ 0,0027). Nella direzione opposta la stessa KAN e' quinta
  su sei. `results/crossdomain_significativita.csv`, nuovo, contiene i 30
  confronti appaiati.
- *"LightGBM, prima in-domain, e' ultima cross-domain"* — **falsa**: e'
  quarta su TON→BoT e seconda su BoT→TON.
- *"400 alberi non stanno affatto in un ATmega2560"* — **falsa**: 60.400 B
  sono il 23,8% della Flash. Vero e' che non stanno negli 8 KB di SRAM e
  che l'unico export in C tentato non compila per AVR.
- *"MITM e' il ceiling per tutti"* — stesso errore dello 0,8184: da
  "nessuno dei sei modelli supera 0,77" si deduceva un limite dello spazio
  di feature, mentre la dispersione fra modelli su quella classe e' da
  0,151 a 0,767, un fattore cinque.

**4. Benchmark di energia** (`mcu_pio/src/main_energy.cpp`, nuovo). I sette
firmware di latenza cronometrano una inferenza per volta e fra una misura e
l'altra stampano 5-9 valori su Serial; l'integratore INA219 accumulava
proprio su quegli intervalli e chiamava l'I2C dentro il conteggio, quindi
misurava la UART. Il firmware nuovo produce due finestre adiacenti di pari
durata, marcate su un pin: una di `EB_BATCH` inferenze consecutive e nulla
altro — niente Serial, niente Wire, niente `delay`, ingressi gia' in RAM —
e una di riferimento con la CPU sveglia sui `nop`. Nove environment
PlatformIO per entrambe le schede. Poiche' togliendo la Serial sparisce
l'unica cosa che consumava il risultato, ogni predizione e' accumulata in
un `volatile` e la somma confrontata con quella attesa dai golden vector:
senza quel controllo una finestra svuotata dall'ottimizzatore sarebbe
indistinguibile da un modello velocissimo.

**5. Ambiente bloccato e tabelle riproducibili.** `requirements.txt` e
`requirements-lock.txt` divergevano in tre modi: il lock fissava
`pytest==9.1.1` contro un vincolo dichiarato `<9`; `pyarrow`, `reportlab` e
`pillow` erano solo nel lock — e **senza pyarrow i blocchi cross-domain e
joint non partono affatto**, si fermano alla prima cache parquet; `torch` e
`m2cgen` erano dichiarati obbligatori e non servono a nulla, dato che i
KAN, multi-layer compreso, sono in numpy puro. `reproduce.py` non aveva
alcuno stage per il joint training, quindi la tabella finale a sette
colonne era l'unica tabella dell'articolo non riproducibile da clone
pulito: ora ci sono gli stage `joint` e `tabelle`, e
`scripts/tabella_finale.py` la compone dai CSV invece che a mano.

**Fuori richiesta, emerso strada facendo.** Il PDF del report era
l'artefatto rimasto piu' indietro — 15 numeri fermi a 3 seed, fra cui
l'affermazione che la KAN multi-layer fosse il peggior modello in transfer,
che il README ritratta per iscritto; il suo blocco cross-domain ora legge
dai CSV. E la verifica "zero istruzioni in virgola mobile" girava su
assembly x86 dell'host: su AVR il floating point compare come chiamate
soft-float di libgcc (`__addsf3`, `__mulsf3`) e quella regex **non ne
trovava nemmeno una su codice pieno di float**. Il controllo ora le cerca,
e i cinque kernel sono compilati davvero per ATmega2560 con `avr-g++`:
zero istruzioni FP e zero chiamate soft-float.

**Un limite di riproducibilita', misurato invece che ipotizzato.** Chiudendo
il punto 5 e' emerso che `artifacts/mlcat_state.pkl` — lo stato addestrato da
cui derivano entrambi gli header C a 10 classi — non era piu' su nessuna
macchina, e non e' versionato perche' e' uno stato dell'ottimizzatore.
Riaddestrarlo funziona (`reproduce.py --stage multiclass-state`) ma **non
riproduce gli stessi pesi**: l'addestramento e' deterministico come algoritmo
— full-batch, `RandomState(0)`, nessun mescolamento — e non lo e' in virgola
mobile, perche' l'ordine delle riduzioni BLAS dipende da thread e versione, e
300 epoche di Adam amplificano gli ultimi bit. Misura: macro-F1 0,9384 contro
0,9378, F1 pesato e numero di parametri identici. Lo scarto vale uno o due
campioni MITM su 208 che cambiano lato, ed e' per questo che il pesato non si
muove: MITM e' lo 0,49% del test set.

I due header restano quindi **artefatti congelati**, verificati bit-esatti dai
check host, e i due stage che li rigenerano sono fuori da `--stage all` di
proposito. Rigenerarli dentro una riproduzione di routine sostituirebbe in
silenzio un artefatto di deployment verificato con uno equivalente ma diverso,
senza guadagnare riproducibilita'. L'unica riga corretta a mano e'
l'intestazione di `kan14_mc_coeff_int8.h`, che riportava il macro-F1 0,9409
del protocollo v1: ora dice 0,9378 e spiega perche' non e' stata rigenerata.

**Due ambienti, verificati l'uno contro l'altro.** Questa sessione e' girata
su Python 3.13.2 con numpy 2.3.4, scipy 1.16.2, lightgbm 4.6.0 e pyarrow
23.0.1: cinque pacchetti e un minor di Python diversi da
`requirements-lock.txt`. Tutto cio' che e' stato rigenerato e' venuto
identico — i due header C byte per byte, i CSV degli esportatori invariati, i
i cinque check host bit-esatti ancora 200/200. Si e' mosso un numero solo: la
stima dei byte di XGBoost, 49.905 -> 50.120 B, perche' l'ensemble passa da
9.921 a 9.964 nodi interni. Non e' xgboost, identico in entrambi gli
ambienti: sono numpy e scipy che spostano gli ultimi bit delle feature
preprocessate, e la selezione greedy degli split amplifica la differenza.
L'F1 resta 0,9989 ± 0,0001. Il valore committato e' quello dell'ambiente del
lock; la deriva e' annotata dentro il lock stesso.

La suite passa da 38 a 86 test.

## Joint training, generalizzazione e CIC-IoT-2023 (agosto 2026)

Richiesta del Prof. Kuznetsov per chiudere il primo progetto: joint training
TON_IoT + BoT-IoT, generalizzazione a un terzo dominio mai visto, CIC-IoT-2023
come quarto per nome. `adattamento-drift/` resta un progetto separato, non
integrato in questa pipeline; il README lo documenta come tale.

**Joint training** (`scripts/joint_training.py`, nuovo). Ordine imposto e
verificato da un test (`tests/test_joint_training.py`): split train/test per
dominio, bilanciamento a pari dimensione e pari rapporto normal/attack
(`balance_joint`, vincolato da BoT-IoT: ~382 normali in training su 3,67 M
flussi), unione, solo allora preprocessing e fit. **Il rapporto principale è
1:5, non 1:50**: la configurazione storica è stata testata per prima insieme
a 1:20/1:100, ma tre modelli su sei (LightGBM, XGBoost, MLP) degradavano in
modo significativo all'aumentare del rapporto senza segni di arresto fino al
pavimento testato — la griglia è stata estesa a 1:10/1:5 e la scelta rifatta
sui dati (paired t-test su 10 seed). Non spinto a 1:1: quel regime è dove
adattamento-drift ha già documentato un collasso della selezione delle
etichette.

**Generalizzazione a UNSW-NB15** (`--eval-extra unsw`), congelando feature,
preprocessing e iperparametri: nessun retraining. Il soffitto — UNSW-NB15
arriva a 0,8184 anche in-domain in questo spazio (`adattamento-drift/RISULTATI.md`,
sezione 11) — è dichiarato prima di ogni numero.

**`cross_domain.py` rilanciato a 10 seed** per TON→BoT e BoT→TON (BoT→BoT
incluso per rigore). Una conclusione della fase 2 non regge più: a 3 seed il
KAN multi-layer sembrava il peggior modello cross-domain; a 10 è MLP(16), sia
in valore assoluto sia in δ. Il pattern generale (capacità in-domain costa
transfer) si rafforza: riguarda due famiglie di modelli, non una.
TON_IoT→TON_IoT resta al protocollo originale (3 seed × 5 fold), mai
segnalato come insufficiente.

**CIC-IoT-2023** (`kanids/harmonized.py::build_ridotto_cic`, portato dalla
versione già esistente in `adattamento-drift/kanids/`; verificato — non solo
dichiarato — che nessuna feature mancante è riempita con un valore inventato:
solleva `KeyError` se le colonne vere non ci sono, e sul file reale non
scatta mai). Lo spazio comune ai quattro dataset è **6+2, non 3+2**:
`test.csv` ha una `flow_duration` genuina (mediana 26,1 s benigni contro 0,0 s
attacchi), a differenza di `Duration` che è il TTL. Il costo della riduzione,
misurato sugli stessi tre domini già analizzati nello spazio ricco con t
appaiati per seed, non contando i segni: su UNSW-NB15 lo spazio ridotto vince
in modo significativo per 3 modelli su 6 (KAN single-layer, KAN multi-layer,
LightGBM; p da <0,0001 a 0,049) e non perde in modo significativo per
nessuno — il contrario di quanto assunto in precedenza, sui modelli dove la
differenza è distinguibile dal rumore. Su TON_test la direzione non è
uniforme: KAN single-layer preferisce il ridotto (p<0,0001), LightGBM e MLP
preferiscono il ricco (p=0,0006, p=0,027) — costo comunque piccolo (≤0,036)
ma non lo stesso segno per tutti i modelli. CIC-IoT-2023
stesso resta vicino al caso (0,41–0,51 di balanced accuracy) per un modello
congiunto zero-shot, non in tensione con la misura di adattamento-drift (che
riguarda un pipeline diverso: cross-domain a singolo dominio, con
adattamento).

Un bug corretto durante il lavoro: il CSV di bilanciamento non univa con la
versione già su disco, perdendo silenziosamente un seed quando un run isolato
precedeva quello completo — stessa classe di errore già vista tre volte in
adattamento-drift. Un secondo bug (KAN costruita con `in_dim` fisso a 10
invece del numero reale di feature selezionate) è emerso solo nello spazio
ridotto, dove le candidate sono 6: corretto prima di qualunque run lungo.

Dettagli, tabelle e riproduzione: `README.md`, sezioni "Joint training" e
"A fourth dataset, in a smaller space".

## Protocollo v2 — consolidamento sperimentale (agosto 2026)

Fase di consolidamento richiesta dal Prof. Kuznetsov. Nessuna nuova variante
della KAN: solo rigore metodologico, un secondo dataset e verifica del
deployment. I sei punti della revisione e il loro stato sono in
`report_KAN-IDS_fase2.pdf`.

### 1. Protocollo leakage-free

* Selezione delle feature per mutual information, vocabolari delle categoriche
  e quantili sono ora fittati **esclusivamente sul training** di ogni fold
  (`kanids/preprocessing.py`).
* Introdotto lo **slot UNK** (indice 0) in ogni tabella categorica: rende
  l'edge una funzione totale, necessario per il cross-domain e per il runtime
  su MCU. Costo: +4 byte sul modello binario compilato (246 → 250 B).
* Il difetto era replicato in **tredici script**. Ora la selezione passa da un
  solo punto (`kanids.preprocessing.rank_by_mi`) e un test fa fallire la build
  se qualcuno ne apre un'altra copia.
* **Effetto misurato del difetto: nullo.** La selezione per-fold sceglie le
  stesse 10 feature in 15 fold su 15 e non esiste alcuna categoria non vista
  in-domain (`results/leakage_audit_stability.csv`). I numeri precedenti non
  sono invalidati; il protocollo andava corretto comunque.

### 2. Validazione uniforme, 5-fold × 3 seed

Quindici fit per modello, media ± deviazione standard, stratificazione sempre
sull'etichetta a 10 classi così che binario e multiclass condividano i fold.

| Modello | F1 binario | Macro-F1 10 classi |
|---|---|---|
| LightGBM | 0,9991 ± 0,0001 | 0,9680 ± 0,0021 |
| XGBoost | 0,9989 ± 0,0001 | 0,9666 ± 0,0021 |
| KAN multi-layer | 0,9976 ± 0,0002 | 0,9374 ± 0,0036 |
| MLP (16) | 0,9964 ± 0,0009 | 0,9182 ± 0,0107 |
| Decision Tree (d=5) | 0,9944 ± 0,0004 | 0,7633 ± 0,0033 |
| KAN single-layer | 0,9835 ± 0,0007 | 0,8767 ± 0,0014 |

### 3. Cross-domain TON_IoT ↔ BoT-IoT

Spazio armonizzato a 13 feature candidate calcolate con la stessa formula sui
due dataset; nel cross-domain il target entra solo nella valutazione. Quattro
direzioni, più l'ablation senza categoriche.

**Il transfer collassa**: TON→BoT lascia ogni modello fra 0,40 e 0,56 di
balanced accuracy, cioè al caso o sotto. Il δ del paper precedente era ≤ 5,95
punti; qui è di 40–59 punti.

### 4. Correzioni alle conclusioni precedenti

Due claim del README sono risultati non sostenibili e sono stati corretti:

* Il confronto con le baseline metteva la KAN sullo spazio grezzo a 14 feature
  e gli alberi su quello derivato a 10 del paper. **Con input identici
  LightGBM fa 0,9991 contro 0,9835.**
* A parità di regola di conteggio l'albero profondo 5 occupa **141 byte** con
  F1 0,9944: più piccolo *e* più accurato del modello KAN da 250 byte, che è
  **dominato** sulla frontiera di Pareto in-domain (`results/footprint.csv`).

Cosa regge: la multi-layer sta sulla frontiera (5,2 KB, 0,9976, contro l'MLP
TFLite del paper a 13 KB e 0,9959 con 95 feature); e nel cross-domain la
classifica si ribalta.

### 5. Inferenza integer-only end-to-end, in C

Entrambe le catene — binaria e a 10 classi — vanno dai contatori grezzi alla
decisione in aritmetica intera, verificate **bit per bit** contro il
riferimento Python su 200 golden vector, con **zero istruzioni in virgola
mobile** nell'assembly del percorso di inferenza (`tools/check_no_float.sh`).

Il percorso precedente (`mcu_e2e/`) interpolava 10.000 knot del
QuantileTransformer in doppia precisione: end-to-end nella struttura, non
integer-only nel runtime. È conservato solo per riferimento.

Difetti emersi dal porting, tutti invisibili in Python:
* moltiplicatore di scala che quantizza a esattamente 32768 → overflow
  silenzioso in `int16` (due occorrenze distinte);
* divisione intera: Python arrotonda verso −∞, il C tronca verso zero;
* soglie a 64 bit necessarie perché `src_bytes` e `dst_bytes` superano 2³¹.

### 6. Riproducibilità

* 42 riferimenti a `/tmp` in 21 script → `artifacts/` dentro il repository.
* Cache con **invalidazione automatica** per impronta di configurazione.
* `reproduce.py` con stage dichiarati; `--stage smoke` esegue l'intera catena
  su dati sintetici in ~20 s **senza scaricare nulla**.
* `models/` versionata con `MANIFEST.json` (protocollo, seed, spazio di
  feature, metriche); `artifacts/` è cache cancellabile.
* **30 test** impediscono la ricomparsa dei difetti: percorsi `/tmp`, percorsi
  assoluti, dipendenze non vincolate, selezione feature fuori da `kanids`,
  virgola mobile negli header integer, risultati v1 mescolati ai correnti.
* I risultati prodotti prima della correzione sono in `results/protocol_v1/`
  con un README che spiega quali numeri sono superati.

### 7. Indipendenza della stima, misurata

`scripts/nested_cv.py` esegue una **cross-validation annidata**: dentro ogni
fold esterno una CV interna sceglie k sul solo training, e la valutazione
avviene su dati che non hanno partecipato alla scelta. La differenza rispetto
alla stima piatta e' l'ottimismo di selezione.

**Risultato: l'ottimismo e' negativo.** 0,9845 ± 0,0006 contro 0,9835 per la
KAN single-layer (−0,0009) e 0,9992 contro 0,9991 per LightGBM (−0,0001). I
numeri pubblicati sono semmai conservativi.

La misura smentisce pero' un altro claim: la selezione interna **non sceglie
mai k = 10**, prende tutte e 16 le feature candidate in 15 fold su 15. La curva
e' monotona, non ha un picco. **k = 10 e' una scelta di deployment** — dieci
statistiche di flusso a bordo invece di sedici — e costa 0,0009 di F1.

### 8. La catena end-to-end e' anche deployata, non solo verificata

`mcu_pio/src/main_e2e.cpp` e' una variante di firmware che parte dai
**contatori grezzi** ed esegue a bordo l'intera catena, feature engineering
compreso, sotto il protocollo di benchmark del paper. Tutte le altre varianti
ricevono vettori gia' normalizzati fuori dal dispositivo: senza questa, la
catena integer sarebbe dimostrata corretta ma non sarebbe mai stata *la*
pipeline in esecuzione sull'MCU.

Firmware e harness di verifica includono lo **stesso** kernel
(`mcu_pio/include/kan_e2e_infer.h`): cio' che e' verificato bit per bit e'
cio' che gira sulla board, non una copia che puo' divergere.

### 9. Verifiche che impediscono le regressioni

* `tests/test_leakage.py::test_crossdomain_target_does_not_influence_training`
  fitta la pipeline due volte sullo stesso source con target radicalmente
  diversi e pretende che tutto l'appreso sia identico bit per bit. Introdurre
  la violazione classica (fit su source ∪ target) lo fa fallire.
* `requirements-lock.txt` con le versioni esatte dell'ambiente che ha prodotto
  i numeri pubblicati, piu' un test che impedisce alla promessa di tornare a
  vuoto: prima `requirements.txt` citava un lock che non esisteva.
* `tools/audit_richieste.py` (`reproduce.py --stage audit`) verifica
  meccanicamente i sei punti della revisione e stampa, requisito per
  requisito, l'evidenza che lo sostiene. **26 su 27**; l'unico non fatto e'
  CIC-IoT-2023, indicato come obiettivo secondario.

La suite e' passata da 22 a **30 test**, e l'audit da 23 a **27 controlli**.

### 10. Ogni modello e' esportato in C e flashabile

Prima alcuni modelli esistevano come header ma senza un `main` che li usasse:
esportati sulla carta, non testabili fisicamente. Ora sono **sette su sette**,
ognuno con il suo environment PlatformIO, verificato da un test che fallisce se
qualcuno aggiunge un header senza il firmware corrispondente.

Fra questi il **Decision Tree profondo 5** (`main_dt5.cpp`), che non era mai
stato esportato pur essendo il modello che **domina** la KAN da 250 byte sulla
frontiera in-domain. Senza, l'obiezione piu' seria al lavoro non sarebbe stata
chiudibile su hardware. L'export ha gia' prodotto un risultato: quantizzare le
soglie a Q7 per il dispositivo costa all'albero **0,0028 di F1**
(0,9944 -> 0,9916, agreement 99,55%), e il divario con la KAN compilata scende
da 0,0109 a **0,0081**. La KAN paga la quantizzazione meno dell'albero, ed e'
un argomento che prima non avevamo perche' confrontavamo un albero float con
una KAN gia' quantizzata.

Aggiunto anche il firmware per la catena end-to-end a 10 classi
(`main_mc_e2e.cpp`), che esisteva solo come verifica offline.

### 11. Coerenza degli artefatti deployati

L'export in C del modello multiclass era rimasto al **protocollo v1**: tabelle
categoriche con 28 righe (3+9+13+3), senza slot UNK, contro le 32 (4+10+14+4)
del protocollo attuale. Il firmware girava quindi su un modello incompatibile
con il preprocessing v2 — senza errori visibili, perche' modello e vettori di
test erano coerenti fra loro. Rigenerato, e un test ora vieta la presenza di
qualunque header con tabelle a 28 righe.

Corretto anche `main.cpp`, che era l'unico firmware non compilabile su host e
quindi l'unico che nessuno poteva verificare prima di flasharlo. Un test ora
compila tutti e sette i firmware a ogni esecuzione della suite.

### Cosa resta aperto

* Benchmark fisici su Mega 2560 ed ESP32-C3: latenza, SRAM, energia, code size.
  Tutti i firmware necessari esistono e sono compilabili; manca l'esecuzione
  sulle board. `results/latency_benchmark.csv` contiene le misure del paper
  precedente, relative alle varianti v1.
* Larghezza nascosta, grado e clip non sono stati riselezionati dentro il
  ciclo: restano ereditati dalla fase precedente. L'effetto complessivo
  dell'esposizione ai dati e' pero' misurato ed e' sotto il millesimo di F1,
  in direzione conservativa (vedi punto 7).
* Analisi di sensibilità sul rapporto di undersampling (fissato a 1:50).
* CIC-IoT-2023 come terzo dataset (obiettivo secondario).

# Le sei richieste, valutate sul repository

Audit condotto su un clone pulito di `main` (`b502227`), leggendo le prove
nei file invece di affidarsi allo strumento di audit del progetto. Voti su
10 per **svolgimento** (è stato fatto?), **professionalità** (è stato fatto
bene?), **completezza** (manca qualcosa?).

| Punto | Svolgimento | Professionalità | Completezza |
|---|---|---|---|
| 1 — Protocollo leakage-free | 10 | 10 | 9 |
| 2 — KAN multi-layer | 10 | 10 | 9 |
| 3 — Secondo dataset BoT-IoT | 10 | 9 | 10 |
| 4 — Baseline identiche | 10 | 9 | 9 |
| 5 — Integer-only end-to-end | 10 | 10 | 10 |
| 6 — Riproducibilità | 9 | 8 | 7 |

**Media 9,4.** Il punto 5 è impeccabile, il punto 6 è quello con il divario
più grande fra quello che il lavoro contiene e quello che un lettore esterno
riesce a raggiungere.

---

## 1. Protocollo leakage-free — 10 / 10 / 9

**Cosa c'è.** Quantili, vocabolari delle categoriche e selezione per mutual
information sono fittati esclusivamente sul training di ogni fold
(`kanids/preprocessing.py`). La selezione passa da un solo punto,
`rank_by_mi`, e un test fa fallire la build se qualcuno ne apre un'altra
copia — il difetto originale era replicato in tredici script. Protocollo
5-fold × 3 seed: **15 fit per modello**, media e deviazione standard su
tutte le metriche (`results/cv_leakagefree_summary_*.csv`). Trenta test,
fra cui un permutation test che riproduce il difetto v1 e deve fallire se
il difetto rientra.

**Oltre la richiesta.** È stata eseguita anche una **cross-validation
annidata** (`scripts/nested_cv.py`) per misurare l'ottimismo di selezione:
risulta **−0,0009**, cioè negativo — i numeri pubblicati sono semmai
conservativi. Il professore non l'aveva chiesta; è la prova più forte che
l'indipendenza della valutazione non è solo dichiarata.

**Il punto debole.** Larghezza nascosta, grado della spline e clip **non
sono stati riselezionati dentro il ciclo**: restano ereditati dalla fase
precedente, quindi hanno visto i dati. L'effetto complessivo è misurato ed
è sotto il millesimo di F1, in direzione conservativa, ed è dichiarato nel
CHANGELOG — ma resta l'unica trasformazione che usa informazione dai dati
non riappresa nel protocollo, che è letteralmente ciò che il punto 1
chiede.

**Cosa farei.** Una CV annidata anche su quei tre iperparametri, o in
alternativa una frase esplicita nel README (non solo nel CHANGELOG) che
dichiari il residuo e ne citi la misura. Oggi un revisore che legge solo il
README non lo scopre.

---

## 2. KAN multi-layer con lo stesso protocollo — 10 / 10 / 9

**Cosa c'è.** `KAN(cat,ML)` è nella stessa tabella 5×3 degli altri, con
**F1 = 0,9976 ± 0,0002** su 15 fit e macro-F1 multiclass 0,9374 ± 0,0036.
Il ≈0,997 che il professore voleva veder confermato è confermato con la
stessa procedura statistica, non con una rieccuzione separata.

**Il punto debole.** La CV annidata del punto 1 copre **solo `KAN(cat,1L)`
e LightGBM**, non la multi-layer. Il modello di cui il professore chiedeva
esplicitamente la conferma statistica è quindi l'unico dei principali senza
la misura più rigorosa disponibile nel repository.

**Cosa farei.** Estendere `nested_cv.py` a `KAN(cat,ML)`. È lo stesso
codice, un modello in più nella lista: mezz'ora di calcolo per chiudere
un'asimmetria che si nota.

---

## 3. Secondo dataset BoT-IoT — 10 / 9 / 10

**Cosa c'è.** Tutti e quattro gli esperimenti richiesti, con i nomi esatti
della richiesta: `ton->ton`, `bot->bot`, `ton->bot`, `bot->ton`, in task
binario normal-vs-attack, su 6 modelli × 3 seed.

Lo spazio armonizzato è la parte migliore: **13 feature numeriche derivate
con la stessa formula sui due dataset** più 2 categoriche su alfabeto
comune, con le esclusioni motivate una per una — porte e indirizzi (sono
identificatori del testbed), aggregati a finestra di BoT-IoT (non hanno
corrispettivo), metadati DNS/SSL/HTTP di TON_IoT (BoT-IoT non li ha). E il
costo dell'esclusione è **misurato**, non assunto: togliere porte e
indirizzi costa 0,0008 di F1 a LightGBM e fa *guadagnare* 0,0025 alla KAN.

Il vincolo cross-domain non è dichiarato, è **imposto da un test**: si fitta
due volte sullo stesso source con target radicalmente diversi e si pretende
che tutto l'appreso sia identico bit per bit. Introducendo la violazione
classica (fit su source ∪ target) il test fallisce.

**Oltre la richiesta.** CIC-IoT-2023 è stato fatto — l'obiettivo secondario
— e prima di adottarlo ne è stato misurato il costo: mancandogli i conteggi
direzionali, sette delle tredici feature cadrebbero, e quella riduzione
costa 0,009 in-domain ma **fino a 0,29 sul risultato adattato**. Da lì la
scelta di aggiungere **UNSW-NB15** come terzo dominio nello spazio ricco e
CIC come quarto nel suo. Il professore aveva scritto «preferisco due
dataset analizzati correttamente piuttosto che tre analizzati
superficialmente»: qui ce ne sono quattro, e la ragione per cui il terzo
non è quello che aveva indicato è una misura.

**Il punto debole.** Il cross-domain della richiesta gira su **3 seed e
senza fold** (18 run = 6 modelli × 3 seed), contro i 90 in-domain. Il
lavoro successivo sull'adattamento ha poi dimostrato, sugli stessi dati,
che **3 seed possono ingannare**: l'affermazione «questa direzione
fallisce sempre» si è rivelata un artefatto del campione in più di un caso.
Le tabelle cross-domain che il professore leggerà sono ancora quelle a 3
seed, mentre il sottoprogetto le ha rifatte a 10.

**Cosa farei.** Rilanciare `scripts/cross_domain.py` a 10 seed. È lo stesso
comando con `--seeds` diverso, ed elimina l'incoerenza fra la tabella
principale e quella del sottoprogetto.

---

## 4. Baseline identiche — 10 / 9 / 9

**Cosa c'è.** Sei modelli sullo stesso spazio di feature: KAN single-layer,
KAN multi-layer, LightGBM, XGBoost, DecisionTree(d=5), MLP(16) — i cinque
richiesti più uno. Tutte le metriche chieste sono presenti e per entrambi i
task: `f1`, `precision`, `recall`, `pr_auc` nel binario;
`macro_f1`, `macro_precision`, `macro_recall`, `pr_auc_macro` nel
multiclass, ciascuna con la sua deviazione standard.

Le **matrici di confusione nel cross-domain** ci sono (colonne `tn/fp/fn/tp`
in ogni run, 24 righe di sintesi), e c'è materiale sul perché degrada:
sovrapposizione delle marginali per feature
(`results/crossdomain_shift.csv`, minimo 0,085 su `byte_rate`), tasso di
categorie non viste, ablation con e senza edge categorici, e una tabella di
degrado per modello (`crossdomain_degradation.csv`).

Un risultato onesto che vale la pena notare: il confronto ha **smentito due
claim precedenti del progetto** — con input identici LightGBM fa 0,9991
contro 0,9835 della KAN single-layer, e l'albero profondo 5 occupa 141 byte
con F1 0,9944, cioè **domina** la KAN da 250 byte sulla frontiera in-domain.
Sono scritti nel CHANGELOG invece che nascosti.

**Il punto debole, ed è il più significativo dei sei.** Il professore ha
scritto: «è particolarmente importante analizzare anche le confusion matrix
e **capire perché il modello degrada**, non soltanto riportare il numero
finale». La risposta nell'analisi principale è **descrittiva**: le
distribuzioni non si sovrappongono. Vera, ma è una constatazione, non una
spiegazione.

La spiegazione vera esiste, ed è nel sottoprogetto: **gli edge sopravvivono
al drift, è la loro combinazione che si rompe** — in `ton→bot` il modello è
al caso (ROC-AUC 0,54) eppure i suoi stessi edge, solo ripesati, arrivano a
0,92; e con tre domini emerge che in due direzioni il ROC-AUC è 0,26–0,27,
cioè l'ordinamento è **sistematicamente rovesciato**, il modello usa
l'informazione col segno sbagliato. È esattamente la risposta che il punto
4 chiedeva, ed è la cosa più interessante di tutto il lavoro — ma non
compare da nessuna parte nella sezione che risponde al punto 4.

**Cosa farei.** Un paragrafo nel README e nel CHANGELOG, sotto il punto 4,
che dia la risposta mecanicistica e rimandi a `adattamento-drift/`. Costa
dieci righe e trasforma la risposta al punto 4 da sufficiente a forte.

---

## 5. Integer-only end-to-end — 10 / 10 / 10

**Cosa c'è.** Entrambe le catene, binaria e a 10 classi, vanno dai contatori
grezzi alla decisione in aritmetica intera, verificate **bit per bit** su
200 golden vector, con **zero istruzioni in virgola mobile** nell'assembly
del percorso di inferenza (verificato su assembly, non su sorgente).

Sono le tre scelte a monte a rendere questo punto il migliore dei sei:

- **Firmware e harness di verifica includono lo stesso header**
  (`kan_e2e_infer.h`): ciò che è verificato bit per bit è ciò che gira sulla
  board, non una copia che può divergere.
- **Sette modelli su sette hanno un firmware flashabile**, ognuno con il suo
  environment PlatformIO, e un test fallisce se qualcuno aggiunge un header
  senza il firmware corrispondente. Fra questi il Decision Tree d=5, cioè
  il modello che *batte* la KAN sulla frontiera in-domain: senza,
  l'obiezione più seria al lavoro non sarebbe chiudibile su hardware.
- **Gli host check compilano e girano senza dataset e senza Python**, che è
  esattamente il vincolo del punto 5 sul ruolo di Python.

Il percorso precedente (`mcu_e2e/`) interpolava 10 000 knot in doppia
precisione — end-to-end nella struttura, non integer-only nel runtime — ed è
conservato solo per riferimento, dichiarato come tale. I tre difetti emersi
dal porting (moltiplicatore che quantizza a 32768 → overflow silenzioso in
int16; divisione intera che arrotonda verso −∞ in Python e tronca verso zero
in C; soglie a 64 bit perché `src_bytes` supera 2³¹) sono documentati: sono
tutti invisibili in Python, ed elencarli è ciò che distingue un porting
verificato da uno sperato.

**Non ho trovato punti deboli sostanziali.** L'unico appunto è cosmetico:
`tools/check_no_float.sh` richiede un argomento e fallisce se invocato senza
— vale la pena dargli un default, perché è lo script che sostiene
l'affermazione centrale del punto.

---

## 6. Riproducibilità — 9 / 8 / 7

**Cosa c'è.** L'organizzazione richiesta è tutta presente: `scripts/`,
`models/` con `MANIFEST.json`, `results/`, `mcu_pio/` per il firmware,
`requirements.txt` più `requirements-lock.txt` con le versioni esatte
(14 pacchetti pinnati). `reproduce.py` con **17 stage dichiarati**, di cui
`--stage smoke` esegue l'intera catena su dati sintetici in ~20 secondi
**senza scaricare nulla**. I seed sono in `models/MANIFEST.json`
(`seed_split: 42`, `seed_cross_validation: [42, 43, 44]`) e stampati a ogni
run. Quarantadue riferimenti a `/tmp` in 21 script sono stati migrati in
`artifacts/`, e un test impedisce che rientrino. I risultati prodotti prima
della correzione sono in `results/protocol_v1/` con un README che spiega
quali numeri sono superati.

**Il punto debole, ed è il più serio dell'intero audit.** Il sottoprogetto
`adattamento-drift/` — 114 file, 4,1 MB, diciannove sezioni di risultati, la
parte più originale del lavoro — è **invisibile all'infrastruttura di
riproducibilità della radice**:

| | occorrenze di `adattamento-drift` |
|---|---|
| `reproduce.py` | **0** |
| `README.md` | **0** |
| `CHANGELOG.md` | **0** |
| test in `tests/` | **0** |

Chi clona il repository e segue le istruzioni — che è precisamente lo
scenario del punto 6 — riproduce la fase 2 e non sa che il resto esiste. Il
sottoprogetto ha un suo README serio, ma bisogna già sapere di aprirlo. In
più i suoi script leggono i dataset da una variabile d'ambiente
(`KANIDS_DATA`) documentata solo lì dentro, quindi non c'è un percorso
unico dichiarato dalla radice.

**Cosa farei, in ordine di rapporto valore/costo:**

1. **Uno stage `drift` in `reproduce.py`** che lanci almeno
   `cross_domain.py` e `drift_int_adapt.py` con il seed documentato. È la
   differenza fra «il repository si riproduce» e «la fase 2 del repository
   si riproduce».
2. **Un paragrafo nel README** che dica che esiste, cosa contiene e dove
   sono i suoi risultati. Dieci righe.
3. **Una voce nel CHANGELOG**, dato che tutte le altre fasi ne hanno una.
4. **Un test minimo** che verifichi almeno che i suoi script si importino
   e che l'header MCU sia coerente — oggi la suite da 30 test non tocca la
   parte più nuova del lavoro.
5. Documentare `KANIDS_DATA` nel README della radice, non solo in quello
   del sottoprogetto.

---

## La chiusura della richiesta, che vale come settimo punto

Il professore aveva concluso: «*se il cross-domain test mostrerà un degrado
significativo — cosa che considero molto probabile e scientificamente
interessante — avremo già la direzione naturale per il passo successivo:
studiare una KAN integer-only capace di adattarsi al domain/concept drift
direttamente su dispositivi embedded, aggiornando localmente solo una
piccola parte dei coefficienti*».

Il degrado c'è stato, ed è generale: sei direzioni su sei, da 0,22 a 0,74 di
balanced accuracy contro 0,82–0,99 in-domain. E il passo successivo **è già
stato fatto**, nei termini esatti in cui era stato descritto: la KAN
single-layer è additiva, quindi ripesare un guadagno per edge più un termine
noto sono **13 coefficienti, 24 byte** — «una piccola parte dei
coefficienti», letteralmente — che con 32 etichette recuperano in cinque
direzioni su sei. L'intera catena gira senza un solo float ed è verificata
bit-esatta. Il riadattamento continuo in aritmetica intera batte il modello
statico in **6 direzioni su 6**, con t appaiati per seed da 3,3 a 91,5.

Non era richiesto entro fine agosto. È l'anticipo del passo che il
professore aveva già indicato, con la stessa disciplina metodologica delle
sei richieste — inclusa una sezione che elenca le proprie affermazioni
cadute quando il campione è stato allargato da 3 a 10 seed.

**Il paradosso da risolvere è che questa parte è quella meno raggiungibile
dal repository.** Le cinque cose elencate al punto 6 costano complessivamente
poche ore e cambiano il modo in cui il lavoro viene letto.

# Changelog

## Terza revisione: verso v2.1-rc2 (agosto 2026)

Sette richieste del Prof. Kuznetsov prima delle misure sulle schede. Questa
voce cresce man mano che vengono soddisfatte.

**Benchmark di energia: calibrazione sullo stesso ciclo, e le due energie
separate.** La finestra di riferimento veniva calibrata cronometrando un
ciclo diverso da quello poi eseguito — con una chiamata a `micros()` per
giro, che su AVR costa circa 3,4 us. In 20 ms si contavano cosi' ~5.900 giri
invece dei ~320.000 di un ciclo di soli `nop`, la divisione intera dava zero
e il guard la portava a uno: la finestra di riferimento durava un sedicesimo
di quella attiva, mentre `E = (P_alta - P_bassa) * T / N` presuppone che le
due durate coincidano. Adesso calibrazione e misura passano per la stessa
funzione `eb_nop_loop()`, `micros()` viene chiamato due volte in tutto e
fuori dal ciclo, e il rapporto sta in virgola fissa Q8. Il CSV riporta anche
la durata **misurata** della finestra di riferimento e lo scarto fra le due
in parti per mille: "stessa durata" era una promessa e adesso e' un numero.
L'intestazione distingue esplicitamente l'energia **totale** per inferenza
(`P_alta * T_alta / batch`) da quella **dinamica** rispetto al baseline
(`(P_alta - P_bassa) * T_alta / batch`), e dice quale delle due include il
consumo statico del core sveglio.

**La KAN multi-layer entra fra i firmware consegnati.** Gli environment di
energia in `platformio.ini` erano nove, ma l'elenco di
`scripts/pacchetto_finale.py` ne includeva sei: il modello che il relatore
considera il miglior compromesso KAN non finiva nel pacchetto delle misure.
Un test confronta adesso quell'elenco con `platformio.ini`.

**L'MLP piccolo esportato in C intero: la baseline densa che mancava a
bordo.** Il confronto sul dispositivo era fra albero, KAN single-layer, KAN
multi-layer e LUT. La rete densa — cioe' proprio l'architettura che la KAN
vuole sostituire — esisteva solo in cross-validation, con i byte **stimati**
a un byte per parametro. `scripts/export_mlp_int_c.py` quantizza lo stesso
fit che `footprint.py` gia' addestrava (stesso seed, stesso split, stesso
wrapper: nessun iperparametro nuovo) e produce `mlp16_int8.h`,
`mlp16_infer.h`, `mlp16_test_vectors.h`, un check host, `main_mlp.cpp`, la
variante `EB_MLP` del firmware di energia e quattro environment.

I byte passano da **705 stimati a 760 misurati**. I 55 di differenza sono i
bias del primo layer, tenuti in int32, e le righe della tabella in cui le 32
colonne one-hot del design vengono compilate. E' lo stesso scarto gia' visto
sull'albero — 141 stimati contro 285 misurati — e nella stessa direzione.

Due scelte riguardano cosa viene misurato, non la matematica. Il one-hot non
esiste a bordo: ogni categorica seleziona una riga di `MLP16_CAT`, una somma
per nascosto invece di 32 moltiplicazioni, altrimenti si misurerebbe una
trasposizione ingenua e non l'MLP. E tutti gli accumulatori stanno in int32:
l'attivazione nascosta viene ridotta di `MLP16_HSHIFT` bit prima del secondo
layer, perche' con l'accumulatore a 64 bit il kernel chiamerebbe `__adddi3`,
`__ashrdi3` e `__mulsidi3` di libgcc su AVR — la latenza misurata sarebbe
quella di un tipo che il processore non ha. Lo shift esce da un bound
calcolato sui pesi quantizzati e su |xq| <= 2^12, non dai dati; se il bound
non entrasse in int32 l'esportatore si ferma invece di scrivere un header che
sborda solo su certi ingressi.

La verifica non aspetta i dati: `tests/test_mlp_int.py` costruisce un MLP con
pesi casuali, lo quantizza con la stessa funzione dell'esportatore, ne emette
l'header e compila **il kernel vero del repository**, confrontando il *logit*
— non la predizione — su 400 vettori che coprono tutto il dominio degli
ingressi. Un logit uguale ovunque e' molto piu' forte di una predizione
uguale: la predizione e' un bit, e due implementazioni diverse la azzeccano
lo stesso quasi sempre. C'e' anche il controllo del controllo: spostando di
uno un solo peso, il confronto deve fallire.

Trovati verificando: `models/MANIFEST.json` elencava gli "header C
deployabili" con un glob `kan*.h`, quindi lasciava fuori `dt5_model.h` — il
modello con cui la KAN viene confrontata piu' spesso — e avrebbe lasciato
fuori anche l'MLP; `scripts/export_tree_c.py` non compariva in nessuno stage
di `reproduce.py`, cioe' l'header dell'albero esisteva nel repository ma
nessuna riproduzione lo rigenerava (adesso c'e' lo stage `baseline-c`); e la
regex delle costanti scalari di `c_footprint.py` non prevedeva `PROGMEM` e
avrebbe saltato in silenzio il bias di uscita.

**La terza forma dello stesso difetto: `to_csv` e i terminatori di riga.**
Trovata da un warning di git durante il commit di questa stessa revisione.
`DataFrame.to_csv(percorso)` usa `os.linesep`: CRLF su Windows, LF altrove, e
ottanta chiamate del progetto lo facevano. Le prime due forme — l'encoding non
dichiarato e `newline=None` in `write_text` — erano state chiuse credendo di
aver chiuso il problema.

Il sintomo qui era piu' insidioso delle altre due. `.gitattributes` normalizza
in fase di commit, quindi il file NELL'INDICE risulta pulito e il test a valle
passava; a differire era il file SUL DISCO. Il che significa che le somme
SHA-256 stampate in `SOMME.sha256` dentro il pacchetto consegnato dipendevano
dal sistema che lo aveva costruito: chi avesse ricostruito il pacchetto su
un'altra macchina avrebbe trovato somme diverse a parita' di contenuto. Il
requisito "gli artefatti non cambiano a seconda della macchina che li
rigenera" era verde ed era falso.

Ottanta chiamate dichiarano adesso `lineterminator="\n"`, un test le conta
sull'AST (`to_csv(` compare anche nei commenti), e un secondo test verifica
che il difetto sia ancora riproducibile: se pandas smettesse di usare
`os.linesep`, il primo non starebbe piu' impedendo niente.

**Due file di appoggio finiti nel repository.** `commit_v21rc2.txt` e
`tag_v21rc2.txt` sono stati committati da un `git add -A`: `.gitignore`
copriva `*.patch` ma non i messaggi. Sono artefatti del processo, non del
progetto, e in un repository consegnato a un revisore dicono solo che nessuno
ha guardato cosa stava committando. Adesso sono ignorati e un test fallisce se
ne ricompaiono.

**Gli host check non compilavano dal pacchetto estratto.** Era la richiesta
piu' concreta della lista, ed era anche vera: gli header finivano in
`header_c/` mentre i sorgenti li cercano in `../include/`, cioe' con il
percorso relativo che hanno nel repository. Il primo comando che un lettore
prova — `cd host_check && g++ -O2 -o check run_coeff_check.cpp` — moriva su un
include mancante. La cartella del pacchetto si chiama adesso `include/`, i tre
check che usavano include senza percorso sono stati normalizzati agli altri, e
la sottocartella `avr/` con lo stub di `pgmspace.h` viene copiata anche lei.
Sei check su sette compilano ed eseguono dal pacchetto **senza una sola
opzione**; il settimo e' quello dell'MLP, che aspetta i suoi header generati.
Un test costruisce il pacchetto, lo copia altrove per togliergli il
repository intorno, e li compila ed esegue tutti da li'.

**CIC-IoT-2023 era ancora descritto nello spazio sbagliato.** Nella sezione
delle limitazioni il README diceva che la valutazione su CIC "is restricted to
the 10 harmonised numeric features", e aggiungeva che le categoriche restano
fuori. Nessuna delle due cose e' vera da quando lo spazio ridotto e' stato
misurato: sono **sei numeriche piu' i due edge categorici**, il 6+2, e la
sezione che lo spiega sta trenta pagine sopra nello stesso file. Un test
adesso legge ogni riga che nomina CIC insieme a un conteggio di feature e
pretende il 6+2.

**Il PDF diceva che una cosa fatta non era iniziata.** La tabella di stato del
report elencava "CIC-IoT-2023 come terzo dataset: non iniziato (obiettivo
secondario)" mentre i risultati su CIC erano gia' otto file in `results/`. Una
tabella di stato sbagliata e' peggio di nessuna tabella: dice al lettore di
non cercare. Ora elenca anche le quattro voci nuove di questa revisione, e un
test fallisce se dichiara "non iniziato" qualcosa i cui artefatti esistono.
L'audit segnala inoltre quando il PDF e' piu' vecchio dei CSV che cita.

**Conteggi invecchiati in silenzio.** "I sette firmware di latenza" erano
diventati otto con l'MLP, "sono nove environment" erano diventati undici. Sono
frasi che stanno esattamente dove un lettore cerca di capire quante cose deve
flashare. Un test le conta da `platformio.ini` e da `src/` invece di
rileggerle.

**L'interpretabilita' della single-layer, senza spiegazioni post-hoc.** Il
kernel deployato somma quattordici termini e nient'altro: dieci edge spline
sulle numeriche e quattro tabelle sulle categoriche, senza interazioni e senza
bias residuo. La scomposizione per feature non e' quindi una *stima* del
contributo — che e' cio' che SHAP, LIME e le mappe di salienza producono,
approssimando una funzione opaca con un surrogato locale — ma sono gli addendi
stessi della somma che il microcontrollore esegue.

E' un'affermazione verificabile, e viene verificata nel modo piu' duro
disponibile: `tests/test_interpretabilita.py` compila il kernel C vero, lo
esegue sui 200 vettori committati e pretende che i quattordici addendi
sommino al logit `int32` del kernel su **200 casi su 200**. Non "circa": lo
stesso intero. Un test fallisce anche se `shap`, `lime`, `captum` o `eli5`
comparissero fra le dipendenze, perche' dichiarare una spiegazione diretta e
dipendere da uno strumento che ne stima una sarebbe una contraddizione.

Le due figure sono `figures/fig_kan_funzioni_apprese.png` (le quattordici
funzioni) e `figures/fig_kan_contributi_locali.png` (tre flussi reali: un
attacco netto, un normale netto e quello piu' vicino alla soglia). Il terzo e'
il piu' istruttivo: il suo logit vale l'uno per cento di quello del caso netto
ed e' il residuo di termini che si oppongono — `duration` e `proto` spingono
verso *normale*, `dst_ip_bytes`, `src_pkts` e `dns_rejected` verso *attacco*.
Un modello che emettesse solo un punteggio direbbe "attacco, appena"; questo
dice quali quattro termini dovrebbero muoversi, e di quanto.

Niente di tutto questo richiede il dataset: coefficienti e vettori stanno
negli header committati.

**Due formulazioni tenute prudenti, di proposito.** Le curve apprese oscillano
— grado 8 senza penalizzazione di smoothness, compilato su 16 segmenti — e il
README lo dice: per un dato ingresso il contributo e' esatto, ma la forma
*fra* i modi della densita' di training non e' la prova di una legge del
dominio. E per la KAN multi-layer una scomposizione additiva esatta **non
esiste**, perche' il secondo strato vede combinazioni delle unita' nascoste:
lo strumento si rifiuta di produrre una figura per quel modello invece di
generarne una che le somiglia e non lo e'. Cio' che si puo' dire di quel
modello e' l'affermazione piu' debole e vera: e' piu' accurato (F1 0,9976
contro 0,9835) e non e' direttamente interpretabile.

**La statistica: l'unita' di analisi era sbagliata, e in due modi diversi.**
La richiesta era di evitare p-value estremi ottenuti trattando fold ripetuti
come osservazioni indipendenti. Guardando i run archiviati il difetto e'
risultato piu' profondo della frase.

*Primo modo, quello nominato.* La selezione del rapporto confrontava 120
"coppie" appaiate: 10 seed x 6 modelli x 2 domini in una lista sola. Modelli e
domini non sono repliche della stessa quantita', e il criterio dichiarato e'
gia' la loro media: l'unita' e' il seed, e i seed sono dieci. Con dodici volte
meno gradi di liberta' i p passano da 9,6·10⁻⁸ a 2,7·10⁻⁷ nel caso estremo e
a 6,2·10⁻³ nel piu' stretto — e restano separabili anche dopo la correzione di
Holm sulla famiglia dei quattro candidati. La conclusione non cambia, perche'
non dipendeva da quei p: 1:5 e' l'argmax della media e **vince in 10 seed su
10** contro ogni candidato, che e' la parte dell'evidenza che non assume
niente. Il conteggio "78/120" era piu' impressionante e diceva meno.

*Secondo modo, trovato leggendo i run.* Nella direzione ton->bot `n_train` ha
UN solo valore (211.043, tutto TON) e `n_test` UN solo valore (3.668.522,
tutto BoT-IoT): i dieci seed riaddestrano sugli stessi identici dati e
valutano sugli stessi identici dati. La dispersione fra seed e' quindi
variabilita' di **riaddestramento**, non di campionamento, ed e' piccola per
costruzione: e' cosi' che nasceva un t = -58. Quel test resta negli artefatti
perche' e' informativo, ma con scritto in ogni riga che cosa misura, e la
domanda a cui non risponde — "questo modello generalizza meglio" — ha una
coppia sorgente-bersaglio sola, cioe' n = 1.

*E i p-value scritti come zero.* `round(p, 4)` aveva prodotto `p_value = 0.0`
sei volte su trenta. Un p nullo non esiste: i CSV portano ora il valore pieno
e una colonna formattata che sotto 1e-12 scrive una disuguaglianza.

`kanids/statistica.py` tiene tutto questo in un posto solo, insieme alla
correzione di Holm per famiglia e a quella di Nadeau-Bengio per gli split
ripetuti. Quest'ultima si applica **solo dove il suo regime esiste**: il
termine `n_test/n_train` e' un surrogato della sovrapposizione fra i training
set delle ripetizioni, quindi vale per `ton->ton` (rho = 0,25, cioe'
esattamente una 5-fold) e non per `bot->bot`, dove il training e' un
sottocampione di 19.431 righe estratto da un pool di 733.000 e le ripetizioni
quasi non si sovrappongono. Applicarla anche li' dava p = 0,72 su tutto: un
numero prudente per la ragione sbagliata. L'ho fatto, ho guardato il
risultato, e il CSV adesso spiega perche' quella colonna e' vuota.

**Un'affermazione ritirata, trovata correggendo l'unita'.** Il README diceva
"la dispersione cresce col rapporto — 0,0228 a 1:5, 0,0563 a 1:100 — quindi i
rapporti alti sono anche meno ripetibili". Quelle cifre erano la dispersione
su tutte le 120 misure, che mescola due cose. Separate: **fra i seed** non
cresce (0,00405 -> 0,00488 -> 0,00639 -> 0,00603 -> 0,00421, non monotona);
**fra modelli e domini** cresce (0,0221 -> 0,0566). Un rapporto alto non rende
la run meno ripetibile: fa disaccordare di piu' i sei modelli fra loro. E'
un'affermazione diversa, ed e' quella che i dati sostengono.

I confronti si ricalcolano dai run archiviati, senza riaddestrare niente:
`python scripts/statistica_confronti.py`, o `reproduce.py --stage statistica`.

**Quanto costa davvero la configurazione che la selezione sceglie.** La
selezione su validation sceglie h=32 grado=6, il progetto deploya h=16
grado=8, e la giustificazione era "il vincolo di dimensione di un
microcontrollore" — un'affermazione, non un numero: nessuno aveva mai
compilato la configurazione scelta. `scripts/footprint_architettura.py`
addestra entrambe con il protocollo della selezione (split esterno, validation
ritagliata dentro il training, test mai letto), le compila con la stessa
procedura dell'esportatore e ne misura l'ingombro in tre modi indipendenti: il
parser del progetto, le sezioni che avr-g++ emette per ATmega2560, e lo stack
che il kernel consuma. Il risultato sta in `results/arch_footprint.csv`, e la
riga della deployata deve riprodurre i byte dell'header committato — se non li
riproduce lo script lo dice, perche' senza quel controllo le due righe non
sarebbero confrontabili con `results/footprint.csv`.

Il modello h=32 grado=6 **non e' stato valutato sul test set**, ne' prima ne'
adesso: le sue cifre di accuratezza restano quelle della selezione, in
validation, su cinque seed. Un test perturba i valori delle righe destinate al
test e pretende che ne' l'ingombro ne' il punteggio si muovano.

**La compilazione della KAN multi-layer sta in un posto solo.** Era scritta di
seguito dentro `scripts/export_kan14_ml_coeff_c.py` e serviva solo all'header
deployato; compilare una seconda architettura con una seconda copia avrebbe
trasformato il confronto fra due architetture in un confronto fra due
compilatori — che e' esattamente l'errore gia' fatto tre volte con la formula
dei byte. Adesso e' in `kanids/compila_ml.py` e la usano entrambi. Lo
spostamento e' verificato nel modo piu' diretto possibile: un test legge
l'header **committato**, ne estrae tutti i numeri, li rida' all'emettitore
nuovo e pretende il file identico byte per byte.

Trovato spostandola: l'esportatore chiamava `cheb_T(x)` con il grado 8 preso
dal default. Su un modello di grado diverso — cioe' proprio quello che il
punto 3 chiede di compilare — sarebbe morto dentro un einsum con un messaggio
che non c'entrava niente. Adesso il grado si legge dalla forma di C1.

**Un header deployato che non era rigenerabile.** Il comando che doveva
verificare lo spostamento della compilazione e' morto con un
`FileNotFoundError` su `artifacts/kan14_mlbin.pkl`: `artifacts/` e' cache
rigenerabile e non versionata, e su quella macchina era vuota. Lo stesso
identico stato di training era pero' nel repository, committato, come
`models/kan14_binary_multilayer.pkl` — `export_models.py` ce lo copia
apposta. Gli esportatori leggevano solo dalla cache, quindi l'header C piu'
grosso fra quelli deployati risultava non rigenerabile da un clone pulito, e
non per una ragione vera. `kanids/checkpoint.py` tiene la corrispondenza in un
posto solo — la usano sia `export_models.py` per sapere cosa copiare sia gli
esportatori per sapere dove cercare — e quando uno stato manca davvero dice
quale script lo produce invece di dire soltanto che un file non c'e'.

**Cinque cifre sbagliate nel file che definisce l'architettura.** Il commento
che giustifica `ARCH_EREDITATA` in `kanids/config.py` diceva 0,99632 e 0,99600
di balanced accuracy, uno scarto di 0,00032, una soglia 1-SE mancata per
0,00020 e p = 0,067. I valori dell'artefatto sono 0,99631, 0,99602, 0,00028,
0,00015 e p = 0,083. Sono gli stessi ricalcoli su output arrotondato gia'
corretti nel README durante la revisione precedente: erano rimasti qui, cioe'
nella giustificazione della configurazione che tutta la pipeline usa, dove
nessuno li guardava. Adesso un test li confronta con
`results/arch_selection_scelta.json` uno per uno, come gia' fa per il README.

**E la formulazione e' cambiata, perche' i numeri lo impongono.** "Non ci
sta" non e' vero: 9.452 B sono il 3,6% dei 256 KB di Flash dell'ATmega2560, e
meno sull'ESP32-C3. Cio' che e' vero e' che la configurazione scelta costa
l'80% di Flash in piu' e il 41% di SRAM in piu' sul percorso di inferenza per
comprare 2,8·10⁻⁴ di balanced accuracy che un t appaiato non separa — e che il
criterio di parita' della regola 1-SE e' "a punteggio praticamente uguale
vince la piu' piccola". E' una preferenza dichiarata con un prezzo misurato,
non un limite fisico, e README, `config.py`, audit e indice del pacchetto
adesso lo dicono nello stesso modo.

**Una nota sulla rappresentazione.** Dopo la compilazione a B-spline ogni
funzione appresa e' descritta da NSEG+3 coefficienti, qualunque fosse il grado
del polinomio da cui proviene: l'ingombro compilato dipende dalla larghezza
nascosta, dai segmenti e dalle cardinalita', **non dal grado**. E' la ragione
per cui il punto 3 si chiude compilando due configurazioni invece di rifare
gli esperimenti, ed e' verificato sulle forme emesse invece che affermato.

**Una stima che cambiava da sola.** Rigenerare `results/footprint.csv` su una
macchina diversa da quella del lock riscriveva in silenzio la riga di
XGBoost: 49.905 -> 50.120 B (+0,43%), perche' con numpy e scipy piu' recenti
l'ensemble passa da 9.921 a 9.964 nodi interni. Il fenomeno era gia'
documentato in fondo a `requirements-lock.txt`, ma niente lo impediva, e il
sintomo arrivava altrove e dopo: la tabella del README, scritta a mano,
cominciava a divergere da un CSV che nessuno aveva modificato. Adesso una
riga *stimata* gia' presente nel CSV non viene sovrascritta — lo script dice
cosa avrebbe scritto e di quanto differisce, e tiene il valore dell'ambiente
del lock; `--aggiorna-stime` la adotta di proposito e ricorda di aggiornare
il README. Le righe *misurate* restano sempre riscritte: si leggono da un
header, sono deterministiche, e un test le confronta con l'header stesso.

Il test che vietava la frase "sei check host bit-esatti" e' stato riscritto:
era giusto finche' i check bit-esatti erano cinque e sarebbe diventato
sbagliato da solo appena ne fosse arrivato un altro — che e' quello che e'
successo. Adesso il numero si conta dai sorgenti di `host_check/` e i
documenti che descrivono lo stato corrente devono dire quello.

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

# Ambiente, costi e comandi riproducibili

ADAPT-01, aggiornato al 10 ottobre 2026. Tutti i risultati di questo PR sono
prodotti dai comandi qui sotto, nell'ordine indicato.

## Ambiente

Registrato automaticamente dentro ogni rendiconto del replay, campo `ambiente`,
insieme all'hardware letto dal sistema in `ambiente.hardware_misurato`:

- Python 3.11.15, CPython
- numpy 2.4.4, scikit-learn 1.8.0
- Linux x86_64, **solo CPU**: nessuna libreria di accelerazione, nessun uso di GPU
- un Intel Xeon a 2,80 GHz, due CPU utilizzabili dal processo, 8.031 MiB di RAM
- nessuna variabile `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` o
  `MKL_NUM_THREADS` impostata: i thread di BLAS sono quelli predefiniti

La macchina è virtuale e le CPU sono condivise. Questo basta per i tempi come
ordine di grandezza e per i rapporti fra modelli, **non** come misura di
latenza di un sistema in esercizio: a parità di modello il massimo del tempo di
un singolo aggiornamento sta al minimo come 29 a 1 sulle corse dedicate, e come
178 a 1 sulle corse eseguite mentre girava altro lavoro.
`costi_aggiornamento.py` riporta quel rapporto per ciascun modello.

La nuova serie C e i controlli appaiati del 9 ottobre sono stati eseguiti in
un secondo ambiente, registrato nei nuovi rendiconti e distinto da quello del
pilota storico:

- commit di partenza dei rendiconti: `ef0eaad07cfde5e2324df07d66b648e3c2a533b0`
- Python 3.13.2, CPython
- numpy 2.3.4, scikit-learn 1.7.2
- Windows 11, AMD64, solo CPU
- CPU `Intel64 Family 6 Model 186 Stepping 3, GenuineIntel`, 12 thread visibili
  al processo, 16.067 MiB di RAM
- nessuna variabile `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` o
  `MKL_NUM_THREADS` impostata

Gli input usati da quella serie sono gli stessi tre flussi ricostruiti dai CSV
originali, con queste impronte SHA-256:

| flusso | SHA-256 |
|---|---|
| A.npz | `ee79d298451cd2e243b4609f8d5cb35986b15fcedb825a467c77c8e79ced5432` |
| B.npz | `e6f4bf163fff89503e0e216d4ec8381146f6e3b71a6acf5e6729f4854b0923e0` |
| C.npz | `3fd24016c085b650ec9a54d4f21bed5b49764cb6df171bcc715e27feebfa939d` |

La campagna completa ADAPT-02 sulla memoria per classe e' stata eseguita nello
stesso ambiente Windows, con commit di avvio
`078a9d8447aee8493a9da7b98ba76d1bdec27678`. I 20 rendiconti completi stanno
fuori Git in
`adattamento-drift/archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8/`;
la copia compressa e'
`adattamento-drift/archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8.zip`,
scaricabile dal collegamento diretto al file condiviso
`https://drive.google.com/file/d/1C4uZIcW7rtWk5j1tROcHrCrhdzvzZq5E/view?usp=drive_link`,
SHA-256
`212B9270EF47D0B997667E83B38CB6C899A5B6ADD00DF61B881AA14DA905AFE0`.
Il riepilogo leggero versionato e'
`adapt02_memoria_per_classe/riepilogo_campagna_C.json`.
L'analisi finale derivata dai rendiconti completi, con tassi complessivi,
medie per blocco, dispersione sui cinque semi, differenze appaiate e figure, si
rigenera con:

```
cd <repo>/adattamento-drift/ton_temporal
python analisi_memoria_per_classe.py
```

Il comando scrive `adapt02_memoria_per_classe/analisi_campagna_C.json`, il CSV
`adapt02_memoria_per_classe/differenze_appaiate_per_seme_C.csv` e le figure in
`adapt02_memoria_per_classe/figure_campagna_C/`.

## ADAPT-03: sola soglia, C/seed42

Il protocollo e' stato fissato in `852f91c`. Smoke e replay completi sono stati
eseguiti dal commit pulito `90597f03413fa5884c6ad7e2895140695e3e14fe`;
l'analisi dei risultati e' stata eseguita inizialmente dallo stesso commit. La
sola contabilizzazione dello stato logico e' stata poi rigenerata dai
rendiconti originali con il commit pulito
`f265c4692dce67de29da55b1eaa3c911b651f3ce`: 11.816 byte a fine blocco e
15.016 byte prima del `pop` del lotto maturato. Nessun replay e' stato ripetuto.

Smoke preregistrato sui primi 20 blocchi:

```bash
cd <repo>/adattamento-drift/ton_temporal
python smoke_soglia_adattiva.py --iniziale flussi/A.npz --calibrazione flussi/B.npz --flusso flussi/C.npz --riferimenti-adapt02 ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8 --uscita-dir ../archivio_esterno/adapt03_soglia_adattiva/smoke_C_seed42_90597f0
```

Il riepilogo dello smoke dichiara PASS su esattamente i blocchi `0..19`: gli
SHA-256 di A, B e C e i parametri coincidono con i valori preregistrati; score,
richieste, `row_id`, soglie B e frozen coincidono fra le nuove corse e con i
primi 20 blocchi ADAPT-02.

Replay completi, in due directory logiche separate ma nello stesso checkpoint:

```bash
cd <repo>/adattamento-drift/ton_temporal
python replay_soglia_adattiva.py --iniziale flussi/A.npz --calibrazione flussi/B.npz --flusso flussi/C.npz --uscita ../archivio_esterno/adapt03_soglia_adattiva/diagnostico_C_seed42_90597f0/C_seed42_fifo_soglia_adattiva.json --memoria-strategia fifo --richiedi-worktree-pulito
python replay_soglia_adattiva.py --iniziale flussi/A.npz --calibrazione flussi/B.npz --flusso flussi/C.npz --uscita ../archivio_esterno/adapt03_soglia_adattiva/diagnostico_C_seed42_90597f0/C_seed42_per_classe_soglia_adattiva.json --memoria-strategia per_classe --richiedi-worktree-pulito
```

Analisi e figure:

```bash
cd <repo>/adattamento-drift/ton_temporal
python analisi_soglia_adattiva.py --fifo ../archivio_esterno/adapt03_soglia_adattiva/diagnostico_C_seed42_90597f0/C_seed42_fifo_soglia_adattiva.json --per-classe ../archivio_esterno/adapt03_soglia_adattiva/diagnostico_C_seed42_90597f0/C_seed42_per_classe_soglia_adattiva.json --riferimenti-adapt02 ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8 --uscita-dir adapt03_soglia_adattiva
```

I rendiconti completi restano fuori Git nell'archivio
`checkpoint_C_seed42_90597f0.zip` (1.088.717 byte), SHA-256
`88d5c4df0e637e6c0c0b61f4bae5ceb9e0c3ee99d21f3b54b3df03a77f4add91`.
Collegamento:
`https://drive.google.com/file/d/1AcK2b5AqezPmoMJWjL0SXFkaRG_1VBMy/view?usp=drive_link`.
`adapt03_soglia_adattiva/manifest_checkpoint.json` registra anche le impronte
dei cinque JSON contenuti. In Git sono conservati protocollo, manifest, analisi
derivata, CSV per blocco, figure e rapporto; D non e' stato letto.

## ADAPT-03: pesi adattati + soglia, smoke C/seed42

Il protocollo e' stato fissato prima del codice nel commit `8918902`. Lo smoke
usa il commit pulito `f9efbe4b1312a82805a532dadd45b8eb11778c1b`: KAN,
memoria per classe 256, politica `ogni_blocco`, budget dell'1%, ritardo di un
blocco e primi 20 blocchi di C. Un'unica traiettoria dei pesi produce un unico
score per blocco; il controllo mantiene la soglia B iniziale e la variante
ricalibra sull'intero B dopo ogni aggiornamento effettivo, con effetto dal
blocco successivo.

B contiene 5.681.198 righe ed e' un archivio aggiuntivo rispetto ai 256 esempi
online. La rappresentazione KAN di B viene costruita una volta in cache; score,
selezione BA, cache, array del fit online e picco RSS sono rendicontati
separatamente. Il valore di 24.904 byte misura soltanto feature ed etichette
della memoria, rappresentazione KAN della memoria e parametri della testa KAN;
non rappresenta tutto lo stato online.

Smoke ufficiale:

```powershell
cd <repo>/adattamento-drift/ton_temporal
python replay_pesi_piu_soglia.py --iniziale flussi/A.npz --calibrazione flussi/B.npz --flusso flussi/C.npz --riferimento-adapt02 ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8/C_seme42_per_classe_ogni_blocco.json --uscita ../archivio_esterno/adapt03_pesi_piu_soglia/smoke_C_seed42_f9efbe4/smoke.json --richiedi-worktree-pulito
```

Analisi riproducibile dal rendiconto originale:

```powershell
cd <repo>/adattamento-drift/ton_temporal
python analisi_smoke_pesi_soglia.py --smoke ../archivio_esterno/adapt03_pesi_piu_soglia/smoke_C_seed42_f9efbe4/smoke.json --riferimento-adapt02 ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8/C_seme42_per_classe_ogni_blocco.json --archivio ../archivio_esterno/adapt03_pesi_piu_soglia/smoke_C_seed42_f9efbe4.zip --uscita-dir adapt03_pesi_piu_soglia
```

Il JSON originale resta fuori Git nello ZIP
`smoke_C_seed42_f9efbe4.zip`, 11.267 byte, SHA-256
`acdeb051288f4b4e8774358020af6f6f1cda90204471894bd0962291596f14c1`.
L'archivio e' scaricabile dal collegamento diretto al file condiviso
`https://drive.google.com/file/d/1RU2QNkTEDIEDJRQW3vJz82kzPlHQn3Ab/view?usp=drive_link`.
Il file interno e' 60.151 byte, SHA-256
`7a2c8424a963e01088baded953287eada6276884994dabd18f2661f441db8ece`.
In Git restano protocollo, manifest, riepilogo, CSV per blocco, figura e
rapporto. Al momento di questo smoke il replay completo non era ancora stato
avviato; l'esecuzione successiva e' documentata nella sezione seguente. D non
e' stato letto.

## ADAPT-03: replay completo pesi adattati + soglia, C/seed42

Il PR #4 e' stato integrato in `main` con merge commit
`9550decf06156664839affd8db5566fe2fbc8bd0`, mantenendo la versione approvata
`d11c59c`. Il protocollo completo e' in
`adapt03_pesi_piu_soglia/protocollo_replay_completo.md`.

Il runner distingue esplicitamente `smoke` e `completo`. La seconda modalita'
richiede esattamente 916 blocchi e confronta l'intera sequenza con ADAPT-02.
Prima dell'avvio si fissa lo SHA e si usa una directory nuova:

```powershell
cd <repo>/adattamento-drift/ton_temporal
git rev-parse --short HEAD | Tee-Object -Variable sha
$uscita = "../archivio_esterno/adapt03_pesi_piu_soglia/replay_completo_C_seed42_$sha"
if (Test-Path $uscita) { throw "la directory di uscita esiste gia': $uscita" }
python replay_pesi_piu_soglia.py --modalita completo --iniziale flussi/A.npz --calibrazione flussi/B.npz --flusso flussi/C.npz --riferimento-adapt02 ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8/C_seme42_per_classe_ogni_blocco.json --uscita "$uscita/replay_completo.json" --richiedi-worktree-pulito
```

Il rendiconto originale viene compresso senza modificarlo:

```powershell
Compress-Archive -LiteralPath "$uscita/replay_completo.json" -DestinationPath "$uscita.zip" -CompressionLevel Optimal
```

L'analisi non riesegue il replay:

```powershell
cd <repo>/adattamento-drift/ton_temporal
python analisi_replay_completo_pesi_soglia.py --replay "$uscita/replay_completo.json" --riferimento-adapt02 ../archivio_esterno/adapt02_memoria_per_classe/campagna_C_078a9d8/C_seme42_per_classe_ogni_blocco.json --archivio "$uscita.zip" --collegamento-archivio "https://drive.google.com/file/d/1qt-qRB7KWpVEg2DrbaSantbbzRJVZgKI/view?usp=drive_link" --uscita-dir adapt03_pesi_piu_soglia/replay_completo
```

Nel riferimento completo il recall e' definito in 916/916 blocchi; FPR,
balanced accuracy e AUROC sono definiti in 875/916. I 41 valori mancanti
restano `null` nel JSON e celle vuote nel CSV: non vengono sostituiti con zero.
B resta un archivio aggiuntivo; D, i cinque semi e la variante 192/64 sono
fuori dal protocollo.

L'esecuzione effettiva usa il commit pulito
`106ee1ccce2f116b7fb9b296df0a403c789f40ab` e la directory nuova
`replay_completo_C_seed42_106ee1c`. Ha prodotto 916 blocchi, 915 aggiornamenti
dei pesi e 915 ricalibrazioni su B in 2.297,616 secondi (38,294 minuti). Il
controllo a soglia B fissa ottiene recall complessivo 20,608%, FPR complessivo
0,474% e balanced accuracy complessiva 60,067%; con gli stessi pesi e soglia
ricalibrata i valori sono 81,082%, 30,141% e 75,471%. L'AUROC media resta
identica, 0,889025 su 875/916 blocchi, come richiesto dall'appaiamento degli
score. L'aumento del recall si accompagna quindi a un forte aumento del FPR e
non dimostra una superiorita' generale.

Il rendiconto originale `replay_completo.json` e' 2.562.150 byte, SHA-256
`996263a0dcccbfa005853c93d93467972889f3d0ff7c1908070825b778af851e`.
L'archivio `replay_completo_C_seed42_106ee1c.zip` e' 381.478 byte, SHA-256
`b084c4bfb0955698965aacb16e8a6a2ff78fe8f22e31a18ff6a12fd8b1e01d38`.
Il collegamento scaricabile all'archivio verificato e'
`https://drive.google.com/file/d/1qt-qRB7KWpVEg2DrbaSantbbzRJVZgKI/view?usp=drive_link`.

## ADAPT-04: calibrazione recente separata 192+64

ADAPT-04 parte dal merge `d7bcd716dc416359474f12023cf0e5766d3aa78a`.
Protocollo e configurazione sono stati fissati nel commit `6434664`, prima del
runner. `fit192` e `calibration64` partono vuoti, restano separati e ricevono
ruoli determinati dal salt `ADAPT04-C-seed42-v1` prima delle etichette.

Il runner carica A e C, ma non B. La soglia B iniziale e le invarianti di
inizializzazione sono lette dal rendiconto ADAPT-03 verificato. Smoke ufficiale:

```powershell
cd <repo>/adattamento-drift/ton_temporal
git rev-parse --short HEAD | Tee-Object -Variable sha
$uscita = "../archivio_esterno/adapt04_calibrazione_recente/smoke_C_seed42_$sha"
if (Test-Path $uscita) { throw "la directory di uscita esiste gia': $uscita" }
python replay_calibrazione_recente.py --modalita smoke --iniziale flussi/A.npz --flusso flussi/C.npz --riferimento-adapt03 ../archivio_esterno/adapt03_pesi_piu_soglia/replay_completo_C_seed42_106ee1c/replay_completo.json --uscita "$uscita/smoke.json" --richiedi-worktree-pulito
python analisi_calibrazione_recente.py --replay "$uscita/smoke.json" --riferimento-adapt03 ../archivio_esterno/adapt03_pesi_piu_soglia/replay_completo_C_seed42_106ee1c/replay_completo.json --analisi-soglia-adapt03 adapt03_soglia_adattiva/analisi_checkpoint.json --riepilogo-pesi-adapt03 adapt03_pesi_piu_soglia/replay_completo/riepilogo_replay_completo.json --uscita-dir "$uscita/analisi"
```

Se i controlli dello smoke passano, lo stesso commit esegue direttamente il
replay completo autorizzato in una directory nuova:

```powershell
cd <repo>/adattamento-drift/ton_temporal
git rev-parse --short HEAD | Tee-Object -Variable sha
$uscita = "../archivio_esterno/adapt04_calibrazione_recente/replay_completo_C_seed42_$sha"
if (Test-Path $uscita) { throw "la directory di uscita esiste gia': $uscita" }
python replay_calibrazione_recente.py --modalita completo --iniziale flussi/A.npz --flusso flussi/C.npz --riferimento-adapt03 ../archivio_esterno/adapt03_pesi_piu_soglia/replay_completo_C_seed42_106ee1c/replay_completo.json --uscita "$uscita/replay_completo.json" --richiedi-worktree-pulito
python analisi_calibrazione_recente.py --replay "$uscita/replay_completo.json" --riferimento-adapt03 ../archivio_esterno/adapt03_pesi_piu_soglia/replay_completo_C_seed42_106ee1c/replay_completo.json --analisi-soglia-adapt03 adapt03_soglia_adattiva/analisi_checkpoint.json --riepilogo-pesi-adapt03 adapt03_pesi_piu_soglia/replay_completo/riepilogo_replay_completo.json --uscita-dir adapt04_calibrazione_recente/replay_completo
```

Entrambe le corse usano solo KAN e C/seed42. D resta riservato; non sono
autorizzati cinque seed o ricerca di iperparametri. Il secondo comando di ogni
blocco ricalcola le aggregazioni e richiede sequenza completa, appaiamento dei
campioni, causalita' e atomicita', identita' degli score delle politiche 2/3,
budget 192+64, metriche monoclasse non sostituite con zero e conteggi effettivi
di aggiornamenti e skip. Il replay completo si avvia soltanto dopo il PASS
dello smoke sullo stesso commit pulito.

## La suite di test

```
cd <repo>
python -m pytest adattamento-drift/ton_temporal -q
```

Su Windows i compilatori richiamati dalla suite devono avere una directory
temporanea esistente e scrivibile. Prima dei test si puo' fissarla in modo
esplicito per la sessione PowerShell:

```powershell
$tmp = "$env:LOCALAPPDATA\Temp\kan-ids-pytest"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp
```

Attesi: **355 test superati** nella cartella `ton_temporal/`: i 251 test
verificati di ADAPT-01, 8 test del checkpoint ADAPT-02 sulla memoria per classe,
5 test sull'analisi finale, 28 test causali e di validazione dello smoke, 8
test dell'analisi ADAPT-03, 27 test del diagnostico pesi+soglia e 28 test
ADAPT-04. Per file:
`test_documenti_coerenti.py`
67, `test_figure.py` 38, `test_soglia.py` 26, `test_politiche.py` 26,
`test_sovrapposizioni.py` 23, `test_riepiloghi_semi.py` 21,
`test_misure_di_costo.py` 17, `test_impronta_campione.py` 16,
`test_guardia_monoclasse.py` 11, `test_riepilogo_finale_C.py` 6,
`test_memoria_per_classe.py` 8, `test_analisi_memoria_per_classe.py` 5,
`test_soglia_adattiva_online.py` 28, `test_analisi_soglia_adattiva.py` 8,
`test_pesi_piu_soglia.py` 27, `test_calibrazione_recente.py` 28.

La storia dei conteggi, perché il numero cambia a ogni commit: **139** al commit
`c5e6d02`, **193** a `7ea4a94`, **195** a `5922992`, **239** a `5a24cf1`,
**240** a `3861a5d`, **242** a `7f154ed`, **245** a `ef0eaad`, **251** a
chiusura di ADAPT-01, **259** con il checkpoint ADAPT-02, **264** con i controlli
dell'analisi finale, **272** con i primi controlli causali, **279** con i
controlli dell'analisi del checkpoint ADAPT-03, **299** dopo le controprove
negative sullo smoke, **300** con la controprova sul picco prima del `pop` e
**320** con il calibratore B e le controprove del diagnostico pesi + soglia,
**327** con modalita' completa, denominatori mancanti e confronto sui 916
blocchi, **345** con ruoli deterministici, buffer separati e transazioni
atomiche di ADAPT-04, **355** con il validatore indipendente del rendiconto e le
relative controprove negative.
I 56 del ciclo ADAPT-01 e dal precedente sono le 38 di `test_figure.py`, 9 aggiunte a
`test_documenti_coerenti.py` e 3 aggiunte a `test_politiche.py` per il seme
separato della politica — la cartella dichiarata in ogni blocco di comandi,
l'esistenza dei file che i comandi citano, le frasi troncate, il confronto fra
il numero di prove dichiarato qui e quello che la raccolta trova davvero —
perché questo conteggio è già rimasto indietro due volte — e l'encoding
dell'uscita dei controlli, presidiato sia sul programma sia **sul file
consegnato**; e i percorsi dei comandi, che devono attraversare
`adattamento-drift/` e citare i flussi relativamente alla cartella da cui il
comando si esegue. Le 6 prove nuove presidiano la serie finale C: 40 rendiconti
originali conservati, 60 riallineati, e test negativi su soglie, misure
frozen, campioni, aggiornamenti richiesti e aggiornamenti effettivamente
applicati. Le 8 prove di ADAPT-02 presidiano invece la memoria per classe:
limite, quote, espulsione, causalita', retrocompatibilita' FIFO, dtype delle
feature, confronto con la FIFO storica e appaiamento di richieste e frozen. Le
5 prove finali distinguono FPR complessivo e medio per blocco, controllano la
dispersione campionaria, il denominatore AUROC, i tipi di attacco, i 30 delta
per seme e le figure consegnate. Le 28 prove ADAPT-03 controllano classe
assente, eta' dell'evidenza dal blocco di osservazione, minimo per classe,
tie-break, invarianza di score e AUROC, appaiamento delle memorie, ritardo
causale, etichette future, richieste in attesa e conteggio dello stato
aggiuntivo. Venti casi aggiunti rifiutano SHA-256 o file diversi, ciascun
parametro preregistrato alterato e smoke vuoti, incompleti, duplicati, traslati,
fuori ordine o con un conteggio dichiarato incoerente; un caso positivo accetta
soltanto la sequenza `0..19` con i metadati fissati.
Gli 8 controlli dell'analisi distinguono aggregazioni globali e medie per blocco,
ricostruiscono periodi contigui di peggioramento, verificano le differenze e
rifiutano score frozen non appaiati; una controprova separa 11.816 byte a fine
blocco dai 15.016 byte raggiunti prima del `pop`; gli ultimi due controllano il
JSON, le 5.496 righe dati del CSV, i denominatori AUROC, i quattro PNG e la
tabella del rapporto consegnato.

Le 7 prove del replay completo richiedono 916 blocchi in ordine, rifiutano un
riferimento incompleto, conservano come mancanti le metriche dei 41 blocchi
monoclasse, verificano la nuova rendicontazione degli array del fit e
ricostruiscono il riepilogo completo dal rendiconto originale.

Le 28 prove ADAPT-04 presidiano serializzazione canonica, zeri e NaN, ruolo
immutabile dei duplicati, separazione dei buffer, quote e posti liberi, eta'
dall'osservazione, causalita' `j+2`, cause degli skip, atomicita', ricalcolo
degli score, metriche mancanti e payload distinti. Le dieci aggiunte fanno
fallire blocchi incompleti o fuori ordine, score 2/3 diversi, soglie applicate
in anticipo, capacita' superate, conteggi falsi, input B aggiunto, campioni non
appaiati e zeri al posto di metriche non definite; inoltre ricostruiscono i
periodi contigui usando soltanto coppie definite.

**I 355 non sono il totale del repository.** Lanciando `pytest` dalla radice del
repository si raccolgono anche le suite delle altre linee di lavoro. L'ultima
misura completa di ADAPT-03 era **961 passate, 1 saltata e 5 sottotest passati**;
il totale ADAPT-04 viene registrato dopo la suite completa sul commit eseguito.
Alla chiusura del precedente diagnostico ADAPT-03 erano 934 passate, 1 saltata
e 5 sottotest passati. Alla
chiusura di ADAPT-02 erano 898 passate, 1 saltata e 5 sottotest passati. Al
commit `5922992` il totale era 829 passate e 1 saltata, di cui 195 erano queste;
a `3861a5d` erano state misurate 874 prove con 240 di ADAPT-01. I due numeri
attuali vanno citati separatamente, perché misurano cose diverse: 345 e' la
cartella `ton_temporal/`, mentre il totale dalla radice include anche le altre
linee di lavoro. Il
totale della radice dipende anche dalle altre cartelle presenti nella copia di
lavoro.

Una versione precedente di questo documento dichiarava «11 test superati» e
citava conteggi di 634 e 643 test misurati su un altro ramo: nessuno dei due
numeri era verificabile qui, ed entrambi sono stati sostituiti con il conteggio
misurato su questo ramo.

La suite completa di ADAPT-01 attraversa anche la serie finale C. Per eseguirla
in una copia nuova non bastano i riepiloghi in Git: gli archivi esterni devono
essere presenti ed estratti nelle cartelle documentate sotto
`adattamento-drift/archivio_esterno/`.

## Costi misurati

Dal campo `costi` del rendiconto, corsa di riferimento con seme 42 su C
(`replay_C_seme42.json`):

- addestramento iniziale dei tre modelli su A: **33 secondi**
- replay su 916 blocchi: **87 secondi**, cioè **0,09
  secondi per blocco**
- etichette richieste: **91.506**, cento per blocco completo e sei sul blocco
  finale di 673 righe; la quota è `91.506 / 9.150.673`, registrata nel campo
  `quota_etichette_effettiva` accanto a `righe_scorse`
- aggiornamenti applicati: **522** per modello; saltati per memoria monoclasse:
  **393**

I tempi variano di qualche secondo fra esecuzioni: sono un ordine di grandezza,
non una misura ripetibile.

**Costo del singolo aggiornamento.** I numeri qui sotto vengono da **una misura
dedicata**, non da una media sui semi: una sola corsa, **seme 42**, politica
`ogni blocco`, **522 aggiornamenti applicati** e 393 tentativi fermati dalla
guardia, eseguita con la macchina altrimenti inattiva e pubblicata in
`replay_evidenza/tempi/ogni_blocco_seme42_costi.json`. La colonna che conta è la
mediana.

| misura dedicata, seme 42, 522 aggiornamenti | ms per aggiornamento, mediana | secondi in aggiornamento | quota del replay |
|---|---:|---:|---:|
| logistica | 5,19 ms | 4,9 s | 5,6% |
| MLP | 5,27 ms | 3,1 s | 3,5% |
| additivo | 11,08 ms | 6,8 s | 7,8% |

Le stesse misure su tutti e venti i rendiconti si ottengono con
`costi_aggiornamento.py` e stanno in `replay_evidenza/costi_aggiornamento.json`.
Là gli aggiornamenti applicati sono in media 276,4 sulle sessanta misure, cioè
venti rendiconti per tre modelli: 520 nelle sole dieci corse `ogni blocco` e
32,8 nelle altre dieci. Una versione precedente di questa riga attribuiva il 520
a tutti e venti i rendiconti.

**Memoria.** Picco di memoria residente del processo: circa 1.338 MiB,
dominato dagli array dello stream che il banco di prova tiene in memoria. Lo
stato che l'adattamento muove è 10.240 byte di memoria FIFO e, per modello,
8.264 byte di parametri per la logistica, 16.520 per l'MLP e 16.456 per
l'additivo. Il numero di aggiornamenti **non dimostra un guadagno di latenza né
di memoria**: il punteggio di ogni blocco si paga in tutte le politiche, e il
risparmio sugli aggiornamenti è di 15 secondi su 86,6 secondi di replay.

L'inventario completo dei 23 file costa circa 13 minuti; la ricostruzione delle
impronte, 2 secondi. Le venti corse di questo ciclo, in sequenza su questa
macchina, circa 48 minuti.

## Comandi, nell'ordine

I dati grezzi non sono redistribuiti, quindi quattro percorsi dipendono dalla
macchina e sono le **sole** cose da sostituire: `<repo>` è la cartella del
clone di questo repository — quella che contiene `adattamento-drift/`, non la
cartella di lavoro — `<dati>` è la cartella con i 23
`Network_dataset_*.csv`, `<lab>` un clone di `ids-update-lab`, `<noto>` il
`train_test_network.csv` il cui SHA-256 è
`26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974`. Tutto il
resto dei comandi è letterale e si incolla così com'è.

**Ogni blocco comincia con il `cd` da cui va eseguito**, perché lo stesso
comando riesce da una cartella e fallisce da un'altra: senza quella riga chi
legge deve indovinarla. I comandi sono su una riga sola di proposito: una riga
spezzata, incollata, diventa due comandi sbagliati.

**1. Ricostruire le impronte degli input già visti**

```
cd <repo>/adattamento-drift/ton_temporal
python impronte_note.py --sorgente <noto> --audit <lab>/research028/vendor/audit_ton_full.py --uscita impronte_note.npz --esigi-sorgente
```

Esce diverso da zero se la sorgente o la cardinalità non coincidono con i
metadati dichiarati: 92.330 impronte uniche da 211.043 righe.

**2. Inventario dei 23 file**

```
cd <repo>/adattamento-drift/ton_temporal
python inventario_ton.py --dati <dati> --audit <lab>/research028/vendor/audit_ton_full.py --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json --impronte impronte_note.npz --uscita inventario --riprendi
```

Ripartibile: rilanciandolo salta i file già fatti. Produce
`data_inventory.csv`, `daily_counts.csv`, `hourly_counts.csv`,
`hourly_counts_frt.csv`, `diagnostics.json`, `feature_ready_time.json` e
`per_file/`.

**3. Copertura, con lo strumento del referente, senza modifiche**

```
cd <repo>/adattamento-drift/ton_temporal
python <lab>/research028/tools/coverage028.py --input inventario/hourly_counts_frt.csv --output inventario/copertura_frt
```

**4. Costruire i flussi ordinati per feature_ready_time**

```
cd <repo>/adattamento-drift/ton_temporal
python costruisci_flusso.py --dati <dati> --audit <lab>/research028/vendor/audit_ton_full.py --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json --inizio 2019-04-02 --fine-esclusa 2019-04-05 --giorni-extra 2019-04-23 --uscita flussi/A.npz
```

Per B: `--inizio 2019-04-24 --fine-esclusa 2019-04-25`.
Per C: `--inizio 2019-04-25 --fine-esclusa 2019-04-27`.
Per D il comando sarebbe `--inizio 2019-04-27 --fine-esclusa 2019-04-30`, ma
**D non è stato costruito**: le sue righe sono state soltanto lette e contate
dai CSV originali da `sovrapposizioni_abcd.py`, per il controllo delle
sovrapposizioni del §5bis. D non è stato usato né per addestrare né per
valutare, e il flusso si costruirà quando la valutazione finale sarà
autorizzata.

**4bis. Le sovrapposizioni dirette fra A, B, C e D**

```
cd <repo>/adattamento-drift/ton_temporal
python sovrapposizioni_abcd.py --csv <dati> --manifest <lab>/research028/inputs/ton_iot_mirror_v1_manifest.json --audit <lab>/research028/vendor/audit_ton_full.py --uscita sovrapposizioni_abcd.json
```

Scorre i 23 CSV una volta, assegna ogni riga al proprio intervallo per
`feature_ready_time` e confronta le impronte dei vettori di feature fra gli
intervalli, usando `normalize_raw` e `fingerprint_rows` del modulo di
riferimento senza modificarli.

**5. Il replay: le quattro configurazioni, per ogni seme**

Le quattro corse del **seme 42**, da incollare così come sono. Per gli altri
semi si cambiano i due `42` di ciascuna riga — il valore di `--seme` e quello
nel nome del file — e i conteggi di `--quanti-aggiornamenti`, che sono quelli
della politica su evidenza dello stesso seme.

```
cd <repo>/adattamento-drift/ton_temporal
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --seme 42 --uscita replay_C_seme42.json
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --seme 42 --uscita replay_C_calibrato_seme42.json
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --politica evidenza_inversione --seme 42 --uscita replay_C_evidenza_seme42.json
python replay.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --politica casuale --quanti-aggiornamenti "lr=54,mlp=20,kan=23" --ammissibili replay_C_calibrato_seme42.json --seme 42 --uscita replay_C_casuale_seme42.json
```

I conteggi da passare al controllo casuale, letti in
`costi.decisioni_della_politica` della corsa su evidenza dello stesso seme:
seme 42 `lr=54,mlp=20,kan=23`; seme 43 `lr=43,mlp=35,kan=16`; seme 44
`lr=44,mlp=45,kan=24`; seme 45 `lr=49,mlp=30,kan=13`; seme 46
`lr=36,mlp=39,kan=21`.

I parametri predefiniti sono quelli della scheda: blocchi di 10.000, budget 1%,
memoria 256, ritardo di un blocco. I semi sono 42, 43, 44, 45 e 46, e **tutte
le venti corse sono pubblicate** in `replay_evidenza/`. I conteggi da passare
al controllo casuale sono quelli applicati dalla politica su evidenza **dello
stesso seme**, che si leggono in `costi.decisioni_della_politica`: cambiarli
renderebbe il controllo non pareggiato. Per la variante scartata,
`--campionamento strati_punteggio`.

**5bis. Estensione C, controlli appaiati e riallineamento mirato**

La campagna estesa conserva i cinque semi di base 42, 43, 44, 45 e 46 e produce
venti calendari casuali per ciascun seme. Il seme della politica e' separato
dal seme di base con la regola `seme_base * 1000 + calendario`, cosi'
l'inizializzazione, i campioni di etichette e le soglie restano legati al seme
di base, mentre cambia soltanto il calendario del controllo casuale.

Prima della campagna completa e' stata eseguita una corsa singola per stimare i
tempi: 333,26 secondi, stima sequenziale per 100 corse 9,26 ore sulla macchina
Windows descritta sopra. La campagna completa scrive fuori Git i rendiconti
interi e in Git solo il riepilogo leggero:

```
cd <repo>/adattamento-drift/ton_temporal
python estensione_calendari_C.py --fermati-dopo 1 --collegamento-archivio "https://drive.google.com/file/d/1X5d8sQDtt7iSowaEKs2oKqqQSQBxOJnZ/view?usp=drive_link"
python estensione_calendari_C.py --salta-esistenti --collegamento-archivio "https://drive.google.com/file/d/1X5d8sQDtt7iSowaEKs2oKqqQSQBxOJnZ/view?usp=drive_link"
```

I cento rendiconti completi della campagna casuale restano in
`adattamento-drift/archivio_esterno/estensione_C_calendari/`; lo zip
`rendiconti_estensione_C_calendari.zip` ha SHA-256
`cb21394a8c8c0fc9acd3361f6ae14dddb385971a36a6d3929450b796ee43bfda` ed e'
collegato qui:
`https://drive.google.com/file/d/1X5d8sQDtt7iSowaEKs2oKqqQSQBxOJnZ/view?usp=drive_link`.
Il riepilogo versionato e' `estensione_C_calendari/riepilogo.json` e registra
configurazione, semi, seme della politica, blocchi scelti, metriche e SHA-256
dei rendiconti.

Il confronto appaiato non usa i vecchi rendiconti evidence, perche' la nuova
serie e' stata eseguita in un ambiente diverso e le soglie non coincidono con
quelle storiche. I controlli evidence e calibrato dei cinque semi vengono quindi
ricostruiti nello stesso ambiente della nuova serie, poi confrontati
automaticamente con i cento casuali:

```
cd <repo>/adattamento-drift/ton_temporal
python controlli_appaiati_C.py --salta-esistenti
```

Il controllo verifica, per ciascun seme e calendario, soglie, digest delle
misure frozen per blocco, digest dei campioni di etichette, aggiornamenti
richiesti e aggiornamenti effettivamente applicati. Soglie, frozen e campioni
coincidono; i conteggi degli aggiornamenti richiedono il riallineamento mirato
di 60 casuali, tutti dei semi 42, 43 e 45. I 100 rendiconti originali non
vengono sovrascritti.

L'esito di `controlli_appaiati_C.py` non basta, da solo, a validare la serie
finale: quel programma controlla i confronti appaiati e prepara il
riallineamento. La validazione finale e' il passo successivo, con
`riepilogo_finale_C.py`, poi richiamato anche da `verifica_richieste.py`.

```
cd <repo>/adattamento-drift/ton_temporal
python controlli_appaiati_C.py --solo-riepilogo --riallinea-casuali --salta-esistenti --workers 2
```

Gli archivi completi dei controlli appaiati stanno sotto
`adattamento-drift/archivio_esterno/controlli_appaiati_C/`, fuori Git:

| contenuto | archivio | SHA-256 | collegamento |
|---|---|---|---|
| 10 rendiconti calibrato/evidence nuovi | `rendiconti_controlli_appaiati_C.zip` | `4d29bd4cdee29cfd5e686d409c1b5b6ff5c75ed1a3f94ca41b0d0240bee4237e` | `https://drive.google.com/file/d/1OgVKO7t74Yi7Q8Gy_k-aWZQg9mb1i4vb/view?usp=drive_link` |
| 60 casuali riallineati | `rendiconti_casuali_riallineati_C.zip` | `6eb5ee0ae8a73731e300823eda7cc0adaf211e001a777978b5a57b3958459fe5` | `https://drive.google.com/file/d/1Wum41yN4wHKJb3tsNTPqndY59MYzjqC_/view?usp=drive_link` |

Il riepilogo versionato `controlli_appaiati_C/riepilogo.json` registra commit,
versioni dell'ambiente tramite i rendiconti, hash degli input, archivi, SHA-256,
semi, calendari e confronti. Il campo
`tutti_appaiati_dopo_riallineamento` vale `true`: dopo i 60 rendiconti
riallineati, soglie, frozen, digest dei campioni e conteggi degli aggiornamenti
sono appaiati rispetto ai nuovi controlli evidence.

La serie finale usata nelle sintesi e nei confronti si costruisce dai
rendiconti completi conservati, senza nuovi training:

```
cd <repo>/adattamento-drift/ton_temporal
python riepilogo_finale_C.py
```

Il riepilogo leggero versionato è `serie_finale_C/riepilogo.json`. Registra i
100 rendiconti originali conservati, la serie effettiva di confronto
(`40 originali conservati + 60 riallineati`), i 10 controlli
calibrato/evidence, le impronte SHA-256 dei file usati e le distribuzioni per
seme. Lo stesso controllo è richiamato da `verifica_richieste.py`: le invarianti
richieste sono soglie, frozen, campioni, aggiornamenti richiesti e
aggiornamenti effettivamente applicati su tutta la serie finale.

**6. I candidati della soglia e la regola di confronto**

```
cd <repo>/adattamento-drift/ton_temporal
python documenta_soglie.py --iniziale flussi/A.npz --calibrazione flussi/B.npz --semi 42 43 44 45 46 --uscita replay_evidenza/soglie_candidati.json
```

**7. I riepiloghi, i confronti e i costi**

Su Linux e macOS, da dentro `replay_evidenza`:

```
cd <repo>/adattamento-drift/ton_temporal/replay_evidenza
Z="replay_C_seme42.json replay_C_seme43.json replay_C_seme44.json replay_C_seme45.json replay_C_seme46.json"
B="replay_C_calibrato_seme42.json replay_C_calibrato_seme43.json replay_C_calibrato_seme44.json replay_C_calibrato_seme45.json replay_C_calibrato_seme46.json"
python ../riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json --verifica
python ../riepiloghi_semi.py --rendiconti $Z --confronto $B --uscita confronto_soglia.json --verifica
python ../confronto_politiche.py --ogni-blocco $B --evidenza replay_C_evidenza_seme4?.json --casuale replay_C_casuale_seme4?.json --uscita confronto_politiche.json
python ../costi_aggiornamento.py --rendiconti replay_C_*seme4?.json --uscita costi_aggiornamento.json
```

In PowerShell, dove le liste si scrivono con le virgole e il globbing va risolto
prima:

```
cd <repo>/adattamento-drift/ton_temporal/replay_evidenza
$Z = "replay_C_seme42.json","replay_C_seme43.json","replay_C_seme44.json","replay_C_seme45.json","replay_C_seme46.json"
$B = "replay_C_calibrato_seme42.json","replay_C_calibrato_seme43.json","replay_C_calibrato_seme44.json","replay_C_calibrato_seme45.json","replay_C_calibrato_seme46.json"
python ..\riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json --verifica
python ..\riepiloghi_semi.py --rendiconti $Z --confronto $B --uscita confronto_soglia.json --verifica
python ..\confronto_politiche.py --ogni-blocco $B --evidenza (Get-Item replay_C_evidenza_seme4*.json).Name --casuale (Get-Item replay_C_casuale_seme4*.json).Name --uscita confronto_politiche.json
python ..\costi_aggiornamento.py --rendiconti (Get-Item replay_C_*seme4?.json).Name --uscita costi_aggiornamento.json
```

Il glob `replay_C_*seme4?.json` prende le venti corse e lascia fuori la variante
scartata, il cui nome finisce in `_strati_punteggio`.

**8. Le figure**

```
cd <repo>/adattamento-drift/ton_temporal
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello lr --uscita figure
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello mlp --uscita figure
python curve_replay.py --rendiconto replay_evidenza/replay_C_calibrato_seme42.json --politiche ogni_blocco=replay_evidenza/replay_C_calibrato_seme42.json evidenza=replay_evidenza/replay_C_evidenza_seme42.json casuale=replay_evidenza/replay_C_casuale_seme42.json --modello kan --uscita figure
```

Tre comandi, uno per modello, che producono i quattordici PNG di `figure/`.
Il terzo stampa anche l'astensione dichiarata su `recupero_kan.png`: sul modello
additivo gli episodi di inversione sono tre, lunghi 7, 2 e 1 blocchi, e uno solo
arriva al minimo di cinque, quindi la figura non viene prodotta.
`figure/LEGGIMI.md` dichiara per ciascuna figura la provenienza e l'impronta dei
rendiconti.

Queste figure appartengono alla serie storica del pilota sul seme 42: servono a
illustrare il meccanismo dei blocchi e delle politiche, non sono le
distribuzioni della serie finale C sui 100 rendiconti effettivi.

La versione precedente di questo passo era **un comando solo e sbagliato**:
`python curve_replay.py --rendiconto replay_C_seme42.json`. Lo script sta in
`ton_temporal/`, il rendiconto in `ton_temporal/replay_evidenza/`, quindi quel
comando falliva da entrambe le cartelle; e citava la corsa a soglia zero, che è
la ragione per cui le sei figure pubblicate erano state prodotte da quella
corsa invece che dalle corse calibrate.

**9. Le richieste del referente, controllate sul materiale**

```
cd <repo>/adattamento-drift/ton_temporal
python verifica_richieste.py --cartella .
python verifica_richieste.py --cartella . --uscita verifica_richieste.txt
```

Traduce ogni richiesta delle lettere e del riesame in un controllo eseguibile
sui file del ramo, ed elenca a parte quello che un programma non può
verificare.

Il secondo comando salva il rendiconto nel file versionato. **Si usa `--uscita`
e non la redirezione della shell**: su PowerShell `> file.txt` riscrive l'uscita
in UTF-16 e la rilegge con una code page che non è UTF-8, e il risultato è un
file che git tratta come binario e in cui il meno tipografico diventa `Ôê`. È
successo ed è stato pubblicato così, dopo che una correzione precedente aveva
tolto l'errore rumoroso — il comando cadeva con `UnicodeEncodeError` — lasciando
al suo posto un errore muto. Con `--uscita` il file lo scrive il programma, in
UTF-8 senza BOM, e una prova controlla il file consegnato e non solo il
programma che lo produce.

## Note di riproducibilità

Il seme controlla il campionamento delle etichette, l'inizializzazione della
memoria e il sottocampione usato per l'addestramento iniziale di MLP e KAN. La
regressione logistica è addestrata su tutto A ed è quindi deterministica: il
suo AUROC congelato vale 0,7149 in tutti e cinque i semi.

Le righe campionate sono le stesse in tutte le configurazioni dello stesso
seme, e lo si verifica sullo SHA-256 registrato per blocco in
`row_id_campionati_sha256` — non più sulla somma
`row_id_campionati_somma_storica`, che è un controllo più debole e resta solo
per continuità con i rendiconti pubblicati prima di questo ciclo.

# Evidenza del replay su C

## Il formato dei rendiconti per blocco

I metadati sono indentati, perché sono quelli che si leggono; **ogni blocco sta
su una riga sola**. Resta JSON valido — `json.load` lo apre senza sapere nulla
di questa scelta — ma un rendiconto di 916 blocchi occupa **1.133 righe**
invece di 135.735, e un diff mostra quali blocchi sono cambiati invece di
spalmare la differenza su migliaia di righe. Con venti rendiconti la differenza
è fra 22.660 righe e 2,7 milioni. I due numeri sono misurati sui rendiconti di
questo ciclo, che portano più campi per blocco dei precedenti: sui rendiconti
del commit `c5e6d02` erano 1.100 e 93.500.

`replay_C_seme42.json` era pubblicato nel formato indentato: la prima versione
di questo commit lo riscrive compatto, a contenuto invariato.

## Tutte le corse sono state rieseguite con il codice corretto

Il riesame del 3 ottobre ha segnalato cinque difetti; tre riguardavano i
rendiconti. I ventuno rendiconti per blocco di questa cartella — le venti corse
più la variante scartata sul campionamento stratificato — **non sono quelli
pubblicati in precedenza**: sono una riesecuzione completa, una sola, fatta
dopo aver corretto il codice. Che cosa cambia nei file:

- ogni blocco porta `row_id_campionati_sha256`, lo SHA-256 dei `row_id`
  campionati su una serializzazione dichiarata, e il campo storico — che era una
  **somma** modulo 10⁹ — si chiama ora `row_id_campionati_somma_storica`;
- la quota di etichette ha per denominatore `righe_scorse`, le righe
  effettivamente scorse, e non `blocchi × dimensione del blocco`;
- `costi` porta il tempo del **singolo aggiornamento** per modello, i tempi dei
  tentativi fermati dalla guardia, la memoria misurata e, in
  `ambiente.hardware_misurato`, l'hardware letto dal sistema. Il picco di
  memoria qui registrato viene da `resource.getrusage` su Linux; dalle corse
  successive a questo commit il rendiconto porta anche
  `costi.memoria.picco_rss_metodo`, aggiunto perche' su Windows la lettura passa
  per `GetProcessMemoryInfo` e, dove nessuna delle due vie e' disponibile, il
  campo dice che la misura non c'e' invece di restare vuoto;
- ogni blocco in cui una politica ha tentato un aggiornamento porta
  `ms_aggiornamento_<modello>`.

**Le misure non cambiano.** `riepiloghi_semi.json` rigenerato da questi
rendiconti non differisce in nessun campo da quello pubblicato prima: AUROC,
matrici di confusione, salti, bande e regime di inversione si riproducono
identici. Le correzioni riguardano i campi del rendiconto e la formulazione dei
numeri, non i risultati.

## I rendiconti

Venti corse, quattro configurazioni per cinque semi da 42 a 46. Tutte sugli
stessi blocchi, con gli stessi `row_id` campionati.

| nome | soglia | politica di aggiornamento |
|---|---|---|
| `replay_C_seme<n>.json` | zero | ogni blocco |
| `replay_C_calibrato_seme<n>.json` | scelta su B | ogni blocco |
| `replay_C_evidenza_seme<n>.json` | scelta su B | solo su evidenza di inversione |
| `replay_C_casuale_seme<n>.json` | scelta su B | su altrettanti blocchi a sorte |

La prima riga è il **riferimento senza calibrazione**. Le altre tre usano la
soglia scelta su B col massimo della balanced accuracy, una per modello,
identica e fissa nella coppia congelato / adattivo; il campo
`soglia_di_decisione` di ciascun rendiconto porta la soglia, la balanced
accuracy raggiunta, quella che si avrebbe a soglia zero e il numero di
candidate a pari merito.

La quarta è il **controllo negativo**: aggiorna sullo stesso numero di blocchi
della terza, modello per modello, estratti fra i blocchi in cui un
aggiornamento è davvero possibile. Estraendoli fra tutti i blocchi, il 43%
delle estrazioni cadrebbe su un blocco con memoria monoclasse, la guardia lo
salterebbe, e il controllo applicherebbe quasi la metà degli aggiornamenti che
deve pareggiare.

C'è inoltre `replay_C_seme42_strati_punteggio.json`, la variante con il budget
distribuito su dieci fasce del punteggio: risultato negativo, i salti salgono
da 393 a 421. **È anch'esso seme 42**, quindi non va passato insieme a
`replay_C_seme42.json` agli script, che rifiutano due rendiconti con lo stesso
seme.

## La soglia: candidati e regola di confronto

- `soglie_candidati.json` — per ciascuna delle quindici coppie (modello, seme):
  i **candidati distinti** sull'intero B, quanti pareggiano il massimo della
  balanced accuracy a uguaglianza esatta e con tolleranza 1e-12, 1e-9 e 1e-6,
  l'indice scelto dalla **mediana inferiore**, e i tre candidati immediatamente
  prima e dopo quello scelto. Prodotto da `documenta_soglie.py`.

  Il dato che conta: a uguaglianza esatta il massimo è sempre raggiunto da **un
  candidato solo**, quindi la regola di parità non è mai stata applicata; con una
  tolleranza di 1e-6 pareggerebbero da 1 a 27 candidati e la soglia cambierebbe.
  Per questo la regola di confronto è dichiarata e non sottintesa.

  Il campo `soglia_di_decisione.regola` dentro i rendiconti per blocco descrive
  la stessa regola con la formulazione usata quando quelle corse sono state
  eseguite — «la mediana delle candidate, e con numero pari il minore dei due
  valori centrali» — che è equivalente a «la mediana inferiore». La formulazione
  autorevole è quella di `soglie_candidati.json` e del §10bis del protocollo.

## La misura dedicata dei tempi: `tempi/`

Il tempo del singolo aggiornamento è registrato in tutte le venti corse, ma
quelle corse sono state eseguite mentre sulla stessa macchina a due CPU girava
altro lavoro: là il rapporto fra il massimo e il minimo dello stesso modello
arriva a 178 volte, con massimi fino a 384 ms, ed è rumore di scheduling e non
il calcolo. Per avere una misura utilizzabile sono state
rifatte due corse sole, in sequenza, con la macchina altrimenti inattiva:

- `tempi/ogni_blocco_seme42_costi.json` e `tempi/evidenza_seme42_costi.json` —
  metadati, `costi` e i tempi per blocco. Le misure per blocco **non** sono
  ripubblicate: sono identiche, campo per campo sui 916 blocchi, a quelle di
  `replay_C_calibrato_seme42.json` e `replay_C_evidenza_seme42.json`, e il campo
  `per_blocco_rimosso` lo dichiara.
- `tempi/prof_ogni_blocco.txt` e `tempi/prof_evidenza_inversione.txt` — l'esito
  di `cProfile` su 200 blocchi delle due politiche, usato per attribuire la
  differenza fra i due tempi di replay. Serve a mostrare che la differenza sta
  nei rifitting della logistica (587 chiamate contro 24) e nella
  rappresentazione a B-spline della memoria (625 contro 431), cioè nello stesso
  lavoro che il cronometro misura.

## I riepiloghi

- `ripetizioni_cinque_semi.json` — il consolidato prodotto durante le prime
  corse: AUROC, richiamo sui normali medio per blocco, salti, esito
  dell'esperimento sul campionamento stratificato.
- `riepiloghi_semi.json` e `.txt` — le **due aggregazioni** del tasso di falsi
  allarmi, il regime di inversione **per seme**, le bande di ricchezza di
  normali.
- `confronto_soglia.json` e `.txt` — soglia zero contro soglia scelta su B.
- `confronto_politiche.json` e `.txt` — le quattro politiche, più il primo passo:
  quanto il verso sia stimabile dalle sole etichette già arrivate. Il campo
  `per_seme` riporta le misure **seme per seme** e conta su quanti semi ciascun
  confronto vale: serve perché una media favorevole non autorizza a dire che il
  confronto valga sempre, e su un seme si rovescia.
- `costi_aggiornamento.json` e `.txt` — il costo del singolo aggiornamento per
  modello e per politica, la memoria, l'hardware, e che cosa il numero di
  aggiornamenti **non** dimostra.

## Come rifare tutto

Da dentro questa cartella, su Linux e macOS. Ogni comando sta su una riga sola:
si incolla come e' scritto.

```
cd <repo>/adattamento-drift/ton_temporal/replay_evidenza
Z="replay_C_seme42.json replay_C_seme43.json replay_C_seme44.json replay_C_seme45.json replay_C_seme46.json"
B="replay_C_calibrato_seme42.json replay_C_calibrato_seme43.json replay_C_calibrato_seme44.json replay_C_calibrato_seme45.json replay_C_calibrato_seme46.json"
python ../riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json --verifica
python ../riepiloghi_semi.py --rendiconti $Z --confronto $B --uscita confronto_soglia.json --verifica
python ../confronto_politiche.py --ogni-blocco $B --evidenza replay_C_evidenza_seme4?.json --casuale replay_C_casuale_seme4?.json --uscita confronto_politiche.json
python ../costi_aggiornamento.py --rendiconti replay_C_*seme4?.json --uscita costi_aggiornamento.json
```

In PowerShell, con le liste separate da virgole e il globbing risolto prima:

```
cd <repo>/adattamento-drift/ton_temporal/replay_evidenza
$Z = "replay_C_seme42.json","replay_C_seme43.json","replay_C_seme44.json","replay_C_seme45.json","replay_C_seme46.json"
$B = "replay_C_calibrato_seme42.json","replay_C_calibrato_seme43.json","replay_C_calibrato_seme44.json","replay_C_calibrato_seme45.json","replay_C_calibrato_seme46.json"
python ..\riepiloghi_semi.py --rendiconti $Z --confronto $B --verifica
python ..\confronto_politiche.py --ogni-blocco $B --evidenza (Get-Item replay_C_evidenza_seme4*.json).Name --casuale (Get-Item replay_C_casuale_seme4*.json).Name
python ..\costi_aggiornamento.py --rendiconti (Get-Item replay_C_*seme4?.json).Name
```

Gli script non si limitano a calcolare. `riepiloghi_semi.py` verifica che in
ogni blocco `fp + vn` faccia i normali e `vp + fn` gli attacchi, che le bande
coprano tutti i blocchi, e che la variazione di falsi positivi sommata sulle
bande coincida con quella totale; con `--confronto` verifica inoltre che
**l'AUROC e gli indici campionati siano identici** fra i due insiemi — 27.480
valori — perché una soglia non può cambiare una misura di ordinamento.
`confronto_politiche.py` verifica che la copia congelata sia identica nelle tre
politiche e che il controllo casuale pareggi gli aggiornamenti. Entrambi si
fermano al primo scostamento.

## Come rifare le corse da zero

Le quattro corse del **seme 42**. Per gli altri semi si cambiano i `42` e i
conteggi di `--quanti-aggiornamenti`, elencati sotto.

```
cd <repo>/adattamento-drift/ton_temporal/replay_evidenza
python ../replay.py --iniziale ../flussi/A.npz --flusso ../flussi/C.npz --seme 42 --uscita replay_C_seme42.json
python ../replay.py --iniziale ../flussi/A.npz --flusso ../flussi/C.npz --calibrazione ../flussi/B.npz --seme 42 --uscita replay_C_calibrato_seme42.json
python ../replay.py --iniziale ../flussi/A.npz --flusso ../flussi/C.npz --calibrazione ../flussi/B.npz --politica evidenza_inversione --seme 42 --uscita replay_C_evidenza_seme42.json
python ../replay.py --iniziale ../flussi/A.npz --flusso ../flussi/C.npz --calibrazione ../flussi/B.npz --politica casuale --quanti-aggiornamenti "lr=54,mlp=20,kan=23" --ammissibili replay_C_calibrato_seme42.json --seme 42 --uscita replay_C_casuale_seme42.json
```

I conteggi per il controllo casuale, uno per seme: 42 `lr=54,mlp=20,kan=23`;
43 `lr=43,mlp=35,kan=16`; 44 `lr=44,mlp=45,kan=24`; 45 `lr=49,mlp=30,kan=13`;
46 `lr=36,mlp=39,kan=21`. Sono gli aggiornamenti applicati dalla politica su
evidenza dello stesso seme, che si leggono in `costi.decisioni_della_politica`:
cambiarli renderebbe il controllo non pareggiato.

I conteggi da passare al controllo casuale sono quelli applicati dalla politica
su evidenza dello stesso seme, che si leggono in
`costi.decisioni_della_politica`. I flussi `A.npz`, `B.npz` e `C.npz` si
ricostruiscono con `costruisci_flusso.py` dai CSV originali, che **non vengono
redistribuiti**: sono identificati per SHA-256 in `../data_inventory.csv` e in
`../verifica_sha256_fonte.json`.

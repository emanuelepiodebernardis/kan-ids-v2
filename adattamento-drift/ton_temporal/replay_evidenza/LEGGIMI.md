# Evidenza del replay su C

## Il formato dei rendiconti per blocco

I metadati sono indentati, perché sono quelli che si leggono; **ogni blocco sta
su una riga sola**. Resta JSON valido — `json.load` lo apre senza sapere nulla
di questa scelta — ma un rendiconto di 916 blocchi occupa circa 1.100 righe
invece di 93.500, e un diff mostra quali blocchi sono cambiati invece di
spalmare la differenza su migliaia di righe. Con venti rendiconti la differenza
è fra 22.000 righe e 1.870.000.

`replay_C_seme42.json` era pubblicato nel formato indentato: la prima versione
di questo commit lo riscrive compatto, a contenuto invariato.

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
accuracy raggiunta, quella che si avrebbe a soglia zero e il numero di candidate
a pari merito.

La quarta è il **controllo negativo**: aggiorna sullo stesso numero di blocchi
della terza, modello per modello, estratti fra i blocchi in cui un aggiornamento
è davvero possibile. Estraendoli fra tutti i blocchi, il 43% delle estrazioni
cadrebbe su un blocco con memoria monoclasse, la guardia lo salterebbe, e il
controllo applicherebbe quasi la metà degli aggiornamenti che deve pareggiare.

C'è inoltre `replay_C_seme42_strati_punteggio.json`, la variante con il budget
distribuito su dieci fasce del punteggio: risultato negativo, i salti salgono da
393 a 421. **È anch'esso seme 42**, quindi non va passato insieme a
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

## I riepiloghi

- `ripetizioni_cinque_semi.json` — il consolidato prodotto durante le prime
  corse: AUROC, richiamo sui normali medio per blocco, salti, esito
  dell'esperimento sul campionamento stratificato.
- `riepiloghi_semi.json` e `.txt` — le **due aggregazioni** del tasso di falsi
  allarmi, il regime di inversione **per seme**, le bande di ricchezza di
  normali.
- `confronto_soglia.json` e `.txt` — soglia zero contro soglia scelta su B.
- `confronto_politiche.json` e `.txt` — le quattro politiche, più il primo passo:
  quanto il verso sia stimabile dalle sole etichette già arrivate.

## Come rifare tutto

```
cd replay_evidenza
Z="replay_C_seme42.json replay_C_seme43.json replay_C_seme44.json replay_C_seme45.json replay_C_seme46.json"
B="replay_C_calibrato_seme42.json replay_C_calibrato_seme43.json replay_C_calibrato_seme44.json replay_C_calibrato_seme45.json replay_C_calibrato_seme46.json"

python ../riepiloghi_semi.py --rendiconti $Z --uscita riepiloghi_semi.json --verifica
python ../riepiloghi_semi.py --rendiconti $Z --confronto $B --uscita confronto_soglia.json --verifica
python ../confronto_politiche.py --ogni-blocco $B \
    --evidenza replay_C_evidenza_seme4*.json --casuale replay_C_casuale_seme4*.json \
    --uscita confronto_politiche.json
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

```
B="--iniziale A.npz --flusso C.npz"
python replay.py $B --seme <n> --uscita replay_C_seme<n>.json
python replay.py $B --calibrazione B.npz --seme <n> --uscita replay_C_calibrato_seme<n>.json
python replay.py $B --calibrazione B.npz --politica evidenza_inversione --seme <n> \
    --uscita replay_C_evidenza_seme<n>.json
python replay.py $B --calibrazione B.npz --politica casuale \
    --quanti-aggiornamenti "lr=54,mlp=20,kan=23" \
    --ammissibili replay_C_calibrato_seme<n>.json --seme <n> \
    --uscita replay_C_casuale_seme<n>.json
```

I conteggi da passare al controllo casuale sono quelli applicati dalla politica
su evidenza dello stesso seme, che si leggono in
`costi.decisioni_della_politica`. I flussi `A.npz`, `B.npz` e `C.npz` si
ricostruiscono con `costruisci_flusso.py` dai CSV originali, che **non vengono
redistribuiti**: sono identificati per SHA-256 in `../data_inventory.csv` e in
`../verifica_sha256_fonte.json`.

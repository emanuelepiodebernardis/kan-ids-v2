# ADAPT-06 — acquisizione informativa e promozione prospettica

Il protocollo è stato approvato al commit
`3199f1eeba952a20f2be5fd60fee8b0317123627` e integrato in `main` col merge
`e5163bce892a0aafdd5136ee7c6a618b9867eadc`. Questa directory contiene ora
anche il checkpoint separato di implementazione e test e i riepiloghi leggeri
del pilot seed 42 autorizzato dopo il merge della PR #13. Non contiene una
campagna sui cinque seed.

TON e UNSW restano stream di sviluppo già esaminati; una conferma indipendente
appartiene ad ADAPT-07.

## Protocollo congelato

- [`PROTOCOL.md`](PROTOCOL.md): ipotesi primaria, budget, gate e criteri;
- [`TEMPORAL_SCHEMA.md`](TEMPORAL_SCHEMA.md): stato e pseudocodice causale;
- [`LABEL_BUDGET.csv`](LABEL_BUDGET.csv): contabilità esatta delle richieste;
- [`MATRIX.csv`](MATRIX.csv): matrice minima proposta, non ancora eseguita;
- [`COST_ESTIMATE.md`](COST_ESTIMATE.md): costo previsto e basi della stima;
- [`LITERATURE_COMPARISON.md`](LITERATURE_COMPARISON.md): confronto con le
  fonti indicate nella scheda, senza claim di priorità.

## Checkpoint codice e test

- [`core.py`](core.py): budget, RNG, selezione, memorie, calibrazione, gate e
  aggregazione dei cinque seed;
- [`models.py`](models.py): adattatori degli stati A/B archiviati, senza nuovi
  training iniziali;
- [`runner.py`](runner.py): quattro tracce appaiate GAM/LR, percorso ARF
  FIT-only, baseline Frozen/B e Frozen/CAL256-target5;
- [`test_core.py`](test_core.py), [`test_models.py`](test_models.py),
  [`test_runner.py`](test_runner.py) e [`test_contract.py`](test_contract.py):
  casi sintetici nominali e negativi e coerenza dei file preregistrati;
- [`REPRODUCTION.md`](REPRODUCTION.md): comandi e confine di autorizzazione;
- [`TEST_LOG.txt`](TEST_LOG.txt) e [`CHECKPOINT_MANIFEST.json`](CHECKPOINT_MANIFEST.json):
  esito consegnato e hash dei file del checkpoint;
- [`verify_checkpoint.py`](verify_checkpoint.py): verifica locale di hash,
  matrice e log; con un'opzione separata riesegue soltanto i test sintetici.

Il primo anchor ACTIVE, quando FIT192 è ancora vuota, è l'osservazione con
`row_id` minimo nella shortlist; dai passi successivi si applica farthest-first
rispetto agli anchor già presenti. Questo tie-break rende deterministico il
caso degenere senza usare etichette.

Prima di un replay il runner verifica tutti gli otto array di `D.npz` contro
la provenienza archiviata. Un contenitore NPZ ricostruito può avere byte ZIP
diversi; in tal caso vengono conservati entrambi gli hash del contenitore, ma
l'esecuzione è ammessa soltanto se schema, dtype, forme, conteggi e SHA-256 di
ogni array coincidono.

## Pilot seed 42, primi 100 blocchi

Il pilot è stato eseguito dal merge della PR #13
`56cbfd9de62569bb1b72b63c0ec1640e4df2d318`, sul commit congelato
`ceaf0e14a2619bbc38189a7cee046abcb0647994`, usando `--max-blocks 100` su TON
e UNSW. I 12 JSON corrispondono a 20 traiettorie adattive. I riepiloghi,
le decisioni, i costi e il puntatore al pacchetto completo sono in
[`pilot_seed42/`](pilot_seed42/README.md).

## Confine di autorizzazione

L'autorizzazione copre soltanto il pilot seed 42 sui primi 100 blocchi dei due
stream. La matrice completa sui cinque seed richiede una decisione ulteriore e
non è stata avviata. La configurazione non viene modificata in base agli esiti
del pilot.

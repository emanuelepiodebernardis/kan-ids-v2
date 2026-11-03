# ADAPT-05 — occurrence-wise FIT/CAL roles

Questa directory è separata e additiva: non modifica runner, modelli, report o
pacchetti DTRAP. Leggere `PROTOCOL.md` prima di eseguire codice.

## Test sintetici

Da `<repo>`:

```powershell
python -m pytest -q adattamento-drift/adapt05_occurrence/test_runner.py
```

I test coprono compatibilità vettoriale, codifica versionata, ripetizioni,
disgiunzione per `observation_id`, indipendenza dalle label, delay, quote,
espulsione, età, classi mancanti, determinismo/restart, query comuni e ultimo
blocco parziale. Verificano inoltre struttura, conteggi, identità delle righe e
SHA-256 ricalcolati dagli array realmente letti dall'NPZ, comprese controprove
su `X`, `y` e `row_id`, e un replay sintetico end-to-end con le quattro tracce
appaiate.

## Smoke preregistrato

I dati ricostruiti non sono conservati in Git. La verifica di compatibilità
numerica usa l'ambiente archiviato (Python 3.12.14 e dipendenze di
`review_extension/requirements.txt`) con BLAS a un thread. Il rendiconto
registra piattaforma, processore e runtime BLAS; la verifica numerica non viene
confusa con le invarianti strutturali:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python adattamento-drift/adapt05_occurrence/runner.py `
  --data <data-root>/ton --dataset ton --seed 42 --salt default `
  --max-blocks 30 --out <out>/ton_seed42_default.json
python adattamento-drift/adapt05_occurrence/runner.py `
  --data <data-root>/unsw --dataset unsw --seed 42 --salt default `
  --max-blocks 30 --out <out>/unsw_seed42_default.json
```

Il runner fallisce se query, memorie limitate o stati di aggiornamento del
controllo vettoriale divergono dai rendiconti archiviati del Paper 2. Digest
dei parametri, soglie e confusion matrix sono confrontati e quantificati
separatamente; una differenza numerica fra processori non viene presentata
come identita' byte per byte.

I due rendiconti consegnati, gli hash, l'esito della compatibilità e la stima
dei tempi della matrice completa sono in [`smoke/`](smoke/README.md).

## Campagna completa

Dopo l'approvazione del checkpoint `e030bd8`, la matrice congelata si esegue
con `campaign.py` in una directory nuova. `analyze.py` valida tutte le 30
combinazioni e produce tabelle, differenze appaiate, copertura, andamento
temporale e figure; `package_campaign.py` genera manifest e archivio
deterministico. Comandi e ambiente sono descritti in
[`REPRODUCTION.md`](REPRODUCTION.md).

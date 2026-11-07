# Riproduzione del checkpoint codice/test ADAPT-06

## Perimetro

Questi comandi verificano esclusivamente implementazione, invarianti sintetiche
e provenienza del checkpoint. Non aprono TON o UNSW, non eseguono fit iniziali,
smoke, pilot o campagne.

Eseguire dalla radice del clone, cioè dalla directory che contiene
`adattamento-drift/`.

## Ambiente usato su Windows

- Windows 11;
- Python 3.13;
- dipendenze già fissate dal repository per il Paper 2.

PowerShell, usando il launcher Python quando disponibile:

```powershell
py -3.13 -m pytest `
  adattamento-drift/adapt06_prospective_gate/test_core.py `
  adattamento-drift/adapt06_prospective_gate/test_contract.py `
  adattamento-drift/adapt06_prospective_gate/test_models.py `
  adattamento-drift/adapt06_prospective_gate/test_runner.py -q

py -3.13 adattamento-drift/adapt06_prospective_gate/verify_checkpoint.py
```

Se `py` non è disponibile, sostituire `py -3.13` col percorso dell'interprete
dell'ambiente. Non usare l'alias Microsoft Store `python.exe` se non punta a un
interprete eseguibile.

Il primo comando deve coincidere con [`TEST_LOG.txt`](TEST_LOG.txt); il secondo
verifica hash, matrice non autorizzata e contenuto minimo del log.

## Comando del futuro pilot

Il runner accetta una singola cella. Il comando seguente documenta l'interfaccia
ma **non è autorizzato in questo checkpoint**:

```powershell
py -3.13 adattamento-drift/adapt06_prospective_gate/runner.py `
  --data <data-root>/ton `
  --model-root adattamento-drift/review_extension/models `
  --dataset ton `
  --family spline `
  --acquisition active `
  --seed 42 `
  --max-blocks <numero-blocchi-pilot-autorizzato> `
  --out <nuova-directory>/ton_spline_seed42_active.json
```

Il percorso di output deve essere nuovo. Per il pilot autorizzato saranno
eseguite le celle prefissate su TON e UNSW, conservando comando effettivo,
commit, ambiente, tempi e memoria. Non si sostituisce `--max-blocks` con una
corsa completa senza una decisione successiva.

La fine del prefisso produce `PENDING_AT_PILOT_END` per un candidato ancora
aperto: non simula la fine reale dello stream, non forza un withholding e non
trasferisce retroattivamente VERIFY in CAL.

## Controlli coperti

I test verificano:

1. causalità delay-1 e primo effetto in `k+2`;
2. budget intero e ultimo blocco parziale;
3. CONTROL comune, FIT disgiunto e separazione FIT/CAL/VERIFY;
4. memorie per classe, prestiti FIFO ed età dall'osservazione;
5. riferimento primario allo stato operativo congelato, con incumbent
   ricalibrato soltanto diagnostico;
6. adozione atomica di modello e soglia dal blocco successivo e divieto di un
   secondo candidato durante VERIFY;
7. vincoli FPR/recall, ultimo look, scadenza e fine stream;
8. inclusione dei seed con FPR sfavorevole nell'aggregazione;
9. calendario causale Frozen/CAL256-target5, compreso il ricalcolo terminale
   privo di effetto valutato;
10. percorso ARF FIT-only e assenza del gate prospettico su ARF;
11. presenza obbligatoria del digest SHA-256 di `D.npz` nei metadati A/B;
12. uguaglianza e valori appena sotto/sopra i margini inclusivi di recall e
    FPR, usando conteggi esatti nel gate e una tolleranza limitata al solo
    arrotondamento nella media fra seed.

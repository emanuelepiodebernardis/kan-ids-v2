# ADAPT-06D — preflight shadow FIT192/FIT256

Questa directory prepara il confronto shadow sui tre `FREEZE` originali di
UNSW/seed42. La consegna contiene soltanto protocollo, configurazione e
inventari causali: non esegue fit, scoring o replay.

## Dipendenze

Nel repository devono essere presenti il report decoupled verificato e gli
stati A/B `model.pkl` e `ready.json`. L'input esterno ricostruito richiede
`D.npz` e `D.json`; i loro SHA-256 sono registrati in `CONFIG.json` e
`MANIFEST.json`. In assenza di questi file la rigenerazione deve fermarsi,
senza sostituzioni.

Il manifest registra inoltre le implementazioni effettivamente usate per
memoria, modello, verifica degli input e preprocessing, oltre all'hash della
sequenza di richieste conservato nel report.

## Rigenerazione su Windows

Dalla radice del clone:

```powershell
python adattamento-drift/adapt06d_shadow_fit_preflight/build_preflight.py `
  --data-dir C:\percorso\unsw
python -m pytest adattamento-drift/adapt06d_shadow_fit_preflight -q
```

La cartella passata a `--data-dir` deve contenere `D.npz` e `D.json`. Il
comando verifica prima provenienza, array, modello e metadata; ricostruisce poi
le memorie con la classe originale e rigenera:

- `PREFLIGHT_CASES.csv`: tre casi, stati e controlli di separazione;
- `PREFLIGHT.md`: tabella leggibile e stato dei controlli;
- `FIT_INVENTORY.csv`: composizione e digest FIT192/FIT256;
- `FIT_MEMBERS.csv`: appartenenza completa e ordine dei due buffer;
- `CAL_MEMBERS.csv` e `VERIFY_MEMBERS.csv`: osservazioni fissate per caso;
- `CONFIG.json`, con gli stati completi fissati e la semantica del confronto,
  e `MANIFEST.json`.

I JSON sorgente e gli artefatti esterni sono aperti in sola lettura e non sono
copiati nel pacchetto Git.

## Pilot shadow autorizzato

Il pilot conserva `89394d542f5b3eaa56e63b06348de251cc48e52a` come
riferimento del protocollo. Deve essere eseguito da un commit pulito, con gli
stessi limiti ai thread del report storico. In PowerShell, dalla radice del
clone:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python adattamento-drift/adapt06d_shadow_fit_preflight/run_shadow_pilot.py `
  --data-dir C:\percorso\unsw
```

Il runner rigenera prima gli otto prodotti del preflight e ne richiede
l'identità byte per byte. Esegue quindi i tre controlli FIT192 e consente i tre
FIT256 soltanto se parametri, calibrazioni, confusioni e decisioni FIT192
coincidono con il riferimento entro le tolleranze preregistrate. Non effettua
promozioni e non modifica la traiettoria.

Dopo i test, il manifest della consegna si rigenera con:

```powershell
python adattamento-drift/adapt06d_shadow_fit_preflight/build_shadow_manifest.py `
  --data-dir C:\percorso\unsw
```

I prodotti del pilot sono `SHADOW_RESULTS.json`, `SHADOW_COMPARISON.csv`,
`SHADOW_REPORT.md`, `SHADOW_ENVIRONMENT.json`, `SHADOW_RUN_LOG.txt`,
`SHADOW_TEST_LOG.txt` e `SHADOW_MANIFEST.json`.

## Riscontro di portabilita' Windows/Linux

La verifica successiva al pilot non ripete alcun fit. Le tre righe richieste,
espresse come parole IEEE 754 binary32 per preservarne esattamente i bit, sono
in `TRANSFORM_PORTABILITY.md`; `NUMPY_SHOW_RUNTIME_WINDOWS.txt` registra il
runtime numerico dell'esecuzione Windows. La nota separa le differenze della
trasformazione dal risultato del pilot e non modifica protocollo, margini,
decisioni o traiettoria.

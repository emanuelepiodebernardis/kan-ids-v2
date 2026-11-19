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

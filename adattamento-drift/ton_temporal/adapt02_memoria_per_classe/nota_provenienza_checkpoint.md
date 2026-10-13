# ADAPT-02: nota di provenienza dello smoke approvato

Questa nota separa lo SHA del codice di replay effettivamente usato nello smoke
dal commit che ha poi versionato lo script di confronto storico e i rendiconti.

## Punti fermi

Lo smoke approvato ha eseguito `replay.py` con il commit:

`a30898523f4573efaacfda45f5c1cc77a7382992`

Il PR e' stato poi aggiornato al commit:

`831ee2597744e4c9ca8aafcc463f0ebc657eefb5`

Fra questi due commit `adattamento-drift/ton_temporal/replay.py` non cambia. La
correzione del dtype della memoria e' quindi la stessa in entrambi.

## Modifiche presenti durante lo smoke

Durante la ripetizione dello smoke il working tree conteneva, oltre a
`replay.py` gia' corretto in `a308985`, modifiche locali allo script
`smoke_memoria_per_classe.py`. Queste modifiche sono state poi versionate nel
commit `831ee2597744e4c9ca8aafcc463f0ebc657eefb5`.

Le modifiche locali riguardavano solo il controllo di retrocompatibilita':

- caricamento della FIFO storica da
  `1679f65697ccdf1e92710ebfcbb940ce36233e7e` tramite `git show`;
- copia di `soglia_bilanciata.py` nella directory temporanea usata dal modulo
  storico;
- scrittura dei due rendiconti FIFO storici in `smoke_seed42/fifo_storica/`;
- confronto automatico fra FIFO corrente e FIFO storica su riepilogo e campi
  per blocco;
- registrazione in `riepilogo_smoke.json` di `commit_codice_eseguito` e
  `commit_fifo_storica`.

Lo smoke finale non ha usato `--riusa-rendiconti-esistenti`: le quattro corse
correnti e le due FIFO storiche sono state rieseguite.

## Comando dello smoke approvato

```bash
cd <repo>/adattamento-drift/ton_temporal
python smoke_memoria_per_classe.py --iniziale flussi/A.npz --flusso flussi/C.npz --calibrazione flussi/B.npz --uscita-dir adapt02_memoria_per_classe/smoke_seed42 --seme 42 --max-blocchi 24
```

Lo SHA `a30898523f4573efaacfda45f5c1cc77a7382992` indica il codice di replay
eseguito; lo SHA `831ee2597744e4c9ca8aafcc463f0ebc657eefb5` indica il commit
che conserva lo script di confronto storico e gli artefatti dello smoke.

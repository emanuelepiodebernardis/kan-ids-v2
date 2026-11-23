# Riscontro Windows/Linux sulle feature trasformate

## Perimetro

Questa verifica riguarda esclusivamente la portabilita' della trasformazione
archiviata per UNSW/spline/seed42. Non sono stati ripetuti fit, calibrazioni,
scoring o decisioni del pilot. Il protocollo resta `89394d5`; il pilot resta
quello eseguito a `b21cdad` e consegnato a `707f726`.

I due input letti nell'ambiente Windows hanno gli SHA-256 attesi:

- `D.npz`: `c20c87d2eb977531bb101ad9170efcfafdb728036dd1e6963a812675242ee3e6`;
- `D.json`: `31f3c76177a1056b8952eac68a6f4f1d74a85e6b788e7ba35dbe52b617328b04`.

Per ciascun `observation_id` e' stata individuata la riga in `D.npz/row_id`,
applicata una sola volta la trasformazione contenuta nello stato A/B
archiviato e convertito il risultato con
`np.asarray(riga_trasformata, dtype="<f4").view("<u4")`. Le colonne sono
nell'ordine del modello. Ogni parola seguente rappresenta esattamente i 32 bit
IEEE 754 della feature; non e' una stampa decimale arrotondata.

## Tre righe richieste

| observation_id | Linux, otto parole binary32 | Windows, otto parole binary32 |
|---:|---|---|
| 1670187 | `3f02822a be94318f bf67820c be996b53 bf3f66b4 bec16a8a bf2d3821 4157111c` | `3f02822c be94318f bf67820c be996b53 bf3f66b4 bec16a8a bf2d3823 4157111c` |
| 1869967 | `bef9cdda bfa2f868 c042113c bfa926ed bf9bfedb bfaef6b7 c0027d1d be02ceb8` | `bef9cdda bfa2f86a c042113c bfa926ed bf9bfedb bfaef6b7 c0027d1d be02ceb8` |
| 2093794 | `3f7a7137 beab9609 bdf97356 be996b53 bef1b8a1 bec16a8a beb86f0e 41134a3c` | `3f7a7137 beab9609 bdf9736f be996b53 bef1b8a1 bec16a8a beb86f0e 41134a3c` |

## Esito del confronto

Venti delle 24 feature coincidono bit per bit. Le quattro differenze, con
indici di colonna 0-based, sono:

| observation_id | colonna | Linux `float.hex` | Windows `float.hex` | distanza binary32 | differenza assoluta |
|---:|---:|---|---|---:|---:|
| 1670187 | 0 | `0x1.0504540000000p-1` | `0x1.0504580000000p-1` | 2 ULP | `1.1920928955078125e-07` |
| 1670187 | 6 | `-0x1.5a70420000000p-1` | `-0x1.5a70460000000p-1` | 2 ULP | `1.1920928955078125e-07` |
| 1869967 | 1 | `-0x1.45f0d00000000p+0` | `-0x1.45f0d40000000p+0` | 2 ULP | `2.384185791015625e-07` |
| 2093794 | 2 | `-0x1.f2e6ac0000000p-4` | `-0x1.f2e6de0000000p-4` | 25 ULP | `1.862645149230957e-07` |

Il riscontro localizza quindi il fallimento del preflight Linux in differenze
di ultimo bit della trasformazione numerica: input, `observation_id`, ordine ed
etichette restano identici. Queste differenze bastano a cambiare i digest
SHA-256 delle righe trasformate. Il confronto non dimostra l'identita' dei fit
fra piattaforme e non la assume: i sei fit del pilot non sono stati ripetuti.

L'ambiente numerico Windows e' registrato integralmente in
`NUMPY_SHOW_RUNTIME_WINDOWS.txt`. In quel file sono normalizzati soltanto il
nome host e il prefisso assoluto dell'ambiente; versioni, backend, architettura,
thread e nomi delle librerie sono quelli restituiti da `numpy.show_runtime()`
dopo il caricamento dello stato A/B.

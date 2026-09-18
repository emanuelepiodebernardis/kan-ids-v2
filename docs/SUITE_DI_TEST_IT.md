# Le suite di test del repository, e come eseguirle

Le suite sono due gruppi, con radici diverse. Eseguirle tutte insieme dalla
radice del repository non funziona, e non per un difetto da correggere: i
progetti sotto `experiments/` sono snapshot congelati di campagne di
validazione hardware, ciascuno con la propria radice, i propri
`requirements.txt` e i propri test, e sono fatti per essere eseguiti dalla
loro cartella.

## 1. La suite principale

Dalla radice del repository:

```
pytest -q -rs
```

Raccoglie `tests/` e tutto ciò che sta fuori da `experiments/`, che
`conftest.py` esclude dichiarandone il motivo.

Esito su Linux, Python 3.11, con `g++` disponibile e senza toolchain AVR:

```
609 passed, 26 skipped, 5 subtests passed
```

I 26 skip non sono fallimenti mascherati: ognuno dichiara che cosa manca e
dove è stato cercato. Si dividono in tre famiglie.

| Quanti | Motivo | Come farli diventare esecuzioni |
|---|---|---|
| 21 | `avr-g++` non trovato | installare un environment AVR di PlatformIO: `pio pkg install -t toolchain-atmelavr`. La ricerca guarda `$AVR_CXX`, il PATH e i pacchetti PlatformIO |
| 4 | dataset o artefatti di calibrazione assenti | servono `data/train_test_network.csv` (TON_IoT) e `artifacts/finalization/`, che si producono con `python reproduce.py --stage lut` |
| 1 | `os.linesep` è già LF | è voluto: il test protegge Windows e su Linux non avrebbe niente da osservare |

## 2. I progetti annidati sotto `experiments/`

Cinque radici indipendenti. Ciascuna si esegue dalla propria cartella, e il
comando è sempre lo stesso; cambia solo da dove lo si lancia.

```
cd experiments/hardware_validation_20260916/<progetto>
pytest -q -rs
```

| Progetto | Esito |
|---|---|
| `hw500/runner/project` | 1 passed, 10 subtests |
| `hw500/runner` | 77 passed, 18 subtests |
| `ram500/runner/project` | 2 passed, 60 subtests |
| `ram500/runner` | 49 passed, 150 subtests |
| `paired_software/source_snapshot` | 19 passed, 4 subtests |

In totale 148 test e 242 subtest, senza fallimenti e senza skip, nello stesso
ambiente in cui gira la suite principale.

### Perché non si raccolgono dalla radice

Per due ragioni indipendenti, entrambe strutturali e nessuna correggibile
senza toccare evidenza congelata.

**Nomi di modulo omonimi fra progetti diversi.** `hw500/runner/project/tests/
test_firmware_host.py` e il suo gemello in `ram500/` hanno lo stesso nome di
modulo; senza pacchetti, pytest li vede come lo stesso modulo e si ferma con
`import file mismatch`. Lo stesso vale per `test_runner.py`. Rinominare quei
file significherebbe modificare l'evidenza di una campagna di misura per
comodità di uno strumento.

**Import che valgono solo dalla radice del progetto.** In
`paired_software/source_snapshot/tests/test_archive_integrity.py` c'è
`from run_stage2 import archive_run`: `run_stage2` sta nella radice di quel
progetto ed è importabile solo da lì.

Una collisione, però, stava **dentro** un singolo progetto e impediva a
`hw500/runner` di eseguire i propri test anche dalla sua cartella:
`tests_continuous/test_runner.py` e `tests_runner/test_runner.py`. È stata
risolta rendendo pacchetti le due cartelle — un `__init__.py` vuoto ciascuna
— così i nomi di modulo diventano `tests_continuous.test_runner` e
`tests_runner.test_runner`, distinti. Nessun file è stato rinominato.

## 3. I test che compilano

Sei test della suite principale e tre dei progetti annidati compilano
davvero: cinque kernel per ATmega2560, per verificare che nel percorso di
inferenza non ci sia una sola istruzione in virgola mobile, e i firmware per
l'host, che vengono eseguiti e confrontati con i golden vector.

Tutti cercano il compilatore con `kanids/toolchain.py`, che guarda nell'ordine
la variabile d'ambiente convenzionale (`$CXX`, `$AVR_CXX`), il PATH e i
pacchetti PlatformIO, e che mette la cartella del compilatore nel PATH del
**solo** sottoprocesso.

Quest'ultimo punto non è un dettaglio. `g++` non è un eseguibile solo: chiama
`as` per assemblare e `ld` per linkare, e li cerca nel PATH. Invocarlo con un
percorso assoluto mentre la sua cartella non è nel PATH produce

```
g++: fatal error: cannot execute 'as'
```

che sembra un compilatore rotto e invece è un compilatore monco. Non è una
differenza fra sistemi operativi: si riproduce identico su Linux indicando il
compilatore con `$CXX` e togliendo la sua cartella dal PATH.

Se il compilatore non c'è, i test si saltano dicendo quale manca e dove è
stato cercato, invece di saltarsi in silenzio.

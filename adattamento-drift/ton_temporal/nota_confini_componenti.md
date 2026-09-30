# Confini con le linee vicine, e componenti riutilizzati

ADAPT-01, 30 settembre 2026.

## 1. Tre domande distinte

Il referente ha posto la distinzione così, e questa attività la rispetta:

- **Paola** studia la **scelta dei modelli e le conseguenze degli errori**.
- **IDS Update Lab** studia la **coerenza della pipeline installata**: che un
  aggiornamento distribuito a un dispositivo preservi la semantica del
  rilevatore.
- **ADAPT-01**, questa attività, studia **come e quando apprendere dai dati
  nuovi**: se piccoli aggiornamenti, con poche etichette e in ritardo, migliorino
  il rilevamento dopo un cambiamento del traffico.

La differenza non è di argomento ma di variabile indipendente. Paola fa variare
il modello a dati fermi; IDS Update Lab fa variare il percorso di distribuzione a
modello fermo; qui si fa variare **il momento e la quantità dell'apprendimento**,
tenendo fissi modello e percorso. Di conseguenza le tre linee non condividono
una misura di merito: là l'errore e il costo dell'errore, là la coerenza fra
host e dispositivo, qui il confronto fra un modello aggiornato e la sua copia
congelata sugli stessi esempi.

## 2. Il confine con Kamila-5, che è concreto e va dichiarato

Le otto feature usate qui — `duration`, `src_bytes`, `dst_bytes`,
`missed_bytes`, `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes` — sono
quelle del contratto **Kamila-5**, cioè del lavoro di riferimento, non quelle
armonizzate di kan-ids-v2. La scheda avvertiva di non darle per uguali, e
infatti non lo sono.

La scelta è deliberata: adottando il contratto di Kamila-5 si può misurare la
sovrapposizione con il campione già visto usando le sue stesse funzioni, e il
confronto ha senso. Il prezzo è che **i risultati di questa attività non si
confrontano direttamente con quelli di kan-ids-v2**, che lavora su un altro
spazio di feature. Ogni confronto fra le due linee va mediato da un
riallineamento esplicito, che qui non è stato fatto.

Il contratto non è assunto ma verificato: il campo `required_features` del
manifest della fonte elenca esattamente quegli otto nomi in quell'ordine, e 34
record pubblicati nel repository di riferimento li dichiarano identici.

## 3. Componenti riutilizzati senza modifiche

Dal modulo `research028/vendor/audit_ton_full.py` del referente:

- `audit_csv` — lo scorrimento a flusso dei CSV e la classificazione in stage;
- `normalize_raw` — il contratto sui valori, compreso il trattamento dello zero
  negativo e i codici di scarto;
- `fingerprint_rows` — le impronte degli otto input;
- `parse_timestamp` e `timestamp_iso` — la lettura dei tempi.

E lo strumento `research028/tools/coverage028.py`, **eseguito così com'è**
sull'aggregato prodotto, senza una riga modificata.

Il riuso non è nominale: le misure di contratto, di classe e di sovrapposizione
di questa attività sono prodotte dal suo codice, non da una riscrittura. Il
codice nostro si limita a chiamarlo, a binare per ora invece che per giorno, e
ad assemblare le consegne.

## 4. Componenti ricostruiti, perché non pubblicati

L'archivio delle impronte degli input già visti, il suo file di controllo e il
descrittore del protocollo non sono pubblicati: il README dichiara che gli array
NPZ campionati non vengono redistribuiti e che gli array di input mancanti vanno
ricostruiti da sorgenti ottenute lecitamente. La ricostruzione è quindi la
procedura prevista.

Riscontro esatto: il codice di riferimento dichiara che l'archivio originale
contiene **92.330 impronte uniche ricavate da 211.043 righe**, e la ricostruzione
— a partire dal campione il cui SHA-256 coincide con quello atteso — ne produce
92.330 da 211.043.

Limite: resta una ricostruzione, non l'artefatto originale, e il controllo di
integrità che lo strumento di valutazione esterna esegue sul proprio `.npz` non
è riproducibile.

## 5. Un componente che non si trasferisce

`coverage028.py` funziona, ma sui dati completi produce 129 candidati e **zero**
con tutti e quattro i suoi gruppi di test. La causa è misurata: la sua regola
fissa il test alla coda finale della raccolta, e in TON_IoT i tipi di attacco
non ricorrono — ciascuno compare in al massimo due giorni consecutivi, e
l'ultimo giorno contiene solo `backdoor` e `mitm`. I suoi gruppi presuppongono
che i tipi tornino.

Non è un difetto del suo strumento: è la firma di una domanda diversa. Il
protocollo di ADAPT-01 è progressivo e non usa un test finale fisso, quindi non
ha bisogno di quei gruppi. Lo strumento resta utile per la misura delle
sovrapposizioni e per l'enumerazione dei candidati, non per la selezione della
partizione.

## 6. Componenti del ramo precedente, e perché quasi nulla è stato trasferito

Il trasferimento selettivo da `drift-protocollo` si è ridotto a un file solo,
`kanids/valutazione.py`, e la ragione è verificata su un clone pulito a
`69618ee`:

- `int_adapt.py`, `metrics.py`, `models.py` e `src/kan_bspline.py` sono **già
  byte per byte identici** su `main`, perché arrivati con il Paper 1;
- `tests/test_rigenera.py` non arriva alla raccolta e `tests/test_protocollo.py`
  dà 13 fallimenti su 28, perché dipendono dagli script e dai file del ramo di
  origine;
- le altre cinque suite leggono la cartella `results/` e i documenti di quel
  ramo.

`valutazione.py` è stato trasferito perché separa i dati su cui si **scelgono** i
parametri da quelli su cui si **riporta**, che è esattamente la disciplina degli
intervalli A/B/C/D.

Al posto delle suite non trasferibili è stata scritta
`ton_temporal/test_guardia_monoclasse.py`: undici prove sulla regola del salto
per memoria monoclasse e sulle richieste comuni ai tre metodi, senza dipendenze
da evidenza preesistente.

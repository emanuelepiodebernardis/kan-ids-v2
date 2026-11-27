# ADAPT-07 — riesame dell’idoneità

## Cosa richiede la prima consegna

Il messaggio chiede di identificare due catture raccolte separatamente,
mai usate nello sviluppo, con fonte, periodo, etichette e caratteristiche
compatibili. Il confronto tra Frozen/B, memoria fissata e memoria FIT192
aggiornata è autorizzato dopo il commit del protocollo, se entrambe
superano il preflight. La scheda chiede di preservare anche le catture
con poche ricorrenze e di segnalare ogni requisito essenziale mancante
prima del confronto. Il budget dell’1%, il ritardo di un blocco e il
feedback da etichette originali appartengono al confronto primario.

La scheda chiede di fissare il warm-up, senza imporne la durata. I 20
blocchi da 10.000 record riprendono ADAPT-06I. La ripartizione
cronologica 40/20/40 è una scelta del nostro protocollo, fissata prima
di leggere i conteggi di idoneità nei log completi. Il
[rendiconto congelato](PREFLIGHT.json) conserva l’esito di queste scelte.
Il codice del replay va congelato prima del confronto primario. Gli
stress test seguono la lettura del primario.

## La revisione del publisher della 1-1

Il log ufficiale v2 pubblica la stessa acquisizione del 9 maggio 2018,
con revisione delle etichette. Il file scaricato durante il riesame è
lungo 144.270.428 byte e ha SHA-256
`803577ecc31cd67293feffa04cd0f4e26866f5fd1ed4b79eba48229b22ed9546`.
L’export v2 usa `detailed_label`. Per leggere il file con il
verificatore congelato si è rinominata solo questa intestazione
in una copia diagnostica. I numeri grezzi, i conteggi per classe e
i supporti A/B/D coincidono con la v1. D contiene ancora 84.980
record completi, di cui 7.761 benigni e 77.219 malevoli.
[Archivio v2 della sessione 1-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/CTU-IoT-Malware-Capture-1-1/)

Nei due log, una lettura separata conta 796.300 righe con i tre
campi `duration`, `orig_bytes` e `resp_bytes` tutti non valorizzati.
Di queste, 780.280 risultano in stato `S0`, che Zeek definisce come
un tentativo di connessione senza risposta. Il segnaposto `-`
indica un campo non valorizzato. Attribuirgli valore zero
cambierebbe la chiave e la popolazione del confronto. Anche
accorciare il warm-up dopo il preflight cambierebbe il protocollo
e lascerebbe la forte selezione per classe già documentata nella 1-1.
[Campi e stati Zeek](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html) ·
[Formato dei log Zeek](https://docs.zeek.org/en/current/tutorial/logs.html)

## Candidata da sottoporre al professore

La proposta è conservare la `48-1` e sottoporre al preflight la sessione
`CTU-IoT-Malware-Capture-35-1` della stessa release v1. È un’acquisizione
distinta del 21 dicembre 2018, mentre la 48-1 inizia il 28 febbraio 2019.
Il catalogo riporta circa 10,45 milioni di flussi e 8,26 milioni di
flussi benigni, con anche etichette malevole. Nel log v1, di circa 1,2
GiB secondo il publisher, un controllo dell’intestazione ha verificato
le otto feature, `ts`, `uid` e i due campi di etichetta. Resta da
verificare sui record completi la distribuzione cronologica di entrambe
le classi, l’unicità degli UID e il supporto dopo il warm-up. Questa è
una proposta basata su metadati e schema. Nessuna prestazione o
frequenza di ricorrenza è stata calcolata per la 35-1.
[Catalogo delle etichette v1](https://www.stratosphereips.org/datasets-iot23) ·
[Date delle acquisizioni](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/README.html) ·
[Log v1 della 35-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-35-1/bro/)

Se il professore concorda, si registra un emendamento con la nuova coppia
e si applica lo stesso controllo strutturale. Il protocollo originale e
il suo esito rimangono consultabili. Solo con due idoneità confermate
si completa il congelamento del runner e si esegue il confronto primario.

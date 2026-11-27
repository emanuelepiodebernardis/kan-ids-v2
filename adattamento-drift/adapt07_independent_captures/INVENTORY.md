# Inventario delle due catture candidate

Selezione documentale del 27 novembre 2026. Si usano due sessioni IoT-23
rilevate separatamente, con ID diversi, periodi non sovrapposti secondo
la fonte e volumi dichiarati compatibili con il riscaldamento previsto.
La scelta privilegia il formato Zeek con mappa diretta verso le otto
feature TON e due log individuali di dimensioni gestibili. I conteggi
di ricorrenza e le prestazioni non entrano nella selezione.

| Candidata | Fonte e periodo dichiarato | Etichette disponibili | Verifica dello schema |
|---|---|---|---|
| `CTU-IoT-Malware-Capture-1-1` | Stratosphere Laboratory, CTU Praga. Inizio 9 maggio 2018, durata di catalogo 112 ore. Il preflight ricava gli estremi effettivi. | Classe binaria benigno/malevolo. Il catalogo riporta benigno, C&C e port scan. | Log Zeek originale `bro/conn.log.labeled`. Verifica delle otto colonne e dei record effettivi nel preflight. |
| `CTU-IoT-Malware-Capture-48-1` | Stesso laboratorio, sessione distinta. Timestamp di avvio in timeline 28 febbraio 2019, durata di catalogo 24 ore. L’intestazione della timeline riportava 2018. Il preflight usa i timestamp Unix del log per risolvere la discordanza. | Classe binaria benigno/malevolo. Il catalogo riporta benigno, attacco, port scan e categorie C&C. | Log Zeek originale `bro/conn.log.labeled`. Verifica delle otto colonne e dei record effettivi nel preflight. |

[Catalogo IoT-23 e definizioni delle etichette](https://www.stratosphereips.org/datasets-iot23) ·
[Timeline 1-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1/README.md) ·
[Timeline 48-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-48-1/README.md)

La release pubblica IoT-23 v1.0.0 è attribuita a Garcia, Parmisano
ed Erquiaga ed è distribuita con CC BY 4.0. Le note di acquisizione
dei singoli scenari rinviano a condizioni anteriori al rilascio
pubblico. Per i log inclusi nella release pubblica valgono il deposito
versionato e il chiarimento attuale del laboratorio sull’attribuzione.
[Deposito e licenza](https://doi.org/10.5281/zenodo.4743746) ·
[FAQ del publisher](https://www.stratosphereips.org/datasets-faq#)

## Caratteristiche e identità

| Colonna chiave TON | Campo Zeek | Unità e significato |
|---|---|---|
| duration | duration | secondi di durata della connessione |
| src_bytes | orig_bytes | byte di payload dell’originatore |
| dst_bytes | resp_bytes | byte di payload del risponditore |
| missed_bytes | missed_bytes | byte mancanti nei gap rilevati |
| src_pkts | orig_pkts | pacchetti dell’originatore |
| src_ip_bytes | orig_ip_bytes | byte a livello IP dell’originatore |
| dst_pkts | resp_pkts | pacchetti del risponditore |
| dst_ip_bytes | resp_ip_bytes | byte a livello IP del risponditore |

`uid` identifica la connessione entro acquisizione e resta fuori
dalla chiave, come timestamp, indirizzi, porte e label. Una connessione
ha un identificatore conservato anche dopo l’ordinamento. Si leggono
le etichette originali per il supporto strutturale e, nel replay,
per il feedback causale e la valutazione offline. L’etichettatura deriva
dall’analisi del publisher, con possibili effetti di scenario e di regole.
[Documentazione Zeek Conn::Info](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html#type-Conn::Info)

## Esposizione del progetto e registro

La ricerca su `main` al commit `166a4431dd7622543c0c5aad94656e521c8fe751`
e sui candidati nella cronologia Git locale (`IoT-23`, `IoT23`,
`CTU-IoT-Malware-Capture`) non ha evidenziato uso degli scenari 1-1 o
48-1. La verifica riguarda il materiale accessibile del progetto.
TON e UNSW sono sviluppo. Anche BoT-IoT, CIC-IoT-2023 e CICIoMT2024
compaiono nelle analisi storiche e sono esclusi. La presenza di
CICIDS2017 nella bibliografia del progetto è un ulteriore motivo per
preferire candidati con esposizione documentata più chiaramente.

Il 27 novembre Emanuele ha consultato, per questa consegna, catalogo,
timeline, licenza e documentazione dello schema. L’accesso agli input
grezzi è circoscritto all’acquisizione e al preflight. Il registro del
preflight identifica file, hash, supporti e decisione. Nessuna metrica
del confronto è stata usata per la scelta delle sessioni.

## Costi preventivi

Il publisher indica log per circa 648 MiB complessivi, oltre a prodotti
di lavoro locali. Riservare circa 3 GiB per sorgenti, elaborazione e
riepiloghi. Come ordine di grandezza, i replay TON di 06I sui suoi
3,55 milioni di record hanno richiesto da circa 68 a 260 secondi per
seed nel primo passaggio Windows, compresi controlli sugli input.
Per ADAPT-07 si prevede un lavoro seriale dell’ordine di 20–40 minuti
per preflight, eventuali fit A/B e due replay, con download a parte.
È una stima preliminare. Il manifest del preflight e, se ammessi, i
report del confronto registrano tempi e ambiente effettivi.

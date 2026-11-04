# ADAPT-05 — consegna completa

Campagna eseguita dal commit
`3d4b4cb6f4ca08f7095785dc433c43919b4dc85f` con Python 3.12.14,
BLAS a un thread e due celle concorrenti. Il workflow ha completato le 30
combinazioni fissate: due dataset, seed 42–46 e tre sali. Nessun esito e' stato
escluso in base al segno dell'effetto.

## Controlli

- 127 test mirati superati nella finalizzazione;
- 30/30 rendiconti completi;
- 22.215 controlli sui rendiconti e 240 controlli di copertura;
- 13.650 snapshot CAL64 estratti dai log e 14 analisi previste mappate;
- input verificati ricalcolando gli SHA-256 degli otto array letti;
- 94/94 file del manifest verificati anche dopo il ripristino dello ZIP in una
  directory vuota.

I 30 JSON e le sette tabelle scientifiche preesistenti sono byte-identici alla
consegna verificata al commit `7c22482`: non sono stati eseguiti fit, tuning o
replay. La finalizzazione rilegge i report originali e aggiunge soltanto le
diagnostiche richieste.

Il rapporto interpretativo e' in [`REPORT.md`](REPORT.md). Le tabelle CSV
conservano aggregazioni, dispersione e differenze appaiate; `figures/` contiene
le sei figure prodotte dallo stesso generatore. La copertura cumulativa, gli
snapshot CAL64, la tabella per digest e l'aderenza alle analisi previste sono
distinti esplicitamente. `campaign_status.json`
registra ambiente, comandi, tempi e SHA di ciascuna cella.

## Archivio completo

- file: `adapt05_occurrence_full_finalized.zip`;
- dimensione: 13.879.041 byte;
- SHA-256:
  `03d0d5e9b6eb896d15a53ae954086461186c6daeaf918bc10cb670570abdccae`;
- rianalisi senza fit o replay:
  [GitHub Actions 41877615465](https://github.com/emanuelepiodebernardis/kan-ids-v2/actions/runs/41877615465).

Lo ZIP non e' conservato in Git. La copia locale esterna si trova in
`adattamento-drift/archivio_esterno/adapt05_occurrence/ea46b3f/`; il file
`adapt05_occurrence_full_finalized.zip.sha256` e il manifest completo sono
conservati qui per il controllo della consegna. Il pacchetto originario al
commit `7c22482` resta conservato separatamente e non viene sovrascritto. I
comandi di ricostruzione sono in
[`../REPRODUCTION.md`](../REPRODUCTION.md).

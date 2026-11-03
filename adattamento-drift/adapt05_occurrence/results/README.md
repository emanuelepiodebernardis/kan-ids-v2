# ADAPT-05 — consegna completa

Campagna eseguita dal commit
`3d4b4cb6f4ca08f7095785dc433c43919b4dc85f` con Python 3.12.14,
BLAS a un thread e due celle concorrenti. Il workflow ha completato le 30
combinazioni fissate: due dataset, seed 42–46 e tre sali. Nessun esito e' stato
escluso in base al segno dell'effetto.

## Controlli

- 124 test mirati superati prima della matrice;
- 30/30 rendiconti completi;
- 22.215 controlli sui rendiconti e 240 controlli di copertura;
- input verificati ricalcolando gli SHA-256 degli otto array letti;
- 89/89 file del manifest verificati anche dopo il ripristino dello ZIP in una
  directory vuota.

Il rapporto interpretativo e' in [`REPORT.md`](REPORT.md). Le tabelle CSV
conservano aggregazioni, dispersione e differenze appaiate; `figures/` contiene
le cinque figure prodotte dallo stesso generatore. `campaign_status.json`
registra ambiente, comandi, tempi e SHA di ciascuna cella.

## Archivio completo

- file: `adapt05_occurrence_full.zip`;
- dimensione: 10.447.388 byte;
- SHA-256:
  `01220c7e09936fa762dd6f87d03e5d76856649bb2227269c1312f5b0ea8c743d`;
- copia persistente:
  [Google Drive](https://drive.google.com/file/d/1TgAlPP_YEyEaGYHnzLnBT2yxwaH-6A5Q/view?usp=drive_link);
- esecuzione: [GitHub Actions 41715167666](https://github.com/emanuelepiodebernardis/kan-ids-v2/actions/runs/41715167666).

Lo ZIP non e' conservato in Git. La copia locale esterna si trova in
`adattamento-drift/archivio_esterno/adapt05_occurrence/3d4b4cb/`; il file
`adapt05_occurrence_full.zip.sha256` e il manifest completo sono conservati qui
per il controllo della consegna. I comandi di ricostruzione sono in
[`../REPRODUCTION.md`](../REPRODUCTION.md).

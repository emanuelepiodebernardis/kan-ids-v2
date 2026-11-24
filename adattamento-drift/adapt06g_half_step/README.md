# ADAPT-06G

Questa cartella preregistra il confronto fra aggiornamento completo e
aggiornamento dei pesi a metà passo per GAM/RANDOM, a parità di budget e regole
ADAPT-06E. `PROTOCOL.md` e `config.json` sono stati committati prima
dell'implementazione e prima di leggere i nuovi risultati. Il percorso legacy
mantiene il coefficiente implicito 1,0; la variante 0,5 è ammessa soltanto per
GAM/RANDOM con soglia disaccoppiata.

## Prodotti

- `RESULTS.csv`: esattamente dieci righe, una per dataset e seed;
- `TEMPORAL_BLOCKS.csv`: localizzazione integrale per blocco;
- `EPISODES.csv`: intervalli contigui favorevoli per seed;
- `DECISION.json`: applicazione meccanica dei criteri preregistrati;
- `REPORT.md`: sintesi descrittiva, inclusi esiti nulli o sfavorevoli;
- `MANIFEST.json`: hash e dimensioni dei prodotti leggeri.

I report completi sono conservati fuori da Git nella directory documentata in
`COMMANDS.md`; non vengono riutilizzate directory di esecuzioni precedenti.


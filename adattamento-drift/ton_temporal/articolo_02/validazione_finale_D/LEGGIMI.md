# Validazione finale su D

Questa cartella conserva gli artefatti leggeri della campagna finale
preregistrata per il secondo articolo. Il codice eseguito e' il commit
`ea216914f756433fe69fd825146de5c1e325cbd8`; il disegno non e' stato
modificato dopo l'apertura di D.

- `riepilogo.json`: aggregazioni ricalcolabili dai cinque rendiconti completi;
- `analisi_finale.json`: risultati assoluti, confronti e controlli temporali;
- `tabella_risultati_assoluti.csv`: otto righe, quattro tracce per due ambiti;
- `rapporto.md`: risultati appaiati, disponibilita' degli update e costi;
- `manifest.json`: commit, comando, ambiente, hash degli input e degli output.

I cinque JSON completi e lo ZIP che li contiene non sono versionati. Restano
nella cartella documentata
`../../../archivio_esterno/articolo_02_validazione_finale/campagna_D_ea21691/`
e sono identificati nel manifest tramite dimensione e SHA-256. D resta
riservato e non e' incluso nel repository o nell'archivio dei rendiconti.

`D_completo` e `D_non_visto_ABC` non sono due replay: sono due aggregazioni
delle stesse predizioni. Il secondo ambito filtra offline le righe il cui
vettore non compare in A, B o C.

## Rigenerazione

Dalla cartella `adattamento-drift/ton_temporal`, con i sei JSON originali
nella directory esterna registrata nel manifest:

```powershell
python analisi_validazione_finale_D.py
python -m pytest test_analisi_validazione_finale_D.py -q
```

Il generatore legge soltanto i rendiconti gia' prodotti: non carica D, non
addestra modelli e non esegue replay o tuning.

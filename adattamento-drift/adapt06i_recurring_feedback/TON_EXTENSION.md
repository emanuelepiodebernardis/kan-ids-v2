# ADAPT-06I — estensione TON ai seed 43–46

## Perimetro

Si conserva il pilot approvato nella PR #24 all’HEAD
`81fbbbacc31efe004de4d51ce373092e3c4b98e2`. Per TON si eseguono
quattro nuovi passaggi, seed 43, 44, 45 e 46. Il seed 42 proviene
dai prodotti originali in `delivery/`, verificati contro il manifest
e il commit della PR. Il pilot UNSW si conclude con il seed 42.

Il [protocollo del pilot](PROTOCOL.md), fissato nel commit
`4f97a0b73b50ebe85e00a3ace0ac1987640f2d58`, si applica a tutti i
seed. Per ciascun seed si usano il modello GAM A, la soglia B
e il calendario di richieste e ruoli ADAPT-06E corrispondenti.
La stessa cattura ordinata è condivisa dai cinque seed. Gli hash
dell’archivio, dei report, degli input, dei modelli e dei metadati
sono verificati secondo il protocollo originale.

## Confronti

Su tutti i blocchi dal 20, incluso l’ultimo blocco parziale, si
confrontano Frozen/B, memoria fissata all’inizio del blocco 20 e
memoria aggiornata. Le richieste del blocco k possono entrare in FIT
da k+2. FIT192, regola di ammissibilità, gestione dei conflitti, ruoli
e budget dell’1% restano quelli del pilot. Le due memorie sono confrontate
sugli stessi ID e con la stessa politica per ricavare le classi dalle
chiavi. I pesi e la soglia B sono quelli archiviati, senza fit o tuning.

## Rendicontazione

Una riga per seed riporta i conteggi TP/FP/FN/TN dei tre bracci, i delta
appaiati, la copertura, i cambi e gli episodi favorevoli. Un episodio
raggruppa i blocchi consecutivi che soddisfano la regola del pilot.
La tabella conserva gli intervalli e le chiavi qualificanti per seed.
La nota indica anche gli intervalli distinti sull’unica cattura,
unendo tra seed i blocchi favorevoli sovrapposti o consecutivi.

Per misurare la concentrazione si riportano digest, conteggio,
denominatore e quota della chiave più frequente nella finestra, della
chiave con più righe ammissibili alla memoria aggiornata e di quella
con più decisioni cambiate rispetto a Frozen/B. Il criterio del pilot
viene riportato per ciascun seed come descrizione del risultato.
I seed misurano la sensibilità al calendario delle richieste sulla
stessa cattura. Si pubblicano tutti gli esiti e ADAPT-07 resta in backlog.

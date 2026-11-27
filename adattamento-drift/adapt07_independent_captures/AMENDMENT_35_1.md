# ADAPT-07 — sostituzione della cattura 1-1 con la 35-1

L’autorizzazione del professore riguarda la coppia IoT-23 **35-1 e 48-1**.
Questo emendamento è registrato sul branch della Draft PR #26 prima di
acquisire ed esaminare il log completo della 35-1. Il precedente controllo
dell’intestazione è documentato nel [riesame](RIESAME.md). La scelta della
35-1 si basa sul catalogo, sullo schema Zeek e sull’acquisizione separata,
con inizio dichiarato il 21 dicembre 2018. Il preflight rileverà gli
estremi effettivi e i supporti. [Catalogo IoT-23](https://www.stratosphereips.org/datasets-iot23) ·
[Date delle acquisizioni](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/README.html) ·
[Log v1 della 35-1](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-35-1/bro/)

Il [preflight originale](PREFLIGHT.json) conserva l’esito di entrambe
le sessioni iniziali. La 1-1 rimane bloccata con 84.980 record D a
fronte del warm-up di 200.000. Per la 48-1 si riusa il log v1 già
verificato con SHA-256
`9f69946243df93c199876c0783cb4b5fb188e6d2325c593858f2db8fb5fa847f`.
La 35-1 appartiene alla stessa release v1 con attribuzione CC BY 4.0.
Si acquisisce soltanto il log Zeek etichettato. [Release e licenza](https://doi.org/10.5281/zenodo.4743746)

Si applica il [protocollo originale](PROTOCOL.md) per intero. Restano
gli otto campi, l’ordine per `ts + duration`, i tagli cronologici 40/20/40
con gestione dei pari, entrambe le classi in A, B e D valutato e
20 blocchi da 10.000 record per il warm-up. Il controllo si applica ai
due log completi. La 35-1 usa il codice scenario 35 nella formula
PCG64 del protocollo in luogo del codice 1. Il codice della 48-1 resta
48. La frequenza delle ricorrenze non interviene nell’idoneità.

Solo con entrambe le catture idonee si congela il codice e si procede
al confronto appaiato tra Frozen/B, memoria fissata e memoria FIT192
aggiornata, con feedback originale, budget 1% e ritardo di un blocco.
Per ciascuna cattura, la consegna riporta TP sul totale degli attacchi
e FP sul totale dei flussi benigni per ogni braccio. Per la 48-1 il
denominatore benigno verificato dopo il warm-up è 1.131.

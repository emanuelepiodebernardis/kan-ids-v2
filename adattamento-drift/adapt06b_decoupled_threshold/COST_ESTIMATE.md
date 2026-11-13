# Stima dei costi del pilot ADAPT-06B

La stima usa esclusivamente i tempi misurati nella matrice ADAPT-06 sullo stesso
hardware, GAM/RANDOM, seed 42. Non è un risultato ADAPT-06B e sarà sostituita
dai tempi effettivi del pilot.

| Dataset | Wall time legacy | Replay legacy | Picco RSS legacy |
|---|---:|---:|---:|
| TON | 278,763 s | 254,051 s | 521,3 MiB |
| UNSW | 111,960 s | 100,092 s | 252,6 MiB |

Il comando ADAPT-06B eseguirà in sequenza un report legacy e uno decoupled per
dataset. Assumendo costo simile per le due modalità, la proiezione seriale è:

- TON: 557,526 s;
- UNSW: 223,920 s;
- totale: **781,446 s = 13,024 minuti**, prima di verifica appaiata e I/O.

Con i due dataset in parallelo il limite inferiore è 557,526 s = 9,292 minuti.
La previsione prudenziale è **15–20 minuti seriali** oppure **12–15 minuti di
wall time con due worker**, includendo caricamento, JSON e hash.

I picchi RSS futuri di legacy e variante saranno misurati in processi figli
nuovi e distinti; nessun valore includerà gli oggetti rimasti dal replay
precedente. La variante condivide la CAL256 col controllo threshold-only matched e aggiunge
al massimo 24 byte di stato numerico per la soglia matched corrente e le due
soglie proposte in attesa; non aggiunge un
secondo buffer CAL. La baseline threshold-only autonoma conserva la propria
CAL256, come nel legacy. Considerando oggetti Python, due processi e margine per
i report, si riservano 650 MiB per TON, 350 MiB per UNSW e **1,2 GiB** totali.
I quattro report attesi occupano circa 50 MiB; la riserva su disco è 200 MiB.

Il budget delle label resta deterministico e non raddoppia in un deployment:
35.499 richieste su TON e 10.160 su UNSW. Le due modalità sono repliche
sperimentali appaiate delle stesse richieste, non seconde annotazioni.


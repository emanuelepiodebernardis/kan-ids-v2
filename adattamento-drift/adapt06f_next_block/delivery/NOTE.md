# ADAPT-06F — aggiornamenti scartati nel blocco successivo

## Perimetro e controllo

Sono stati riconciliati tutti i 38 cicli dei dieci rendiconti originali ADAPT-06E. Trentaquattro cicli hanno un blocco successivo valutabile; i quattro chiusi alla fine dello stream rimangono nel CSV come `NO_NEXT_BLOCK`. I tre `ACCEPT` riproducono stato, soglia e confusioni registrati in d+1.

Il confronto usa soltanto d+1. Le etichette di quel blocco servono esclusivamente al conteggio; non sono eseguiti fit o ricalibrazioni e la traiettoria originale non viene modificata.

## Esito

Fra i 31 cicli respinti o sospesi con d+1 disponibile, 4 soddisfano il criterio operativo `delta_tp > 0` e `delta_fp <= 0`. I casi sono: ton/42/freeze302, ton/43/freeze302, ton/45/freeze302, ton/46/freeze302.

Nei respinti i casi favorevoli sono 0 su 24; nei sospesi sono 4 su 7. Con la soglia diagnostica dell'incumbent, calibrata sugli stessi dati del candidato, restano favorevoli 5 casi: ton/42/freeze302, ton/43/freeze302, ton/45/freeze302, ton/46/freeze302, unsw/45/freeze65.

I quattro casi operativamente favorevoli corrispondono tutti a un solo episodio di traffico, ton/decision352/next353, ripetuto in quattro semi TON. In queste righe il guadagno varia da +6 a +18 TP e la differenza dei FP da -1994 a -1508. Il controllo a calibrazione comparabile aggiunge un secondo episodio, ton/decision352/next353, unsw/decision75/next76, perché rende favorevole anche il candidato respinto unsw/45/freeze65: rispetto all'incumbent operativo produce +87 TP e +11 FP; rispetto allo stesso incumbent con soglia diagnostica produce +9 TP e -5 FP. Il quadro dipende dunque da due episodi e cambia con il riferimento di soglia.

## Interpretazione

I risultati sono descrittivi e circoscritti al blocco immediatamente successivo. I cinque semi per dataset condividono la stessa cattura: le ripetizioni fra semi non sono repliche indipendenti del traffico. Un eventuale episodio favorevole isolato non dimostra quindi un beneficio generale del candidato né modifica il NO-GO preregistrato di ADAPT-06E.

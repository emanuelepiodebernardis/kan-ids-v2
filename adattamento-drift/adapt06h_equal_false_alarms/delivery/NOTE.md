# ADAPT-06H — diagnosi a parità di falsi allarmi

Analizzati 34 cicli osservabili su 38, con 34 coppie seme/blocco nei totali e 0 fallback. I 4 cicli di fine stream sono conservati nel CSV con metriche vuote.

| Stream | Blocchi con entrambe le classi | Guadagno retrospettivo | Guadagno retrospettivo mantenuto | Semi favorevoli | Episodi qualificanti |
|---|---:|---:|---:|---:|---:|
| TON | 16 | 5 | 4 | 0/5 | 0 |
| UNSW | 18 | 6 | 4 | 1/5 | 1 |

## Interpretazione

- TON. Emergono limiti sia di ordinamento sia di trasferimento della soglia. Il candidato guadagna TP al tetto retrospettivo in 5/16 blocchi appaiati. Nei restanti 11 il migliore ordinamento al tetto osservato non è dimostrato. In 1 dei blocchi con guadagno retrospettivo e CAL disponibile, le soglie scelte al FREEZE non mantengono il guadagno contro entrambi i controlli.
- UNSW. Emergono limiti sia di ordinamento sia di trasferimento della soglia. Il candidato guadagna TP al tetto retrospettivo in 6/18 blocchi appaiati. Nei restanti 12 il migliore ordinamento al tetto osservato non è dimostrato. In 2 dei blocchi con guadagno retrospettivo e CAL disponibile, le soglie scelte al FREEZE non mantengono il guadagno contro entrambi i controlli.
- Episodio descrittivo TON, blocchi 353–353 (indici da zero), semi 42,43,45,46.
- Episodio descrittivo UNSW, blocchi 64–64 (indici da zero), semi 45.
- Episodio descrittivo UNSW, blocchi 76–76 (indici da zero), semi 45.
- Episodio descrittivo UNSW, blocchi 89–89 (indici da zero), semi 43,46.

I semi condividono la stessa cattura. L’analisi riguarda i blocchi successivi alle decisioni storiche del gate e misura un potenziale su quel sottoinsieme.
Il tetto retrospettivo è comune alle due teste. I falsi allarmi effettivi possono essere inferiori al tetto quando più punteggi sono uguali.

**Decisione.** Il criterio preregistrato per proporre un replay causale completo non è soddisfatto.

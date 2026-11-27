# ADAPT-07 — confronto primario

Il confronto usa le etichette originali, budget 1% e ritardo di un blocco. Ogni sessione mantiene il suo ordine e i tre bracci condividono record e richieste.

| Cattura | Politica | TP / attacchi | FP / benigni | Recall | FPR |
|---|---|---:|---:|---:|---:|
| iot23-35-1 | Frozen/B | 1163815 / 1163903 | 0 / 1042166 | 99.9924% | 0.0000% |
| iot23-35-1 | Memoria fissata | 1163815 / 1163903 | 0 / 1042166 | 99.9924% | 0.0000% |
| iot23-35-1 | FIT192 aggiornata | 1163815 / 1163903 | 0 / 1042166 | 99.9924% | 0.0000% |
| iot23-48-1 | Frozen/B | 1155740 / 1156604 | 2 / 1131 | 99.9253% | 0.1768% |
| iot23-48-1 | Memoria fissata | 1155740 / 1156604 | 2 / 1131 | 99.9253% | 0.1768% |
| iot23-48-1 | FIT192 aggiornata | 1155740 / 1156604 | 2 / 1131 | 99.9253% | 0.1768% |

## iot23-35-1

Contro Frozen/B ΔTP +0 e ΔFP +0. Contro memoria fissata ΔTP +0 e ΔFP +0. Gli episodi favorevoli sono 0 e le chiavi qualificanti 0. Il criterio è insoddisfatto.

## iot23-48-1

Contro Frozen/B ΔTP +0 e ΔFP +0. Contro memoria fissata ΔTP +0 e ΔFP +0. Gli episodi favorevoli sono 0 e le chiavi qualificanti 0. Il criterio è insoddisfatto.

Il criterio completo è soddisfatto in 0 catture su due.
La 48-1 contiene 1.131 flussi benigni nel segmento valutato. Su questa cattura una differenza di un falso allarme equivale a circa 0,088 punti percentuali di FPR.
Le due sessioni sono acquisizioni distinte dello stesso laboratorio. Gli esiti sono riferiti separatamente.

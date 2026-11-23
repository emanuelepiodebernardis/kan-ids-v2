# ADAPT-06D — pilot shadow FIT192/FIT256

Protocollo: `89394d542f5b3eaa56e63b06348de251cc48e52a`. Commit eseguito: `b21cdad66deb46eec571b0eb9a7475339bac4b75`.

Il preflight è stato rigenerato dagli input con gli hash dichiarati e tutti
gli otto prodotti coincidono byte per byte. I tre controlli FIT192 hanno
riprodotto parametri, calibrazioni, confusioni e decisioni entro le
tolleranze prefissate; solo dopo questo PASS sono stati eseguiti i FIT256.

| FREEZE → look | Memoria | TP/FP candidato | TP/FP incumbent | Recall candidato | FPR candidato | Δ recall | Δ FPR | Decisione |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 18 → 37 | 192 | 94/8 | 89/6 | 94.000% | 2.286% | +5.000 pp | +0.571 pp | `ACCEPT` |
| 18 → 37 | 256 | 94/20 | 89/6 | 94.000% | 5.714% | +5.000 pp | +4.000 pp | `REJECT_HARM` |
| 39 → 62 | 192 | 89/9 | 98/12 | 86.408% | 2.013% | -8.738 pp | -0.671 pp | `REJECT_HARM` |
| 39 → 62 | 256 | 94/10 | 98/12 | 91.262% | 2.237% | -3.883 pp | -0.447 pp | `REJECT_HARM` |
| 64 → 89 | 192 | 101/20 | 100/30 | 97.115% | 4.032% | +0.962 pp | -2.016 pp | `WAIT` |
| 64 → 89 | 256 | 101/28 | 100/30 | 97.115% | 5.645% | +0.962 pp | -0.403 pp | `WAIT` |

## Confronto diretto FIT256 − FIT192

| FREEZE → look | Δ TP | Δ FP | Δ recall | Δ FPR | Transizione gate |
|---|---:|---:|---:|---:|---|
| 18 → 37 | +0 | +12 | +0.000 pp | +3.429 pp | `ACCEPT -> REJECT_HARM` |
| 39 → 62 | +5 | +1 | +4.854 pp | +0.224 pp | `REJECT_HARM -> REJECT_HARM` |
| 64 → 89 | +0 | +8 | +0.000 pp | +1.613 pp | `WAIT -> WAIT` |

## Esito preregistrato

- casi aggiuntivi con tutti i margini: `[]`;
- caso già accettabile preservato: `False`;
- nuove violazioni di danno: `[18]`;
- criterio complessivo soddisfatto: `False`.

Esito nullo per i tre casi preregistrati; nessuna conclusione generale.

Nessun candidato shadow è stato promosso e nessuna traiettoria è stata
modificata. Il risultato riguarda soltanto i tre casi UNSW/seed42
selezionati dopo la diagnosi; capacità, composizione ed età cambiano
insieme e non consentono un'attribuzione causale alla sola numerosità.

## Tempi

- rigenerazione e controllo preflight: 7.493 s;
- caricamento input: 0.467 s;
- ricostruzione e trasformazioni: 0.215 s;
- sei fit: 0.204 s;
- sei calibrazioni: 0.223 s;
- scoring VERIFY e gate: 0.323 s;
- esecuzione totale prima del confezionamento: 9.375 s.

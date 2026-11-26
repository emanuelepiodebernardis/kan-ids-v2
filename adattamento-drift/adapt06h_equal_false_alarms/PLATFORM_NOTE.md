# ADAPT-06H — riscontro numerico sui due confini CAL

La consegna originale `1fd6ff7` resta il riferimento. L’audit rilegge solo
i due casi segnalati, con gli stessi pesi al FREEZE, array, ordine delle
osservazioni e CAL. La regola preregistrata rimane `score > soglia`, con un
unico esito per tutti i pari.

## Confronto tra ambienti

Le misure sono state riprodotte con Windows 11, Python 3.12.7, NumPy 2.3.5,
scikit-learn 1.7.2 e OpenBLAS 0.3.30 a un thread. Per entrambi i cicli il
ricalcolo riproduce tetto, soglie e confusioni della consegna.
[Audit Windows con valori esatti binary64](PLATFORM_AUDIT_WINDOWS.json).

Il riesame indipendente ha usato Linux x86-64, Python 3.12.14, NumPy 2.3.5,
scikit-learn 1.7.2 e OpenBLAS 0.3.30 a un thread. Il revisore ha ricostruito
i dati dai CSV e usato i dieci report originali ADAPT-06E.
[Valori riportati nel riesame Linux](PLATFORM_REVIEW_LINUX.json).

| Caso | Ambiente | ID normale CAL | Soglia operativa archiviata | Score incumbent al confine | Tetto su 128 normali | TP/FP candidato nel blocco successivo |
|---|---|---:|---:|---:|---:|---:|
| TON/44, FREEZE 302 → blocco 353 | Windows | 21770852 | 8.465551138302391 | 8.465551138302391 | 0 | 10/1112 |
| TON/44, FREEZE 302 → blocco 353 | Linux | 21770852 | 8.465551138302391 | ≈8.4655511569 (margine +1,86×10⁻⁸) | 1 | 28/2423 |
| UNSW/43, FREEZE 37 → blocco 47 | Windows | 1842453 | 1.980076144867696 | 1.980076144867696 | 4 | 1776/631 |
| UNSW/43, FREEZE 37 → blocco 47 | Linux | 1842453 | 1.980076144867696 | ≈1.980077041 (margine +8,96×10⁻⁷) | 5 | 1806/699 |

Gli score Linux in tabella sono ricostruiti sommando alla soglia archiviata
il margine approssimato della scheda. La scheda non riporta i bit degli
score Linux né le soglie ricalibrate esatte.

Su Windows il normale al confine è il primo in ordine decrescente nel
caso TON e il quinto nel caso UNSW. In entrambi i casi lo score è uguale
alla soglia archiviata bit per bit. La regola stretta lo esclude. Nel
riesame Linux il margine è positivo e quel normale viene contato. I test
verificano l’uguaglianza, il primo numero rappresentabile sopra soglia,
il margine riferito, il nuovo tetto e le coppie TP/FP.

## Sensibilità al tetto con gli score Windows

Per isolare l’effetto del tetto, qui si ricalibrano entrambe le teste
sugli stessi score CAL Windows usando il tetto alternativo riferito dal
professore. Anche i conteggi della tabella provengono da score Windows.

| Caso | Tetto | Soglia candidato | Soglia incumbent ricalibrato | TP/FP candidato | TP/FP incumbent ricalibrato |
|---|---:|---:|---:|---:|---:|
| TON/44 · blocco 353 | 0 | 3.8115358144911164 | 8.465551138302391 | 10/1112 | 19/2306 |
| TON/44 · blocco 353 | 1 | −2.5717475891180497 | 7.279004777499558 | 28/2423 | 19/2378 |
| UNSW/43 · blocco 47 | 4 | 1.1651327896580863 | 1.980076144867696 | 1776/631 | 1667/261 |
| UNSW/43 · blocco 47 | 5 | −0.023585666475995026 | −1.4734129346824436 | 1806/699 | 1824/1014 |

In entrambi i casi le coppie TP/FP del candidato con il tetto alternativo
coincidono con quelle riferite su Linux. Su TON, a tetto 1, il candidato
totalizza 45 FP in più
dell’incumbent ricalibrato e 117 FP in più dell’incumbent operativo originale
(2.306 FP). Nel test è sufficiente spostare lo score al confine di un valore
rappresentabile per cambiare il tetto. Il salto della soglia del candidato
deriva dalla distanza fra il primo e il secondo score normale CAL. I sette
normali pari al secondo score sono esclusi dalla disuguaglianza stretta.

La corrispondenza riconcilia l’effetto dei due tetti sui blocchi segnalati.
Con i margini approssimati disponibili, la causa aritmetica esatta delle
divergenze e l’identità bit per bit delle soglie ricalibrate restano aperte.

## Limiti di riproduzione

Sul Linux del riesame `DECISION.json`, `EPISODES.csv` e `NOTE.md` coincidono
byte per byte. `CYCLES.csv` e `SEEDS.csv` presentano differenze. Il riesame
ha confermato indipendentemente le 170 coppie TP/FP delle cinque tracce sui
34 blocchi Linux. I criteri per seme, gli episodi e la decisione coincidono.

La prova a tetto alternativo resta fuori dai totali per seme. `delivery/`
conserva i prodotti e il manifest originali. Il [manifest supplementare](PLATFORM_AUDIT_MANIFEST.json)
identifica la nota, i riferimenti dei due ambienti e i limiti del confronto.

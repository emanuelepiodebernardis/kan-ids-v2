# ADAPT-06H — riscontro numerico sui due confini CAL

La consegna originale `1fd6ff7` resta il riferimento. L’audit rilegge solo
i due casi segnalati, con gli stessi pesi al FREEZE, array, ordine delle
osservazioni e CAL. La regola preregistrata rimane `score > soglia`, con un
unico esito per tutti i pari.

## Misure Windows

Le misure sono state riprodotte con Windows 11, Python 3.12.7, NumPy 2.3.5,
scikit-learn 1.7.2 e OpenBLAS 0.3.30 a un thread. Il JSON
[dell’audit Windows](PLATFORM_AUDIT_WINDOWS.json) registra le identità delle sorgenti e i valori
decimali ed esadecimali binary64. Per entrambi i cicli il ricalcolo riproduce
tetto, soglie e confusioni della consegna.

| Caso | ID normale CAL al confine | Score incumbent Windows | Soglia operativa archiviata | Tetto CAL Windows | Tetto CAL Linux riferito |
|---|---:|---|---|---:|---:|
| TON/44, FREEZE 302 | 21770852 | `0x1.0ee5cb8033faap+3` | `0x1.0ee5cb8033faap+3` | 0 | 1 |
| UNSW/43, FREEZE 37 | 1842453 | `0x1.fae6452dcbe84p+0` | `0x1.fae6452dcbe84p+0` | 4 | 5 |

Su Windows il normale al confine è il primo in ordine decrescente nel
caso TON e il quinto nel caso UNSW. In entrambi i casi lo score è uguale
alla soglia archiviata bit per bit. La regola stretta lo esclude. I test
spostano soltanto quello score al successivo numero rappresentabile sopra
la soglia e verificano l’incremento di una unità nel tetto.

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

Nel caso TON il passaggio da 1.112 a 2.423 FP coincide con il conteggio
riferito su Linux. A tetto 1, il candidato totalizza 45 FP in più
dell’incumbent ricalibrato e 117 FP in più dell’incumbent operativo originale
(2.306 FP). Nel test è sufficiente spostare lo score al confine di un valore
rappresentabile per cambiare il tetto. Il salto della soglia del candidato
deriva dalla distanza fra il primo e il secondo score normale CAL. I sette
normali pari al secondo score sono esclusi dalla disuguaglianza stretta.

Per UNSW la tabella quantifica l’effetto del passaggio da 4 a 5 sui medesimi
score Windows. Il messaggio di revisione indica il tetto su Linux.
Il confronto bit per bit di score e soglie Linux richiede i valori della
scheda Trello. L’audit distingue dunque le misure Windows, i conteggi
riferiti dal revisore e la prova a tetto alternativo. La corrispondenza dei
conteggi TON non identifica da sola la fonte della divergenza numerica.

La prova aggiuntiva non entra nei totali per seme né nel criterio
preregistrato. `delivery/` conserva tutti i prodotti originali.

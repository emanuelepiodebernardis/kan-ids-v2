# ADAPT-06I — pilot con feedback passato

Seed 42, su tutti i blocchi a partire dal 20. I tre confronti usano gli stessi ID. La memoria fissata conserva il dizionario di inizio blocco 20, mentre quella aggiornata incorpora le sole etichette FIT disponibili prima della predizione.

| Stream | ID valutati | Righe ammissibili con memoria aggiornata | Cambi rispetto a Frozen/B | ΔTP/ΔFP contro Frozen/B | ΔTP/ΔFP contro memoria fissata | Episodi favorevoli | Chiavi qualificanti | Estensione proponibile |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| TON | 3.349.973 | 74.403 (2,22%) | 25.809 | +15695/-6930 | +1919/-9047 | 3 | 5 | Sì |
| UNSW | 816.018 | 145.768 (17,86%) | 0 | +0/+0 | +0/+0 | 0 | 0 | No |

| Stream | Decisione | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|
| TON | Frozen/B | 662.716 | 38.848 | 2.383.650 | 264.759 |
| TON | Memoria fissata | 676.492 | 40.965 | 2.369.874 | 262.642 |
| TON | Memoria aggiornata | 678.411 | 31.918 | 2.367.955 | 271.689 |
| UNSW | Frozen/B | 153.676 | 12.528 | 9.508 | 640.306 |
| UNSW | Memoria fissata | 153.676 | 12.528 | 9.508 | 640.306 |
| UNSW | Memoria aggiornata | 153.676 | 12.528 | 9.508 | 640.306 |

## TON

Dal blocco 20 al 354 la memoria aggiornata ammette 94 chiavi distinte. Le decisioni diverse da Frozen/B riguardano 12 chiavi. Il controllo fisso copre 29.207 righe (0,87%).

Contro Frozen/B si recuperano 16.230 attacchi, se ne perdono 535, si correggono 7.987 falsi allarmi e se ne aggiungono 1.057.

- Chiave più frequente nella finestra. La chiave `ff4880edb8fb994dbef59c54249dd751250b9bb7dc2b66352f4582e10022e898` rappresenta 49.389 righe su 3.349.973 (1,47%).
- Chiave con più righe ammissibili. La chiave `f0b15f79c9b6a8dcf9a9113dbdc6a94424db5cff323b1d133ea7c9ca3df20dba` rappresenta 18.694 righe su 74.403 (25,13%).
- Chiave con più cambi contro Frozen/B. La chiave `edc7f6a1fe7755376f5aeb16e91c4ff59fe3dfdbe45ca9a6b27ff23eded29df4` rappresenta 8.522 righe su 25.809 (33,02%).

Gli episodi favorevoli contro entrambi i controlli sono nei blocchi 59–60, 281, 295, con indici da zero.
Il pilot soddisfa i criteri preregistrati per proporre l’estensione.

## UNSW

Dal blocco 20 al 101 la memoria aggiornata ammette 22 chiavi distinte. Le decisioni diverse da Frozen/B riguardano 0 chiavi. Il controllo fisso copre 155.294 righe (19,03%).

Contro Frozen/B si recuperano 0 attacchi, se ne perdono 0, si correggono 0 falsi allarmi e se ne aggiungono 0.

- Chiave più frequente nella finestra. La chiave `dadecaa620aafb5593de5c87d8ec3a8fc24e16ecf6fb6e27f0bdf107adf95183` rappresenta 35.146 righe su 816.018 (4,31%).
- Chiave con più righe ammissibili. La chiave `dadecaa620aafb5593de5c87d8ec3a8fc24e16ecf6fb6e27f0bdf107adf95183` rappresenta 35.146 righe su 145.768 (24,11%).

Nessun episodio soddisfa il criterio di vantaggio appaiato contro entrambi i controlli.
L’estensione resta esclusa. Condizioni mancanti (vantaggio nei totali contro entrambi i controlli, almeno due episodi distinti, almeno due chiavi qualificanti).

Il pilot riguarda una cattura per stream e il solo seme 42. Usa il calendario storico delle richieste. I confronti misurano il contributo della memoria a calendario fissato sugli stream di sviluppo. ADAPT-07 rimane in backlog.

# Evidenza del replay su C

- `replay_C_seme42.json` — la corsa di riferimento, seme 42, 916 blocchi,
  con le misure blocco per blocco per adattivo e congelato.
- `replay_C_seme42_strati_punteggio.json` — la variante con lo stesso budget
  distribuito su dieci fasce del punteggio della logistica congelata. Risultato
  negativo: i salti salgono da 393 a 421.
- `ripetizioni_cinque_semi.json` — il consolidato dei semi da 42 a 46.

Le corse dei semi 43, 44, 45 e 46 non sono incluse per non gonfiare il
repository: si riproducono con
`python replay.py --iniziale A.npz --flusso C.npz --uscita <file> --seme <n>`.
I flussi `A.npz`, `B.npz` e `C.npz` si ricostruiscono con `costruisci_flusso.py`
dai CSV originali, che non vengono redistribuiti.

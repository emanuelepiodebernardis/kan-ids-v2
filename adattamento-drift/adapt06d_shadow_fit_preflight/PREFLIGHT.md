# ADAPT-06D — tabella di preflight

La tabella è derivata dai report e dagli artefatti verificati. Non sono stati
eseguiti fit, calibrazioni di candidati, scoring o replay.

| FREEZE → look | Esito originale | RANDOM_FIT disponibili (normali/attacchi) | FIT192 (normali/attacchi) | FIT256 (normali/attacchi) | Età FIT192 / FIT256 | FIT256: overlap + nuovi | CAL (normali/attacchi) | VERIFY (normali/attacchi) | Stato |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 18 → 37 | ACCEPT | 1047/303 | 96/96 | 128/128 | 1–4 / 1–6 blocchi | 192 + 64 | 141/115 | 350/100 | PASS |
| 39 → 62 | REJECT_HARM | 2281/644 | 96/96 | 128/128 | 1–5 / 1–8 blocchi | 192 + 64 | 128/128 | 447/103 | PASS |
| 64 → 89 | WAIT | 3787/1013 | 96/96 | 128/128 | 1–8 / 1–12 blocchi | 192 + 64 | 128/128 | 496/104 | PASS |

## Controlli conclusi

- i tre FIT192 coincidono per ID e conteggi con gli snapshot originali;
- FIT256 è stato ricostruito causalmente e non riempito per assunzione;
- ogni FIT256 contiene i 192 membri FIT192 e 64 membri aggiuntivi;
- FIT, CAL e VERIFY sono disgiunti in tutti i casi;
- modello, metadata, report e input esterni hanno gli SHA-256 attesi;
- il confronto futuro resta bloccato finché FIT192 non riproduce il riferimento.

Il report originale consente di confrontare esattamente insieme di ID e
conteggi FIT192, ma non contiene l'ordine interno della FIFO né il digest della
matrice trasformata. Questi due elementi sono nuovi prodotti della
ricostruzione causale e dovranno essere confermati riproducendo FIT192 prima di
qualsiasi FIT256.

I digest completi e gli ID sono in `FIT_INVENTORY.csv`, `FIT_MEMBERS.csv`,
`CAL_MEMBERS.csv` e `VERIFY_MEMBERS.csv`.

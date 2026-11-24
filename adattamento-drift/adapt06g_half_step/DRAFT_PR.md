## Domanda

ADAPT-06F ha mostrato che, in 19 dei 24 cicli respinti di ADAPT-06E, il
candidato completo avrebbe aumentato i veri positivi nel blocco successivo ma
anche i falsi positivi. Questa consegna verifica una sola variante fissata prima
dei risultati: la media, coefficiente 0,5, fra i nove parametri della testa GAM
operativa e quelli del nuovo fit, intercetta inclusa.

## Protocollo e implementazione

Il protocollo è stato fissato nel commit
`66228d22cbaa29f84902cd69730294531f24ce0e`. Il commit eseguito è
`cb651dd2bc0bd0ffe70384f0ac9b00a46d7c0f5b`.
La base della consegna è il commit `5878d78` di ADAPT-06F ([PR #21](https://github.com/emanuelepiodebernardis/kan-ids-v2/pull/21)).

Restano invariati GAM/RANDOM, budget dell'1%, ritardo di un blocco, FIT192,
CAL256, allocazione VERIFY 50/100, look, scadenza e margini di ADAPT-06E. Il
candidato a metà passo viene calibrato sulla CAL già disponibile prima di
VERIFY. Le decisioni modificano causalmente la traiettoria successiva. Il
percorso preesistente conserva come default il passo completo.

Il preflight verifica A/B/D, identità degli array, dieci modelli iniziali e i
dieci report ADAPT-06E originali. I dieci replay completi, TON e UNSW seed
42–46, sono stati eseguiti da A/B in processi nuovi; non sono stati eseguiti
nuovi fit iniziali, tuning, ACTIVE o altri dataset.

## Risultati

L'esito preregistrato è **NO-GO**.

- Su TON le confusioni del mezzo passo coincidono con ADAPT-06E e con il
  controllo matched in tutti e cinque i seed. Le promozioni sono zero.
- Su UNSW nessun seed soddisfa insieme `ΔTP > 0` e `ΔFP <= 0` contro sia
  ADAPT-06E sia il controllo a pesi fissi con lo stesso calendario e la stessa
  evidenza CAL. Il seed 44 migliora rispetto a 06E (+426 TP, −365 FP), con
  +889 FP rispetto al controllo matched. Nel seed 45 la perdita rispetto a
  06E è di 4.535 TP, di cui 4.125 concentrati nel blocco 87 (indice da zero).
- L'unico blocco favorevole contro entrambi i riferimenti è UNSW seed 46,
  blocco 89 (indice da zero), con +29 TP e −1 FP contro ciascuno. Sullo stream
  completo il confronto con matched dà +528 TP e +426 FP.
- Non esistono tre seed favorevoli nello stesso stream né due episodi temporali
  qualificanti; non è quindi soddisfatto il criterio per proporre ADAPT-07.

`RESULTS.csv` contiene le dieci righe con TP, FP, FN, FPR, promozioni e confronti
con ADAPT-06E, matched, Frozen/B e ADAPT-06B. `TEMPORAL_BLOCKS.csv` conserva
entrambi gli stream integralmente; `EPISODES.csv` localizza i blocchi contigui.
La sequenza degli ID richiesti coincide con ADAPT-06E in tutti i blocchi. Le
variazioni di ruolo interessano 45 coppie seme/blocco e sono riportate nei CSV.

## Controlli

- preflight: PASS;
- matrice: 10/10 report PASS, 1.191,062 s di wall time;
- analisi: PASS, 10 righe e decisione NO-GO;
- suite ADAPT-06B, ADAPT-06E, ADAPT-06F e ADAPT-06G: 105 test superati;
- rigenerazione dei sei file della consegna con corrispondenza byte per byte;
- Ruff: PASS.

Il pacchetto completo `adapt06g_half_step_cb651dd.zip` contiene i dieci JSON,
il preflight e i log. Il suo SHA-256 è
`4606bdd9f1a8eab3333d8ac014a60c9edbb5a2205defaeea2f9f2e929cfaa7d9`.
I comandi, i percorsi di ripristino e gli hash dei riferimenti sono in
`adattamento-drift/adapt06g_half_step/COMMANDS.md`.

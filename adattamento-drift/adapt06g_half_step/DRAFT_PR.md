# Titolo

adapt-06g: valutare il mezzo passo a parità di etichette

# Descrizione

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

- Su TON il mezzo passo non viene mai promosso e coincide con ADAPT-06E e con
  il controllo matched in tutti i cinque seed.
- Su UNSW nessun seed soddisfa insieme `ΔTP > 0` e `ΔFP <= 0` contro sia
  ADAPT-06E sia il controllo a pesi fissi con lo stesso calendario e la stessa
  evidenza CAL.
- L'unico intervallo locale favorevole è UNSW seed 46, blocco 89. Sullo stream
  completo lo stesso seed aumenta di 528 TP e 426 FP rispetto al matched, perciò
  rimane sfavorevole.
- Non esistono tre seed favorevoli nello stesso stream né due episodi temporali
  qualificanti; non è quindi soddisfatto il criterio per proporre ADAPT-07.

`RESULTS.csv` contiene le dieci righe con TP, FP, FN, FPR, promozioni e confronti
con ADAPT-06E, matched, Frozen/B e ADAPT-06B. `TEMPORAL_BLOCKS.csv` conserva
entrambi gli stream integralmente; `EPISODES.csv` localizza i blocchi contigui.
Gli ID richiesti coincidono con ADAPT-06E in tutti i blocchi; le variazioni di
ruolo dovute alle diverse decisioni sono registrate esplicitamente.

## Controlli

- preflight: PASS;
- matrice: 10/10 report PASS, 1.191,062 s di wall time;
- analisi: PASS, 10 righe e decisione NO-GO;
- test mirati e regressione ADAPT-06E: 44 superati;
- Ruff: PASS.

I report completi restano nell'archivio esterno; Git conserva protocollo,
configurazione, codice, comandi, tabella, analisi temporale, decisione e
manifest leggero.

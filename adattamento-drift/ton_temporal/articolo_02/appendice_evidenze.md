# Appendice delle evidenze ADAPT-01--04

## Regole di lettura

Ogni riga e' un confronto variante meno controllo. Recall e FPR sono espressi
in punti percentuali; AUROC e' una differenza assoluta. FN e FP sono conteggi
globali oppure, dove dichiarato nella nota, medie delle differenze appaiate per
seme. Un valore negativo indica una diminuzione nella variante.

Un confronto e' diretto soltanto entro la stessa riga. Le righe possono essere
discusse insieme solo quando coincidono `gruppo_comparabilita`, modello e
perimetro dei semi; i gruppi non formano una graduatoria unica. I denominatori
AUROC restano espliciti per non trasformare i blocchi monoclasse mancanti in
zeri.

## Tabella completa

| Fase | Gruppo | Modello | Semi | Controllo -> variante | Delta recall (pp) | Delta FPR (pp) | Delta AUROC | Delta FN | Delta FP | Denominatore AUROC | Nota |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| ADAPT-01 | `A1-C-cinque-semi` | LR | 42-46 | `congelato` -> `ogni_blocco` | 16,110 | -30,180 | 0,0590 | n.d. | n.d. | 875/916 blocchi per seme | beneficio dell'aggiornamento continuo |
| ADAPT-01 | `A1-C-cinque-semi` | LR | 42-46 | `casuale` -> `evidenza` | 0,630 | -5,230 | 0,0257 | n.d. | n.d. | 875/916 blocchi per seme | calendari con uguale numero di aggiornamenti |
| ADAPT-01 | `A1-C-cinque-semi` | LR | 42-46 | `ogni_blocco` -> `evidenza` | -6,250 | 0,900 | 0,0017 | n.d. | n.d. | 875/916 blocchi per seme | momento selettivo contro aggiornamento continuo |
| ADAPT-01 | `A1-C-cinque-semi` | MLP | 42-46 | `congelato` -> `ogni_blocco` | 42,780 | -13,200 | -0,0257 | n.d. | n.d. | 875/916 blocchi per seme | beneficio dell'aggiornamento continuo |
| ADAPT-01 | `A1-C-cinque-semi` | MLP | 42-46 | `casuale` -> `evidenza` | -7,010 | 2,190 | 0,0691 | n.d. | n.d. | 875/916 blocchi per seme | calendari con uguale numero di aggiornamenti |
| ADAPT-01 | `A1-C-cinque-semi` | MLP | 42-46 | `ogni_blocco` -> `evidenza` | -11,870 | 3,360 | 0,0292 | n.d. | n.d. | 875/916 blocchi per seme | momento selettivo contro aggiornamento continuo |
| ADAPT-01 | `A1-C-cinque-semi` | KAN | 42-46 | `congelato` -> `ogni_blocco` | 10,510 | -2,250 | -0,0841 | n.d. | n.d. | 875/916 blocchi per seme | beneficio dell'aggiornamento continuo |
| ADAPT-01 | `A1-C-cinque-semi` | KAN | 42-46 | `casuale` -> `evidenza` | -5,320 | -11,700 | 0,0974 | n.d. | n.d. | 875/916 blocchi per seme | calendari con uguale numero di aggiornamenti |
| ADAPT-01 | `A1-C-cinque-semi` | KAN | 42-46 | `ogni_blocco` -> `evidenza` | -10,450 | -1,460 | 0,0290 | n.d. | n.d. | 875/916 blocchi per seme | momento selettivo contro aggiornamento continuo |
| ADAPT-02 | `A2-ogni_blocco-cinque-semi` | LR | 42-46 | `FIFO/ogni_blocco` -> `per-classe/ogni_blocco` | -64,632 | -16,566 | 0,0511 | 5802237,6 | -28710,8 | 875/916 blocchi per seme | media delle differenze appaiate per seme; FN e FP sono differenze medie |
| ADAPT-02 | `A2-ogni_blocco-cinque-semi` | MLP | 42-46 | `FIFO/ogni_blocco` -> `per-classe/ogni_blocco` | -57,739 | -15,723 | 0,1419 | 5183444,6 | -27250,8 | 875/916 blocchi per seme | media delle differenze appaiate per seme; FN e FP sono differenze medie |
| ADAPT-02 | `A2-ogni_blocco-cinque-semi` | KAN | 42-46 | `FIFO/ogni_blocco` -> `per-classe/ogni_blocco` | -56,659 | -11,027 | 0,0843 | 5.086.487 | -19111,2 | 875/916 blocchi per seme | media delle differenze appaiate per seme; FN e FP sono differenze medie |
| ADAPT-02 | `A2-evidenza_inversione-cinque-semi` | LR | 42-46 | `FIFO/evidenza_inversione` -> `per-classe/evidenza_inversione` | -63,627 | -17,249 | 0,0481 | 5.711.979 | -29.895 | 875/916 blocchi per seme | media delle differenze appaiate per seme; FN e FP sono differenze medie |
| ADAPT-02 | `A2-evidenza_inversione-cinque-semi` | MLP | 42-46 | `FIFO/evidenza_inversione` -> `per-classe/evidenza_inversione` | -56,550 | -17,564 | 0,1024 | 5076687,8 | -30441,2 | 875/916 blocchi per seme | media delle differenze appaiate per seme; FN e FP sono differenze medie |
| ADAPT-02 | `A2-evidenza_inversione-cinque-semi` | KAN | 42-46 | `FIFO/evidenza_inversione` -> `per-classe/evidenza_inversione` | -45,055 | -7,816 | 0,0589 | 4044722,2 | -13547,2 | 875/916 blocchi per seme | media delle differenze appaiate per seme; FN e FP sono differenze medie |
| ADAPT-03 | `A3-sola-soglia-C-seed42` | LR | 42 | `frozen+soglia_B` -> `frozen+soglia_adattiva/per-classe` | 11,548 | -14,426 | 0 | -1.036.709 | -25.003 | 875/916 blocchi | score e AUROC invariati; cambia soltanto la soglia |
| ADAPT-03 | `A3-sola-soglia-C-seed42` | MLP | 42 | `frozen+soglia_B` -> `frozen+soglia_adattiva/per-classe` | 55,366 | 11,783 | 0 | -4.970.433 | 20.422 | 875/916 blocchi | score e AUROC invariati; cambia soltanto la soglia |
| ADAPT-03 | `A3-sola-soglia-C-seed42` | KAN | 42 | `frozen+soglia_B` -> `frozen+soglia_adattiva/per-classe` | 17,727 | 1,478 | 0 | -1.591.419 | 2.561 | 875/916 blocchi | score e AUROC invariati; cambia soltanto la soglia |
| ADAPT-03 | `A3-pesi-piu-soglia-C-seed42` | KAN | 42 | `pesi_adattati+soglia_B` -> `pesi_adattati+soglia_B_ricalibrata` | 60,474 | 29,667 | 0 | -5.428.963 | 51.417 | 875/916 blocchi | unica traiettoria dei pesi; B e' archivio aggiuntivo |
| ADAPT-04 | `A4-192-64-C-seed42` | KAN | 42 | `frozen_cal64` -> `fit192_cal64` | -2,599 | -16,080 | -0,1093 | 233.335 | -27.870 | 875/916 blocchi | effetto congiunto dei pesi rispetto alla sola soglia |
| ADAPT-04 | `A4-192-64-C-seed42` | KAN | 42 | `fit192_b` -> `fit192_cal64` | 34,331 | 12,354 | 0 | -3.081.988 | 21.412 | 875/916 blocchi | effetto della soglia a pesi identici |

## Provenienza

- ADAPT-01: `replay_evidenza/confronto_politiche.json`;
- ADAPT-02: `adapt02_memoria_per_classe/analisi_campagna_C.json`;
- ADAPT-03 soglia: `adapt03_soglia_adattiva/analisi_checkpoint.json`;
- ADAPT-03 pesi+soglia:
  `adapt03_pesi_piu_soglia/replay_completo/riepilogo_replay_completo.json`;
- ADAPT-04:
  `adapt04_calibrazione_recente/replay_completo/riepilogo.json`.

Dimensioni e SHA-256 delle fonti versionate e dei raw esterni sono registrati
in `manifest.json`. Questa appendice non usa D e non aggiunge esperimenti.

# Autorizzazione della matrice completa ADAPT-06

La matrice definita nel protocollo e in `MATRIX.csv` è stata autorizzata il
9 novembre 2026 dopo la verifica indipendente del pilot e il merge della PR
#14 (`4fea0dc3300ca3f8d6a36d538b1786364ea7e932`).

L'autorizzazione copre esclusivamente:

- stream completi TON e UNSW;
- seed 42–46;
- acquisizione RANDOM e ACTIVE per GAM, LR e ARF FIT-only;
- ALWAYS e GATED per GAM/LR;
- riferimenti Frozen/B e threshold-only già previsti;
- 60 report JSON e 100 traiettorie adattive.

Ogni cella deve partire dagli stati A/B archiviati in un processo nuovo e
scrivere in una directory nuova. Protocollo, budget, ritardo, memorie, criteri
del gate e regola GO/NO-GO/INCONCLUSIVE restano invariati. Assenza di
promozioni, insufficienze e risultati sfavorevoli sono esiti da conservare.

Non sono autorizzati nuovi fit iniziali, tuning, ADAPT-07 o riuso dello stato
finale del pilot.

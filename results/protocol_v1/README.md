# Risultati del protocollo v1 (superati)

I file in questa cartella sono stati prodotti **prima** della correzione del
protocollo (commit del 20 luglio 2026) e non sono stati rigenerati sotto il
protocollo v2. Sono conservati per tracciabilita', **non vanno citati come
risultati correnti**.

Il difetto di v1: il ranking per mutual information che seleziona le 10
feature numeriche era calcolato su un campione dell'intero dataset prima
dello split, e i vocabolari delle categoriche erano costruiti su train+test.
Effetto misurato su TON_IoT: la selezione per-fold sceglie le stesse 10
feature in 15 fold su 15 (`../leakage_audit_stability.csv`), quindi i numeri
non sono invalidati — ma il protocollo non e' quello dichiarato oggi.

## Attenzione a due file in particolare

* `cv_multiseed_summary_real.csv` — confronta la KAN con le baseline su
  spazi di feature DIVERSI (KAN sulle 14 grezze, baseline sulle 10 derivate
  del paper) e riporta LightGBM a 0,9818. Con input identici LightGBM fa
  **0,9991**: vedi `../cv_leakagefree_summary_binary_ALL.csv`. E' il
  confronto viziato che la fase 2 ha corretto.
* `kan14_cv_summary_real.csv` — riporta 0,9837 per la KAN single-layer.
  Il valore v2 e' 0,9835 +/- 0,0007, statisticamente identico, ma prodotto
  con selezione delle feature dentro ogni fold.

## Dove sono i risultati correnti

| Tema | File v2 |
|---|---|
| Binario in-domain, tutti i modelli | `../cv_leakagefree_summary_binary_ALL.csv` |
| Multiclass in-domain, tutti i modelli | `../cv_leakagefree_summary_multiclass_ALL.csv` |
| Cross-domain, 4 direzioni | `../crossdomain_table.csv`, `../crossdomain_degradation.csv` |
| Ingombro dei modelli | `../footprint.csv` |
| Effetto del leakage v1 | `../leakage_audit_stability.csv` |

# Valutazione Full Dataset — KAN-IDS Multi-Layer

**Data**: 2026-05-29  
**Dataset**: `train_test_network.csv` — 211043 sample totali  
**Modello**: KAN multi-layer (INDIM=10, HIDDEN=16, C=10, KSEG=8, L=64)  
**Training**: 48000 sample (60k campionati seed=42, split 80/20)  
**Preprocessing**: QuantileTransformer(normal, n_quantiles=1000, seed=42), fittato su training  

---

## Risultati globali

| Metrica | 12k test set | 211k full dataset |
|---|---|---|
| Accuracy | 0.9623 (11547/12000) | **0.9644** (203533/211043) |
| Macro-F1 | 0.9118 | **0.9177** |
| Weighted-F1 | — | 0.9663 |

---

## Report per classe (211k sample)

| Classe | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| backdoor | 20000 | 1.00 | 1.00 | **1.00** |
| ddos | 20000 | 0.95 | 0.93 | 0.94 |
| dos | 20000 | 0.99 | 0.97 | 0.98 |
| injection | 20000 | 0.90 | 0.89 | 0.90 |
| mitm | 1043 | 0.34 | 0.88 | 0.49 |
| normal | 50000 | 0.99 | 0.98 | **0.99** |
| password | 20000 | 1.00 | 0.98 | **0.99** |
| ransomware | 20000 | 1.00 | 1.00 | **1.00** |
| scanning | 20000 | 0.98 | 0.99 | **0.99** |
| xss | 20000 | 0.91 | 0.91 | 0.91 |

**Nota**: `mitm` è fortemente sbilanciato (1043 sample, 0.5% del dataset).  
Macro-F1 senza mitm: **0.9652**

---

## Principali confusioni

| Reale | Predetto come | Errori | % |
|---|---|---|---|
| xss | injection | 1296 | 6.5% |
| injection | xss | 1219 | 6.1% |
| injection | ddos | 514 | 2.6% |
| ddos | injection | 477 | 2.4% |
| ddos | xss | 376 | 1.9% |
| ddos | mitm | 342 | 1.7% |
| dos | mitm | 248 | 1.2% |
| xss | mitm | 258 | 1.3% |

La principale fonte di errori è la confusione **xss ↔ injection** e la classe **mitm** 
(pochissimi sample nel training → alta recall ma precision molto bassa).

---

## Riproducibilità

```bash
python scripts/passo5_eval.py   # rigenera preprocessing e harness
# Poi modificare per usare il full CSV invece del test set
```

Pipeline completa riproducibile con `random_state=42` in tutti i passaggi.

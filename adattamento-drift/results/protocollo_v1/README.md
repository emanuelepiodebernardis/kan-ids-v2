# Risultati del protocollo v1 (superati)

I file in questa cartella sono stati prodotti **prima** della correzione del
protocollo di selezione degli iperparametri, quando `iters` e `ridge`
venivano scelti massimizzando la balanced accuracy calcolata sulle stesse
righe target riportate come risultato finale.

Sono conservati per tracciabilita', come `results/protocol_v1/` nella radice
del repository. **Non vanno citati**: i valori corrispondenti del protocollo
corretto stanno in `results/` senza suffisso.

Cosa e' cambiato:

- `kanids/valutazione.py` divide il complemento delle righe selezionate in
  validation (30 %) e test (70 %), disgiunti, stratificati sull'etichetta e
  determinati dal solo seed.
- `scripts/sweep_iperparametri.py` gira in modo selezione: leggere il test
  solleva `AccessoAlTestVietato`. Lo sweep di `iters` legge la validation;
  quello di `ridge`, che gira sulla simulazione prequenziale dove non esiste
  un complemento, usa seed di calibrazione (90-94) disgiunti dai seed di
  riporto (42-51).

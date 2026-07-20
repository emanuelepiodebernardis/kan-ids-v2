# Dati: TON_IoT Network Dataset

Questo repository **non** include i dati (CC BY 4.0, ma è buona pratica non
ridistribuirli). Scaricali dalla fonte ufficiale e mettili nella root del repo.

## File necessario

`train_test_network.csv` — il subset network di TON_IoT (44 colonne grezze:
statistiche di flusso TCP/UDP/ICMP, metadati DNS/SSL/HTTP). È lo stesso file
usato nel lavoro precedente.

## Dove scaricarlo

Fonte ufficiale (UNSW Canberra, Moustafa et al.):
https://research.unsw.edu.au/projects/toniot-datasets

Il file si trova nella sezione "Processed datasets" / "Train_Test_datasets"
della cartella Network dataset.

## Verifica rapida

Dopo il download, controlla che le colonne grezze attese siano presenti:
`src_bytes`, `dst_bytes`, `src_pkts`, `dst_pkts`, `duration`, `proto`,
`label`, `type`. Il feature engineering unificato si basa su queste.

```python
import pandas as pd
df = pd.read_csv("train_test_network.csv", nrows=5)
print(df.shape)          # atteso: 44 colonne
print("label" in df.columns, "proto" in df.columns)
```

## Riferimento

N. Moustafa et al., "A New Threat Intelligence Scheme for Safeguarding
Industry 4.0 Systems", IEEE Access, 2018.

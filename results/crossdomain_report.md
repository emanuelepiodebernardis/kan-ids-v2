# Cross-domain TON_IoT <-> BoT-IoT

Task binario normal vs attack, spazio armonizzato a 13 feature candidate (10 selezionate per MI **sul solo source domain**).


## Balanced accuracy (media dei due recall)


### ton->ton

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| LightGBM | 0.9962 ± 0.0003 | 0.9970 | 0.9954 | 0.9978 | 15 |
| XGBoost | 0.9948 ± 0.0004 | 0.9977 | 0.9920 | 0.9976 | 15 |
| KAN(cat,ML) | 0.9933 ± 0.0005 | 0.9939 | 0.9927 | 0.9958 | 15 |
| MLP(16) | 0.9885 ± 0.0018 | 0.9956 | 0.9814 | 0.9949 | 15 |
| DecisionTree(d=5) | 0.9828 ± 0.0013 | 0.9921 | 0.9735 | 0.9919 | 15 |
| KAN(cat,1L) | 0.9700 ± 0.0011 | 0.9686 | 0.9715 | 0.9796 | 15 |

### bot->bot

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| KAN(cat,ML) | 0.9979 ± 0.0013 | 0.9966 | 0.9993 | 0.9983 | 15 |
| LightGBM | 0.9966 ± 0.0038 | 0.9967 | 0.9965 | 0.9984 | 15 |
| DecisionTree(d=5) | 0.9953 ± 0.0044 | 0.9947 | 0.9958 | 0.9974 | 15 |
| KAN(cat,1L) | 0.9935 ± 0.0010 | 0.9870 | 1.0000 | 0.9934 | 15 |
| XGBoost | 0.9769 ± 0.0071 | 0.9999 | 0.9539 | 0.9999 | 15 |
| MLP(16) | 0.9460 ± 0.0165 | 0.9997 | 0.8924 | 0.9999 | 15 |

### ton->bot

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| KAN(cat,1L) | 0.5632 ± 0.0061 | 0.4416 | 0.6848 | 0.6126 | 3 |
| XGBoost | 0.5597 ± 0.0184 | 0.4254 | 0.6939 | 0.5965 | 3 |
| DecisionTree(d=5) | 0.5466 ± 0.0000 | 0.4558 | 0.6373 | 0.6262 | 3 |
| LightGBM | 0.4815 ± 0.0095 | 0.2900 | 0.6730 | 0.4495 | 3 |
| MLP(16) | 0.4703 ± 0.0782 | 0.4668 | 0.4738 | 0.6323 | 3 |
| KAN(cat,ML) | 0.4026 ± 0.0337 | 0.1999 | 0.6052 | 0.3306 | 3 |

### bot->ton

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| MLP(16) | 0.7343 ± 0.0149 | 0.9254 | 0.5432 | 0.8953 | 3 |
| LightGBM | 0.7171 ± 0.0442 | 0.6450 | 0.7892 | 0.7510 | 3 |
| KAN(cat,ML) | 0.6581 ± 0.0752 | 0.5334 | 0.7828 | 0.6531 | 3 |
| XGBoost | 0.6508 ± 0.0122 | 0.8132 | 0.4883 | 0.8242 | 3 |
| KAN(cat,1L) | 0.5989 ± 0.1077 | 0.3674 | 0.8303 | 0.4936 | 3 |
| DecisionTree(d=5) | 0.4651 ± 0.0140 | 0.1023 | 0.8280 | 0.1763 | 3 |

## Degrado in-domain -> cross-domain

| modello | ton_in_domain | ton->bot | delta_ton->bot | bot_in_domain | bot->ton | delta_bot->ton |
|---|---|---|---|---|---|---|
| MLP(16) | 0.9885 | 0.4703 | 0.5182 | 0.9460 | 0.7343 | 0.2117 |
| LightGBM | 0.9962 | 0.4815 | 0.5148 | 0.9966 | 0.7171 | 0.2795 |
| XGBoost | 0.9948 | 0.5597 | 0.4352 | 0.9769 | 0.6508 | 0.3261 |
| KAN(cat,ML) | 0.9933 | 0.4026 | 0.5907 | 0.9979 | 0.6581 | 0.3398 |
| KAN(cat,1L) | 0.9700 | 0.5632 | 0.4068 | 0.9935 | 0.5989 | 0.3946 |
| DecisionTree(d=5) | 0.9828 | 0.5466 | 0.4362 | 0.9953 | 0.4651 | 0.5301 |

## Sovrapposizione delle marginali (0 = disgiunte, 1 = identiche)

| feature | mediana TON | mediana BoT | sovrapposizione |
|---|---|---|---|
| byte_rate | 544217.687 | 32.431 | 0.085 |
| duration | 0.000 | 15.509 | 0.106 |
| bytes_total | 172.000 | 600.000 | 0.153 |
| pkt_asymmetry | 0.000 | 0.857 | 0.162 |
| payload_mean_src | 37.333 | 56.250 | 0.175 |
| flow_rate | 7978.723 | 0.404 | 0.178 |
| bytes_src | 82.000 | 600.000 | 0.254 |
| pkts_src | 1.000 | 6.000 | 0.327 |
| pkts_total | 2.000 | 7.000 | 0.360 |
| byte_asymmetry | 0.090 | 0.998 | 0.381 |
| payload_mean_dst | 20.000 | 0.000 | 0.496 |
| bytes_dst | 40.000 | 0.000 | 0.505 |
| pkts_dst | 1.000 | 0.000 | 0.626 |
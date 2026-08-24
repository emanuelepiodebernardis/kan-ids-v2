# Cross-domain TON_IoT <-> BoT-IoT

Task binario normal vs attack, spazio armonizzato a 13 feature candidate (10 selezionate per MI **sul solo source domain**).


## Balanced accuracy (media dei due recall)


### ton->ton

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| LightGBM | 0.9963 ± 0.0004 | 0.9970 | 0.9955 | 0.9978 | 50 |
| XGBoost | 0.9947 ± 0.0004 | 0.9977 | 0.9918 | 0.9976 | 50 |
| KAN(cat,ML) | 0.9932 ± 0.0007 | 0.9938 | 0.9927 | 0.9957 | 50 |
| MLP(16) | 0.9879 ± 0.0016 | 0.9958 | 0.9799 | 0.9948 | 50 |
| DecisionTree(d=5) | 0.9828 ± 0.0015 | 0.9919 | 0.9738 | 0.9919 | 50 |
| KAN(cat,1L) | 0.9701 ± 0.0012 | 0.9688 | 0.9715 | 0.9797 | 50 |

### bot->bot

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| LightGBM | 0.9971 ± 0.0031 | 0.9967 | 0.9975 | 0.9984 | 50 |
| KAN(cat,ML) | 0.9971 ± 0.0027 | 0.9966 | 0.9977 | 0.9983 | 50 |
| DecisionTree(d=5) | 0.9952 ± 0.0034 | 0.9948 | 0.9956 | 0.9974 | 50 |
| KAN(cat,1L) | 0.9934 ± 0.0010 | 0.9868 | 1.0000 | 0.9934 | 50 |
| XGBoost | 0.9779 ± 0.0087 | 0.9999 | 0.9560 | 0.9999 | 50 |
| MLP(16) | 0.9426 ± 0.0361 | 0.9997 | 0.8855 | 0.9998 | 50 |

### ton->bot

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| KAN(cat,1L) | 0.5573 ± 0.0095 | 0.4407 | 0.6740 | 0.6117 | 10 |
| XGBoost | 0.5528 ± 0.0269 | 0.4137 | 0.6918 | 0.5834 | 10 |
| DecisionTree(d=5) | 0.5494 ± 0.0089 | 0.4558 | 0.6430 | 0.6262 | 10 |
| LightGBM | 0.4779 ± 0.0083 | 0.2880 | 0.6677 | 0.4472 | 10 |
| KAN(cat,ML) | 0.4588 ± 0.0548 | 0.2605 | 0.6572 | 0.4098 | 10 |
| MLP(16) | 0.4369 ± 0.0634 | 0.4211 | 0.4526 | 0.5898 | 10 |

### bot->ton

| modello | balanced acc | recall attack | recall normal | F1 | n |
|---|---|---|---|---|---|
| MLP(16) | 0.7343 ± 0.0144 | 0.9145 | 0.5541 | 0.8909 | 10 |
| LightGBM | 0.6964 ± 0.0545 | 0.5898 | 0.8030 | 0.7087 | 10 |
| KAN(cat,ML) | 0.6855 ± 0.0977 | 0.5733 | 0.7978 | 0.6875 | 10 |
| XGBoost | 0.6487 ± 0.0461 | 0.7851 | 0.5124 | 0.8071 | 10 |
| KAN(cat,1L) | 0.6112 ± 0.0695 | 0.3211 | 0.9013 | 0.4626 | 10 |
| DecisionTree(d=5) | 0.4597 ± 0.0107 | 0.0935 | 0.8260 | 0.1624 | 10 |

## Degrado in-domain -> cross-domain

| modello | ton_in_domain | ton->bot | delta_ton->bot | bot_in_domain | bot->ton | delta_bot->ton |
|---|---|---|---|---|---|---|
| MLP(16) | 0.9879 | 0.4369 | 0.5510 | 0.9426 | 0.7343 | 0.2083 |
| LightGBM | 0.9963 | 0.4779 | 0.5184 | 0.9971 | 0.6964 | 0.3007 |
| KAN(cat,ML) | 0.9932 | 0.4588 | 0.5344 | 0.9971 | 0.6855 | 0.3116 |
| XGBoost | 0.9947 | 0.5528 | 0.4420 | 0.9779 | 0.6487 | 0.3292 |
| KAN(cat,1L) | 0.9701 | 0.5573 | 0.4128 | 0.9934 | 0.6112 | 0.3822 |
| DecisionTree(d=5) | 0.9828 | 0.5494 | 0.4334 | 0.9952 | 0.4597 | 0.5355 |

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
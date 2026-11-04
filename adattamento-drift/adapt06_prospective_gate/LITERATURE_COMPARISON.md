# Confronto iniziale con la letteratura

La tabella serve a delimitare il protocollo, non a rivendicare priorità. Le
differenze riguardano il disegno proposto; una claim di novità richiederebbe una
revisione bibliografica più ampia.

| Lavoro | Selezione delle label | Ritardo/memoria | Validazione dell'update | FPR | Differenza rispetto ad ADAPT-06 |
|---|---|---|---|---|---|
| Žliobaitė et al. (2014), [DOI](https://doi.org/10.1109/TNNLS.2012.2236570) | incertezza, allocazione dinamica e randomizzazione per stream con drift | stream generico; il ritardo IDS non è il centro del confronto | strategie di acquisizione, non gate prospettico IDS con CAL/VERIFY disgiunti | non è il vincolo operativo primario | ADAPT-06 conserva esplorazione e separa acquisizione dalla certificazione dell'update |
| Olímpio et al. (2023), [DOI](https://doi.org/10.1016/j.cose.2023.103451) | active learning su IDS packet-level con budget ridotto | studia esplicitamente delayed labeling | valuta impatto di delay e acquisizione sui learner; non usa il successivo controllo casuale come certificazione separata del candidato | riporta FAR insieme a recall/precision | ADAPT-06 appaia candidate/incumbent sulle stesse label CONTROL future e può non promuovere |
| Camarda et al. (2025), [PDF](https://ceur-ws.org/Vol-3962/paper42.pdf) | incertezza; label selezionate alimentano drift detector e finestra di retraining | batch online e retraining window | drift error-rate attiva pruning/partial fit di una Random Forest incrementale | TNR/F1 e costo label; non gate al target FPR raggiunto | ADAPT-06 rende disgiunte FIT, CAL e VERIFY per il candidato e mantiene ARF come baseline |
| Jeon e Kim (2026), [DOI](https://doi.org/10.3390/electronics15173812) | EDG disponibili cronologicamente, buffer e budget few-shot | esplicita arrival bias e orizzonte di raccolta | escalation fra ricalibrazione, retraining misto e target-only con time-ordered CV; la sufficiency finale usa una validazione retrospettiva | recall sotto vincolo FPR | ADAPT-06 usa label CONTROL successive al freeze per una decisione strettamente prospettica sotto delay |

## Posizionamento prudente

ADAPT-06 combina elementi noti — active learning, esplorazione, delayed labels,
memorie recenti, calibrazione e update selettivo — in un controllo causale
appaiato. L'oggetto da verificare è se l'allocazione congiunta delle label e il
gate prospettico evitino aggiornamenti dannosi sul caso studio, non se active
learning o drift-aware updating siano nuovi in sé.

Il confronto finale dovrà riportare per ogni lavoro almeno: unità selezionata,
pool disponibile, riuso delle label, ritardo, stato conservato, criterio di
promozione, metrica FPR effettiva e costo. Le fonti indicate qui sono il punto
di partenza richiesto dalla scheda, non una revisione sistematica.

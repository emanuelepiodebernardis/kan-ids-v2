# Nota del pilota — ADAPT-01, stream di sviluppo C

10 ottobre 2026. I numeri vengono dalla serie finale C: 100 rendiconti
effettivi dei calendari casuali, 40 originali conservati e 60 riallineati, più
i 10 controlli calibrato/evidence nello stesso ambiente. Il riepilogo leggero è
`serie_finale_C/riepilogo.json`; i rendiconti completi stanno negli archivi
esterni dichiarati in `ambiente_e_comandi.md`.

## Osservazioni

**Il confronto è ora appaiato.** La serie finale conserva tutti i 100 casuali
originali, ma per il confronto usa i 40 originali dei semi 44 e 46 e i 60
riallineati dei semi 42, 43 e 45. Per ogni rendiconto finale coincidono con il
controllo evidence dello stesso seme: soglie, misure frozen per blocco, digest
dei campioni di etichette, aggiornamenti richiesti e aggiornamenti
effettivamente applicati.

**Evidence migliora l'AUROC media rispetto ai calendari casuali, ma non è una
superiorità generale.** Le medie sui cinque semi sono: logistica 0,7756 contro
0,7539, MLP 0,7939 contro 0,7380, additivo 0,8341 contro 0,7621. Nello stesso
confronto evidence riduce il FPR complessivo, ma riduce anche il richiamo degli
attacchi: −0,0198 per la logistica, −0,0564 per l'MLP, −0,0424 per l'additivo.
Il risultato da evidenziare è quindi il compromesso: migliore ordinamento medio
e meno falsi allarmi complessivi, al prezzo di meno attacchi richiamati.

**Distribuzioni AUROC dei 20 calendari casuali per seme.** Ogni cella casuale è
`min/mediana/media/max`; le altre colonne sono i controlli appaiati.

| modello | seme | casuale | evidence | frozen | ogni blocco |
|---|---:|---:|---:|---:|---:|
| LR | 42 | 0,690/0,766/0,767/0,831 | 0,765 | 0,715 | 0,781 |
| LR | 43 | 0,688/0,754/0,748/0,814 | 0,785 | 0,715 | 0,765 |
| LR | 44 | 0,687/0,765/0,758/0,805 | 0,764 | 0,715 | 0,783 |
| LR | 45 | 0,643/0,744/0,728/0,794 | 0,743 | 0,715 | 0,759 |
| LR | 46 | 0,701/0,772/0,768/0,831 | 0,820 | 0,715 | 0,782 |
| MLP | 42 | 0,663/0,799/0,787/0,864 | 0,835 | 0,803 | 0,839 |
| MLP | 43 | 0,645/0,723/0,718/0,795 | 0,797 | 0,763 | 0,748 |
| MLP | 44 | 0,631/0,722/0,718/0,787 | 0,739 | 0,794 | 0,757 |
| MLP | 45 | 0,613/0,698/0,696/0,770 | 0,786 | 0,806 | 0,702 |
| MLP | 46 | 0,664/0,776/0,771/0,857 | 0,812 | 0,801 | 0,791 |
| additivo | 42 | 0,690/0,747/0,745/0,844 | 0,806 | 0,884 | 0,797 |
| additivo | 43 | 0,698/0,765/0,768/0,873 | 0,855 | 0,889 | 0,811 |
| additivo | 44 | 0,614/0,744/0,746/0,860 | 0,790 | 0,893 | 0,793 |
| additivo | 45 | 0,421/0,749/0,742/0,875 | 0,837 | 0,895 | 0,812 |
| additivo | 46 | 0,731/0,820/0,810/0,881 | 0,881 | 0,894 | 0,824 |

**Il congelato dell'additivo resta il riferimento più alto sull'AUROC.** In
media sui semi vale 0,8911, contro 0,8341 di evidence, 0,8073 di ogni blocco e
0,7621 dei calendari casuali. L'adattamento sull'additivo riduce i falsi
allarmi complessivi, ma non supera il modello congelato sull'ordinamento.

## Limiti

Sono risultati esplorativi su C, senza significatività statistica calcolata e
senza conferma su uno stream successivo. Il controllo casuale ha ora una
distribuzione di 20 calendari per seme, ma resta dentro C. Il rivelatore di
inversione è debole: esiste solo nel 41,6% dei blocchi e ha precisione 15–18%.
Le misure di tempo vengono da macchine condivise e valgono come ordine di
grandezza, non come latenza in esercizio. D non è indipendente da A, B e C: il
37,23% delle sue righe e il 58,03% dei suoi normali hanno un vettore già visto.

## Che cosa è bloccato

**Lo stream D è riservato e resta intoccato.** Il protocollo è fissato: D intero
come analisi principale, sottoinsieme a vettore non visto come analisi
supplementare sulle stesse predizioni, nessuna scelta di modello o politica sui
risultati di D. L'esecuzione su D resta bloccata fino ad autorizzazione.

## Il prossimo esperimento

Il passo naturale è decidere se migliorare il rivelatore dentro C, senza toccare
D: più normali etichettati nei blocchi poveri, oppure memoria che conservi
campioni per classe a parità di dimensione. La serie finale chiude il confronto
casuale; il punto aperto ora è se il compromesso evidence — AUROC più alta e
meno falsi allarmi, ma meno richiamo degli attacchi — sia accettabile o vada
corretto cambiando il rivelatore.

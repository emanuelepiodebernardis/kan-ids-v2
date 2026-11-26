# ADAPT-06I — feedback passato sui vettori ricorrenti

Il pilot misura quanto feedback FIT già disponibile corregga le decisioni
Frozen/B quando lo stesso vettore ricompare. Confronta la memoria aggiornata
causalmente con Frozen/B e con il dizionario fissato all’inizio del blocco
20, sugli stessi esempi e sullo stesso calendario archiviato. I pesi e la
soglia B sono gli stessi in tutti i confronti.

| Richiesta | Evidenza |
|---|---|
| Preregistrare politica, controlli e regola di estensione | [Protocollo](PROTOCOL.md), commit `4f97a0b73b50ebe85e00a3ace0ac1987640f2d58) |
| Riprodurre entrambi gli stream con seed 42 e budget storico | [Comandi](COMMANDS.md), [manifest](delivery/MANIFEST.json) e riepilogo delle richieste in [SUMMARY.json](delivery/SUMMARY.json) |
| Verificare ritardo, duplicati e conflitto | [Test sintetici](test_pilot.py), [riconciliazione dei prodotti](test_delivery.py) |
| Rendicontare supporti, confusioni e differenze | [Per blocco](delivery/BLOCKS.csv), [differenze totali](delivery/DIFFERENCES.csv), [nota](delivery/NOTE.md) |
| Quantificare concentrazione ed episodi | [Per chiave ammissibile](delivery/KEYS.csv), [per episodio favorevole](delivery/EPISODES.csv) |

Il runner ha verificato gli snapshot di FIT e le confusioni Frozen/B in
tutti i **355 blocchi di TON** e i **102 di UNSW**, compreso il
riscaldamento. La valutazione appaiata parte dal blocco 20 e include i
blocchi finali parziali.

| Stream | Blocchi valutati | ΔTP/ΔFP contro Frozen/B | ΔTP/ΔFP contro memoria fissata | Episodi / chiavi | Estensione |
|---|---:|---:|---:|---:|---|
| TON | 335 | +15.695 / −6.930 | +1.919 / −9.047 | 3 / 5 | Criteri soddisfatti |
| UNSW | 82 | 0 / 0 | 0 / 0 | 0 / 0 | Criteri non soddisfatti |

Per TON, gli episodi favorevoli ricadono nei blocchi 59–60, 281 e 295,
con indici da zero. La chiave che cambia più decisioni pesa 8.522 su
25.809 cambi contro Frozen/B. Su UNSW la memoria aggiornata è ammissibile
nel 17,86% delle righe, ma lascia invariata la decisione su ciascuna.
I risultati riguardano il seed 42 sulle due catture di sviluppo. ADAPT-07
resta in backlog.

`BLOCKS.csv` conserva la frontiera temporale di FIT a inizio blocco e
l’hash del suo snapshot a fine blocco. `KEYS.csv` include una riga per
chiave risultata ammissibile in almeno uno dei bracci con memoria e la
frequenza sull’intera finestra di valutazione. `SUMMARY.json` riporta
anche la chiave più frequente a prescindere dall’ammissibilità.

La riproduzione dall’archivio estratto ha confermato byte per byte i
quattro CSV e il riepilogo JSON. La suite mirata con i moduli di
riferimento ha superato 120 test, di cui 15 per ADAPT-06I. Ruff e
`git diff --check` hanno dato esito positivo.

## Estensione TON

I risultati del seed 42 sono stati verificati e riutilizzati dalla PR #24.
I quattro nuovi replay TON, seed 43–46, seguono il
[protocollo dell’estensione](TON_EXTENSION.md). La
[tabella dei cinque seed](ton_matrix/SEEDS.csv) e la [nota](ton_matrix/NOTE.md)
riportano i conteggi appaiati, gli episodi favorevoli e la concentrazione per
chiave. I [comandi](COMMANDS.md#estensione-ton-ai-cinque-seed) e il
[manifest](ton_matrix/MANIFEST.json) identificano fonti e prodotti.
Per UNSW vale il pilot sopra.

Su TON tutti e cinque i seed migliorano sia TP sia FP rispetto a
Frozen/B. Nei totali contro entrambi i controlli risultano favorevoli
quattro seed su cinque. Il criterio del pilot è soddisfatto nei seed
42, 43 e 46. Il seed 44 migliora entrambi i totali, con un solo episodio
e una sola chiave qualificante. Nel seed 45 la memoria aggiornata
riduce i falsi allarmi di 4.971 e i veri positivi di 1.693 rispetto
alla memoria fissata.

Il tratto 59–60 comprende blocchi favorevoli in tutti e cinque i seed.
Il riepilogo conta nove intervalli sulla cattura, unendo i blocchi
sovrapposti o consecutivi tra seed. La chiave che modifica più
decisioni pesa dal 27,90% al 42,14% dei cambi rispetto a Frozen/B,
a seconda del seed.

Tra seed variano i pesi A, la soglia B e le richieste ADAPT-06E.
Entro ogni seed i tre bracci condividono questi elementi. Le differenze
appaiate misurano quindi il contributo della memoria per ciascuna
configurazione sulla stessa cattura.

Il controllo sullo ZIP originale ha riconciliato tutti i 1.675 blocchi
valutati nei cinque seed con i report ADAPT-06E per budget, ruoli, FIT192
e conteggi Frozen/B. La suite mirata ha superato 128 test, inclusi gli
otto controlli aggiunti per la matrice.
La riproduzione da un’estrazione pulita dello ZIP ha confermato
byte per byte i quattro CSV e il riepilogo JSON della prima consegna.

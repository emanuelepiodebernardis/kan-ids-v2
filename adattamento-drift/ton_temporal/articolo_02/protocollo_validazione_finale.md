# Proposta minima di validazione finale

## Domanda e disegno

Validare su D, una sola volta e dopo il congelamento del protocollo, il
compromesso osservato per KAN fra `frozen+CAL64`, `FIT192+B` e
`FIT192+CAL64`. Usare i semi 42--46, un calendario comune per seme, memoria
totale 192+64, quota di etichette 1%, ritardo `j+2`, ruoli immutabili per
vettore e la regola BA con il tie-break gia' fissato. B non viene riletto
durante il replay: `FIT192+B` usa la soglia fissa su B preregistrata prima
dell'avvio. D non interviene in alcuna scelta.

## Controlli preregistrati

1. Registrare commit, comando, ambiente e SHA-256 di A, B e D prima dei run.
2. Richiedere blocchi completi e ordinati, digest identici di righe ed etichette
   fra le tre braccia e uguale inizializzazione per seme.
3. Verificare causalita' `j+2`, separazione FIT/CAL, atomicita' degli update e
   identita' di pesi e score fra le due braccia a pesi comuni.
4. Conservare i valori mancanti nei blocchi monoclasse e riportare per ogni
   metrica il denominatore valido; non sostituirli con zero.
5. Conservare raw, confronto per blocco, manifest, hash e tempi effettivi in un
   archivio esterno immutabile.

## Metriche e criteri

Le metriche primarie sono FN, FP, recall e FPR globali. Balanced accuracy e
AUROC per blocco, con denominatore, sono secondarie. Riportare differenze
appaiate per seme, dispersione, segno in almeno quattro semi su cinque, peggiori
periodi consecutivi, strati update/skip, aggiornamenti disponibili/applicati e
costi.

Una variante puo' essere detta superiore solo se non peggiora ne' FN ne' FP e
migliora almeno uno dei due aggregati, con lo stesso segno in almeno quattro
semi su cinque. Segni opposti fra FN e FP definiscono un compromesso. Un aumento
di balanced accuracy o del solo recall non basta; l'AUROC va interpretata
separatamente dalle soglie. I criteri non saranno modificati dopo l'apertura di
D.

## Costo previsto

ADAPT-04 ha richiesto 190,088 s per 916 blocchi: 35,162 s di addestramento
iniziale, 147,160 s di replay e 7,766 s residui. Per cinque semi e un D di 916
blocchi la stima e' 950,44 s, circa 15,8 minuti seriali. Per `N` blocchi per
seme: `5 * (42,928 + 0,160655 * N)` secondi. Il raw atteso e' circa 19 MB e lo
ZIP circa 2,3 MB. Il benchmark ha raggiunto circa 1,25 GiB RSS; la somma
conservativa dei massimi osservati del solo stato logico del controllore e'
29.120 byte, non un picco simultaneo e senza overhead Python.

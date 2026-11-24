# ADAPT-06H — rilevamento a parità di falsi allarmi

## Domanda e perimetro

Verificare se la testa GAM candidata completa può collocare più attacchi sopra
soglia a parità di falsi allarmi, e se un tetto stimato con le sole etichette già
disponibili permette di ottenere quel vantaggio sui dati successivi. La base è il
merge ADAPT-06G `00f8fb729caeec845ad2052ed59b0f6c3acc64df`. Fonti: i dieci JSON
originali ADAPT-06E `28a81d5` (ZIP SHA-256
`91c2cf5c3fae62b1b6009714ed4a33560e5077e8cad647d87ad1c1cb9fb30eef`), il
manifest e il CSV dei 38 cicli ADAPT-06F integrati nella PR #21. TON e UNSW,
semi 42–46, GAM/RANDOM, stessi flussi, ID e ritardo di un blocco. Si valutano le
due teste salvate al FREEZE, sulla rappresentazione archiviata. La traiettoria
storica e le decisioni del gate rimangono fisse. Sono esclusi nuovi fit,
replay della politica, nuove richieste di etichette, modifiche del coefficiente
e ADAPT-07.

## Allineamento temporale

Si riconciliano tutti i FREEZE e tutti gli esiti terminali con il CSV di
ADAPT-06F. Per i 34 cicli osservabili, si valuta il `next_block` di ADAPT-06F,
pari a `decision_block + 1` con indici da zero. Si utilizzano anche eventuali
blocchi parziali. I quattro cicli a fine stream mantengono identificativi, esito
e motivo di assenza, con campi scientifici vuoti. I pesi candidati, i pesi
operativi e la soglia operativa originale sono quelli al FREEZE.

## Unica regola di soglia

Allarme se `score > soglia`. Ogni gruppo di punteggi identici ha un unico
esito. Per `n` normali di calibrazione e un tetto intero `0 ≤ k ≤ n`, se
`k < n` la soglia è il punteggio normale in posizione `k+1` nell’ordine
decrescente (con ripetizioni). La disuguaglianza stretta conserva al massimo
`k` falsi allarmi, anche quando il pareggio lascia parte del tetto inutilizzata.
Con `k = 0 < n` la soglia è il massimo score normale. Con `k = n` si usa
`−∞`, serializzato come `-inf`, e si ammettono tutti i punteggi finiti. Si
rifiutano punteggi non finiti.

## Confronti per ciclo

1. **Limite retrospettivo.** Sul blocco di valutazione, il tetto è il numero di
   FP del modello operativo alla soglia originale. Si applica la regola
   separatamente a entrambe le teste sugli stessi normali. La soglia più
   permissiva massimizza i TP al tetto dato senza separare i pari. Si riportano
   soglie, TP, FP, TN e FN effettivi. Le etichette del blocco servono a questo
   solo confronto e alla misura delle confusioni. Con blocco senza normali,
   il limite è banale (zero FP con tutti gli attacchi segnalati). Senza
   attacchi, i TP sono zero. Si registrano entrambi i casi e si interpreta
   l’ordinamento solo nei blocchi con almeno un normale e un attacco.
2. **Calibrazione disponibile al FREEZE.** Si usano esclusivamente gli ID
   `per_block[freeze_block].memory_observation_ids.cal`. Si verifica che
   ciascun ID sia unico, provenga da una richiesta CONTROL e che la sua
   etichetta sia arrivata entro la fine del FREEZE, con ritardo di un blocco.
   Si registra il supporto per classe e si usano solo i normali confermati
   per fissare il tetto. Il tetto è il numero di FP di quella testa operativa
   alla soglia originale sui normali CAL. Si applica la stessa regola
   separatamente a candidato e incumbent, fissando entrambe le soglie prima
   di accedere al blocco di valutazione. Si confrontano candidato calibrato,
   incumbent originale e incumbent ricalibrato. Si mantengono gli eventuali
   aumenti di FP al trasferimento della soglia. Basta un normale CAL per
   calcolare una soglia. Senza normali CAL, entrambi i bracci calibrati
   ripiegano sul modello operativo con soglia originale e il ciclo è
   marcato come fallback. Gli attacchi CAL non determinano le soglie.
   Le confusioni rimangono definite con una classe assente. I tassi con
   denominatore zero sono vuoti.

## Aggregazione e decisione

Si conservano tutti i cicli. Nei totali appaiati per stream e seme si conta
una volta ogni blocco osservabile. In caso di duplicato si seleziona il ciclo
con FREEZE più precoce, registrando la selezione nel CSV. Si sommano TP, FP, TN
e FN, mantenendo i fallback come parità rispetto alla politica originale. Un
seme ha supporto per il criterio se il totale contiene almeno un attacco, un
normale e un blocco con calibrazione eseguibile. Un seme è favorevole se il
candidato guadagna TP e non aggiunge FP rispetto a **entrambi** i controlli
sugli stessi blocchi. Si propone un replay solo con almeno tre semi favorevoli
su cinque nello stesso stream e almeno due episodi distinti fra quei semi.
Un blocco è favorevole con la medesima regola sui conteggi del blocco. Per gli
episodi si usa l’unione degli indici favorevoli dei semi qualificanti, quindi
si raggruppano i blocchi consecutivi. Sovrapposizioni e adiacenze tra repliche
formano un solo episodio. Si pubblicano anche gli episodi descrittivi degli
altri semi. Il limite retrospettivo non abilita il replay. Si riportano le
frequenze dei guadagni retrospettivi, i casi che si trasferiscono al confronto
operativo e quelli che si perdono. Nei blocchi senza guadagno retrospettivo
si segnala l’assenza di evidenza di migliore ordinamento al tetto osservato.
Queste conclusioni sono descrittive dei blocchi selezionati dal gate, su
catture condivise tra semi.

## Consegna

CSV per ciclo, tabella per seme, esito per episodio, nota e comando di
riproduzione. Manifest con SHA del protocollo, del codice, delle sorgenti e
dei prodotti. I criteri sono fissati prima di calcolare i nuovi risultati.

# ADAPT-06F — valutazione controfattuale nel blocco successivo

## Domanda e perimetro

L'analisi stabilisce quali candidati terminalmente respinti o sospesi in
ADAPT-06E avrebbero aumentato gli attacchi rilevati senza aggiungere falsi
allarmi nel solo blocco immediatamente successivo alla decisione. Sono usati i
dieci rendiconti originali di ADAPT-06E: TON e UNSW, semi 42–46. Non si
eseguono fit, ricalibrazioni, replay, tuning o modifiche della traiettoria.

I 38 cicli sono riconciliati uno a uno con `CYCLES.csv`: 24 `REJECT_HARM`, 11
stati `WITHHOLD_*` e 3 `ACCEPT`. La chiave è dataset, seme e blocco `FREEZE`;
sono verificati anche stato terminale e blocco della decisione.

## Convenzione temporale e confronto principale

Per ogni ciclo, `d` è il blocco della decisione terminale. Si valuta soltanto
`d+1`, incluse tutte le righe disponibili se è l'ultimo blocco parziale. I
quattro cicli chiusi alla fine dello stream rimangono nel CSV come
`NO_NEXT_BLOCK` e non entrano nei conteggi dei casi favorevoli.

Dal record `FREEZE` si recuperano senza ricostruzione:

- candidato: pesi e soglia del candidato;
- riferimento principale: pesi e soglia dell'incumbent operativo;
- controllo a calibrazione comparabile: gli stessi pesi dell'incumbent e la
  soglia diagnostica salvata, calibrata sugli stessi dati CAL del candidato.

Le tre configurazioni sono valutate sulle stesse righe e sulle stesse
etichette di `d+1`. Le etichette sono usate esclusivamente per contare TP, FP,
attacchi e normali. Un caso è favorevole quando `delta_tp > 0` e
`delta_fp <= 0`, dove le differenze sono candidato meno riferimento.

## Controlli vincolanti

Prima dello scoring sono verificati SHA-256 e dimensioni dei dieci JSON tramite
`RUN_MANIFEST.json`, lo SHA-256 dello ZIP completo, la provenienza degli input
e dei modelli A/B e l'identità specifica del rendiconto Windows UNSW/45. Lo
script rifiuta cicli mancanti, duplicati o non riconciliati.

Per ognuno dei tre `ACCEPT`, il candidato deve coincidere nel blocco `d+1` con
lo stato, la soglia e i parametri `gated` registrati. Anche TP e FP ricalcolati
devono coincidere con il rendiconto. Questo controllo presidia la frontiera
temporale e la regola di predizione `score > threshold`.

I semi sulla stessa cattura ripetono il medesimo traffico: i conteggi per seme
misurano la sensibilità alla selezione casuale delle etichette e non sono
repliche indipendenti del traffico.


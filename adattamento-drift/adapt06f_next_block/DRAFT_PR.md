## Perché

ADAPT-06E ha chiuso 38 cicli del gate: 24 `REJECT_HARM`, 11 `WITHHOLD` e
3 `ACCEPT`. La valutazione precedente descriveva l'evidenza VERIFY usata per
decidere. Questa consegna misura invece che cosa avrebbero prodotto il
candidato e l'incumbent salvati al `FREEZE` nel solo blocco immediatamente
successivo alla decisione terminale.

## Provenienza e metodo

L'analisi parte dai dieci JSON originali TON/UNSW, semi 42–46, eseguiti al
commit `28a81d5de70642d6f76d31a79d865103e4eb5bd2`. Lo ZIP sorgente ha SHA-256
`91c2cf5c3fae62b1b6009714ed4a33560e5077e8cad647d87ad1c1cb9fb30eef`.
UNSW/45 usa esplicitamente il report Windows identificato nella PR #20,
SHA-256 `1259d7a476ffa61a9d04d2355ba80bd366c019d23d109adbfca8a0ab29166f27`.
I dieci JSON locali, `RUN_MANIFEST.json` e `CYCLES.csv` sono inoltre
confrontati direttamente con le copie contenute nello ZIP e devono coincidere
byte per byte.

Per ciascun ciclo lo script recupera dal `FREEZE`:

- pesi e soglia del candidato;
- pesi e soglia dell'incumbent operativo;
- gli stessi pesi dell'incumbent con la soglia diagnostica calibrata sugli
  stessi dati CAL del candidato.

Le tre configurazioni sono valutate sulle stesse righe di `d+1`. Le etichette
servono soltanto al conteggio di TP e FP. Non sono eseguiti fit,
ricalibrazioni, replay, tuning o modifiche della traiettoria. Il criterio
favorevole è `delta_tp > 0` e `delta_fp <= 0`.

## Risultati

Tutti i 38 cicli sono riconciliati. Trentaquattro hanno un blocco successivo;
i quattro chiusi a fine stream restano nel CSV come `NO_NEXT_BLOCK`. I tre
`ACCEPT` riproducono digest, parametri, soglia e confusioni registrati in
`d+1`, incluso l'ultimo blocco parziale UNSW/46.

Fra i 31 cicli respinti o sospesi valutabili:

- nessuno dei 24 `REJECT_HARM` è favorevole rispetto all'incumbent operativo;
- quattro sospesi sono favorevoli, tutti nello stesso episodio TON
  decisione 352 / blocco successivo 353, ripetuto nei semi 42, 43, 45 e 46;
- a calibrazione comparabile restano quei quattro casi e si aggiunge
  UNSW/45, candidato congelato al blocco 65 e valutato nel blocco 76: rispetto
  all'incumbent operativo produce +87 TP e +11 FP, mentre rispetto allo stesso
  incumbent con soglia diagnostica produce +9 TP e -5 FP.

Nel singolo episodio TON, le quattro righe favorevoli producono da +6 a +18 TP
e da -1.994 a -1.508 FP rispetto all'incumbent operativo.

Il segnale operativo dipende quindi da un solo episodio di traffico ripetuto
fra semi, non da quattro catture indipendenti. Il quadro è descrittivo e non
modifica il NO-GO preregistrato di ADAPT-06E.

## Prodotti e controlli

- un unico CSV con una riga per ciclo;
- nota breve, riepilogo JSON e manifest con la catena degli hash;
- script e comando PowerShell riproducibili;
- 12 test mirati su estensione, riconciliazione, casi terminali, regola
  favorevole, blocco parziale, tre `ACCEPT`, provenienza e portabilità;
- `ruff check` superato.

I report originali e lo ZIP sorgente rimangono invariati. La PR resta Draft
fino al riesame.


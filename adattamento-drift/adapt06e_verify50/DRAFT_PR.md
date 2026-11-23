# adapt-06e: verificare metà delle etichette mentre un candidato è aperto

## Stato

Draft. Protocollo, codice e test sono separati dai risultati. I replay reali
possono iniziare soltanto dopo il commit di questo checkpoint, il preflight e
la riproduzione dei due riferimenti ADAPT-06B seed 42.

## Modifica

ADAPT-06E conserva GAM/RANDOM, FIT192, CAL256, budget dell'1%, ritardo,
margini, scadenza e ID richiesti di ADAPT-06B. Quando un candidato è aperto
all'inizio del blocco, la quota complessiva VERIFY passa da un quarto a metà
delle richieste: i CONTROL originali restano invariati e gli ID aggiuntivi
provengono dalla testa dell'ordine RANDOM_FIT già determinato. Le etichette
aggiuntive appartengono esclusivamente al ciclo e non entrano in FIT o CAL.

## Controlli eseguiti

- 14 test ADAPT-06E: budget, identità degli ID, ritardo, separazione dei ruoli,
  campioni tardivi, fine stream, causalità e validatore negativo;
- regressione del percorso storico e controllo del manifest ADAPT-06B;
- controlli statici Ruff superati.

## Risultati

Non ancora eseguiti in questo checkpoint. Sono predisposti la riproduzione dei
riferimenti TON/UNSW seed 42 e i dieci replay completi seed 42–46, ma questo
testo non anticipa alcun esito.


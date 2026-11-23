# ADAPT-06E — VERIFY 50/100

Questa cartella implementa il confronto preregistrato fra ADAPT-06B e una
variante che, soltanto mentre un candidato è aperto, assegna a VERIFY metà
delle richieste di etichetta. Il budget totale, gli ID richiesti e tutti gli
altri parametri restano invariati.

## Ordine vincolante

1. Eseguire i test sintetici e registrare il commit di protocollo/codice.
2. Eseguire `preflight.py` sugli input e sugli stati A/B archiviati.
3. Rieseguire il riferimento ADAPT-06B per TON e UNSW, seed 42, con
   `verify_reference.py`.
4. Soltanto dopo entrambi i PASS, eseguire i dieci replay con
   `run_matrix.py` in una directory nuova.
5. Verificare e aggregare con `analyze.py`.

Ogni report viene eseguito in un processo separato. `verify_report.py`
ricontrolla budget, ordine temporale, conservazione dei campioni e separazione
fra VERIFY_EXTRA, FIT e CAL. Il runner rifiuta la sovrascrittura di output
esistenti.

## Artefatti

- `PROTOCOL.md` e `config.json`: disegno e parametri congelati;
- `engine.py` e `runner.py`: motore isolato e ingresso della variante;
- `preflight.py` e `verify_reference.py`: controlli prima dei risultati;
- `run_matrix.py`: launcher dei dieci replay completi;
- `verify_report.py`: validatore autonomo dei report;
- `analyze.py`: confronti appaiati, costi e decisione preregistrata;
- `test_verify50.py`: test sintetici e controprove negative;
- `COMMANDS.md`: comandi PowerShell effettivi.

I report JSON completi restano nell'archivio esterno; nella PR entrano
manifest, hash, CSV e rapporto leggero. Nessun fit iniziale, tuning, ACTIVE o
ADAPT-07 appartiene a questa consegna.

## Stima iniziale

I riferimenti storici impiegano circa 2,8 minuti seriali per i primi due
stream seed 42 e 21,3 minuti per i dieci replay. Prima dei risultati è quindi
registrata una stima prudenziale di 15 minuti per ottenere e verificare i
primi due report e di 45 minuti per replay, controlli e aggregazione completi
sulla macchina Windows documentata. La consegna completa è prevista entro il
24 novembre 2026.


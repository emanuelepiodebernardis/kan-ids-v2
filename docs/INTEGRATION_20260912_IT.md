# Paper 1 — stato dell'integrazione del 12 settembre 2026

Questa è una correzione software privata v0.9.1 della copia di finalizzazione
v0.9.0. La bozza dell'articolo resta v0.9.0: non è stato emesso un nuovo PDF,
tag, release GitHub o DOI. Le descrizioni storiche sottostanti nel README
non sono una dichiarazione di validazione completa di questa integrazione.

## Perimetro e base di confronto

La base pubblica verificata è il commit
`df72c7764874efb8de13383d265e3ff0a8ba937c`.
La precedente `reference_software` non era un overlay di soli tre file o di
tre piccole correzioni. Conteneva il livello di finalizzazione del supervisore:
selezione LUT sul train, XAI, coorte comune, kernel corretti, infrastruttura
di misura e documenti/evidenze. «Tre interventi» indicava le aree di lavoro,
non l'intero diff rispetto al repository pubblico.

Confronto byte per byte, prima di questa correzione: 507 file identici,
23 modificati e 71 nuovi file di finalizzazione, più tre backup/ricevute
del precedente applicatore. Separatamente, 115 file di `adattamento-drift/`
e `report_KAN-IDS_fase2.pdf` esistono solo nel repository pubblico. Sono
fuori dal Paper 1, ma **non vanno cancellati** durante l'integrazione.
`data/README.md` è presente e identico. I dataset raw non sono distribuiti
dal clone. Gli environment PlatformIO passano da 29 a 49: dieci aggiunte
di latenza e dieci di energia per la coorte comune.

Integrare il livello Paper 1 in una copia/branch locale, con i suoi file
di provenienza. L'infrastruttura energia può essere conservata ora, con
stato non misurato; non richiede di aspettare le nuove schede. Il protocollo
operativo va comunque adattato allo strumento realmente disponibile:
la traccia sincronizzata prevista dal vecchio protocollo non va data per
disponibile con un misuratore USB generico.

## Correzioni nuove v0.9.1

`reproduce.py --stage lut` esegue ora `select` sul train, confronta l'header
generato con quello congelato e solo dopo esegue `evaluate` sul test. Ogni
run usa una nuova directory; non sovrascrive header o risultati canonici.
Un errore di selezione o un header differente impedisce la valutazione test.
Il confronto footprint continua a riferirsi all'header congelato.

`tools/check_joint_pooling.py` controlla i due gruppi attivi del Paper 1:
identità dei dieci seed 42–51, copertura modello/dominio, unicità delle chiavi,
supporti delle classi e uguaglianza fra matrici pooled e somme dei record.
I test permanenti coprono la ripresa con seed 42, duplicati keep-last,
seed diversi a parità di cardinalità e matrici incomplete. Il writer
`joint_training.py` già corretto non è stato nuovamente modificato.

Verifiche locali, senza training o accesso alle schede:

```powershell
py -3.11 tools/check_joint_pooling.py
py -3.11 -m pytest -q tests/test_reproduce_lut_phases.py tests/test_lut_selection.py tests/test_joint_pooling_integrity.py
py -3.11 reproduce.py --stage lut
```

Richiedono le dipendenze Python esistenti del progetto. Il solo checker
pooling usa la libreria standard. Lo stage LUT richiede gli array congelati
in `artifacts/finalization/`; non riaddestra il classificatore.
`reproduce.py --stage all` non è stato eseguito e non è dichiarato verificato
end-to-end da questa correzione.

## Risultati e limiti

Il nuovo run LUT ha riprodotto L=1025 e gli stessi CSV numerici conservati:
168834 righe train, 42209 test, zero divergenze di decisione coefficienti/LUT.
È un controllo host di riproduzione di risultati esistenti, non una misura
fisica aggiuntiva. Gli header e i sorgenti MCU restano quelli della v0.9.

I sei JSONL ricevuti contengono 900 record per-fit e coincidono con i CSV
dei run. Le 42 matrici attive sono corrette. L'opzione
`--include-historical` verifica anche le vecchie griglie e termina con errore
per dodici matrici `ratio50` che omettono seed 42. Questi artefatti del
protocollo superato, con esposizione al test, restano conservati; le
ricostruzioni corrette sono evidenza separata, non nuovi risultati principali.

La tabella LUT a 1025 punti e i kernel corretti sono già nelle sorgenti
della build Mega del 9 settembre, prima del replay fisico. L'identità
rispetto a quella build è diversa dall'identità rispetto al baseline
pubblico, che contiene L=257. Una futura nuova build richiede la propria
identificazione; le vecchie misure non vengono trasferite automaticamente.

Energia e peak runtime SRAM restano non misurati. Nessun nuovo esperimento
ML, firmware, flash o deposito pubblico è stato eseguito in questa verifica.

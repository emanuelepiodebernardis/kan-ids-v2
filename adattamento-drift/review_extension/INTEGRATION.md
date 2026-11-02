# Paper 2 — consegna per Emanuele Pio De Bernardis

La versione corrente è nel ramo `paper2/dtrap-submission-20261102` del fork
`KuznetsovKarazin/kan-ids-v2`. La base del repository dello studente verificata
è `81d99d9020d60b39d3b5a65fc6f06296f89537db`. La consegna include tutti gli
incrementi precedenti e due nuovi blocchi: codice/evidenze di calibrazione e
manoscritto ACM. Non applicare soltanto l'ultimo commit a una base che non
contiene gli incrementi intermedi.

## Importare senza sovrascrivere il lavoro locale

Da un clone pulito, verificare prima `git status`. Poi:

```sh
git fetch https://github.com/KuznetsovKarazin/kan-ids-v2.git paper2/dtrap-submission-20261102
git switch -c paper2-dtrap-integration FETCH_HEAD
git log --oneline 81d99d9020d60b39d3b5a65fc6f06296f89537db..HEAD
git diff --stat 81d99d9020d60b39d3b5a65fc6f06296f89537db..HEAD
```

Se esiste già `paper2-dtrap-integration`, scegliere un nome nuovo; non usare
reset o push forzati. Il bundle permette lo stesso import offline:
`git bundle verify Paper2_History.bundle`, poi `git fetch` dal percorso del
bundle usando il ramo sopra. I riferimenti di provenienza conservano i
protocolli fissati prima delle rispettive esecuzioni.

## Controllare e ricostruire

1. Seguire `REPRODUCTION.txt`: ambiente, SHA-256, 99 test, ricostruzione delle
   tabelle dai report. I dati grezzi servono solo per rifare le predizioni.
2. Eseguire `make -C adattamento-drift/paper2_submission all` con TeX Live
   completo. `main.pdf` e `supplement.pdf` sono anonimi; `author-version.pdf`
   e `supplement-author.pdf` includono nomi e dichiarazioni. I vecchi PDF in
   `review_extension/manuscript/` sono superati.
3. Leggere le sezioni su copertura CAL, confronto ALL256, violazioni dei target
   e limiti dei due capture. Verificare il proprio CRediT.
4. Verificare l'esportazione con `export_release.py` anche sul sistema usato per
   l'integrazione. L'uscita deve dichiarare un numero non nullo di sorgenti; poi
   eseguire `Numerical_Evidence/verify_numerical_evidence.py` e controllare gli
   SHA-256 di entrambi gli ZIP.
5. Aprire una PR da questo ramo verso `main`, riportando le verifiche. Il merge
   e l'approvazione scientifica spettano ai coautori. Non rilanciare il vecchio
   D per cercare un seed o un peso migliore.

## Risultato nuovo da mantenere nel testo

TON, target FPR 5%: Frozen/ROLE256 realizza 31,98%; Frozen/ALL256 con lo stesso
sistema di abilitazione degli aggiornamenti realizza 4,53%. La capacità non
spiega il divario: cambiano gli esempi ammessi in calibrazione. Head/ALL256
realizza ancora 10,22%, quindi il target non è garantito. Con ALL256 identico e
lambda=2, head/frozen hanno BA 87,94%/86,78%, FPR 9,94%/13,16%, recall
85,83%/86,72%. Il vantaggio residuo è condizionato alla politica; su UNSW i
controlli storici restano più forti.

Set C aggiunge dieci replay: 600.825 controlli durante il replay e 574.280
verifiche di sorgenti, traiettorie e aggregazione. Tutte le traiettorie condivise
coincidono. La consegna originale passa 96/96 test mirati; l'integrazione ne
aggiunge 3 sull'esportazione multipiattaforma, per un totale corrente di 99/99.
La suite legacy più ampia passa 408 test,
ne salta 3 e fallisce 3 controlli di documenti/output e materiali storici che
richiedono archivi esterni assenti dal clone; dettagli in
`calibration_audit/provenance/execution.json`. Non dichiarare verde l'intera
suite storica.

## Prima dell'invio a DTRAP

- Tutti e tre gli autori devono approvare testo, responsabilità e dichiarazioni.
  Per Frontoni sono documentati Funding acquisition e Project administration;
  occorre confermare un contributo intellettuale sostanziale e la responsabilità
  richiesta da ACM. Aggiungere Writing — review & editing soltanto dopo un
  effettivo intervento sul testo.
- Il periodo GAIA verificato è agosto–dicembre 2026. Le relazioni sono dichiarate
  nella versione identificata, non nei file per revisori anonimi.
- Depositare il pacchetto pubblico con i metadati predisposti e registrare il
  DOI, che non è ancora stato ottenuto. Aggiornare Data Availability e ricompilare;
  verificare con la redazione l'anonimato del materiale pubblico durante la revisione.
- Usare i file di `DTRAP_Submission`, non questo documento, il bundle Git o
  documenti interni come response letter. La lettera all'editor non attesta
  approvazioni o esclusività da dichiarare dagli autori nel portale.

Il pacchetto scientifico è completo; restano gli adempimenti e le conferme degli
autori. Non è stata effettuata alcuna submission né alcun merge automatico.

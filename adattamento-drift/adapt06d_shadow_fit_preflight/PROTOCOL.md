# ADAPT-06D — protocollo shadow FIT192/FIT256

## 1. Domanda e perimetro

L'esperimento futuro verifica, nei soli tre candidati già osservati in
UNSW/seed42, se una memoria FIT256 produce candidati migliori di FIT192 quando
traiettoria, richieste, label, incumbent, CAL e VERIFY restano fissati. Questa
prima consegna verifica esclusivamente la fattibilità. Non esegue fit,
calibrazione di candidati, scoring, replay o promozioni.

I casi sono fissati prima dei risultati: `FREEZE` 18, 39 e 64, con confronto
primario rispettivamente al primo look completo 37, 62 e 89 (indici 0-based).
Non si selezionerà successivamente un look più favorevole. Altri look ed
eventuali chiusure parziali saranno soltanto diagnostici.

## 2. Inventario causale e memorie

Si usano esclusivamente richieste `RANDOM_FIT` la cui label è arrivata entro il
relativo `FREEZE`. Devono coincidere `observation_id`, ordine di inserimento,
label, preprocessing e regola `ClassFifoMemory` del runner verificato. Sono
esclusi dati futuri e ruoli `CONTROL`, CAL o VERIFY.

- FIT192: capacità 192, quota 96 per classe;
- FIT256: capacità 256, quota 128 per classe;
- per entrambi: FIFO binaria con lending, senza limite di età FIT.

La composizione non viene presupposta. Il preflight ricostruisce entrambi i
buffer e registra per ogni membro ID, label, blocco di osservazione/arrivo/primo
effetto, età e digest della feature trasformata. Riporta inoltre digest
ordinato e non ordinato, conteggi, età e sovrapposizione. FIT192 deve coincidere
esattamente con gli snapshot originali; una discrepanza rende il preflight
`FAIL`.

Il report storico conserva gli ID FIT ordinati canonicamente, non l'ordine
interno della FIFO, e non conserva il digest della matrice trasformata.
L'ordine e il digest riportati qui sono quindi ricostruiti causalmente da
richieste, arrivi e implementazione verificata. La futura riproduzione FIT192 è
il controllo indipendente previsto per confermarli prima di FIT256.

## 3. Stato, CAL e VERIFY fissati

Per ogni caso si congela l'incumbent operativo originale, inclusi digest,
vettore completo dei parametri e soglia. `CONFIG.json` materializza questi
stati, insieme al candidato FIT192 di riferimento, alla CAL e alla confusione
del primo look: la riproduzione non dipende quindi da un riferimento implicito
al solo digest. Si riusa la CAL originale del `FREEZE`, con gli stessi ID,
label e regola target-FPR. Ogni candidato futuro deve calibrare autonomamente
la propria soglia sulla stessa CAL: la soglia FIT192 non può essere copiata a
FIT256.

Il confronto usa le stesse osservazioni VERIFY e il medesimo intervallo del
primo look originale. I candidati shadow non vengono promossi e non modificano
FREEZE successivi o traiettoria. FIT, CAL e VERIFY devono restare disgiunti.
La CAL è identificata dal suo insieme canonico di ID; il suo ordine non entra
nella regola di selezione della soglia. L'ordine FIT e VERIFY resta invece
quello causale registrato.

## 4. Guardia di riproduzione FIT192

Una futura esecuzione deve procedere in quest'ordine:

1. ricostruire FIT192 e verificarne ID, ordine, label e feature;
2. rifare il solo candidato FIT192 e la sua calibrazione sulla CAL fissata;
3. riprodurre confusioni e decisione del primo look originale;
4. soltanto dopo tutti i `PASS`, eseguire il corrispondente FIT256 shadow.

Ogni candidato usa il preprocessing e la rappresentazione GAM archiviati in
`model.pkl`. Si crea un nuovo `LogisticRegression` senza warm start, con
`ridge=1`, `C=1`, `max_iter=1000`, solver `lbfgs`, intercetta attiva e nessun
`random_state`, nello stesso ambiente e con gli stessi limiti ai thread del
report originale. L'ordine di input è quello versionato in `FIT_MEMBERS.csv`.
La soglia di ciascun candidato viene ricalcolata separatamente sulla CAL
fissata con la regola target-FPR originale.

ID, appartenenza ai ruoli, conteggi, confusioni e decisioni richiedono identità
esatta. Parametri, soglie e metriche derivate usano tolleranza assoluta
`1e-12`, con tolleranza relativa zero. Ogni mancata coincidenza arresta tutti i
FIT256 non ancora eseguiti e deve essere spiegata; non si sostituiscono casi o
dati.

## 5. Criterio futuro prefissato

Sono previsti al massimo tre fit di controllo FIT192 e tre fit FIT256. In ogni
caso entrambi i candidati sono confrontati con lo stesso incumbent operativo
fissato, sulle stesse osservazioni VERIFY del primo look. Per «caso
aggiuntivo» si intende un caso nel quale FIT256 soddisfa tutti e tre i margini
mentre FIT192 non li soddisfa. Il criterio primario richiede almeno un caso
aggiuntivo che soddisfi insieme:

- guadagno di recall almeno 2 punti percentuali;
- delta FPR non oltre 1 punto percentuale;
- FPR candidato non oltre 6%.

FIT256 non deve perdere il caso già accettabile né introdurre una nuova
violazione di danno negli altri casi. Tutte le confusioni e i delta saranno
riportati anche a conteggio invariato. Un esito favorevole giustifica al più un
seguito concordato; un esito nullo vale soltanto per questi tre casi.

## 6. Costi e limiti

I riferimenti registrati sono 1,459 s per il caricamento input, 0,130 s per i
tre fit/calibrazioni candidati storici e 46,219 s per il replay decoupled
completo. Non sono misure del futuro shadow. Il preventivo seriale è:

| Fase | Intervallo previsto |
|---|---:|
| verifica input e ricostruzione causale | 5–15 s |
| 3 FIT192 + 3 FIT256 | 0,5–3 s |
| 6 calibrazioni indipendenti delle soglie | 0,5–2 s |
| scoring dei tre intervalli VERIFY e controlli | 5–20 s |
| test, manifest e confezionamento | 45–120 s |
| **totale** | **56–160 s** |

Il tempo storico di 0,130 s aggrega fit e calibrazione e non permette di
misurarli separatamente; le due righe future sono quindi intervalli preventivi
distinti, non misure osservate. La stima complessiva è un preventivo, non una
misura né una scadenza. I tre casi sono stati
scelti dopo la diagnosi e appartengono allo stesso seed osservato. Capacità,
composizione ed età cambiano insieme: non sarà possibile attribuire un effetto
alla sola numerosità. Nessuna griglia, nuovo seed/dataset o ADAPT-07 rientra in
questo protocollo.

# Le richieste del professore contro quello che è stato fatto

Confronto fra i **sei punti della revisione** del Prof. Kuznetsov (protocollo
v2, `CHANGELOG.md`), la lista *Cosa resta aperto* con cui la fase 2 si era
chiusa, e lo stato dopo il lavoro sull'adattamento al drift.

Le verifiche qui sotto sono state rieseguite su un clone pulito del
repository, non sulla copia di lavoro.

---

## 1. I sei punti della revisione

Eseguito `tools/audit_richieste.py` sul clone: **26 requisiti su 27
soddisfatti**, invariato rispetto alla fase 2.

| Punto | Richiesta | Stato |
|---|---|---|
| 1 | Protocollo leakage-free | soddisfatto — 3/3 requisiti |
| 2 | KAN multi-layer con lo stesso protocollo | soddisfatto — 2/2 |
| 3 | Secondo dataset (BoT-IoT) | soddisfatto — 4/5 |
| 4 | Baseline identiche | soddisfatto — 4/4 |
| 5 | Inferenza integer-only end-to-end | soddisfatto — 7/7 |
| 6 | Riproducibilità | soddisfatto — 4/4 |

L'unico requisito che l'audit segna ancora come **NON FATTO** è, dentro il
punto 3:

> *CIC-IoT-2023 come terzo dataset (secondario) — non iniziato: il
> professore lo indica come obiettivo secondario*

**Ma quel requisito è stato fatto.** Il controllo è questo:

```python
cic = any("cic" in p.name.lower() for p in R.glob("*")) and \
      any("cic" in p.name.lower() for p in (REPO / "scripts").glob("*"))
```

Cerca file con "cic" nel nome in `results/` e `scripts/` **della radice**. Il
lavoro su CIC-IoT-2023 vive in `adattamento-drift/`, che l'audit non guarda:
2 file di risultati e 6 file di codice che trattano CIC sono lì dentro e
restano invisibili.

**Va aggiornato il controllo, non il lavoro.** Finché non lo si fa, chiunque
esegua l'audit — professore incluso — legge che l'unica cosa aperta è ancora
aperta.

---

## 2. Cosa il punto 3 ha effettivamente ricevuto

La richiesta era CIC-IoT-2023 come terzo dataset, obiettivo secondario. È
stato consegnato più di così, e con una motivazione misurata invece che
assunta.

**Prima di scaricarlo, il costo è stato misurato.** Le 47 feature di
CIC-IoT-2023 non hanno i conteggi direzionali (`src_bytes`/`dst_bytes`,
`src_pkts`/`dst_pkts`) né lo stato della connessione: sette delle tredici
feature numeriche armonizzate sarebbero cadute. Proiettando TON_IoT e BoT-IoT
sullo spazio ridotto a parità di righe, seed e modelli, la riduzione costa
**0,009 in-domain ma fino a 0,29 sul risultato adattato**. Sono le feature
direzionali a rendere gli edge *adattabili*: un risultato che non si cercava,
e che dice quali feature contano per l'adattamento invece che per
l'accuratezza.

**Quindi è stato aggiunto UNSW-NB15 come terzo dominio** — stesso strumento
di cattura di BoT-IoT (Argus), spazio ricco intatto, nessuna corrispondenza
nuova da inventare — **e CIC-IoT-2023 come quarto**, nel suo spazio, come
prova di robustezza. La richiesta è soddisfatta con dei numeri, non elusa.

**Sul file di CIC è emerso un errore di diagnosi, poi corretto.** Una prima
analisi aveva concluso che il file caricato fosse CICIoMT2024 e privo di
durata utilizzabile. Rimisurato: `Duration` è davvero il TTL (mediana 64,
massimo 255) e `IAT` è corrotta, ma `test.csv` ha una colonna `flow_duration`
**genuina** — benigni con mediana 26,1 s contro attacchi a 0,0 s, correlazione
con `IAT` pari a 0,008. Il confronto fra le due terne è stato rifatto nello
spazio giusto.

**Il confronto fra le terne, a parità di spazio e di direzioni, su 10 seed:**

| | delta (UNSW − CIC) | t | p |
|---|---|---|---|
| non adattato | −0,130 | −16,5 | <0,0001 |
| 8 etichette | −0,007 | −0,3 | 0,76 |
| 32 etichette | −0,033 | −1,6 | 0,15 |
| 128 etichette | −0,086 | −6,6 | 0,0001 |

CIC-IoT-2023 non trasferisce meglio: **è più facile**, e lo è già prima di
qualunque etichetta. Un dominio verso cui si trasferisce bene senza adattarsi
non è un banco di prova severo. UNSW-NB15 resta la scelta giusta per
l'analisi principale, ma per questa ragione — non perché dia numeri migliori.

---

## 3. La lista *Cosa resta aperto* della fase 2

| Voce | Stato |
|---|---|
| CIC-IoT-2023 come terzo dataset | **fatto** (vedi sopra) — l'audit non lo vede |
| Benchmark fisici su Mega 2560 ed ESP32-C3 | **aperto** — passa al professore |
| Sensibilità al rapporto di undersampling (1:50) | **aperto, e ora pesa di più** |
| Larghezza nascosta, grado, clip non riselezionati | **aperto** |

**La sensibilità all'undersampling merita attenzione.** Era già aperta nella
fase 2, e tutto il lavoro sull'adattamento al drift la eredita senza
verificarla: ogni script usa `--ratio` con default 50.0. Non è più una lacuna
di una sezione, è un'assunzione non testata sotto quattordici sezioni di
risultati nuovi.

**I benchmark fisici hanno un sostituto parziale.** Non essendo disponibile
l'hardware, la sezione 17c costruisce un modello di costo in operazioni —
moltiplicazioni-accumulo int32, lookup in tabella, RAM di picco — in funzione
di *n*, *d* e iterazioni, così che moltiplicando per i cicli-per-operazione
del target si ottenga il tempo. Non sostituisce la misura, dice cosa
aspettarsi da essa.

---

## 4. Cosa è stato aggiunto oltre le richieste

Il lavoro sull'adattamento al drift non era nella lista della revisione:
nasce dal punto 3, dove il transfer collassava, e risponde alla domanda
successiva — *si può recuperare?*

**Diagnosi.** Il collasso non è una soglia mal posizionata. In TON→BoT il
ROC-AUC sul target è 0,54, cioè al caso: nessuna ricalibrazione lo salva.
Ma gli stessi edge, solo ripesati, arrivano a 0,92 — il modello ha imparato
le cose giuste con i pesi sbagliati. Con tre domini emerge un fenomeno
invisibile con due: in `unsw→ton` e `ton→unsw` il ROC-AUC è 0,26 e 0,27,
**sotto il caso**, cioè l'ordinamento è sistematicamente rovesciato.

**Il metodo.** La KAN single-layer è additiva, quindi un guadagno per edge
più un termine noto bastano a riscriverne il comportamento: **13
coefficienti, 24 byte**, senza toccare il firmware — è la tabella `MULT` di
moltiplicatori Q15 che il kernel già usa. Con **32 etichette** scelte da una
regola eseguibile a bordo.

**Verificato bit per bit.** L'header C pubblicato, compilato da un clone
pulito: `200 golden vector, logit diversi: 0, decisioni diverse: 0, 24 byte`.

**Il risultato più solido.** Il riadattamento continuo con buffer, **in
aritmetica intera**, batte il modello statico in **6 direzioni su 6**, con t
appaiati per seed da 3,3 a 91,5 e guadagni da +0,04 a +0,26. Punteggio,
selezione, stima dei guadagni e riadattamento: nessun float in nessun
passaggio.

**Undici metodi senza etichette provati e falliti** — riallineamento dei
quantili, tre regole di soglia, EM sul prior, TENT, TENT filtrato, IM/SHOT,
due varianti di ridge adattivo, k-center. Uno solo (IM) dà un guadagno
parziale. Non è più una lacuna: è il risultato che sostiene l'affermazione
che su questo problema **un piccolo budget di etichette è necessario**.

---

## 5. Le affermazioni che sono cadute strada facendo

Sono nel documento con la stessa evidenza di quelle che reggono.

**«Aggiornare poco batte riaddestrare tutto» non regge.** Misurato su due
direzioni sembrava robusto; su cinque direzioni e 10 seed, con test t
appaiati, il rifit completo vince più spesso e di poco (quattro sconfitte per
un totale di 0,145) e i 13 coefficienti vincono raramente e di molto (una
vittoria da 0,135). Il delta medio è **−0,002: un pareggio**. Quello che
resta è un'affermazione di costo — 24 byte contro 250, nessun riaddestramento
sul dispositivo — non di accuratezza.

**«6000 iterazioni chiudono il 65% del divario» non regge.** A campione
appaiato su 10 seed nessuna direzione si distingue dal rumore. L'effetto
esiste ma è concentrato: otto seed su nove sono rumore puro, uno solo
guadagna 0,183. È un'assicurazione contro la coda rara, non un guadagno
medio.

**«L'innesco conformal fornisce il segnale» era da togliere**, e c'è di più:
con ε in (0,1) la soglia raggiungibile tende a 1/e ≈ 0,368 mentre il p-value
medio sotto deriva si ferma a 0,40. **Nessuna scelta di ε può funzionare con
quel segnale di conformità** — un limite dimostrato, non constatato.

**Una direzione fallisce davvero.** In `unsw→bot` la selezione raccoglie zero
normali in **tutti e 10 i seed**: BoT-IoT ha 477 normali su 3,67 M. Il collo
di bottiglia non è l'adattamento, è trovare cosa etichettare.

---

## 6. Cosa resta da fare, in ordine

1. **Aggiornare `tools/audit_richieste.py`** perché guardi anche in
   `adattamento-drift/`. È l'unica cosa che oggi fa leggere al professore
   che il punto 3 è incompleto quando non lo è. Cinque minuti.
2. **Sensibilità al rapporto di undersampling.** Aperta dalla fase 2 ed
   ereditata da tutto il lavoro nuovo senza verifica.
3. **Benchmark fisici** su Mega 2560 ed ESP32-C3. Il modello di costo dice
   cosa aspettarsi; le misure vere restano da fare.
4. **Rendere affidabile la RLS.** Il modello di costo mostra che è **551
   volte più economica in calcolo e 5,9 volte più piccola in RAM** di
   qualunque alternativa misurata. Non è bloccata dalle risorse: è bloccata
   da un problema di accuratezza in due direzioni su sei, quelle in cui
   BoT-IoT è la sorgente. Risolverlo sblocca l'opzione più economica
   dell'intero lavoro.
5. **Un segnale di conformità diverso** da |z − mediana| per le due direzioni
   dove la martingala non scatta.
6. **Larghezza nascosta, grado e clip**, mai riselezionati dentro il ciclo.

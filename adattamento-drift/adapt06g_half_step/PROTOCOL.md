# ADAPT-06G — aggiornamento parziale dei pesi a parità di etichette

## Domanda

ADAPT-06F ha mostrato che, in 19 dei 24 candidati respinti da ADAPT-06E, il
candidato completo avrebbe rilevato più attacchi nel blocco successivo ma
avrebbe anche aumentato i falsi allarmi. Questa prova verifica se un unico
passo più piccolo conserva parte del guadagno di rilevamento contenendo gli
errori sul traffico normale.

## Variante unica fissata prima dei risultati

La sola variante è GAM/RANDOM con coefficiente `alpha = 0,5`. A ogni `FREEZE`,
dopo il nuovo fit sui 192 esempi della memoria FIT, si costruisce:

```text
half_step = (1 - alpha) * incumbent_operativo + alpha * full_fit
```

La media riguarda tutti i nove parametri della testa spline, intercetta
inclusa. Non sono provati altri coefficienti. Il candidato half-step è
calibrato sulla memoria CAL già disponibile prima di VERIFY, usando la stessa
regola di ADAPT-06E. Il gate valuta e, in caso di `ACCEPT`, promuove il
candidato half-step; la decisione influenza tutte le predizioni successive.

## Matrice e invarianti

Si eseguono dieci replay completi: TON e UNSW, semi 42–46. Ogni processo parte
dagli stati A/B archiviati. Restano invariati:

- budget di etichette dell'1% e ritardo di un blocco;
- FIT192, CAL256, quote per classe ed età CAL;
- assegnazione RANDOM, richieste CONTROL e VERIFY_EXTRA;
- look, scadenza, margini e criterio di danno di ADAPT-06E;
- regola di predizione `score > threshold`.

Sono registrati digest e parametri di incumbent, full fit e half-step. Per ogni
caso si verificano budget, unicità delle richieste, ritardo causale e ogni
eventuale differenza nelle assegnazioni di ruolo rispetto ad ADAPT-06E.

## Confronti e criterio primario

Per ogni dataset e seme si riportano TP, FP, FN, recall, FPR e promozioni per:

1. half-step gated;
2. ADAPT-06E full-step gated;
3. controllo a pesi fissi con lo stesso calendario e la stessa calibrazione;
4. Frozen/B;
5. ADAPT-06B gated archiviato.

Un seme è favorevole soltanto se half-step produce `delta_tp > 0` e
`delta_fp <= 0` sia contro ADAPT-06E sia contro il controllo matched. Il
segnale primario richiede gli stessi tre o più semi favorevoli in uno stream.
Tutti gli esiti dell'altro stream restano visibili.

## Episodi temporali e passaggio eventuale ad ADAPT-07

I guadagni e i danni sono localizzati per blocchi contigui. Un episodio è
identificato da dataset e intervallo contiguo; lo stesso intervallo ripetuto in
più semi conta una sola volta. ADAPT-07 può essere proposto soltanto se il
vantaggio compare in almeno due episodi temporali distinti e, negli stessi
semi, i falsi allarmi non aumentano rispetto ad ADAPT-06B.

## Esiti ammessi

GO, NO-GO ed esito nullo sono tutti risultati validi. Non sono ammessi nuovi
fit iniziali, etichette aggiuntive, tuning, coefficienti alternativi o nuovi
dataset. I risultati completi vengono conservati anche se sfavorevoli.


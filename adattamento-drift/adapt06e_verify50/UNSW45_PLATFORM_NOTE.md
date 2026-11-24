# UNSW/45 — riscontro Windows sul candidato del blocco 41

Il controllo usa esclusivamente il report originale
`unsw_seed45_verify50.json` (SHA-256
`1259d7a476ffa61a9d04d2355ba80bd366c019d23d109adbfca8a0ab29166f27`), eseguito al commit
`28a81d5de70642d6f76d31a79d865103e4eb5bd2`. Non sono stati ripetuti fit o replay.

## Frontiera causale

Il candidato viene congelato al blocco 41 e accettato al blocco 63 (indici
0-based). I nuovi pesi e la nuova soglia entrano nelle predizioni dal blocco
64. Nel report Windows, stato, nove parametri e soglia usati al blocco 64
coincidono esattamente con il candidato congelato.

- digest dello stato: `a1c014c98062407130de456d9be7432519894d45d904ecc1e0d14d5f7ef08c86`;
- pesi: `[0.47488231442321666, 2.9726275388881271, 0.83126095947345013, 1.2193059918441658, -0.088201153393299078, 0.40770624387059434, 1.3136239380529224, 1.3467843089998346, 4.3528503288583833]`;
- soglia: `-0.10554843372708511`;
- FIT: 192 ID, digest
  `aa4ee1a84dba5e98b5e0e546f7a2b880da18aaf60edc4373cf888d72f7381f9e`;
- CAL: 256 ID, digest
  `5377b1482b89accf5c1dcb3d4ca531c57c837f02b461f7a4df4aaad869956666`.

Gli ID completi, nel loro ordine registrato, sono in
`UNSW45_FREEZE41_IDS.csv`.

## Confronto degli artefatti Windows/Linux

I digest FIT e CAL coincidono fra i due ambienti:

| Oggetto | Windows | Linux | Esito |
|---|---|---|---|
| FIT IDs | `aa4ee1a84dba5e98b5e0e546f7a2b880da18aaf60edc4373cf888d72f7381f9e` | `aa4ee1a84dba5e98b5e0e546f7a2b880da18aaf60edc4373cf888d72f7381f9e` | uguale |
| CAL IDs | `5377b1482b89accf5c1dcb3d4ca531c57c837f02b461f7a4df4aaad869956666` | `5377b1482b89accf5c1dcb3d4ca531c57c837f02b461f7a4df4aaad869956666` | uguale |
| Soglia dal blocco 64 | `-0.10554843372708511` | `-0.097844396558292246` | diversa |

La scheda riporta per Linux Python 3.13.5, NumPy 2.3.5,
scikit-learn 1.7.2, SciPy 1.16.2 e BLAS a un thread. Non riporta il vettore
dei nove pesi Linux; il vettore Windows è pubblicato sopra per consentire il
confronto diretto, ma non viene dichiarata un'identità o una differenza dei
pesi senza il secondo vettore. Dai dati disponibili la prima divergenza
osservabile è quindi la soglia applicata dal blocco 64, non la selezione degli
esempi FIT/CAL.

Il report Windows registra Windows-11-10.0.26300-SP0, Python
3.13.2, NumPy 2.3.5,
scikit-learn 1.7.2 e River
0.22.0; i quattro limiti dei thread sono impostati a
uno.

## Confusione sull'intero stream

| Ambiente | TP | FP |
|---|---:|---:|
| Windows consegnato | 201179 | 18590 |
| Linux riportato dal relatore | 201178 | 18631 |
| Windows − Linux | +1 | -41 |

La differenza finale è coerente con una divergenza numerica della traiettoria,
ma questi dati non bastano ad attribuirla al fit oppure alla calibrazione: per
questa distinzione serve il vettore dei pesi Linux.

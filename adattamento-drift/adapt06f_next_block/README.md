# ADAPT-06F — il blocco successivo alle decisioni del gate

Questa cartella contiene un'analisi controfattuale circoscritta ai 38 cicli
terminali dei dieci replay ADAPT-06E. Per ogni ciclo confronta il candidato e
l'incumbent salvati al `FREEZE` sul solo blocco immediatamente successivo alla
decisione terminale. Il codice non modifica i report originali e non esegue
fit, selezione di soglie o replay.

## Prodotti

- `delivery/NEXT_BLOCK_CYCLES.csv`: una riga per ciclo, inclusi i quattro casi
  senza blocco successivo;
- `delivery/NOTE.md`: sintesi descrittiva dei casi favorevoli;
- `delivery/SUMMARY.json`: conteggi elaborabili automaticamente;
- `delivery/MANIFEST.json`: provenienza, hash, ambiente e comando canonico;
- `PROTOCOL.md`: perimetro e criteri fissati;
- `COMMANDS.md`: riproduzione su Windows e controllo della consegna;
- `test_next_block.py`: controprove su riconciliazione, causalità temporale,
  ACCEPT e prodotti consegnati.

I rendiconti completi e gli input non sono duplicati in Git. I percorsi locali
sono parametri espliciti del comando. Lo ZIP ADAPT-06E deve avere SHA-256
`91c2cf5c3fae62b1b6009714ed4a33560e5077e8cad647d87ad1c1cb9fb30eef`.
L'archivio sorgente è disponibile nel
[collegamento già verificato di ADAPT-06E](https://drive.google.com/file/d/1OCTJSaD9blogZM2zcQ4aXeGaZPWJgjCt/view?usp=drive_link).
Lo script richiede che i dieci report, `RUN_MANIFEST.json` e `CYCLES.csv`
coincidano byte per byte con le copie contenute in questo ZIP.

I cinque semi di uno stesso dataset condividono la medesima cattura. Le righe
per seme misurano la sensibilità alla selezione casuale delle etichette e non
sono repliche indipendenti del traffico.


# Figure

Generate da `scripts/figure.py`, che le ricalcola dai CSV in `results/`:
nessun numero e' scritto a mano, quindi una figura non puo' divergere dai
dati che dichiara di mostrare.

    python scripts/figure.py            # tutte
    python scripts/figure.py --solo 3   # una sola

PDF vettoriale per l'articolo, PNG a 300 dpi per il README.

| file | sezione | cosa mostra |
|---|---|---|
| `fig1_diagnosi` | 1 | ROC-AUC sul dominio di arrivo contro quello di partenza, sei modelli, due direzioni: l'ordinamento regge in una e cade al caso nell'altra |
| `fig2_recupero` | 11 | balanced accuracy contro budget di etichette, sei direzioni, con dev.std e seed riusciti; `unsw→bot` marcata come fallita |
| `fig3_transfer_invertito` | 11 | ROC-AUC per direzione attorno al caso: tre direzioni cadono sotto 0,5 |
| `fig4_selezione` | 4 | normali raccolte (meccanismo) accanto alla balanced accuracy (effetto), per regola di selezione e regime di rarita' |
| `fig5_coeff_vs_rifit` | 11 | delta appaiato per seed con intervallo di confidenza, correzione di Holm su 15 confronti |
| `fig6_costo` | 17c | calcolo e RAM per aggiornamento, con la soglia di SRAM dell'ATmega2560; la texture distingue misurato da proiettato |
| `fig7_collo_di_bottiglia` | 9 | la direzione che fallisce con ogni regola normale e i due selettori che la sbloccano, accanto alla prova che altrove non servono |
| `fig8_guardia` | 19 | la griglia direzione x rapporto del guadagno sul modello statico, con e senza la guardia sui batch a una classe sola (7 celle perdenti su 20 contro 0), e il costo in accuratezza dei tre livelli di memoria: nessuna, 728 byte, 12 KB |

## Scelte di resa, dichiarate

- **Palette verificata** per daltonismo (deuteranopia e tritanopia) e per
  contrasto sulla superficie chiara: ogni coppia adiacente supera la soglia
  di separazione e ogni colore supera 3:1 sul fondo. Le figure reggono
  stampate in bianco e nero e lette da chi non distingue rosso e verde.
- **Identita' mai affidata al solo colore**: ogni serie ha anche un
  marcatore proprio, ed e' etichettata direttamente dove entra. In
  `fig6` la distinzione fra misurato e proiettato porta anche una texture.
- **Un solo asse per pannello.** Due grandezze di scala diversa — MAC e
  RAM, normali raccolte e accuratezza — vanno in due pannelli affiancati,
  mai su due scale y sovrapposte.
- **Un confronto senza varianza non è un confronto non significativo.**
  Quando due metodi danno lo stesso identico risultato in ogni seed, il
  p-value non è definito: quel caso viene escluso dalla famiglia di Holm e
  dichiarato per quello che è, invece di essere disegnato come un test
  fallito. Trattarlo come un p qualsiasi non solo gonfia la famiglia — con
  `max(prec, nan)` propaga il NaN e può seppellire il confronto più
  significativo, che è esattamente ciò che è successo nella prima versione
  di `fig7`.
- **Le celle ereditate si dichiarano.** In `fig8` un terzo delle celle non
  e' una misura indipendente: viene per identita' dal rapporto 1:50, perche'
  a quel rapporto il sotto-campionamento non vincola quella sorgente.
  Disegnarle come le altre sarebbe una copertura sovradichiarata; portano un
  tratteggio e una nota.
- **Gli zeri non si disegnano su scala logaritmica.** Le regole che non
  raccolgono nulla sono dichiarate in nota, non appiattite a un valore
  piccolo che sembrerebbe un dato.
- **Griglia recessiva**, cornice aperta: le linee di riferimento non
  competono con i dati.

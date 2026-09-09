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
- **Gli zeri non si disegnano su scala logaritmica.** Le regole che non
  raccolgono nulla sono dichiarate in nota, non appiattite a un valore
  piccolo che sembrerebbe un dato.
- **Griglia recessiva**, cornice aperta: le linee di riferimento non
  competono con i dati.

# `drift_graduale.py` prima della guardia sui batch monoclasse

Copia dei CSV e dei checkpoint prodotti **prima** che `stat_13x13_guardia`
(sezione 19) fosse aggiunta come ottava politica. Non sono superati: sono
la misura di riferimento con cui il run nuovo va confrontato.

**Il confronto che serve**: aggiungere una politica non deve spostare le
altre sette. Nessuna politica consuma il generatore casuale condiviso
(`adaptive_pick` e `balanced_draw` ricevono `seed + k`), quindi le sette
colonne gia' pubblicate devono uscire **identiche**, cella per cella.
Verificato prima di lanciare, su `bot->ton` seed 42 ratio 50: 140 celle su
140 invariate, bit per bit. Il test
`tests/test_guardia_monoclasse.py::test_le_sette_politiche_non_si_muovono`
lo ricontrolla su tutto quello che c'e', e fallisce se una si muove.

Se un giorno queste copie divergono dai file in `results/`, non e' la
guardia: e' qualcos'altro che si e' mosso nella catena, ed e' esattamente
cio' che questo confronto serve a intercettare.

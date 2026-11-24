## Domanda

Verifica se la testa GAM candidata completa rileva più attacchi entro lo
stesso tetto di falsi allarmi e se il vantaggio si conserva con soglie
fissate al FREEZE. Si usano i report originali ADAPT-06E e la riconciliazione
ADAPT-06F, con le traiettorie storiche fisse.

- Protocollo fissato prima dei risultati: `9c9acd64ad5a0cca2bc98a90de1aae873bac8e3d`.
- Codice che ha prodotto la consegna: `cdc7deff2ff8f4f271569eac77dc95b9e7ae1fb0`.
- Base: `00f8fb729caeec845ad2052ed59b0f6c3acc64df`, merge di ADAPT-06G.
- 38 cicli conservati, 34 confronti osservabili e quattro cicli a fine stream.
  Il controllo sulle predizioni storiche passa per tutti e tre gli `ACCEPT`.

## Risultati

Sul blocco successivo il candidato guadagna TP al tetto retrospettivo in
5/16 coppie seme/blocco di TON e in 6/18 di UNSW. Usando i soli normali CAL
disponibili al FREEZE, il vantaggio contro entrambi i controlli si conserva
in quattro di quei casi per stream. Sui blocchi osservati emergono limiti
sia nell’ordinamento sia nel trasferimento della soglia.

Per TON le quattro coppie favorevoli con soglie CAL riguardano il solo blocco
353. Nessun seme soddisfa il criterio sui totali appaiati. Su UNSW passa
soltanto il seme 46, con +53 TP e −1 FP contro ciascuno dei due controlli
sui tre blocchi osservati e un unico episodio qualificante (blocco 89,
indice da zero). Nel seme UNSW 45 il guadagno retrospettivo complessivo è
+72 TP. Con soglie fissate al FREEZE il confronto contro entrambi i controlli
risulta invece −103 TP e +1.560 FP.

Il criterio preregistrato per proporre un replay causale completo richiede
almeno tre semi favorevoli nello stesso stream e due episodi distinti tra
quei semi. La decisione è di fermarsi a questa diagnosi. I semi condividono
le catture e la conclusione riguarda i blocchi selezionati dal gate storico.

## Riproduzione

`COMMANDS.md` documenta come ricalcolare `delivery/CYCLES.csv`,
`delivery/SEEDS.csv`, `delivery/EPISODES.csv`, `delivery/DECISION.json` e
`delivery/NOTE.md`. `delivery/MANIFEST.json` identifica sorgenti, ambiente
e SHA-256. L’analisi usa inferenza sui pesi archiviati e le etichette già
disponibili. ADAPT-07 rimane in backlog.

"""La larghezza e il grado della KAN si scelgono su validation, non sul test.

La revisione aveva corretto il rapporto del joint training. Ricontrollando il
resto e' venuto fuori che il rapporto era il caso piu' visibile, non l'unico:
`hidden = 16` veniva da un confronto fra 0,9784 e 0,9778, e `degree = 8` da
uno fra 0,9409 e 0,9374 — tutti F1 misurati sul 20% held-out, cioe' sullo
stesso insieme poi riportato come risultato.

Questi test tengono in piedi tre cose:

1. la selezione non guarda il test (stessa prova del rapporto: si rendono
   assurdi i valori delle righe destinate al test e si pretende che il
   punteggio di validation non si muova di un bit);
2. la regola di scelta e' quella dichiarata PRIMA di vedere i numeri, e non
   una riscritta dopo perche' il risultato non piaceva;
3. i valori che la pipeline usa davvero coincidono con quelli dell'artefatto
   di selezione, quando l'artefatto esiste. Senza questo, i risultati
   committati potrebbero essere di un'architettura e il codice di un'altra.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SCRIPT = REPO / "scripts" / "select_architettura.py"


def _modulo():
    spec = importlib.util.spec_from_file_location("select_architettura", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lo_script_esiste_e_si_importa():
    assert SCRIPT.exists(), "manca scripts/select_architettura.py"
    _modulo()


def test_il_reticolo_contiene_i_valori_attuali():
    """Se 16 e 8 non fossero fra i candidati, la selezione non potrebbe
    confermarli: sembrerebbe una scelta e sarebbe un cambio forzato."""
    m = _modulo()
    assert 16 in m.HIDDEN_CANDIDATI, "hidden=16 non e' fra i candidati"
    assert 8 in m.DEGREE_CANDIDATI, "degree=8 non e' fra i candidati"


def test_la_regola_e_quella_dichiarata():
    """La regola sta scritta in un posto solo e qui se ne fissa il contenuto.

    Non e' pedanteria: una regola di selezione riscritta dopo aver visto i
    risultati e' esattamente il difetto che tutta questa revisione corregge,
    e sarebbe invisibile in un diff distratto. Se cambia, questo test cade e
    qualcuno deve dire perche'.
    """
    m = _modulo()
    assert "1-SE" in m.REGOLA
    assert "piu' piccola" in m.REGOLA
    sorgente = SCRIPT.read_text(encoding="utf-8")
    assert 'soglia = float(best["media"] - best["errore_standard"])' in sorgente, (
        "la soglia non e' piu' 'migliore meno un errore standard'")
    assert 'ammesse.loc[ammesse["parametri"].idxmin()]' in sorgente, (
        "fra le ammesse non si sceglie piu' la piu' piccola")


def test_la_selezione_non_legge_il_test():
    """La prova vera, costruita come quella del rapporto in
    tests/test_joint_training.py: due dataset identici tranne che per i VALORI
    delle righe destinate al test, resi assurdi. Se il punteggio di validation
    cambiasse anche di un bit, vorrebbe dire che qualcosa — la selezione delle
    feature, i quantili, il fit — ha guardato il test.

    Si perturbano i valori e non le etichette di proposito: `outer_split`
    stratifica su y, quindi capovolgere le etichette del test cambierebbe lo
    split stesso e il test misurerebbe sé stesso invece del leakage. (Prima
    versione di questo test: capovolgeva le etichette, falliva, e la
    differenza era tutta dello split.)
    """
    m = _modulo()
    from kanids.datasets import encode_targets, make_synthetic
    from kanids.splits import outer_split

    df = make_synthetic(4000, seed=0)
    y, _ym, _c = encode_targets(df)
    _tr, te = outer_split(y, seed=42)

    df_assurdo = df.copy()
    numeriche = [c for c in df.columns
                 if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    rng = np.random.RandomState(7)
    for c in numeriche:
        # cast a float prima di scrivere: molte colonne sono int64 e pandas
        # rifiuta di infilarci valori non interi
        df_assurdo[c] = df_assurdo[c].astype("float64")
        df_assurdo.loc[df_assurdo.index[te], c] = (
            df_assurdo.loc[df_assurdo.index[te], c].to_numpy() * -3.0
            + rng.uniform(5000, 9000, size=len(te)))

    a = m.una_run(df, y, 42, "KAN(cat,1L)", 0, 4, k=6)
    b = m.una_run(df_assurdo, y, 42, "KAN(cat,1L)", 0, 4, k=6)
    assert a[m.METRICA] == b[m.METRICA], (
        f"il punteggio di validation cambia se cambio i VALORI del test "
        f"({a[m.METRICA]} contro {b[m.METRICA]}): il test entra nella selezione")
    assert a["n_fit"] == b["n_fit"] and a["n_val"] == b["n_val"]


def test_validation_e_test_sono_disgiunti():
    """Il vincolo strutturale sotto al test precedente: la validation e'
    ritagliata dentro il training, quindi non puo' intersecare il test."""
    m = _modulo()
    from kanids.datasets import encode_targets, make_synthetic
    from kanids.splits import outer_split

    df = make_synthetic(3000, seed=0)
    y, _ym, _c = encode_targets(df)
    tr, te = outer_split(y, seed=42)
    pos_fit, pos_val = m.inner_split(y[tr], seed=42)
    fit, val = tr[pos_fit], tr[pos_val]
    assert not (set(val) & set(te)), "la validation tocca il test set"
    assert not (set(fit) & set(te)), "il fit tocca il test set"
    assert not (set(fit) & set(val)), "fit e validation si sovrappongono"
    assert set(fit) | set(val) == set(tr), "fit + validation != training"


def test_servono_almeno_tre_seed():
    """Con uno o due seed l'errore standard non significa niente e la regola
    1-SE degenera. Meglio fermarsi che scegliere per un motivo che nessuno
    saprebbe ricostruire."""
    import pandas as pd
    m = _modulo()
    finto = pd.DataFrame([
        {"seed": 42, "model": "KAN(cat,ML)", "hidden": h, "degree": 8,
         m.METRICA: 0.9, "n_parametri": h * 100}
        for h in (8, 16)])
    with pytest.raises(SystemExit, match="almeno 3 seed"):
        m.scegli(finto, "KAN(cat,ML)")


def test_la_regola_1se_preferisce_la_piu_piccola_a_parita_pratica():
    """Il comportamento che la regola promette, su numeri costruiti: una
    configurazione grande che vince di pochissimo NON deve vincere."""
    import pandas as pd
    m = _modulo()
    righe = []
    for i, seed in enumerate((42, 43, 44, 45, 46)):
        rumore = [0.002, -0.001, 0.0015, -0.002, 0.0005][i]
        righe.append({"seed": seed, "model": "KAN(cat,ML)", "hidden": 8,
                      "degree": 8, m.METRICA: 0.970 + rumore, "n_parametri": 800})
        righe.append({"seed": seed, "model": "KAN(cat,ML)", "hidden": 32,
                      "degree": 8, m.METRICA: 0.9706 + rumore, "n_parametri": 3200})
    s = m.scegli(pd.DataFrame(righe), "KAN(cat,ML)")
    assert s["hidden"] == 8, (
        f"h=32 vince di 0,0006 (rumore) e la regola l'ha scelto: {s}")
    assert s["migliore_assoluta"]["hidden"] == 32, (
        "l'artefatto deve comunque REGISTRARE quale aveva la media piu' alta")


def test_la_pipeline_legge_larchitettura_invece_di_riscriverla():
    """Nessun runner deve avere 16 o 8 scritti a mano: e' cosi' che una
    scelta diventa una costante che nessuno sa piu' da dove viene."""
    for nome in ("cv_leakagefree.py", "cross_domain.py", "joint_training.py"):
        testo = (REPO / "scripts" / nome).read_text(encoding="utf-8")
        codice = "\n".join(r for r in testo.splitlines()
                           if not r.lstrip().startswith("#"))
        assert "hidden=16" not in codice, f"{nome}: hidden=16 scritto a mano"
        assert "degree=8" not in codice, f"{nome}: degree=8 scritto a mano"
        assert "HIDDEN" in codice or "DEGREE" in codice, (
            f"{nome} non legge l'architettura da kanids/config.py")


def test_lo_scarto_dalla_selezione_e_dichiarato(tmp_path):
    """Il test piu' importante di questo file.

    La selezione su validation sceglie h=32 grado=6; il progetto deploya
    h=16 grado=8, ereditata dalla fase 1. Le due NON coincidono, e va bene:
    la seconda e' una scelta di dimensione su microcontrollore, non un
    risultato della selezione. Cio' che non va bene e' che la differenza
    resti dentro un JSON che nessuno apre.

    Quindi: se l'artefatto di selezione esiste e differisce da cio' che si
    deploya, il README deve dirlo, con i due numeri. Se un giorno qualcuno
    adottasse la selezione, la condizione decade da sola.
    """
    from kanids import ARCH, ARCH_ORIGINE, scarto_dalla_selezione
    scarto = scarto_dalla_selezione()
    if scarto is None:
        pytest.skip("selezione non ancora eseguita")

    diversi = {m: s for m, s in scarto.items() if not s["coincidono"]}
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    if not diversi:
        return                      # coincidono: niente da dichiarare

    assert ARCH_ORIGINE == "ereditata", (
        "ARCH_ORIGINE dice 'validation' ma cio' che si deploya non e' cio' "
        "che la selezione ha scelto")
    assert "selected and deployed are not the same" in readme, (
        "il README non dichiara che architettura selezionata e architettura "
        "deployata sono diverse. I numeri sono in "
        "results/arch_selection_scelta.json e nessuno li leggera' li'.")
    # Cercare "32" e "16" nel README non proverebbe niente: sono numeri che
    # compaiono ovunque, e un'asserzione che non puo' fallire e' peggio che
    # nessuna asserzione (verificato: rimuovendo la tabella, passava lo
    # stesso). Si cerca invece il punteggio di validation della
    # configurazione scelta, che nel README compare solo li'.
    # La prima versione controllava solo la media della configurazione
    # scelta, e ha fatto il suo mestiere: ha bocciato uno 0,99632 ricopiato a
    # mano da un output arrotondato invece che letto dall'artefatto (il vero
    # valore era 0,99631). Ma nella stessa sezione erano sbagliati anche la
    # media della deployata, la soglia e lo scarto: un test che ne guarda uno
    # su quattro lascia passare tre errori dello stesso tipo. Adesso li
    # verifica tutti, e ognuno dalla sua fonte.
    import json as _json
    import pandas as _pd
    grezzo = _json.loads(
        (REPO / "results" / "arch_selection_scelta.json").read_text(encoding="utf-8"))
    tabella = _pd.read_csv(REPO / "results" / "arch_selection.csv")

    mancanti = []
    for m in diversi:
        s = grezzo["scelte"][m]
        attesi = {"media della configurazione scelta": s.get("media_validation"),
                  "soglia 1-SE": s.get("soglia_1se")}
        dep = diversi[m]["deployata"]
        riga = tabella[(tabella.model == m) & (tabella.hidden == dep["hidden"])
                       & (tabella.degree == dep["degree"])]
        if len(riga):
            attesi["media della configurazione deployata"] = float(
                riga["balanced_accuracy_mean"].iloc[0])
        for che_cosa, valore in attesi.items():
            if valore is not None and f"{valore:.5f}" not in readme:
                mancanti.append(f"{m}: {che_cosa} = {valore:.5f}")
    assert not mancanti, (
        "il README non riporta questi numeri della selezione:\n  "
        + "\n  ".join(mancanti)
        + "\n\nVanno letti dagli artefatti, non ricalcolati dall'output "
          "stampato: quello e' arrotondato a quattro decimali.")


def test_i_valori_usati_sono_quelli_ereditati_finche_non_si_adotta():
    """La pipeline deve usare l'architettura deployata, non quella scelta:
    altrimenti rigenerare un qualunque stage produrrebbe numeri di
    un'architettura diversa da quella dei risultati committati, in silenzio."""
    from kanids import ARCH
    from kanids.config import ARCH_EREDITATA
    for m, v in ARCH_EREDITATA.items():
        assert ARCH[m]["hidden"] == v["hidden"] and ARCH[m]["degree"] == v["degree"], (
            f"{m}: la pipeline non usa piu' l'architettura ereditata. Se e' "
            f"voluto, vanno rigenerati CV, cross-domain, joint, header C, "
            f"golden vector, footprint, figure, report e firmware.")


def test_larchitettura_ereditata_e_dichiarata_come_tale():
    """Finche' la selezione non gira, i valori sono quelli della fase 1 e il
    codice deve dirlo — non spacciarli per scelti."""
    testo = (REPO / "kanids" / "config.py").read_text(encoding="utf-8")
    assert "ARCH_EREDITATA" in testo
    assert "held-out" in testo, (
        "config.py non dice piu' da dove vengono i valori ereditati")


def test_anche_config_py_riporta_le_cifre_dell_artefatto():
    """Lo stesso controllo del README, sul file che DEFINISCE l'architettura.

    Il test sul README esiste perche' quattro cifre della sezione erano
    ricalcoli fatti su un output gia' arrotondato. Le stesse quattro erano
    anche nel commento di `kanids/config.py`, che nessuno guardava, e ci sono
    rimaste dopo la correzione: 0,99632 invece di 0,99631, 0,99600 invece di
    0,99602, uno scarto di 0,00032 invece di 0,00028, una soglia mancata per
    0,00020 invece di 0,00015 e p = 0,067 invece di 0,083.

    E' il posto peggiore dove lasciarle: quel commento e' la giustificazione
    di ARCH_EREDITATA, cioe' della configurazione che tutta la pipeline usa.
    """
    from kanids.config import scarto_dalla_selezione
    scarto = scarto_dalla_selezione()
    if scarto is None:
        pytest.skip("selezione non ancora eseguita")
    diversi = {m: s for m, s in scarto.items() if not s["coincidono"]}
    if not diversi:
        return

    import json as _json
    import pandas as _pd
    grezzo = _json.loads(
        (REPO / "results" / "arch_selection_scelta.json").read_text(encoding="utf-8"))
    tabella = _pd.read_csv(REPO / "results" / "arch_selection.csv")
    testo = (REPO / "kanids" / "config.py").read_text(encoding="utf-8")

    # solo il blocco di commento che giustifica ARCH_EREDITATA, non l'intero
    # file: altrove le cifre possono comparire per altre ragioni
    blocco = testo.split("ARCH_EREDITATA", 1)[0] + \
        testo.split("ARCH_EREDITATA", 1)[1].split("ARCH =", 1)[0]

    def in_italiano(v: float) -> str:
        return f"{v:.5f}".replace(".", ",")

    mancanti = []
    for m, s in diversi.items():
        g = grezzo["scelte"][m]
        dep = s["deployata"]
        riga = tabella[(tabella.model == m) & (tabella.hidden == dep["hidden"])
                       & (tabella.degree == dep["degree"])]
        attesi = {"media della scelta": g.get("media_validation"),
                  "soglia 1-SE": g.get("soglia_1se")}
        if len(riga):
            attesi["media della deployata"] = float(
                riga["balanced_accuracy_mean"].iloc[0])
        for confronto in g.get("confronti_appaiati", []):
            if (confronto["hidden"], confronto["degree"]) == (dep["hidden"],
                                                              dep["degree"]):
                attesi["scarto dalla scelta"] = confronto["delta"]
        for che_cosa, valore in attesi.items():
            if valore is None:
                continue
            if in_italiano(valore) not in blocco and \
                    f"{valore:.5f}" not in blocco:
                mancanti.append(f"{m}: {che_cosa} = {in_italiano(valore)}")
    assert not mancanti, (
        "il commento di kanids/config.py non riporta questi numeri della "
        "selezione:\n  " + "\n  ".join(mancanti)
        + "\n\nSono la giustificazione di ARCH_EREDITATA: se sono sbagliati "
          "li' e' giusti nel README, il repository si contraddice nel posto "
          "che conta di piu'.")

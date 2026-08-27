"""La statistica dei confronti: unita' di analisi, correzioni, formattazione.

Richiesta del Prof. Kuznetsov, punto 4: "evitare p-value estremi ottenuti
trattando fold ripetuti come osservazioni indipendenti. Per l'articolo bastano
media +/- std, differenze, conteggi di vittorie e, dove appropriato, confronti
corretti."

Tre difetti concreti, e un test per ciascuno
============================================
1. `round(p, 4)` scriveva `p_value = 0.0` sei volte su trenta in
   `crossdomain_significativita.csv`. Un p-value nullo non esiste.
2. La selezione del rapporto elencava 120 "coppie" appaiate: 10 seed x 6
   modelli x 2 domini. Modelli e domini non sono repliche, e il criterio
   dichiarato e' gia' la media su di essi.
3. Nei run cross-domain training e test sono due interi domini FISSI. Il t
   fra seed misura la variabilita' del riaddestramento, ed e' per questo che
   arriva a -58: la dispersione al denominatore e' piccola per costruzione.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kanids.statistica import (SOGLIA_P, confronto_appaiato,     # noqa: E402
                               formatta_p, holm, riassunto,
                               unita_di_analisi)

RESULTS = REPO / "results"


# ─────────────────────────────────────────────────────────────
# 1. un p-value non diventa mai zero
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("p", [9.61e-08, 1e-13, 2.6e-07, 5e-324, 0.0])
def test_un_p_value_non_si_scrive_mai_come_zero(p):
    s = formatta_p(p)
    assert s not in ("0.0", "0.0000", "0"), f"{p} formattato come {s!r}"
    assert float(s.lstrip("<")) > 0


def test_sotto_la_soglia_si_scrive_una_disuguaglianza():
    assert formatta_p(1e-20).startswith("<")
    assert not formatta_p(0.03).startswith("<")
    assert formatta_p(0.0301) == "0.0301"


def test_il_difetto_originale_e_ancora_riproducibile():
    """Controllo del controllo: la formattazione vecchia deve ancora produrre
    lo zero, altrimenti questo test non sta impedendo niente."""
    assert round(9.61e-08, 4) == 0.0
    assert formatta_p(9.61e-08) == "9.61e-08"


def test_un_p_assente_si_dichiara():
    assert formatta_p(None) == "n/d"
    assert formatta_p(float("nan")) == "n/d"


# ─────────────────────────────────────────────────────────────
# 2. l'unita' di analisi
# ─────────────────────────────────────────────────────────────
def test_le_serie_devono_essere_appaiate():
    with pytest.raises(ValueError):
        confronto_appaiato([1.0, 2.0, 3.0], [1.0, 2.0])


def test_aggregare_prima_riduce_le_osservazioni_a_quelle_vere():
    """10 seed x 6 modelli x 2 domini danno 10 osservazioni, non 120."""
    rs = np.random.RandomState(0)
    righe = [{"seed": s, "model": m, "dst": d,
              "balanced_accuracy": rs.rand()}
             for s in range(10) for m in range(6) for d in ("ton", "bot")]
    ridotto = unita_di_analisi(righe, ["seed"], ["model", "dst"],
                               "balanced_accuracy")
    assert len(ridotto) == 10, "l'aggregazione non ha ridotto le osservazioni"
    assert set(ridotto.columns) == {"seed", "balanced_accuracy"}


def test_gonfiare_le_osservazioni_gonfia_il_t():
    """La dimostrazione numerica del difetto: replicare dodici volte la stessa
    informazione non aggiunge nulla ma moltiplica il t per sqrt(12).

    E' il motivo per cui 120 "coppie" davano p = 9,6e-08 dove i dieci seed
    veri ne danno 2,7e-07 o piu'.
    """
    rs = np.random.RandomState(1)
    per_seed = rs.normal(0.003, 0.002, 10)
    onesto = confronto_appaiato(per_seed, np.zeros(10), valutazione_fissa=True)
    gonfio = confronto_appaiato(np.repeat(per_seed, 12), np.zeros(120),
                                valutazione_fissa=True)
    assert gonfio["n"] == 12 * onesto["n"]
    assert abs(gonfio["t"]) > 3 * abs(onesto["t"]), (
        "replicare le osservazioni non gonfia il t: il test non sta "
        "misurando il difetto che dice di misurare")
    assert gonfio["p"] < onesto["p"]


# ─────────────────────────────────────────────────────────────
# 3. la correzione si applica solo dove il suo regime esiste
# ─────────────────────────────────────────────────────────────
def test_con_valutazione_fissa_non_si_corregge_e_si_dice_perche():
    c = confronto_appaiato(np.arange(10) * 0.01, np.zeros(10),
                           n_train=1000, n_test=100, valutazione_fissa=True)
    assert c["t_corretto"] is None and c["p_corretto"] is None
    assert "valutazione fisso" in c["correzione"]


def test_nadeau_bengio_si_applica_a_una_kfold_e_abbassa_il_t():
    """rho = n_test/n_train = 0,25 e' esattamente una 5-fold: e' il regime
    per cui la correzione e' stata derivata."""
    rs = np.random.RandomState(2)
    a = rs.normal(0.9, 0.01, 50)
    b = a - 0.02 + rs.normal(0, 0.002, 50)
    c = confronto_appaiato(a, b, n_train=168834, n_test=42209)
    assert c["t_corretto"] is not None, c["correzione"]
    assert abs(c["t_corretto"]) < abs(c["t"]), "la correzione non e' conservativa"
    assert c["p_corretto"] > c["p"]
    assert "Nadeau-Bengio" in c["correzione"]


def test_fuori_da_quel_regime_la_correzione_non_si_applica():
    """Con un training di 19.431 righe estratto da un pool di 733.000, rho
    vale 37,7 e le ripetizioni quasi non si sovrappongono: applicare la
    formula darebbe p = 0,72 su tutto, cioe' un numero prudente per la
    ragione sbagliata."""
    rs = np.random.RandomState(3)
    a = rs.normal(0.99, 0.002, 50)
    b = a - 0.019 + rs.normal(0, 0.001, 50)
    c = confronto_appaiato(a, b, n_train=19431, n_test=733704)
    assert c["t_corretto"] is None
    assert "37" in c["correzione"] and "sovrappongono" in c["correzione"]


def test_senza_le_dimensioni_degli_split_non_si_inventa_una_correzione():
    c = confronto_appaiato(np.arange(10) * 0.01, np.zeros(10))
    assert c["t_corretto"] is None
    assert "non note" in c["correzione"]


def test_pochi_dati_o_differenza_costante_non_producono_un_t():
    assert confronto_appaiato([1.0, 2.0], [0.0, 1.0])["t"] is None
    c = confronto_appaiato([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert c["t"] is None and "costante" in c["correzione"]


# ─────────────────────────────────────────────────────────────
# 4. Holm
# ─────────────────────────────────────────────────────────────
def test_holm_su_un_esempio_a_mano():
    assert holm([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.06, 0.06, 0.06])


def test_holm_e_monotono_e_non_supera_uno():
    rs = np.random.RandomState(4)
    p = sorted(rs.rand(20))
    agg = holm(p)
    assert all(x <= 1.0 for x in agg)
    assert agg == sorted(agg), "Holm deve restare monotono sull'ordine dei p"
    assert all(a >= b for a, b in zip(agg, p)), "un p corretto non puo' calare"


def test_holm_ignora_i_confronti_senza_p():
    agg = holm([0.01, None, 0.02])
    assert agg[1] is None
    assert agg[0] == pytest.approx(0.02)      # famiglia di due, non di tre


# ─────────────────────────────────────────────────────────────
# 5. gli artefatti prodotti
# ─────────────────────────────────────────────────────────────
DESCRITTIVE = ["n_unita", "media_a", "dev_a", "media_b", "dev_b",
               "differenza", "dev_differenza", "vince_a", "p_formattato",
               "p_holm", "valutazione_fissa", "correzione"]


@pytest.mark.parametrize("nome", ["crossdomain_significativita.csv",
                                  "joint_ratio_significativita.csv"])
def test_i_csv_dei_confronti_portano_le_quantita_richieste(nome):
    f = RESULTS / nome
    if not f.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f)
    mancanti = [c for c in DESCRITTIVE if c not in d.columns]
    assert not mancanti, (
        f"{nome} non riporta {mancanti}: il relatore ha chiesto media +/- std, "
        f"differenze e conteggi di vittorie, non il solo p")


@pytest.mark.parametrize("nome", ["crossdomain_significativita.csv",
                                  "joint_ratio_significativita.csv",
                                  "indomain_significativita.csv"])
def test_nessun_p_value_e_zero_negli_artefatti(nome):
    f = RESULTS / nome
    if not f.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f)
    zeri = d[d.p_value == 0]
    assert zeri.empty, (
        f"{nome} ha {len(zeri)} p-value esattamente 0: e' un artefatto della "
        f"formattazione, non un risultato")
    assert (d.p_formattato.astype(str).str.strip() != "0.0").all()


def test_la_selezione_del_rapporto_usa_il_seed_come_unita():
    f = RESULTS / "joint_ratio_significativita.csv"
    scelta = RESULTS / "joint_ratio_selection_scelta.json"
    if not f.exists() or not scelta.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    import json
    d = pd.read_csv(f)
    g = json.loads(scelta.read_text(encoding="utf-8"))
    n_seed = len(g["seeds"])
    assert (d.unita == "seed").all(), "l'unita' di analisi non e' il seed"
    assert (d.n_unita == n_seed).all(), (
        f"il confronto usa {sorted(set(d.n_unita))} osservazioni invece dei "
        f"{n_seed} seed: modelli e domini sono stati rimessi in lista")
    assert (d.n_unita < 100).all(), "sono tornate le 120 coppie"


def test_il_json_della_scelta_concorda_con_il_csv():
    """Due artefatti che dicono la stessa cosa devono dirla uguale: e' la
    regola che il progetto si e' dato dopo aver trovato tre copie diverse
    della formula dei byte."""
    import json
    f = RESULTS / "joint_ratio_significativita.csv"
    scelta = RESULTS / "joint_ratio_selection_scelta.json"
    if not f.exists() or not scelta.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f)
    g = json.loads(scelta.read_text(encoding="utf-8"))
    conf = g.get("confronti_appaiati", [])
    assert len(conf) == len(d), (
        f"il JSON elenca {len(conf)} confronti, il CSV {len(d)}")
    for c in conf:
        riga = d[d.modello_b == f"1:{c['contro']:g}"]
        assert len(riga) == 1, c
        assert riga.iloc[0].n_unita == c["n"]
        assert riga.iloc[0].vince_a == c["vince_in"]
        assert float(riga.iloc[0].differenza) == pytest.approx(
            c["differenza_media"], abs=1e-9)


def test_il_cross_domain_dichiara_che_la_valutazione_e_fissa():
    """Nei run di ton->bot il training e il test sono due interi domini, gli
    stessi per tutti i seed. Il CSV deve dirlo, altrimenti quel t si legge
    come una misura di generalizzazione."""
    f = RESULTS / "crossdomain_significativita.csv"
    if not f.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f)
    assert d.valutazione_fissa.all(), (
        "il CSV non dichiara che l'insieme di valutazione e' fisso")
    assert d.correzione.str.contains("riaddestramento").all(), (
        "il CSV non dice che cosa misura davvero quel test")


def test_nessuna_decisione_del_progetto_dipende_da_un_p_value():
    """Il rapporto e' scelto sulla media, l'architettura con la regola 1-SE.
    Se un giorno un p entrasse in un criterio, questo test non se ne
    accorgerebbe da solo — ma i due file che i criteri li scrivono devono
    continuare a dichiararli senza p."""
    import json
    scelta = RESULTS / "joint_ratio_selection_scelta.json"
    arch = RESULTS / "arch_selection_scelta.json"
    if scelta.exists():
        g = json.loads(scelta.read_text(encoding="utf-8"))
        assert "media" in g["criterio"] and "p" not in g["criterio"].split()
    if arch.exists():
        a = json.loads(arch.read_text(encoding="utf-8"))
        assert "1-SE" in a["regola"] and "p-value" not in a["regola"]


# ─────────────────────────────────────────────────────────────
# 6. il README riporta questi numeri, non altri
# ─────────────────────────────────────────────────────────────
def _sezione(marcatore: str, quanto: int = 4000) -> str:
    testo = (REPO / "README.md").read_text(encoding="utf-8")
    i = testo.index(marcatore)
    return testo[i:i + quanto]


def test_la_tabella_del_rapporto_nel_readme_viene_dallartefatto():
    """Quattro differenze, quattro conteggi di vittorie e le medie per
    candidato: ogni cifra deve stare nel CSV. La tabella e' scritta a mano ed
    e' la prima che un revisore guarda."""
    f = RESULTS / "joint_ratio_significativita.csv"
    if not f.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f)
    sez = _sezione("| ratio | bal. acc. on validation")

    mancanti = []
    for _i, r in d.iterrows():
        for che_cosa, testo in (
                ("media del candidato", f"{r.media_b:.5f}"),
                ("dispersione del candidato", f"{r.dev_b:.5f}"),
                ("differenza", f"{abs(r.differenza):.5f}"),
                ("vittorie", str(r.vince_a))):
            if testo not in sez:
                mancanti.append(f"{r.modello_b}: {che_cosa} = {testo}")
    assert not mancanti, (
        "la tabella del rapporto nel README non riporta:\n  "
        + "\n  ".join(mancanti))
    assert "120" in sez, (
        "il README non dice piu' da cosa si viene: senza nominare le 120 "
        "coppie, la correzione diventa invisibile")


def test_la_tabella_cross_domain_nel_readme_viene_dallartefatto():
    f = RESULTS / "crossdomain_significativita.csv"
    if not f.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f)
    tb = d[d.exp == "ton->bot"]
    kan = "KAN(cat,1L)"
    sez = _sezione("**What replaces the ranking claim")

    mancanti = []
    for _i, r in tb.iterrows():
        if kan not in (r.modello_a, r.modello_b):
            continue
        delta = r.differenza if r.modello_a == kan else -r.differenza
        if f"{abs(delta):.4f}" not in sez:
            altro = r.modello_b if r.modello_a == kan else r.modello_a
            mancanti.append(f"differenza contro {altro} = {abs(delta):.4f}")
    assert not mancanti, (
        "la tabella cross-domain del README non riporta:\n  "
        + "\n  ".join(mancanti))
    # e le taglie che spiegano che cosa misura quel test
    for n in ("211,043", "3,668,522"):
        assert n in sez, (
            f"il README non dice che {n} e' fisso su tutti i seed: senza, il "
            f"t si legge come una misura di generalizzazione")


def test_il_readme_non_promette_piu_ripetibilita_dove_non_ce():
    """L'affermazione "higher ratios are less repeatable" veniva dalla
    dispersione fra tutte le 120 misure, che mescola variabilita' fra seed e
    disaccordo fra modelli. Fra i seed la dispersione non cresce."""
    f = RESULTS / "joint_ratio_significativita.csv"
    if not f.exists():
        pytest.skip("python scripts/statistica_confronti.py")
    d = pd.read_csv(f).sort_values("modello_b")
    dev = d.dev_b.tolist()
    cresce = all(x <= y for x, y in zip(dev, dev[1:]))
    sez = _sezione("| ratio | bal. acc. on validation")
    if not cresce:
        assert "not monotone" in sez or "retracted" in sez, (
            "la dispersione fra seed non cresce col rapporto, ma il README "
            "non ritira l'affermazione che diceva il contrario")

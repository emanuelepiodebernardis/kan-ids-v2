"""Test del blocco A (joint training TON_IoT + BoT-IoT).

Due vincoli, entrambi verificati per costruzione e non solo dichiarati:

1. `balance_joint` produce contributi di pari dimensione e pari rapporto
   normal/attack, qualunque sia lo squilibrio naturale fra i due domini.
2. I due test set (TON_test, BoT_test) non entrano MAI in feature
   selection, normalizzazione, vocabolari o training: fittare due volte lo
   stesso training congiunto con test radicalmente diversi deve produrre un
   appreso identico bit per bit. Stesso schema di
   tests/test_leakage.py::test_crossdomain_target_does_not_influence_training,
   applicato al joint training invece che al cross-domain.
3. Nemmeno la SCELTA DEL RAPPORTO vede i test set: la validation e'
   ritagliata dentro il training, e cambiare completamente le righe che
   finiranno nel test non sposta di un indice ne' la validation ne' i
   training bilanciati dei cinque rapporti candidati.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from kanids.harmonized import HARMONIZED_CATEGORICAL, HARMONIZED_NUMERIC  # noqa: E402
from kanids.splits import outer_split  # noqa: E402

from joint_training import (RATIOS_CANDIDATI, balance_joint, fit_eval,  # noqa: E402
                            inner_split)


def _harmonized_synth(n, seed, shift=0.0):
    """Frame sintetico nello schema armonizzato, con label sbilanciata come
    un dominio reale (attacco = classe maggioritaria)."""
    rng = np.random.RandomState(seed)
    d = {c: np.abs(rng.randn(n) * 10 + 20 + shift) for c in HARMONIZED_NUMERIC}
    df = pd.DataFrame(d)
    df["proto_h"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["state_h"] = rng.choice(["established", "closed", "incomplete", "reset"], n)
    for c in HARMONIZED_CATEGORICAL:
        if c not in df:
            df[c] = "other"
    df["label"] = rng.choice([0, 1], n, p=[0.2, 0.8])
    return df


# ── 1. balance_joint: pari dimensione, pari rapporto ──────────────────
def test_balance_joint_equal_size_and_ratio():
    """Due domini con squilibrio naturale MOLTO diverso (come TON_IoT e
    BoT-IoT reali) devono ricevere lo stesso numero di normali, lo stesso
    numero di attacchi, quindi la stessa dimensione totale."""
    seed = 7
    H = {
        "ton": _harmonized_synth(6000, seed=1),                       # normal/attack "normale"
        "bot": pd.concat([                                            # normali scarsissimi
            _harmonized_synth(60, seed=2).assign(label=0),
            _harmonized_synth(6000, seed=3).assign(label=1),
        ], ignore_index=True),
    }
    split_train = {}
    for d in ("ton", "bot"):
        y = H[d]["label"].to_numpy()
        tr, _ = outer_split(y, seed=seed)
        split_train[d] = tr

    ratio = 20.0
    balanced, info = balance_joint(H, split_train, ratio, seed)

    n_ton = len(balanced["ton"])
    n_bot = len(balanced["bot"])
    assert n_ton == n_bot, f"dimensioni diverse: ton={n_ton} bot={n_bot}"

    for d in ("ton", "bot"):
        y = H[d]["label"].to_numpy()[balanced[d]]
        n_normal = int((y == 0).sum())
        n_attack = int((y == 1).sum())
        assert n_normal == info["n_normal"]
        assert n_attack == info["n_attack"]
    # il tetto e' il dominio con meno normali (bot, 60 nel pool grezzo)
    assert info["n_normal"] <= 60
    # il rapporto e' rispettato (a meno del cap sugli attacchi disponibili,
    # qui non vincolante)
    assert info["n_attack"] == int(ratio * info["n_normal"])


def test_balance_joint_indices_stay_inside_the_train_split():
    """Gli indici bilanciati sono un sottoinsieme del train split passato:
    balance_joint non deve mai poter "recuperare" righe dal test."""
    seed = 3
    H = {"ton": _harmonized_synth(2000, seed=1), "bot": _harmonized_synth(500, seed=2)}
    split_train, split_test = {}, {}
    for d in ("ton", "bot"):
        y = H[d]["label"].to_numpy()
        tr, te = outer_split(y, seed=seed)
        split_train[d], split_test[d] = tr, te

    balanced, _ = balance_joint(H, split_train, 50.0, seed)
    for d in ("ton", "bot"):
        assert set(balanced[d]).issubset(set(split_train[d]))
        assert not (set(balanced[d]) & set(split_test[d]))


# ── 2. i due test set non influenzano il training congiunto ───────────
def _fit_on_joint(train_df, test_dfs, seed=42):
    info = {"ratio": 50.0, "n_normal": 0, "n_attack": 0}
    _, prep = fit_eval(train_df, test_dfs, seed, 5, True, "DecisionTree", False, info)
    return prep


def test_joint_training_test_set_does_not_influence_training():
    """Cambiare completamente i due test set (TON_test, BoT_test) non deve
    cambiare nulla di cio' che il preprocessor congiunto ha imparato.

    Stesso schema del vincolo cross-domain: si fitta due volte sullo stesso
    training congiunto con coppie di test radicalmente diverse — una
    riscalata di 5000 e con categorie mai viste — e si pretende che
    selezione delle feature, vocabolari, cardinalita' e quantili siano
    identici bit per bit.
    """
    train = _harmonized_synth(4000, seed=1)
    test_a = {"ton": _harmonized_synth(800, seed=2), "bot": _harmonized_synth(800, seed=3)}
    test_b = {
        "ton": _harmonized_synth(800, seed=4, shift=5000.0),
        "bot": _harmonized_synth(800, seed=5, shift=5000.0),
    }
    for d in ("ton", "bot"):
        test_b[d]["proto_h"] = "sctp"        # categoria mai vista in training
        test_b[d]["state_h"] = "bizarre"

    p1 = _fit_on_joint(train, test_a)
    p2 = _fit_on_joint(train, test_b)

    assert p1.numeric_features_ == p2.numeric_features_, "la selezione feature e' cambiata"
    assert p1.vocabularies_ == p2.vocabularies_, "i vocabolari categorici sono cambiati"
    assert p1.cardinalities_ == p2.cardinalities_, "le cardinalita' sono cambiate"
    np.testing.assert_array_equal(p1.quantile_.quantiles_, p2.quantile_.quantiles_)

    probe = _harmonized_synth(300, seed=99)
    X1, C1 = p1.transform(probe)
    X2, C2 = p2.transform(probe)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(C1, C2)


# ── 3. la scelta del rapporto non vede i test set ─────────────────────
def test_inner_split_partitions_the_train_split_only():
    """La validation interna deve essere una partizione del solo training:
    fit ∪ val = split_train, fit ∩ val = ∅, e nessuna delle due tocca il
    test. E' il vincolo che rende la scelta del rapporto legittima."""
    seed = 11
    H = {"ton": _harmonized_synth(3000, seed=1), "bot": _harmonized_synth(900, seed=2)}
    split_train, split_test = {}, {}
    for d in ("ton", "bot"):
        tr, te = outer_split(H[d]["label"].to_numpy(), seed=seed)
        split_train[d], split_test[d] = tr, te

    fit_idx, val_idx = inner_split(H, split_train, seed)
    for d in ("ton", "bot"):
        f, v, tr, te = (set(fit_idx[d]), set(val_idx[d]),
                        set(split_train[d]), set(split_test[d]))
        assert f | v == tr, "fit e validation non ricoprono il training"
        assert not (f & v), "fit e validation si sovrappongono"
        assert not (v & te), "la validation pesca dal test set"
        assert not (f & te), "il fit pesca dal test set"
        assert 0 < len(v) < len(tr)


def test_inner_split_keeps_both_classes_in_validation():
    """La validation resta alla distribuzione naturale del dominio, ma
    stratificata: se il dominio ha normali, la validation ne ha. Senza
    questo la balanced accuracy di selezione sarebbe indefinita."""
    seed = 5
    H = {
        "ton": _harmonized_synth(4000, seed=1),
        "bot": pd.concat([                              # normali scarsissimi
            _harmonized_synth(120, seed=2).assign(label=0),
            _harmonized_synth(6000, seed=3).assign(label=1),
        ], ignore_index=True),
    }
    split_train = {d: outer_split(H[d]["label"].to_numpy(), seed=seed)[0]
                   for d in ("ton", "bot")}
    _fit, val_idx = inner_split(H, split_train, seed)
    for d in ("ton", "bot"):
        y = H[d]["label"].to_numpy()[val_idx[d]]
        assert set(np.unique(y)) == {0, 1}, f"{d}: la validation ha una sola classe"


def test_ratio_selection_ignores_the_test_sets_entirely():
    """Il vincolo del Prof. Kuznetsov, verificato e non solo dichiarato.

    Si costruisce lo stesso identico problema due volte, cambiando SOLO i
    test set: una volta plausibili, una volta traslati di 5000 e con
    categorie mai viste. Tutto cio' che entra nella scelta del rapporto —
    indici di fit, indici di validation, training bilanciato a ogni
    rapporto candidato — deve risultare identico. Se il test set riuscisse
    a filtrare nella selezione, qui divergerebbe.
    """
    seed = 42
    base = {"ton": _harmonized_synth(3000, seed=1), "bot": _harmonized_synth(900, seed=2)}

    def materiale_di_selezione(H):
        split_train = {}
        for d in ("ton", "bot"):
            tr, _te = outer_split(H[d]["label"].to_numpy(), seed=seed)
            split_train[d] = tr
        fit_idx, val_idx = inner_split(H, split_train, seed)
        per_rapporto = {}
        for ratio in RATIOS_CANDIDATI:
            bal, info = balance_joint(H, fit_idx, ratio, seed)
            per_rapporto[ratio] = ({d: bal[d].copy() for d in bal}, info)
        return fit_idx, val_idx, per_rapporto

    # H_a e H_b differiscono SOLO nelle righe che finiranno nel test set
    H_a = {d: base[d].copy() for d in base}
    H_b = {d: base[d].copy() for d in base}
    for d in ("ton", "bot"):
        _tr, te = outer_split(base[d]["label"].to_numpy(), seed=seed)
        for c in HARMONIZED_NUMERIC:
            H_b[d].loc[H_b[d].index[te], c] = H_b[d].loc[H_b[d].index[te], c] + 5000.0
        H_b[d].loc[H_b[d].index[te], "proto_h"] = "sctp"
        H_b[d].loc[H_b[d].index[te], "state_h"] = "bizarre"

    fa, va, ra = materiale_di_selezione(H_a)
    fb, vb, rb = materiale_di_selezione(H_b)

    for d in ("ton", "bot"):
        np.testing.assert_array_equal(fa[d], fb[d])
        np.testing.assert_array_equal(va[d], vb[d])
    for ratio in RATIOS_CANDIDATI:
        bal_a, info_a = ra[ratio]
        bal_b, info_b = rb[ratio]
        assert info_a == info_b, f"ratio {ratio}: i conteggi di bilanciamento sono cambiati"
        for d in ("ton", "bot"):
            np.testing.assert_array_equal(
                bal_a[d], bal_b[d],
                err_msg=f"ratio {ratio}, {d}: il test set ha cambiato il training")


def test_lo_spazio_ridotto_seleziona_gli_stessi_flussi():
    """Giustifica l'ereditarieta' del rapporto fra spazio ricco e ridotto.

    Lo spazio ridotto e' una proiezione delle stesse righe: `balance_joint`
    guarda solo le etichette e il seed, mai le feature, quindi a pari
    rapporto deve selezionare esattamente gli stessi flussi. Se un giorno
    non fosse piu' vero, ereditare il rapporto scelto nello spazio ricco
    diventerebbe illecito e questo test lo direbbe.
    """
    from kanids.harmonized import build_ridotto_da_ricco

    seed = 13
    H = {"ton": _harmonized_synth(3000, seed=1), "bot": _harmonized_synth(900, seed=2)}
    split_train = {d: outer_split(H[d]["label"].to_numpy(), seed=seed)[0]
                   for d in ("ton", "bot")}
    Hr = {k: build_ridotto_da_ricco(v) for k, v in H.items()}

    for ratio in RATIOS_CANDIDATI:
        a, info_a = balance_joint(H, split_train, ratio, seed)
        b, info_b = balance_joint(Hr, split_train, ratio, seed)
        assert info_a == info_b, f"ratio {ratio}: conteggi diversi fra i due spazi"
        for d in ("ton", "bot"):
            np.testing.assert_array_equal(
                a[d], b[d],
                err_msg=f"ratio {ratio}, {d}: lo spazio ridotto sceglie altre righe")


def test_joint_training_unseen_test_categories_go_to_unk():
    train = _harmonized_synth(2000, seed=1)
    test_dfs = {"ton": _harmonized_synth(500, seed=2), "bot": _harmonized_synth(500, seed=3)}
    test_dfs["bot"]["proto_h"] = "sctp"
    p = _fit_on_joint(train, test_dfs)
    _, C = p.transform(test_dfs["bot"])
    j = p.categorical_.index("proto_h")
    assert (C[:, j] == 0).all(), "categoria nuova non mappata su UNK"
    assert p.unseen_rate(test_dfs["bot"])["proto_h"] == 1.0

"""Modelli con interfaccia uniforme (numerico + categorico).

Le KAN con edge categorici erano finora riscritte a mano dentro sei script
diversi, ognuno con il suo loop di training: impossibile garantire che
stessero addestrando la stessa cosa. Qui c'e' una sola implementazione,
usata sia dagli esperimenti sia dall'export in C.

Tutti i modelli espongono:
    fit(Xnum, Xcat, y)  ->  self
    predict_proba(Xnum, Xcat)
    predict(Xnum, Xcat)
in modo che il runner di cross-validation sia identico per KAN e baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CLIP


# ─────────────────────────────────────────────────────────────
# BASE DI CHEBYSHEV
# ─────────────────────────────────────────────────────────────
def chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """T_0..T_degree valutati in x (atteso in [-1, 1]). Shape (n, degree+1)."""
    x = np.clip(x, -1.0, 1.0)
    out = np.empty((len(x), degree + 1), dtype=np.float64)
    out[:, 0] = 1.0
    if degree >= 1:
        out[:, 1] = x
    for d in range(2, degree + 1):
        out[:, d] = 2.0 * x * out[:, d - 1] - out[:, d - 2]
    return out


def _norm(X: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(X / clip, -1.0, 1.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ─────────────────────────────────────────────────────────────
# KAN SINGLE-LAYER CON EDGE CATEGORICI
# ─────────────────────────────────────────────────────────────
class CategoricalKANBinary:
    """10 edge Chebyshev sulle numeriche + un edge tabellare per categorica.

    L'edge categorico e' phi_j(c) = Tab_j[c]: in una KAN compilata a LUT e'
    gia' nella forma di deployment, quindi non costa nulla in piu' rispetto
    a un edge numerico. Lo slot 0 di ogni tabella e' la categoria mai vista
    (UNK) e viene appreso dai dati di training che finiscono li' (nessuno,
    in-domain: resta al valore di inizializzazione, cioe' contributo nullo).
    """

    def __init__(self, in_dim: int, cardinalities, degree: int = 8,
                 clip: float = CLIP, seed: int = 0):
        self.in_dim = in_dim
        self.cardinalities = list(cardinalities)
        self.degree = degree
        self.clip = clip
        self.seed = seed

    def _init(self):
        rng = np.random.RandomState(self.seed)
        self.coeffs_ = rng.randn(self.in_dim, self.degree + 1) * 0.05
        self.tables_ = [np.random.RandomState(self.seed + j + 1).randn(c) * 0.05
                        for j, c in enumerate(self.cardinalities)]

    def fit(self, Xnum, Xcat, y, epochs: int = 250, lr: float = 0.3,
            l2: float = 1e-4, class_weight: bool = True, verbose: bool = False):
        self._init()
        y = np.asarray(y, dtype=np.float64)
        Xn = _norm(np.asarray(Xnum, np.float64), self.clip)
        B = np.stack([chebyshev_basis(Xn[:, i], self.degree) for i in range(self.in_dim)])
        n = len(y)
        J = len(self.tables_)

        if class_weight:
            pos = max(y.mean(), 1e-6)
            sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-6))
        else:
            sw = np.ones(n)

        for ep in range(epochs):
            z = np.einsum("ind,id->n", B, self.coeffs_)
            for j in range(J):
                z += self.tables_[j][Xcat[:, j]]
            g = sw * (_sigmoid(z) - y)
            self.coeffs_ -= lr * (np.einsum("ind,n->id", B, g) / n + l2 * self.coeffs_)
            for j in range(J):
                gt = np.zeros_like(self.tables_[j])
                np.add.at(gt, Xcat[:, j], g)
                self.tables_[j] -= lr * (gt / n + l2 * self.tables_[j])
            if verbose and (ep + 1) % 50 == 0:
                print(f"  ep {ep+1}/{epochs}")
        return self

    def decision_function(self, Xnum, Xcat):
        Xn = _norm(np.asarray(Xnum, np.float64), self.clip)
        z = sum(chebyshev_basis(Xn[:, i], self.degree) @ self.coeffs_[i]
                for i in range(self.in_dim))
        for j in range(len(self.tables_)):
            z = z + self.tables_[j][Xcat[:, j]]
        return z

    def predict_proba(self, Xnum, Xcat):
        p = _sigmoid(self.decision_function(Xnum, Xcat))
        return np.column_stack([1 - p, p])

    def predict(self, Xnum, Xcat, thr: float = 0.5):
        return (self.predict_proba(Xnum, Xcat)[:, 1] >= thr).astype(int)

    @property
    def n_parameters(self) -> int:
        return self.coeffs_.size + sum(t.size for t in self.tables_)


class CategoricalKANMulticlass:
    """Versione multiclass: ogni edge produce un vettore di C logit."""

    def __init__(self, in_dim: int, n_classes: int, cardinalities,
                 degree: int = 8, clip: float = CLIP, seed: int = 0):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.cardinalities = list(cardinalities)
        self.degree = degree
        self.clip = clip
        self.seed = seed

    def _init(self):
        rng = np.random.RandomState(self.seed)
        self.coeffs_ = rng.randn(self.in_dim, self.n_classes, self.degree + 1) * 0.05
        self.tables_ = [np.random.RandomState(self.seed + j + 1).randn(c, self.n_classes) * 0.05
                        for j, c in enumerate(self.cardinalities)]

    def fit(self, Xnum, Xcat, y, epochs: int = 300, lr: float = 0.3,
            l2: float = 1e-4, class_weight: bool = True, verbose: bool = False):
        self._init()
        y = np.asarray(y, dtype=np.int64)
        C = self.n_classes
        Xn = _norm(np.asarray(Xnum, np.float64), self.clip)
        B = [chebyshev_basis(Xn[:, i], self.degree) for i in range(self.in_dim)]
        n = len(y)
        J = len(self.tables_)

        Y = np.zeros((n, C))
        Y[np.arange(n), y] = 1.0
        if class_weight:
            counts = np.bincount(y, minlength=C).astype(np.float64)
            sw = (n / (C * np.maximum(counts, 1)))[y]
        else:
            sw = np.ones(n)
        s = sw.sum()

        for ep in range(epochs):
            Z = np.zeros((n, C))
            for i in range(self.in_dim):
                Z += B[i] @ self.coeffs_[i].T
            for j in range(J):
                Z += self.tables_[j][Xcat[:, j]]
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            G = (sw[:, None] * (P - Y)) / s
            for i in range(self.in_dim):
                self.coeffs_[i] -= lr * ((B[i].T @ G).T + l2 * self.coeffs_[i])
            for j in range(J):
                gt = np.zeros_like(self.tables_[j])
                np.add.at(gt, Xcat[:, j], G)
                self.tables_[j] -= lr * (gt + l2 * self.tables_[j])
            if verbose and (ep + 1) % 50 == 0:
                print(f"  ep {ep+1}/{epochs}")
        return self

    def decision_function(self, Xnum, Xcat):
        Xn = _norm(np.asarray(Xnum, np.float64), self.clip)
        Z = np.zeros((len(Xn), self.n_classes))
        for i in range(self.in_dim):
            Z += chebyshev_basis(Xn[:, i], self.degree) @ self.coeffs_[i].T
        for j in range(len(self.tables_)):
            Z += self.tables_[j][Xcat[:, j]]
        return Z

    def predict_proba(self, Xnum, Xcat):
        Z = self.decision_function(Xnum, Xcat)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, Xnum, Xcat):
        return self.decision_function(Xnum, Xcat).argmax(axis=1)

    @property
    def n_parameters(self) -> int:
        return self.coeffs_.size + sum(t.size for t in self.tables_)


# ─────────────────────────────────────────────────────────────
# ADATTATORE PER LE BASELINE SKLEARN-LIKE
# ─────────────────────────────────────────────────────────────
class SklearnAdapter:
    """Espone l'interfaccia (Xnum, Xcat) su un estimatore sklearn.

    `cat_encoding`:
      "ordinal" per i modelli ad albero (LightGBM/XGBoost/DecisionTree),
      "onehot"  per l'MLP, che con codici interi inventerebbe un ordine.
    Le colonne one-hot sono costruite dalle cardinalita' apprese sul
    training, quindi una categoria UNK nel test resta rappresentabile.
    """

    def __init__(self, estimator, cardinalities, cat_encoding: str = "ordinal"):
        self.estimator = estimator
        self.cardinalities = list(cardinalities)
        self.cat_encoding = cat_encoding

    def _design(self, Xnum, Xcat):
        Xnum = np.asarray(Xnum, np.float64)
        if Xcat is None or Xcat.shape[1] == 0:
            return Xnum
        if self.cat_encoding == "ordinal":
            return np.hstack([Xnum, Xcat.astype(np.float64)])
        blocks = [Xnum]
        for j, card in enumerate(self.cardinalities):
            oh = np.zeros((len(Xcat), card))
            oh[np.arange(len(Xcat)), np.clip(Xcat[:, j], 0, card - 1)] = 1.0
            blocks.append(oh)
        return np.hstack(blocks)

    def fit(self, Xnum, Xcat, y, **kw):
        self.estimator.fit(self._design(Xnum, Xcat), y)
        return self

    def predict(self, Xnum, Xcat):
        return self.estimator.predict(self._design(Xnum, Xcat))

    def predict_proba(self, Xnum, Xcat):
        return self.estimator.predict_proba(self._design(Xnum, Xcat))


def get_baselines(task: str, cardinalities, seed: int = 42, n_classes: int = 2) -> dict:
    """Le baseline richieste, sullo stesso identico spazio di feature."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    out = {}
    out["DecisionTree(d=5)"] = SklearnAdapter(
        DecisionTreeClassifier(max_depth=5, random_state=seed,
                               class_weight="balanced"),
        cardinalities, "ordinal")

    try:
        from lightgbm import LGBMClassifier
        out["LightGBM"] = SklearnAdapter(
            LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, max_bin=255,
                           class_weight="balanced", random_state=seed,
                           n_jobs=-1, verbose=-1),
            cardinalities, "ordinal")
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = SklearnAdapter(
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          tree_method="hist", max_bin=256, random_state=seed,
                          n_jobs=-1, verbosity=0,
                          eval_metric="logloss" if task == "binary" else "mlogloss",
                          objective="binary:logistic" if task == "binary"
                          else "multi:softprob",
                          num_class=None if task == "binary" else n_classes),
            cardinalities, "ordinal")
    except ImportError:
        pass

    # MLP piccolo: confronto onesto in numero di parametri con la KAN
    out["MLP(16)"] = SklearnAdapter(
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, early_stopping=True,
                      n_iter_no_change=10, random_state=seed),
        cardinalities, "onehot")
    return out


# ─────────────────────────────────────────────────────────────
# KAN MULTI-LAYER CON EDGE CATEGORICI
# ─────────────────────────────────────────────────────────────
def _cheb_dT(x: np.ndarray, degree: int) -> np.ndarray:
    """Derivate T'_d(x) = d * U_{d-1}(x), stessa ricorrenza dello script v1."""
    x = np.clip(x, -1.0, 1.0)
    dT = np.zeros(x.shape + (degree + 1,), dtype=x.dtype)
    if degree >= 1:
        dT[..., 1] = 1.0
    U = [np.ones_like(x), 2 * x]
    for n in range(2, degree + 1):
        dT[..., n] = n * U[n - 1]
        U.append(2 * x * U[-1] - U[-2])
    return dT


def _cheb_T_nd(x: np.ndarray, degree: int) -> np.ndarray:
    """Base di Chebyshev applicata elemento per elemento: (..., degree+1)."""
    x = np.clip(x, -1.0, 1.0)
    T = [np.ones_like(x), x]
    for n in range(2, degree + 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return np.stack(T, axis=-1)


def _scatter_add_rows(idx: np.ndarray, values: np.ndarray, n_rows: int) -> np.ndarray:
    """Somma le righe di `values` nelle righe indicate da `idx`.

    Equivalente a:
        out = np.zeros((n_rows, values.shape[1])); np.add.at(out, idx, values)
    ma passa da np.bincount su indici appiattiti. np.add.at non e' bufferizzato
    ed e' di gran lunga l'operazione piu' lenta del passo di training: su
    168k righe x 16 unita' nascoste costa piu' di tutte le contrazioni BLAS
    messe insieme.
    """
    n, h = values.shape
    flat = (idx.astype(np.int64)[:, None] * h + np.arange(h, dtype=np.int64)[None, :]).ravel()
    out = np.bincount(flat, weights=values.ravel().astype(np.float64),
                      minlength=n_rows * h)
    return out.reshape(n_rows, h).astype(values.dtype)


class MultiLayerKANBinary:
    """[10 num Chebyshev + 4 cat tabellari] -> H hidden -> tanh -> Chebyshev -> 1.

    Riscrittura della rete di `scripts/kan14_ml_binary.py` con l'interfaccia
    comune (Xnum, Xcat), in modo che possa passare per lo stesso runner di
    cross-validation di tutti gli altri modelli. Architettura, iperparametri
    (H=16, grado 8, Adam lr=0.01, 300 epoche, BCE pesata) e inizializzazione
    sono quelli dello script originale: l'obiettivo e' verificare quel
    risultato, non ottimizzarlo.

    A differenza del single-layer questa rete NON e' additiva: il secondo
    strato vede combinazioni delle unita' nascoste, quindi puo' rappresentare
    interazioni fra feature. E' l'ipotesi che spiega il divario con gli
    alberi, e questo modello e' il test di quell'ipotesi.
    """

    def __init__(self, in_dim: int, cardinalities, hidden: int = 16,
                 degree: int = 8, clip: float = CLIP, seed: int = 0,
                 epochs: int = 300, lr: float = 0.01):
        self.in_dim = in_dim
        self.cardinalities = list(cardinalities)
        self.hidden = hidden
        self.degree = degree
        self.clip = clip
        self.seed = seed
        self.epochs = epochs
        self.lr = lr

    #: il runner puo' interrompere e riprendere questo fit
    supports_resume = True

    def fit(self, Xnum, Xcat, y, epochs: int | None = None, verbose: bool = False,
            class_weight: bool = True, state_path=None, max_seconds=None):
        """Come la variante multiclass: con `state_path` e `max_seconds` il
        training si interrompe salvando lo stato completo e riprende in modo
        bit-esatto. `self.finished_` dice se il fit e' completo."""
        import pickle
        import time as _time

        E = epochs or self.epochs
        H, D = self.hidden, self.degree
        rng = np.random.RandomState(self.seed)
        _t0 = _time.time()

        X = (np.asarray(Xnum, np.float32) / np.float32(self.clip))
        y = np.asarray(y, np.float32)
        N = len(y)
        J = len(self.cardinalities)

        if class_weight:
            pos = max(float(y.mean()), 1e-6)
            sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-6)).astype(np.float32)
        else:
            sw = np.ones(N, np.float32)

        state = None
        if state_path is not None and Path(state_path).exists():
            with open(state_path, "rb") as fh:
                state = pickle.load(fh)
        if state is not None:
            params = state["p"]
            C1, C2, tabs = params[0], params[1], list(params[2:])
            m, v = state["m"], state["v"]
            ep0, step0 = state["ep"], state["t"]
        else:
            C1 = (rng.randn(self.in_dim, H, D + 1) * 0.1).astype(np.float32)
            C2 = (rng.randn(H, 1, D + 1) * 0.1).astype(np.float32)
            tabs = [(rng.randn(c, H) * 0.1).astype(np.float32) for c in self.cardinalities]
            params = [C1, C2] + tabs
            m = [np.zeros_like(p) for p in params]
            v = [np.zeros_like(p) for p in params]
            ep0, step0 = 0, 0

        T1 = _cheb_T_nd(X, D)                       # (N, in_dim, D+1), fisso
        T1f = T1.reshape(N, self.in_dim * (D + 1))  # vista piatta per il BLAS
        b1, b2, eps = 0.9, 0.999, 1e-8

        self.finished_ = True
        step = step0
        for ep in range(ep0, E):
            if max_seconds is not None and _time.time() - _t0 > max_seconds:
                if state_path is not None:
                    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(state_path, "wb") as fh:
                        pickle.dump({"p": params, "m": m, "v": v,
                                     "ep": ep, "t": step}, fh)
                self.finished_ = False
                self.epochs_done_ = ep
                self.C1_, self.C2_, self.tables_ = C1, C2, tabs
                return self
            t = step + 1
            # forward: le contrazioni sono le stesse degli einsum di riferimento,
            # riscritte come prodotti matriciali (stesso risultato, ~10x piu' veloce)
            Hh = T1f @ C1.transpose(0, 2, 1).reshape(self.in_dim * (D + 1), H)
            for j in range(J):
                Hh += tabs[j][Xcat[:, j]]
            A = np.tanh(Hh)
            T2 = _cheb_T_nd(A, D)
            T2f = T2.reshape(N, H * (D + 1))
            z = (T2f @ C2.reshape(H * (D + 1), 1))[:, 0]
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (sw * (p - y) / N).astype(np.float32)

            gC2 = (T2f.T @ g).reshape(H, 1, D + 1)
            # sum_d C2[h,d] * T'_d(A) accumulato con la ricorrenza di
            # Chebyshev di seconda specie: evita di materializzare un array
            # (N, H, D+1) per epoca (~100 MB), che era il collo di bottiglia
            c2 = C2[:, 0, :]
            acc = np.broadcast_to(c2[:, 1], A.shape).astype(np.float32).copy()
            U_prev, U_cur = np.ones_like(A), 2.0 * A      # U_0, U_1
            for d in range(2, D + 1):
                acc += (c2[:, d] * d) * U_cur             # T'_d = d * U_{d-1}
                U_prev, U_cur = U_cur, 2.0 * A * U_cur - U_prev
            gA = acc * g[:, None]
            gHh = (gA * (1 - A * A)).astype(np.float32)
            gC1 = (T1f.T @ gHh).reshape(self.in_dim, D + 1, H).transpose(0, 2, 1)
            grads = [np.ascontiguousarray(gC1), gC2]
            for j in range(J):
                gt = np.zeros_like(tabs[j])
                np.add.at(gt, Xcat[:, j], gHh)
                grads.append(gt)

            for k, (P, G) in enumerate(zip(params, grads)):
                m[k] = b1 * m[k] + (1 - b1) * G
                v[k] = b2 * v[k] + (1 - b2) * G * G
                P -= self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
            step = t

            if verbose and t % 50 == 0:
                print(f"  ep {t}/{E}")

        if state_path is not None and Path(state_path).exists():
            Path(state_path).unlink()
        self.epochs_done_ = E
        self.C1_, self.C2_, self.tables_ = C1, C2, tabs
        return self

    def decision_function(self, Xnum, Xcat, batch: int = 250_000):
        """Forward a blocchi: valutare 3,7 M flussi in un colpo allocherebbe
        oltre 3 GB solo per le basi di Chebyshev dei due strati. Il risultato
        e' identico, la memoria resta costante."""
        D = self.degree
        X = (np.asarray(Xnum, np.float32) / np.float32(self.clip))
        H = self.C1_.shape[1]
        W1 = self.C1_.transpose(0, 2, 1).reshape(-1, H)
        W2 = self.C2_.reshape(-1, 1)
        out = np.empty(len(X), dtype=np.float32)
        for a in range(0, len(X), batch):
            b = min(a + batch, len(X))
            Hh = _cheb_T_nd(X[a:b], D).reshape(b - a, -1) @ W1
            for j in range(len(self.tables_)):
                Hh += self.tables_[j][Xcat[a:b, j]]
            A = np.tanh(Hh)
            out[a:b] = (_cheb_T_nd(A, D).reshape(b - a, -1) @ W2)[:, 0]
        return out

    def predict_proba(self, Xnum, Xcat):
        p = _sigmoid(self.decision_function(Xnum, Xcat).astype(np.float64))
        return np.column_stack([1 - p, p])

    def predict(self, Xnum, Xcat, thr: float = 0.5):
        return (self.predict_proba(Xnum, Xcat)[:, 1] >= thr).astype(int)

    @property
    def n_parameters(self) -> int:
        return self.C1_.size + self.C2_.size + sum(t.size for t in self.tables_)


class MultiLayerKANMulticlass:
    """[10 num Chebyshev + 4 cat tabellari] -> H hidden -> tanh -> Chebyshev -> C classi.

    Versione multiclass di MultiLayerKANBinary, fedele a
    `scripts/kan_ml_cat_mc.py` (H=16, grado 8, Adam lr=0.01, 300 epoche,
    cross-entropy pesata per classe) ma riscritta con contrazioni BLAS in
    modo da poter girare 15 volte dentro la cross-validation invece di una
    sola volta su uno split.

    Il gradiente rispetto alle attivazioni non materializza l'array
    (N, H, grado+1) delle derivate della base: accumula per grado con la
    ricorrenza di Chebyshev di seconda specie, come nella variante binaria.
    """

    def __init__(self, in_dim: int, n_classes: int, cardinalities, hidden: int = 16,
                 degree: int = 8, clip: float = CLIP, seed: int = 0,
                 epochs: int = 300, lr: float = 0.01):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.cardinalities = list(cardinalities)
        self.hidden = hidden
        self.degree = degree
        self.clip = clip
        self.seed = seed
        self.epochs = epochs
        self.lr = lr

    #: il runner di cross-validation puo' interrompere e riprendere questo fit
    supports_resume = True

    def fit(self, Xnum, Xcat, y, epochs: int | None = None, verbose: bool = False,
            class_weight: bool = True, state_path=None, max_seconds=None):
        """Addestra, opzionalmente a chunk.

        Con `state_path` e `max_seconds` il training si interrompe allo
        scadere del budget salvando lo stato completo (pesi + momenti di
        Adam + contatore di passo) e riprende esattamente da li' alla
        chiamata successiva. Serve perche' un fit completo qui costa ~170 s
        e la cross-validation ne richiede quindici: su una macchina a tempo
        limitato senza questo non si arriva in fondo.
        `self.finished_` dice se il fit e' completo.
        """
        import pickle
        import time as _time

        E = epochs or self.epochs
        H, D, C, K = self.hidden, self.degree, self.n_classes, self.in_dim
        rng = np.random.RandomState(self.seed)
        _t0 = _time.time()

        X = (np.asarray(Xnum, np.float32) / np.float32(self.clip))
        y = np.asarray(y, np.int64)
        N = len(y)
        J = len(self.cardinalities)

        state = None
        if state_path is not None and Path(state_path).exists():
            with open(state_path, "rb") as fh:
                state = pickle.load(fh)
        if state is not None:
            params = state["p"]
            C1, C2, tabs = params[0], params[1], list(params[2:])
            m, v = state["m"], state["v"]
            ep0, step0 = state["ep"], state["t"]
        else:
            C1 = (rng.randn(K, H, D + 1) * 0.1).astype(np.float32)
            C2 = (rng.randn(H, C, D + 1) * 0.1).astype(np.float32)
            tabs = [(rng.randn(c, H) * 0.1).astype(np.float32) for c in self.cardinalities]
            params = [C1, C2] + tabs
            m = [np.zeros_like(p) for p in params]
            v = [np.zeros_like(p) for p in params]
            ep0, step0 = 0, 0

        Y = np.zeros((N, C), dtype=np.float32)
        Y[np.arange(N), y] = 1.0
        if class_weight:
            counts = np.bincount(y, minlength=C).astype(np.float32)
            sw = ((N / (C * np.maximum(counts, 1)))[y]).astype(np.float32)
        else:
            sw = np.ones(N, np.float32)
        s = sw.sum()

        T1 = _cheb_T_nd(X, D)
        T1f = T1.reshape(N, K * (D + 1))
        b1, b2, eps = 0.9, 0.999, 1e-8

        self.finished_ = True
        step = step0
        for ep in range(ep0, E):
            if max_seconds is not None and _time.time() - _t0 > max_seconds:
                if state_path is not None:
                    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(state_path, "wb") as fh:
                        pickle.dump({"p": params, "m": m, "v": v,
                                     "ep": ep, "t": step}, fh)
                self.finished_ = False
                self.epochs_done_ = ep
                self.C1_, self.C2_, self.tables_ = C1, C2, tabs
                return self
            t = step + 1
            Hh = T1f @ C1.transpose(0, 2, 1).reshape(K * (D + 1), H)
            for j in range(J):
                Hh += tabs[j][Xcat[:, j]]
            A = np.tanh(Hh)

            # Forward del secondo strato accumulato grado per grado: la base
            # di Chebyshev valutata in A non viene mai materializzata come
            # array (N, H, grado+1), che a 168k righe sarebbe ~100 MB per
            # epoca ed era il collo di bottiglia.
            Z = np.zeros((N, C), dtype=np.float32)
            Tp, Tc = np.ones_like(A), A.copy()
            for d in range(D + 1):
                if d == 0:
                    Td = Tp
                elif d == 1:
                    Td = Tc
                else:
                    Td = 2.0 * A * Tc - Tp
                    Tp, Tc = Tc, Td
                Z += Td @ C2[:, :, d]
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            G = ((sw[:, None] * (P - Y)) / s).astype(np.float32)

            # Secondo passaggio sulla stessa ricorrenza per il gradiente:
            # ricalcolare T_d costa molto meno che tenerlo in memoria.
            gC2 = np.empty((H, C, D + 1), dtype=np.float32)
            Tp, Tc = np.ones_like(A), A.copy()
            for d in range(D + 1):
                if d == 0:
                    Td = Tp
                elif d == 1:
                    Td = Tc
                else:
                    Td = 2.0 * A * Tc - Tp
                    Tp, Tc = Tc, Td
                gC2[:, :, d] = Td.T @ G

            # gA[n,h] = sum_d (G[n,:] . C2[h,:,d]) * T'_d(A[n,h]),
            # accumulato grado per grado: nessun array (N, H, D+1).
            gA = np.zeros_like(A)
            U_prev, U_cur = np.ones_like(A), 2.0 * A       # U_0, U_1
            for d in range(1, D + 1):
                Wd = G @ C2[:, :, d].T                      # (N, H)
                if d == 1:
                    gA += Wd
                else:
                    gA += (d * Wd) * U_cur                  # T'_d = d * U_{d-1}
                    U_prev, U_cur = U_cur, 2.0 * A * U_cur - U_prev
            gHh = (gA * (1 - A * A)).astype(np.float32)
            gC1 = (T1f.T @ gHh).reshape(K, D + 1, H).transpose(0, 2, 1)

            grads = [np.ascontiguousarray(gC1), np.ascontiguousarray(gC2)]
            for j in range(J):
                grads.append(_scatter_add_rows(Xcat[:, j], gHh, tabs[j].shape[0]))

            for k, (Pm, Gg) in enumerate(zip(params, grads)):
                m[k] = b1 * m[k] + (1 - b1) * Gg
                v[k] = b2 * v[k] + (1 - b2) * Gg * Gg
                Pm -= self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
            step = t

            if verbose and t % 50 == 0:
                print(f"  ep {t}/{E}")

        if state_path is not None and Path(state_path).exists():
            Path(state_path).unlink()          # fit completo: stato non serve piu'
        self.epochs_done_ = E
        self.C1_, self.C2_, self.tables_ = C1, C2, tabs
        return self

    def decision_function(self, Xnum, Xcat, batch: int = 250_000):
        """Forward a blocchi, come nella variante binaria: memoria costante
        anche su milioni di righe, risultato identico."""
        D, H, C = self.degree, self.hidden, self.n_classes
        X = (np.asarray(Xnum, np.float32) / np.float32(self.clip))
        W1 = self.C1_.transpose(0, 2, 1).reshape(-1, H)
        W2 = self.C2_.transpose(0, 2, 1).reshape(-1, C)
        out = np.empty((len(X), C), dtype=np.float32)
        for a in range(0, len(X), batch):
            b = min(a + batch, len(X))
            Hh = _cheb_T_nd(X[a:b], D).reshape(b - a, -1) @ W1
            for j in range(len(self.tables_)):
                Hh += self.tables_[j][Xcat[a:b, j]]
            A = np.tanh(Hh)
            out[a:b] = _cheb_T_nd(A, D).reshape(b - a, -1) @ W2
        return out

    def predict_proba(self, Xnum, Xcat):
        Z = self.decision_function(Xnum, Xcat).astype(np.float64)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, Xnum, Xcat):
        return self.decision_function(Xnum, Xcat).argmax(axis=1)

    @property
    def n_parameters(self) -> int:
        return self.C1_.size + self.C2_.size + sum(t.size for t in self.tables_)

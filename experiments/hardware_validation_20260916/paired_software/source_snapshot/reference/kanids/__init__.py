"""kanids — core riproducibile e leakage-free della linea KAN-IDS.

Tutto cio' che apprende qualcosa dai dati (feature selection, encoding
categorico, normalizzazione) vive qui e ha una sola regola:

    fit() vede solo il training split. transform() non vede mai y.

Gli script in scripts/ non devono piu' fare feature selection o
preprocessing per conto proprio: importano da qui.
"""
from .console import usa_utf8 as _usa_utf8

# Gli script di questo progetto stampano frecce ("TON→BoT"), em-dash e
# deviazioni standard col ±. Su Windows, quando l'output e' rediretto o
# messo in pipe, Python sceglie cp1252 e quei caratteri fanno terminare il
# programma a meta' — l'audit moriva dopo quaranta righe di "[ok]". Qui si
# decide una volta per tutte: UTF-8. Non tocca stream gia' UTF-8, quindi
# importare kanids da un notebook non cambia nulla.
_usa_utf8()

from .config import (
    REPO_ROOT, ARTIFACTS_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR,
    SEEDS, N_SPLITS, TEST_SIZE, CLIP, K_NUMERIC,
    ARCH, ARCH_ORIGINE, ARCH_SELEZIONATA, HIDDEN, DEGREE, DEGREE_1L,
    scarto_dalla_selezione,
    NUMERIC_RAW, SKEWED, CATEGORICAL, UNK_INDEX,
    set_global_seed, artifact_path,
)
from .preprocessing import LeakageFreePreprocessor
from .splits import outer_split, cv_splits, describe_protocol
from .metrics import binary_metrics, multiclass_metrics, aggregate

__all__ = [
    "REPO_ROOT", "ARTIFACTS_DIR", "RESULTS_DIR", "MODELS_DIR", "FIGURES_DIR",
    "SEEDS", "N_SPLITS", "TEST_SIZE", "CLIP", "K_NUMERIC",
    "ARCH", "ARCH_ORIGINE", "ARCH_SELEZIONATA", "HIDDEN", "DEGREE",
    "DEGREE_1L", "scarto_dalla_selezione",
    "NUMERIC_RAW", "SKEWED", "CATEGORICAL", "UNK_INDEX",
    "set_global_seed", "artifact_path",
    "LeakageFreePreprocessor",
    "outer_split", "cv_splits", "describe_protocol",
    "binary_metrics", "multiclass_metrics", "aggregate",
]

__version__ = "2.0.0"

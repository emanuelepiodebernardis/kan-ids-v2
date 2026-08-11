"""kanids — core riproducibile e leakage-free della linea KAN-IDS.

Tutto cio' che apprende qualcosa dai dati (feature selection, encoding
categorico, normalizzazione) vive qui e ha una sola regola:

    fit() vede solo il training split. transform() non vede mai y.

Gli script in scripts/ non devono piu' fare feature selection o
preprocessing per conto proprio: importano da qui.
"""
from .config import (
    REPO_ROOT, ARTIFACTS_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR,
    SEEDS, N_SPLITS, TEST_SIZE, CLIP, K_NUMERIC,
    NUMERIC_RAW, SKEWED, CATEGORICAL, UNK_INDEX,
    set_global_seed, artifact_path,
)
from .preprocessing import LeakageFreePreprocessor
from .splits import outer_split, cv_splits, describe_protocol
from .metrics import binary_metrics, multiclass_metrics, aggregate

__all__ = [
    "REPO_ROOT", "ARTIFACTS_DIR", "RESULTS_DIR", "MODELS_DIR", "FIGURES_DIR",
    "SEEDS", "N_SPLITS", "TEST_SIZE", "CLIP", "K_NUMERIC",
    "NUMERIC_RAW", "SKEWED", "CATEGORICAL", "UNK_INDEX",
    "set_global_seed", "artifact_path",
    "LeakageFreePreprocessor",
    "outer_split", "cv_splits", "describe_protocol",
    "binary_metrics", "multiclass_metrics", "aggregate",
]

__version__ = "2.0.0"

"""ADAPT-06 prospective update gate.

The package implements the protocol frozen at merge
e5163bce892a0aafdd5136ee7c6a618b9867eadc.  Importing it has no side effects
and never starts a replay.
"""

from .core import (
    BLOCK_SIZE,
    CAL_CAPACITY,
    FIT_CAPACITY,
    GateEvidence,
    GateOutcome,
    LabelBudget,
    ThresholdOnlyBaseline,
    aggregate_seed_outcomes,
    budget_for_block,
    evaluate_gate,
    select_fpr_threshold,
)

__all__ = [
    "BLOCK_SIZE",
    "CAL_CAPACITY",
    "FIT_CAPACITY",
    "GateEvidence",
    "GateOutcome",
    "LabelBudget",
    "ThresholdOnlyBaseline",
    "aggregate_seed_outcomes",
    "budget_for_block",
    "evaluate_gate",
    "select_fpr_threshold",
]

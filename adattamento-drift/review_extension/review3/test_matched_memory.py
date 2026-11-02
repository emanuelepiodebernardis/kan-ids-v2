import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from review3 import runner as r


def populated():
    mem = r.e.cr.MemorieSeparate(); larger = r.e.cr.BufferRuolo(256)
    for block in range(80):
        X = np.full((2, 8), block, dtype=float)
        y = np.array([0, 1]); digests = [f'{2*block+i:064x}' for i in range(2)]
        mem.aggiungi(X, y, np.array(['calibration']*2), block, digests)
        larger.aggiungi(X, y, block, digests)
    return mem, larger


def test_larger_cal_memory_retains_older_labels_with_identical_age_rule():
    mem, larger = populated()
    views = r.validate_memories(mem, larger, 80)
    assert len(views[64][1]) == 64
    assert len(views[256][1]) == 100
    assert views[64][2].max() == 32
    assert views[256][2].max() == 50


def test_fit_cal_overlap_is_rejected():
    mem, larger = populated()
    mem.fit.aggiungi(np.ones((1, 8)), np.array([0]), 79, [larger.digest()[0]])
    with pytest.raises(AssertionError):
        r.validate_memories(mem, larger, 80)


def test_current_block_labels_cannot_enter_calibration():
    mem, larger = populated()
    with pytest.raises(AssertionError):
        r.validate_memories(mem, larger, 79)


def test_one_class_can_borrow_unused_quota_without_changing_contract():
    mem = r.e.cr.MemorieSeparate(); larger = r.e.cr.BufferRuolo(256)
    X = np.zeros((100, 8)); y = np.ones(100, dtype=int)
    digests = [f'{i:064x}' for i in range(100)]
    mem.aggiungi(X, y, np.array(['calibration']*100), 0, digests)
    larger.aggiungi(X, y, 0, digests)
    views = r.validate_memories(mem, larger, 1)
    assert len(views[64][1]) == 64
    assert len(views[256][1]) == 100

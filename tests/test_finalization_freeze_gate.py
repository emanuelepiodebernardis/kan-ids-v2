"""The preparation gate must reject forged or stale freeze evidence before test transforms."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prep_final', ROOT / 'scripts/prepare_finalization_data.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_rejects_nonempty_unrelated_json(tmp_path):
    p = tmp_path / 'protocol.json'
    p.write_text('{"ready": true}')
    with pytest.raises(ValueError, match='Not a valid frozen'):
        prep.validate_freeze(p, tmp_path)


def test_rejects_incomplete_frozen_binding(tmp_path):
    p = tmp_path / 'protocol.json'
    p.write_text(json.dumps({'status': 'FROZEN_BEFORE_TEST_EVALUATION',
                            'source_split': 'train', 'selection_reads_test': False,
                            'selected_L': 1025, 'candidate_grid': [1025]}))
    with pytest.raises(ValueError, match='Incomplete frozen'):
        prep.validate_freeze(p, tmp_path)

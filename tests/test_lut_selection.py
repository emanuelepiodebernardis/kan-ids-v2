"""Regression: test content cannot select or revise a sampled-LUT representation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import export_kan14_lut_c as exp


def _calibration(tmp_path: Path):
    m = exp.leggi_modello(exp.SORGENTE)
    rng = np.random.default_rng(441)
    ids = np.arange(40, dtype=np.int64)
    cards = np.r_[m['CAT_OFF'][1:], len(m['CAT'])] - m['CAT_OFF']
    d = {'Xq': rng.integers(-4096, 4097, size=(40, m['NFEAT']), dtype=np.int64),
         'CAT': np.column_stack([rng.integers(c, size=40) for c in cards]),
         'row_ids': ids, 'y_true': rng.integers(2, size=40, dtype=np.int64)}
    npz = tmp_path / 'train.npz'
    np.savez(npz, **d)
    metadata = {'source_split': 'train', 'source_csv_sha256': '1' * 64,
                'row_ids_sha256': hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                'preprocessing': {'kind': 'synthetic Q12 inputs for regression test'}}
    meta = tmp_path / 'train.json'
    meta.write_text(json.dumps(metadata), encoding='utf-8')
    return npz, meta, d


def test_selection_ignores_replaced_and_unreadable_test_inputs(tmp_path, monkeypatch):
    """Actual select pipeline, with poisoned test readers and replaced test bytes."""
    train, metadata, _ = _calibration(tmp_path)
    fake_test = tmp_path / 'kan14_test_vectors.h'
    fake_test.write_text('arbitrary first test content')
    monkeypatch.setattr(exp, 'VETTORI', fake_test)

    def forbidden(*args, **kwargs):
        raise AssertionError('Selection attempted to read test vectors')

    monkeypatch.setattr(exp, 'leggi_vettori', forbidden)
    first = exp.select(train, metadata, header_path=tmp_path / 'a.h', output_dir=tmp_path / 'a')
    fake_test.write_text('DIFFERENT input values, margins, predictions, and labels')
    second = exp.select(train, metadata, header_path=tmp_path / 'b.h', output_dir=tmp_path / 'b')
    fake_test.unlink()
    third = exp.select(train, metadata, header_path=tmp_path / 'c.h', output_dir=tmp_path / 'c')
    assert first['selected_L'] == second['selected_L'] == third['selected_L']
    assert first['evidence'] == second['evidence'] == third['evidence']
    assert (tmp_path / 'a.h').read_bytes() == (tmp_path / 'b.h').read_bytes() == (tmp_path / 'c.h').read_bytes()
    assert (tmp_path / 'a/lut_vs_coeff_calibration.csv').read_bytes() == (tmp_path / 'b/lut_vs_coeff_calibration.csv').read_bytes()


def test_selection_rejects_test_split_and_bad_row_provenance(tmp_path):
    train, metadata, _ = _calibration(tmp_path)
    meta = json.loads(metadata.read_text())
    meta['source_split'] = 'test'
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='test is forbidden'):
        exp.select(train, metadata, header_path=tmp_path / 'out.h', output_dir=tmp_path / 'out')
    meta['source_split'] = 'train'
    meta['row_ids_sha256'] = '0' * 64
    metadata.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='row_ids hash mismatch'):
        exp.select(train, metadata, header_path=tmp_path / 'out.h', output_dir=tmp_path / 'out')
    assert not (tmp_path / 'out.h').exists()


def test_test_evaluation_cannot_mutate_frozen_selection(tmp_path):
    train, metadata, d = _calibration(tmp_path)
    out = tmp_path / 'out'
    exp.select(train, metadata, header_path=tmp_path / 'out.h', output_dir=out)
    protocol = out / 'lut_selection_protocol.json'
    frozen = protocol.read_bytes()
    frozen_header = (tmp_path / 'out.h').read_bytes()
    d['row_ids'] += 1000
    test = tmp_path / 'test.npz'
    np.savez(test, **d)
    a = exp.evaluate(protocol, test, output_dir=tmp_path / 'eval1')
    d['y_true'] = 1 - d['y_true']
    np.savez(test, **d)
    b = exp.evaluate(protocol, test, output_dir=tmp_path / 'eval2')
    assert a['results']['L'] == b['results']['L']
    assert protocol.read_bytes() == frozen
    assert (tmp_path / 'out.h').read_bytes() == frozen_header
    assert a['selected_L_changed'] is False
    with pytest.raises(FileExistsError, match='Selection is frozen'):
        exp.select(train, metadata, header_path=tmp_path / 'out.h', output_dir=out)
    with pytest.raises(FileExistsError, match='Post-freeze result exists'):
        exp.evaluate(protocol, test, output_dir=tmp_path / 'eval1')
    (tmp_path / 'out.h').write_text(frozen_header.decode() + '\n/* tampered */\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        exp.evaluate(protocol, test, output_dir=tmp_path / 'eval3')


def test_evaluation_rejects_train_test_overlap(tmp_path):
    train, metadata, _ = _calibration(tmp_path)
    out = tmp_path / 'out'
    exp.select(train, metadata, header_path=tmp_path / 'out.h', output_dir=out)
    with pytest.raises(ValueError, match='overlap'):
        exp.evaluate(out / 'lut_selection_protocol.json', train, output_dir=tmp_path / 'eval')

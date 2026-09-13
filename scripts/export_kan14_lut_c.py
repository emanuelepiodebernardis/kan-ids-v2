#!/usr/bin/env python3
"""Sample a frozen coefficient KAN; select on train, then evaluate on test.

    python scripts/export_kan14_lut_c.py select --calibration TRAIN.npz --metadata TRAIN.json
    python scripts/export_kan14_lut_c.py evaluate --protocol results/lut_selection_protocol.json --test-data TEST.npz

NPZ schema: integer Xq, CAT, row_ids; evaluation also needs binary y_true.
No implicit dataset loading, no training, no test-dependent header emission.
Historical RC3 L=257 was chosen using test-vector margins; that history is
retained. The new phase separation does not make previously viewed test data
an untouched holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from kanids import RESULTS_DIR                                  # noqa: E402
from kanids import lut as klut                                  # noqa: E402
from kanids.interpretabilita import (leggi_modello, leggi_vettori,  # noqa: E402
                                     logit as logit_coeff)

INCLUDE = _REPO / 'mcu_pio' / 'include'
SORGENTE = INCLUDE / 'kan14_coeff_int8.h'
VETTORI = INCLUDE / 'kan14_test_vectors.h'  # historical only; never read by select
USCITA = INCLUDE / 'kan14_lut_int16.h'
CANDIDATI = (9, 17, 33, 65, 129, 257, 513, 1025)
RULE = 'minimum_L_with_B_lt_min_abs_train_logit_else_max_fixed_candidate'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8', newline='\n')


def _artifact(path: Path, base: Path) -> dict:
    return {'path': Path(os.path.relpath(path, base)).as_posix(), 'sha256': sha256(path)}


def _checked_artifact(item: dict, base: Path) -> Path:
    path = (base / item['path']).resolve()
    if sha256(path) != item['sha256']:
        raise ValueError(f'Frozen artifact hash mismatch: {path}')
    return path


def leggi_npz(path: Path, m: dict, labels: bool = False) -> dict:
    """Read explicit integer inputs, with no fitting or test fallback."""
    required = ('Xq', 'CAT', 'row_ids') + (('y_true',) if labels else ())
    with np.load(path, allow_pickle=False) as f:
        if any(key not in f for key in required):
            raise ValueError(f'NPZ requires {required}')
        d = {key: np.array(f[key]) for key in required}
    if any(not np.issubdtype(a.dtype, np.integer) for a in d.values()):
        raise ValueError('Xq, CAT, row_ids and labels must contain integers')
    n = len(d['row_ids'])
    if not n or d['row_ids'].shape != (n,):
        raise ValueError('row_ids must be a nonempty one-dimensional array')
    if len(np.unique(d['row_ids'])) != n:
        raise ValueError('Repeated raw row IDs are not allowed')
    if d['Xq'].shape != (n, m['NFEAT']) or d['CAT'].shape != (n, m['NCAT']):
        raise ValueError('Input shape does not match the frozen coefficient model')
    if np.any(d['Xq'] < -klut.Q12) or np.any(d['Xq'] > klut.Q12):
        raise ValueError('Xq must lie on the integer Q12 grid [-4096,4096]')
    cards = np.r_[m['CAT_OFF'][1:], len(m['CAT'])] - m['CAT_OFF']
    if np.any(d['CAT'] < 0) or np.any(d['CAT'] >= cards):
        raise ValueError('Categorical code outside the frozen vocabulary')
    if labels and (d['y_true'].shape != (n,) or not np.isin(d['y_true'], [0, 1]).all()):
        raise ValueError('y_true must contain one binary label per row')
    return {key: a.astype(np.int64) for key, a in d.items()}


def tabella(m: dict, calibration: dict) -> pd.DataFrame:
    """Fixed-grid calibration sweep. Production selection requires train metadata.

    X alias permits retrospective verification of the historical RC3 table;
    it is not used by the train-selection CLI.
    """
    xq = calibration['Xq'] if 'Xq' in calibration else calibration['X']
    z = logit_coeff(m, xq, calibration['CAT'])
    margine = int(np.abs(z).min())
    righe = []
    for L in CANDIDATI:
        lu = klut.campiona(m, L)
        dev = klut.deviazione_esaustiva(lu, m)
        zl = klut.logit(lu, m, xq, calibration['CAT'])
        limite = int(dev.sum())
        righe.append({
            'L': L, 'byte_modello': klut.byte_modello(lu, m),
            'deviazione_max_edge': int(dev.max()),
            'limite_scostamento_logit': limite,
            'margine_minimo_osservato': margine,
            'vettori_entro_il_limite': int((np.abs(z) <= limite).sum()),
            'decisioni_identiche': int(((zl >= 0) == (z >= 0)).sum()),
            'scostamento_max_osservato': int(np.abs(zl - z).max()),
            'decisioni_garantite': bool(limite < margine),
        })
    return pd.DataFrame(righe)


def scegli(t: pd.DataFrame) -> int:
    """Bound-based rule; empirical test agreement cannot affect the result."""
    garantiti = t[t.decisioni_garantite]
    return int(t.L.max() if garantiti.empty else garantiti.L.min())


def select(calibration_path: Path, metadata_path: Path, *, model_path: Path = SORGENTE,
           header_path: Path = USCITA, output_dir: Path = RESULTS_DIR) -> dict:
    """Freeze training-only selection. Never open a test header or dataset."""
    calibration_path, metadata_path = map(Path, (calibration_path, metadata_path))
    model_path, header_path, output_dir = map(Path, (model_path, header_path, output_dir))
    protocol_path = output_dir / 'lut_selection_protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Selection is frozen; use a new output directory for a versioned revision')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('source_split') != 'train':
        raise ValueError("Selection requires source_split='train'; test is forbidden")
    if not isinstance(metadata.get('source_csv_sha256'), str) or len(metadata['source_csv_sha256']) != 64:
        raise ValueError('Metadata requires source_csv_sha256')
    if not metadata.get('row_ids_sha256') or not metadata.get('preprocessing'):
        raise ValueError('Metadata requires row_ids_sha256 and preprocessing provenance')
    if metadata.get('calibration_npz_sha256', sha256(calibration_path)) != sha256(calibration_path):
        raise ValueError('Calibration NPZ differs from its preprocessing metadata')
    if metadata.get('model_header_sha256', sha256(model_path)) != sha256(model_path):
        raise ValueError('Calibration preprocessing metadata references a different model header')
    m = leggi_modello(model_path)
    calibration = leggi_npz(calibration_path, m)
    ids_hash = hashlib.sha256(calibration['row_ids'].astype('<i8').tobytes()).hexdigest()
    if metadata['row_ids_sha256'] != ids_hash:
        raise ValueError('Calibration row_ids hash mismatch (little-endian int64 bytes)')
    t = tabella(m, calibration)
    L = scegli(t)
    r = t[t.L == L].iloc[0]
    lu = klut.campiona(m, L)
    n = len(calibration['row_ids'])
    note = (
        f'/* KAN-IDS binary 14-feature SAMPLED-LUT of the frozen coefficient model\n'
        f' * ({klut.byte_modello(lu, m)} B di modello; stored parameter arrays).\n'
        f' * Selected on TRAIN calibration only: L={L}, n={n}.\n'
        f' * Rule: {RULE}.\n'
        f' * Exhaustive per-edge Q12 deviation sum B={int(r.limite_scostamento_logit)}.\n'
        f' * Minimum calibration |coefficient logit|={int(r.margine_minimo_osservato)};\n'
        f' * calibration rows with |logit|<=B: {int(r.vettori_entro_il_limite)}.\n'
        f' * Bound certifies decision agreement only when |z_coeff(x)|>B,\n'
        f' * with identical categorical terms and exact non-overflowing arithmetic.\n'
        f' * It does not certify classification correctness or all future inputs.\n'
        f' * Test evaluation follows the separate frozen selection protocol.\n'
        f' * Existing coefficient golden predictions are unchanged. */')
    output_dir.mkdir(parents=True, exist_ok=True)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / 'lut_vs_coeff_calibration.csv'
    t.to_csv(sweep_path, index=False, lineterminator='\n')
    header_path.write_text(klut.header(lu, m, note), encoding='utf-8', newline='\n')
    protocol = {
        'schema_version': 1, 'status': 'FROZEN_BEFORE_TEST_EVALUATION',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_split': 'train', 'source_csv_sha256': metadata['source_csv_sha256'],
        'row_ids_sha256': ids_hash, 'row_ids_hash_encoding': 'ordered little-endian int64 raw bytes',
        'n_calibration': n, 'candidate_grid': list(CANDIDATI), 'selection_rule': RULE,
        'selection_uses_labels': False, 'selection_reads_test': False, 'selected_L': L,
        'fallback_max_candidate': not bool(r.decisioni_garantite),
        'evidence': {
            'bound_B': int(r.limite_scostamento_logit),
            'minimum_abs_calibration_logit': int(r.margine_minimo_osservato),
            'calibration_rows_not_certified': int(r.vettori_entro_il_limite),
            'calibration_certified_fraction': 1 - int(r.vettori_entro_il_limite) / n,
            'calibration_decisions_identical': int(r.decisioni_identiche),
            'all_calibration_decisions_certified': bool(r.decisioni_garantite),
            'model_bytes': klut.byte_modello(lu, m),
        },
        'bound_scope': {
            'grid': 'each numeric edge at every integer Q12 value [-4096,4096]',
            'formula': 'B = sum_i max_x |edge_lut_i(x) - edge_coeff_i(x)|',
            'decision_condition': 'abs(z_coeff(x)) > B',
            'assumptions': ['identical categorical terms and valid category codes',
                            'specified integer rounding and arithmetic right shifts',
                            'no overflow or undefined signed arithmetic in the C implementation'],
            'not_guaranteed': ['classification correctness', 'agreement for abs(z_coeff)<=B',
                               'all future decisions'],
        },
        'historical_note': 'RC3 L=257 used test-vector margins; its results remain historical. '
                           'The previously examined test set is not a new untouched holdout.',
        'artifacts': {
            'coefficient_header': _artifact(model_path, output_dir),
            'lut_header': _artifact(header_path, output_dir),
            'calibration_npz': _artifact(calibration_path, output_dir),
            'calibration_metadata': _artifact(metadata_path, output_dir),
            'calibration_sweep': _artifact(sweep_path, output_dir),
        },
    }
    _write_json(protocol_path, protocol)
    print(f'FROZEN L={L}, bytes={klut.byte_modello(lu,m)}, '
          f'calibration certified={n-int(r.vettori_entro_il_limite)}/{n}; '
          f'fallback={protocol["fallback_max_candidate"]}; no test data read')
    return protocol


def evaluate(protocol_path: Path, test_path: Path, *, output_dir: Path = RESULTS_DIR) -> dict:
    """Evaluate one frozen L; disagreement is evidence, never a reselection trigger."""
    from sklearn.metrics import f1_score

    protocol_path, test_path, output_dir = map(Path, (protocol_path, test_path, output_dir))
    protocol_hash = sha256(protocol_path)
    p = json.loads(protocol_path.read_text(encoding='utf-8'))
    if p.get('status') != 'FROZEN_BEFORE_TEST_EVALUATION' or p.get('selection_reads_test') is not False:
        raise ValueError('A train-only frozen selection protocol is required')
    artifacts = {k: _checked_artifact(v, protocol_path.parent) for k, v in p['artifacts'].items()}
    m = leggi_modello(artifacts['coefficient_header'])
    lu = klut.campiona(m, int(p['selected_L']))
    text = artifacts['lut_header'].read_text(encoding='utf-8')
    if text != klut.header(lu, m, text[:text.index('#pragma once')].rstrip('\n')):
        raise ValueError('Frozen LUT header does not match the selected coefficient model and L')
    test = leggi_npz(test_path, m, labels=True)
    calibration = leggi_npz(artifacts['calibration_npz'], m)
    if np.intersect1d(test['row_ids'], calibration['row_ids']).size:
        raise ValueError('Test and calibration row IDs overlap')
    z = logit_coeff(m, test['Xq'], test['CAT'])
    zl = klut.logit(lu, m, test['Xq'], test['CAT'])
    bound = int(klut.deviazione_esaustiva(lu, m).sum())
    if int(np.abs(zl - z).max()) > bound:
        raise ValueError('Observed deviation violates the exhaustive edge bound')
    row = {
        'n_test': len(z), 'L': int(lu['L']),
        'decisioni_identiche': int(((z >= 0) == (zl >= 0)).sum()),
        'decisioni_diverse': int(((z >= 0) != (zl >= 0)).sum()),
        'flussi_entro_il_limite': int((np.abs(z) <= bound).sum()),
        'limite_scostamento_logit': bound,
        'scostamento_max_osservato': int(np.abs(zl - z).max()),
        'f1_coefficienti': float(f1_score(test['y_true'], z >= 0, zero_division=0)),
        'f1_lut': float(f1_score(test['y_true'], zl >= 0, zero_division=0)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'lut_vs_coeff_postfreeze_test.csv'
    result_path = output_dir / 'lut_postfreeze_test_protocol.json'
    if csv_path.exists() or result_path.exists():
        raise FileExistsError('Post-freeze result exists; preserve it and version any rerun')
    pd.DataFrame([row]).to_csv(csv_path, index=False, lineterminator='\n')
    result = {
        'schema_version': 1, 'phase': 'postfreeze_test_evaluation',
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'selection_protocol_sha256': protocol_hash, 'test_npz_sha256': sha256(test_path),
        'test_row_ids_sha256': hashlib.sha256(test['row_ids'].astype('<i8').tobytes()).hexdigest(),
        'selected_L_changed': False, 'results': row, 'results_csv_sha256': sha256(csv_path),
        'measurement_scope': 'host integer simulation; not MCU timing or energy',
    }
    if sha256(protocol_path) != protocol_hash:
        raise RuntimeError('Frozen protocol changed during evaluation')
    _write_json(result_path, result)
    print(pd.DataFrame([row]).to_string(index=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='phase', required=True)
    sel = sub.add_parser('select', help='freeze representation on training calibration only')
    sel.add_argument('--calibration', type=Path, required=True)
    sel.add_argument('--metadata', type=Path, required=True)
    sel.add_argument('--model', type=Path, default=SORGENTE)
    sel.add_argument('--header', type=Path, default=USCITA)
    sel.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    ev = sub.add_parser('evaluate', help='evaluate the frozen L on test inputs')
    ev.add_argument('--protocol', type=Path, required=True)
    ev.add_argument('--test-data', type=Path, required=True)
    ev.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    args = ap.parse_args()
    if args.phase == 'select':
        select(args.calibration, args.metadata, model_path=args.model,
               header_path=args.header, output_dir=args.output_dir)
    else:
        evaluate(args.protocol, args.test_data, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

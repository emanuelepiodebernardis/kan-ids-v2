#!/usr/bin/env python3
"""Recover RC3 preprocessing without fitting classifiers; split selection from evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kanids.datasets import load_ton_iot, encode_targets
from kanids.splits import outer_split, cv_splits
from kanids.preprocessing import LeakageFreePreprocessor
from kanids.interpretabilita import _blocco, _interi, leggi_vettori


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ids_sha(a):
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def quantize(x):
    return np.rint(np.clip(x, -3.5, 3.5) / 3.5 * 4096).astype(np.int64)


def vocab(prep):
    return {c: ['UNK'] + list(prep.vocabularies_[c]) for c in prep.categorical_}


def prepare_train(csv, out):
    """Only transform training rows. Test margins/predictions are never read here."""
    if (out / 'train_calibration.json').exists():
        raise FileExistsError('Use a new output directory for a new calibration preparation.')
    df = load_ton_iot(csv)
    yb, ym, classes = encode_targets(df)
    tr, te = outer_split(ym, seed=42)
    prep = LeakageFreePreprocessor(k_numeric=10, random_state=42,
                                   selection_target='multiclass').fit(df.iloc[tr], ym[tr])
    fs = np.load(ROOT / 'models/feature_space.npz')
    if prep.numeric_features_ != fs['feats'].tolist() or prep.cardinalities_ != fs['cards'].tolist():
        raise ValueError('Recovered KAN feature order/cardinalities differ from RC3.')
    saved_vocab = json.loads((ROOT / 'models/vocabolari_categorici.json').read_text(encoding='utf-8'))['vocabolari']
    if vocab(prep) != saved_vocab:
        raise ValueError('Recovered KAN category vocabulary differs from RC3.')
    X, C = prep.transform(df.iloc[tr])
    joblib.dump(prep, out / 'kan_preprocessor.joblib')
    np.savez_compressed(out / 'kan_preprocessor_arrays.npz',
                        quantiles=prep.quantile_.quantiles_, references=prep.quantile_.references_,
                        train_row_ids=tr, log_mask=prep._log_mask_,
                        numeric_features=np.array(prep.numeric_features_))
    np.savez_compressed(out / 'train_calibration.npz', Xq=quantize(X), CAT=C,
                        row_ids=tr, y_true=yb[tr])
    meta = {
        'source_split': 'train', 'source_csv_sha256': sha(csv),
        'source_csv_bytes': csv.stat().st_size, 'source_rows': len(df),
        'source_columns': df.columns.tolist(),
        'reader': 'pandas.read_csv(low_memory=False), default NA handling; original row order, no row filtering',
        'row_id_definition': 'zero-based pandas data-row position, excluding CSV header',
        'source_split_rule': 'outer_split(type, seed=42, test_size=0.2); sorted indices',
        'selection_policy': 'All 168834 original training rows; no test scores or labels used for LUT selection.',
        'n_rows': len(tr), 'row_ids_sha256': ids_sha(tr),
        'row_ids_hash_encoding': 'contiguous little-endian signed int64',
        'test_row_ids_sha256': ids_sha(te),
        'calibration_npz_sha256': sha(out / 'train_calibration.npz'),
        'preprocessing': prep.summary(), 'vocabolari': vocab(prep),
        'preprocessing_hashes': {p.name: sha(p) for p in [out / 'kan_preprocessor.joblib', out / 'kan_preprocessor_arrays.npz']},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'classifier_fits': 0,
        'history': 'Reconstructed from supplied CSV and RC3 code; original fitted transform was not saved. Feature order and vocab verified; golden input identity checked only after LUT freeze.',
        'model_header_sha256': sha(ROOT / 'mcu_pio/include/kan14_coeff_int8.h'),
    }
    dump(out / 'train_calibration.json', meta)
    print(json.dumps({'stage': 'train_only', 'n_train': len(tr), 'n_test_untransformed': len(te),
                      'features': prep.numeric_features_, 'classifier_fits': 0,
                      'calibration_sha256': meta['calibration_npz_sha256']}, indent=2))


def golden_selection(labels, seed):
    rs = np.random.RandomState(seed)
    return np.r_[rs.choice(np.flatnonzero(labels == 1), 100, replace=False),
                 rs.choice(np.flatnonzero(labels == 0), 100, replace=False)]


def validate_freeze(protocol, out):
    frozen = json.loads(protocol.read_text(encoding='utf-8'))
    if (frozen.get('status') != 'FROZEN_BEFORE_TEST_EVALUATION'
            or frozen.get('source_split') != 'train'
            or frozen.get('selection_reads_test') is not False
            or frozen.get('selected_L') not in frozen.get('candidate_grid', [])):
        raise ValueError('Not a valid frozen train-only LUT selection protocol.')
    required = {'coefficient_header', 'lut_header', 'calibration_npz',
                'calibration_metadata', 'calibration_sweep'}
    if not required.issubset(frozen.get('artifacts', {})):
        raise ValueError('Incomplete frozen artifact bindings.')
    paths = {}
    for name, item in frozen['artifacts'].items():
        p = (protocol.parent / item['path']).resolve()
        if sha(p) != item['sha256']:
            raise ValueError('Frozen artifact hash mismatch: ' + name)
        paths[name] = p
    if paths['calibration_npz'] != (out / 'train_calibration.npz').resolve() or paths['calibration_metadata'] != (out / 'train_calibration.json').resolve():
        raise ValueError('Freeze is bound to a different calibration preparation.')
    meta = json.loads(paths['calibration_metadata'].read_text(encoding='utf-8'))
    if frozen['source_csv_sha256'] != meta['source_csv_sha256'] or frozen['row_ids_sha256'] != meta['row_ids_sha256']:
        raise ValueError('Freeze/calibration source identity mismatch.')
    if sha(paths['coefficient_header']) != meta['model_header_sha256']:
        raise ValueError('Freeze uses a different coefficient model.')
    actual_L = int(re.search(r'#define\s+KLUT_L\s+(\d+)', paths['lut_header'].read_text(encoding='utf-8')).group(1))
    if actual_L != frozen['selected_L']:
        raise ValueError('Freeze and generated LUT length differ.')
    return frozen


def prepare_evaluation(csv, out, protocol):
    """Transform test only after a separately written LUT decision exists."""
    if not protocol or not protocol.is_file():
        raise ValueError('A written LUT selection protocol is required before evaluation preparation.')
    frozen = validate_freeze(protocol, out)
    meta = json.loads((out / 'train_calibration.json').read_text(encoding='utf-8'))
    if sha(csv) != meta['source_csv_sha256']:
        raise ValueError('CSV changed since calibration.')
    for name, h in meta['preprocessing_hashes'].items():
        if sha(out / name) != h:
            raise ValueError('Frozen preprocessing changed: ' + name)
    prep = joblib.load(out / 'kan_preprocessor.joblib')
    df = load_ton_iot(csv)
    yb, ym, _ = encode_targets(df)
    tr, te = outer_split(ym, seed=42)
    Xte, Cte = prep.transform(df.iloc[te]); Qte = quantize(Xte)
    np.savez_compressed(out / 'test_evaluation.npz', Xq=Qte, CAT=Cte, row_ids=te, y_true=yb[te])
    checks = {}
    exclude = []
    for stem, prefix, seed in [('kan14', 'KTV', 1), ('kan14_ml', 'KMLTV', 2)]:
        idx = golden_selection(yb[te], seed)
        txt = (ROOT / f'mcu_pio/include/{stem}_test_vectors.h').read_text(encoding='utf-8')
        expected_x = np.array(_interi(_blocco(txt, prefix + '_X'))).reshape(200, 10)
        expected_c = np.array(_interi(_blocco(txt, prefix + '_CAT'))).reshape(200, 4)
        expected_y = np.array(_interi(_blocco(txt, prefix + '_LABEL')))
        checks[stem] = {'x_equal': bool(np.array_equal(Qte[idx], expected_x)),
                        'categories_equal': bool(np.array_equal(Cte[idx], expected_c)),
                        'labels_equal': bool(np.array_equal(yb[te[idx]], expected_y))}
        exclude.extend(te[idx].tolist())
    # DT/MLP exports use first CV fold and binary-target feature ranking, unlike KAN.
    sp = next(cv_splits(ym, seeds=(42,)))
    btr, bva = sp['train_idx'], sp['val_idx']
    bp = LeakageFreePreprocessor(k_numeric=10, random_state=42,
                                selection_target='binary').fit(df.iloc[btr], yb[btr])
    joblib.dump(bp, out / 'baseline_preprocessor.joblib')
    np.savez_compressed(out / 'baseline_preprocessor_arrays.npz',
                        quantiles=bp.quantile_.quantiles_, references=bp.quantile_.references_,
                        train_row_ids=btr, log_mask=bp._log_mask_,
                        numeric_features=np.array(bp.numeric_features_))
    bi = np.random.RandomState(42).choice(len(bva), 200, replace=False)
    exclude.extend(bva[bi].tolist())
    bx, bc = bp.transform(df.iloc[bva[bi]])
    txt = (ROOT / 'mcu_pio/include/mlp16_test_vectors.h').read_text(encoding='utf-8')
    checks['mlp16'] = {
        'x_equal': bool(np.array_equal(quantize(bx), np.array(_interi(_blocco(txt, 'MLPTV_X'))).reshape(200, 10))),
        'categories_equal': bool(np.array_equal(bc, np.array(_interi(_blocco(txt, 'MLPTV_CAT'))).reshape(200, 4))),
        'labels_equal': bool(np.array_equal(yb[bva[bi]], np.array(_interi(_blocco(txt, 'MLPTV_LABEL'))))),
    }
    txt = (ROOT / 'mcu_pio/include/dt5_model.h').read_text(encoding='utf-8')
    dg = np.array(_interi(_blocco(txt, 'DT5_GOLDEN'))).reshape(200, 16)
    checks['dt5'] = {'x_equal': bool(np.array_equal(np.rint(np.c_[bx, bc] * 128).astype(np.int64), dg[:, :14])),
                     'labels_equal': bool(np.array_equal(yb[bva[bi]], dg[:, 15]))}
    dump(out / 'preprocessing_recovery_checks.json', checks)
    if not all(all(c.values()) for c in checks.values()):
        raise ValueError('Recovered preprocessing does not reproduce RC3 golden inputs; see checks JSON. No cohort exported.')
    eligible = np.setdiff1d(np.intersect1d(te, bva), np.asarray(exclude))
    rs = np.random.RandomState(20260908)
    attack = rs.choice(eligible[yb[eligible] == 1], 250, replace=False)
    normal = rs.choice(eligible[yb[eligible] == 0], 250, replace=False)
    cohort = np.column_stack([attack, normal]).ravel()
    kx, kc = prep.transform(df.iloc[cohort]); bx, bc = bp.transform(df.iloc[cohort])
    np.savez_compressed(out / 'hardware_cohort.npz', row_ids=cohort, y_true=yb[cohort],
                        kan_Xq=quantize(kx), kan_CAT=kc,
                        baseline_Xq=quantize(bx), baseline_CAT=bc,
                        dt_Xq=np.rint(np.c_[bx, bc] * 128).astype(np.int64))
    pd.DataFrame({'order': np.arange(500), 'raw_row_id': cohort, 'label': yb[cohort],
                  'type': df.iloc[cohort]['type'].to_numpy()}).to_csv(out / 'hardware_flow_ids.csv', index=False, lineterminator='\n')
    raw = df.iloc[cohort].copy()
    raw.insert(0, 'raw_row_id', cohort)
    raw.insert(0, 'order', np.arange(500))
    raw.to_csv(out / 'hardware_raw_flows.csv', index=False, lineterminator='\n')
    cohort_meta = {
        'source_csv_sha256': sha(csv), 'source_rows': len(df), 'n_unique_flows': 500,
        'row_id_definition': meta['row_id_definition'], 'row_ids_sha256': ids_sha(cohort),
        'selection': 'RandomState(20260908), 250 attack and 250 normal; interleaved attack/normal. Candidates are intersection of KAN outer test and baseline CV validation; excludes all original binary golden rows. No scores used.',
        'n_eligible': len(eligible), 'npz_sha256': sha(out / 'hardware_cohort.npz'),
        'raw_flows_sha256': sha(out / 'hardware_raw_flows.csv'),
        'kan_preprocessing': prep.summary(), 'baseline_preprocessing': bp.summary(),
        'baseline_vocabolari': vocab(bp),
        'preprocessing_hashes': {p.name: sha(p) for p in out.glob('*preprocessor*')},
        'lut_freeze_protocol_sha256': sha(protocol),
        'golden_input_recovery': checks, 'classifier_fits': 0,
        'energy_cache': 'First 20 interleaved rows (10 attack + 10 normal), repeated; separate warm-cache workload.',
        'statistics': 'Hardware workload only. Balanced subset quality is not full-test performance.',
        'history': 'Dataset has been inspected historically; this operation does not make it historically untouched.',
    }
    dump(out / 'hardware_cohort.json', cohort_meta)
    dump(out / 'test_evaluation.json', {'source_split': 'test', 'source_csv_sha256': sha(csv),
                                      'n_rows': len(te), 'row_ids_sha256': ids_sha(te),
                                      'npz_sha256': sha(out / 'test_evaluation.npz'),
                                      'lut_freeze_protocol_sha256': sha(protocol)})
    print(json.dumps({'stage': 'postfreeze_evaluation_preparation', 'golden_checks': checks,
                      'cohort_unique': len(np.unique(cohort)), 'n_test': len(te),
                      'classifier_fits': 0}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['train', 'evaluation'])
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/finalization')
    ap.add_argument('--lut-protocol', type=Path)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'train': prepare_train(a.csv, a.out)
    else: prepare_evaluation(a.csv, a.out, a.lut_protocol)


if __name__ == '__main__':
    main()

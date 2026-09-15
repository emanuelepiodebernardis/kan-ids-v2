#!/usr/bin/env python3
"""Reproduce the post-hoc exact-raw-row overlap audit without fitting a model.

Requires numpy, pandas, and g++. Default inputs are the accompanying evidence/
directory. Equality uses all 44 parsed raw CSV columns, including labels,
addresses and attack type, with pandas' historical default NA semantics. The
non_overlapping subgroup means only no exact match to any training row under
that criterion; it is not a host-disjoint, time-disjoint or deduplicated split.
L=1025 is frozen deployment; L=513 is diagnostic, never a selection change.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_array(text, name, shape=None):
    match = re.search(r'\b' + name + r'\s*\[[^;=]*=\s*([^;]+);', text)
    if match is None:
        raise ValueError('Header array missing: ' + name)
    values = np.array([int(v) for v in re.findall(r'-?\d+', match[1])], dtype=np.int64)
    return values.reshape(shape) if shape else values


def metrics(y, pred):
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    tnr, tpr = tn / (tn + fp), tp / (tp + fn)
    return dict(n=len(y), normal=tn + fp, attack=tp + fn,
                TN=tn, FP=fp, FN=fn, TP=tp,
                F1=2 * tp / (2 * tp + fp + fn),
                BA=(tnr + tpr) / 2, TNR=tnr, TPR=tpr,
                accuracy=(tn + tp) / len(y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=Path(__file__).parent / 'evidence')
    p.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    p.add_argument('--floating-predictions', type=Path)
    a = p.parse_args()
    hashes = {}

    def read(rel):
        data = (a.evidence / rel).read_bytes()
        hashes[rel] = {'sha256': sha(data), 'bytes': len(data)}
        return data

    raw_bytes = read('raw/train_test_network.csv')
    raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    train = np.load(io.BytesIO(read('finalization/train_calibration.npz')), allow_pickle=False)
    test = np.load(io.BytesIO(read('finalization/test_evaluation.npz')), allow_pickle=False)
    historical = pd.read_csv(io.BytesIO(read('historical/canonical_test_duplicate_row_ids.csv')))
    manifest = json.loads(read('finalization/test_evaluation.json'))
    train_manifest = json.loads(read('finalization/train_calibration.json'))
    old_overlap = json.loads(read('historical/ton_overlap_summary.json'))
    read('historical/analyze_certificate.py')
    assert raw.shape == (211043, 44)
    assert sha(raw_bytes) == manifest['source_csv_sha256'] == old_overlap['raw_sha256']
    assert hashes['finalization/test_evaluation.npz']['sha256'] == manifest['npz_sha256']
    assert hashes['finalization/train_calibration.npz']['sha256'] == train_manifest['calibration_npz_sha256']
    train_ids = train['row_ids'].astype(np.int64)
    row_ids = test['row_ids'].astype(np.int64)
    y = test['y_true'].astype(np.int64)
    q = test['Xq'].astype(np.int64)
    c = test['CAT'].astype(np.int64)
    assert sha(row_ids.astype('<i8').tobytes()) == manifest['row_ids_sha256']
    assert sha(train_ids.astype('<i8').tobytes()) == train_manifest['row_ids_sha256']
    assert len(train_ids) == 168834 and len(row_ids) == 42209
    assert not np.intersect1d(train_ids, row_ids).size
    assert np.array_equal(np.sort(np.r_[train_ids, row_ids]), np.arange(len(raw)))
    assert np.array_equal(raw.iloc[row_ids]['label'].to_numpy(), y)
    assert np.array_equal(raw.iloc[train_ids]['label'].to_numpy(), train['y_true'])
    left = raw.iloc[row_ids].copy()
    left['_canonical_row_id'] = row_ids
    joined = left.merge(raw.iloc[train_ids].drop_duplicates(), on=list(raw.columns),
                        how='left', sort=False, indicator=True, validate='many_to_one')
    assert np.array_equal(joined['_canonical_row_id'], row_ids)
    duplicate = joined['_merge'].eq('both').to_numpy()
    assert duplicate.sum() == 5201
    assert np.array_equal(np.sort(row_ids[duplicate]), np.sort(historical['_canonical_row_id']))
    assert int(((y == 0) & duplicate).sum()) == 1634
    assert int(((y == 1) & duplicate).sum()) == 3567
    assert q.shape == (42209, 10) and c.shape == (42209, 4)
    assert np.all((q >= -4096) & (q <= 4096))

    h = read('headers/kan14_coeff_int8.h').decode()
    lh = read('headers/kan14_lut_int16.h').decode()
    coeff = read_array(h, 'KC_COEF', (10, 19))
    mult = read_array(h, 'KC_MULT')
    cat = read_array(h, 'KC_CAT')
    off = read_array(h, 'KC_CAT_OFF')
    cm = read_array(h, 'KC_CAT_MULT')
    assert np.array_equal(cat, read_array(lh, 'KLUT_CAT'))
    assert np.array_equal(off, read_array(lh, 'KLUT_CAT_OFF'))
    assert np.array_equal(cm, read_array(lh, 'KLUT_CAT_MULT'))
    u = np.arange(8193, dtype=np.int64)
    seg = np.minimum(u // 512, 15)
    t = (u * 16 - seg * 8192) * 4
    om = 32768 - t
    t2, t3 = t * t // 32768, (t * t // 32768) * t // 32768
    basis = np.stack([((om * om // 32768) * om) // 32768,
                      3 * t3 - 6 * t2 + 131072,
                      -3 * t3 + 3 * t2 + 3 * t + 32768, t3], axis=1)
    f = np.stack([((basis * coeff[j, seg[:, None] + np.arange(4)]).sum(1) * mult[j]) // 32768
                  for j in range(10)])
    score_cat = sum(cat[off[j] + c[:, j]] * cm[j] * 6 for j in range(4))
    score = sum(f[j, q[:, j] + 4096] for j in range(10)) + score_cat
    subgroup_masks = {'all': np.ones(len(y), dtype=bool),
                      'exact_duplicate': duplicate, 'non_overlapping': ~duplicate}
    result = {'schema_version': 1, 'scope': __doc__.strip(),
              'analysis_kind': 'post-hoc frozen-model diagnostic; no fitting or model selection',
              'classifier_fits': 0, 'hardware_executions': 0, 'selection_changed': False,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': pd.__version__},
              'overlap': {'method': 'many-to-one full-row merge on all 44 parsed raw columns; default pandas NA handling',
                          'columns': list(raw.columns), 'raw_rows': len(raw),
                          'train_rows': len(train_ids), 'test_rows': len(row_ids),
                          'duplicate_test_rows': int(duplicate.sum()),
                          'duplicate_fraction': float(duplicate.mean()),
                          'non_overlapping_test_rows': int((~duplicate).sum()),
                          'duplicate_mask_sha256_uint8_in_test_order': sha(duplicate.astype(np.uint8).tobytes()),
                          'duplicate_row_ids_sha256_i64le_in_test_order': sha(row_ids[duplicate].astype('<i8').tobytes()),
                          'historical_duplicate_ids_identical': True,
                          'train_test_observation_ids_disjoint': True,
                          'definition_warning': 'Non-overlapping under exact 44-column criterion only; not independently deduplicated, host-disjoint, time-disjoint, or historically untouched.'},
              'subgroups': {}, 'candidates': []}
    metric_rows, certificate_rows = [], []
    audit = pd.DataFrame({'test_position': np.arange(len(y)), 'canonical_row_id': row_ids,
                          'label': y, 'exact_raw44_duplicate': duplicate.astype(np.int8),
                          'coefficient_score': score, 'coefficient_prediction': (score >= 0).astype(np.int8)})
    for subgroup, mask in subgroup_masks.items():
        m = metrics(y[mask], (score[mask] >= 0).astype(np.int8))
        result['subgroups'][subgroup] = {'coefficient': m}
        metric_rows.append(dict(subgroup=subgroup, representation='coefficient', **m))
    errors = {}
    for size in [1025, 513]:
        gap = 8192 // (size - 1)
        scales, tables = [], []
        for j in range(10):
            nodes, shift = f[j, ::gap], 0
            while np.abs(np.rint(nodes / 2**shift)).max() > 32767:
                shift += 1
            scales.append(shift)
            tables.append(np.rint(nodes / 2**shift).astype(np.int64))
        tables = np.asarray(tables)
        idx = np.minimum(u // gap, size - 2)
        frac = u - idx * gap
        flut = np.stack([(tables[j, idx] + (tables[j, idx + 1] - tables[j, idx]) * frac // gap) * 2**scales[j]
                         for j in range(10)])
        err = flut - f
        errors[size] = err
        emin, emax = err.min(axis=1), err.max(axis=1)
        lower, upper = int(emin.sum()), int(emax.sum())
        bound = int(np.abs(err).max(axis=1).sum())
        if size == 1025:
            assert np.array_equal(tables, read_array(lh, 'KLUT_TAB', (10, 1025)))
            assert np.array_equal(scales, read_array(lh, 'KLUT_SHIFT'))
        lut_score = sum(flut[j, q[:, j] + 4096] for j in range(10)) + score_cat
        original_ok = np.abs(score) > bound
        signed_ok = ((score >= 0) & (score + lower >= 0)) | ((score < 0) & (score + upper < 0))
        lut_guard = ((lut_score >= 0) & (lut_score - upper >= 0)) | ((lut_score < 0) & (lut_score - lower < 0))
        mismatch = (score >= 0) != (lut_score >= 0)
        assert not mismatch[original_ok].any() and not mismatch[signed_ok].any() and not mismatch[lut_guard].any()
        assert np.all((lut_score - score >= lower) & (lut_score - score <= upper))
        audit[f'lut{size}_score'] = lut_score
        audit[f'lut{size}_prediction'] = (lut_score >= 0).astype(np.int8)
        audit[f'lut{size}_certified_original'] = original_ok.astype(np.int8)
        audit[f'lut{size}_certified_signed_reference'] = signed_ok.astype(np.int8)
        audit[f'lut{size}_certified_signed_lut_guard'] = lut_guard.astype(np.int8)
        item = {'L': size, 'deployment': size == 1025, 'model_bytes': 20 * size + 54,
                'original_symmetric_B': bound, 'signed_interval': [lower, upper],
                'edge_min_errors': emin.tolist(), 'edge_max_errors': emax.tolist(),
                'signed_lower_witness_q': (err.argmin(axis=1) - 4096).tolist(),
                'signed_upper_witness_q': (err.argmax(axis=1) - 4096).tolist(),
                'edge_error_sha256_i64le': sha(err.astype('<i8').tobytes()), 'subgroups': {}}
        for subgroup, mask in subgroup_masks.items():
            m = metrics(y[mask], (lut_score[mask] >= 0).astype(np.int8))
            result['subgroups'][subgroup][f'lut{size}'] = m
            metric_rows.append(dict(subgroup=subgroup, representation=f'lut{size}', **m))
            row = {'subgroup': subgroup, 'L': size, 'n': int(mask.sum()),
                   'coefficient_lut_disagreements': int(mismatch[mask].sum()),
                   'uncertified_original': int((mask & ~original_ok).sum()),
                   'uncertified_signed_reference': int((mask & ~signed_ok).sum()),
                   'uncertified_signed_lut_guard': int((mask & ~lut_guard).sum()),
                   'observed_max_absolute_score_error': int(np.abs(lut_score[mask] - score[mask]).max())}
            for cname, certified in [('original', original_ok), ('signed_reference', signed_ok), ('signed_lut_guard', lut_guard)]:
                for label in [0, 1]:
                    row[f'uncertified_{cname}_class{label}'] = int((mask & (y == label) & ~certified).sum())
            certificate_rows.append(row)
            item['subgroups'][subgroup] = row
        result['candidates'].append(item)

    program = r'''#include <cstdio>
#include <cstdint>
#include "kan14_coeff_infer.h"
#include "kan14_lut_infer.h"
int main(int argc, char **argv) {
  if (argc > 1) {
    int16_t x[10] = {0}; uint8_t cat[4] = {0};
    for (int j=0; j<10; ++j) {
      for (int q=-4096; q<=4096; ++q) {
        x[j]=(int16_t)q;
        int32_t d=kan14_lut_logit(x,cat)-kan14_coeff_logit(x,cat);
        std::fwrite(&d,sizeof(d),1,stdout);
      }
      x[j]=0;
    }
  } else {
    int32_t input[14];
    while (std::fread(input,sizeof(int32_t),14,stdin)==14) {
      int16_t x[10]; uint8_t cat[4];
      for (int j=0;j<10;++j) x[j]=(int16_t)input[j];
      for (int j=0;j<4;++j) cat[j]=(uint8_t)input[10+j];
      int32_t out[2]={kan14_coeff_logit(x,cat),kan14_lut_logit(x,cat)};
      std::fwrite(out,sizeof(int32_t),2,stdout);
    }
  }
}
'''
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name in ['kan14_coeff_int8.h', 'kan14_coeff_infer.h', 'kan14_lut_int16.h', 'kan14_lut_infer.h', 'q15_mul.h']:
            (td / name).write_bytes(read('headers/' + name))
        (td / 'replay.cpp').write_text(program, encoding="utf-8", newline="\n")
        subprocess.run(['g++', '-std=c++11', '-O2', str(td / 'replay.cpp'), '-o', str(td / 'replay')],
                       check=True, capture_output=True)
        raw_out = subprocess.check_output([str(td / 'replay')], input=np.c_[q, c].astype('<i4').tobytes())
        replay = np.frombuffer(raw_out, dtype='<i4').reshape(-1, 2)
        assert np.array_equal(replay[:, 0], score)
        assert np.array_equal(replay[:, 1], audit['lut1025_score'])
        probes = np.frombuffer(subprocess.check_output([str(td / 'replay'), '--probes']), dtype='<i4').reshape(10, 8193)
        err = errors[1025]
        fixed_other = err[:, 4096].sum() - err[:, 4096]
        assert np.array_equal(probes - fixed_other[:, None], err)
    result['host_verification'] = {'all_42209_coefficient_scores_match': True,
                                  'all_42209_frozen_lut_scores_match': True,
                                  '81930_coordinate_probe_errors_match': True,
                                  'program_sha256': sha(program.encode()),
                                  'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                                  'device_execution': False}
    if a.floating_predictions:
        float_data = a.floating_predictions.read_bytes()
        z = np.load(io.BytesIO(float_data), allow_pickle=False)
        assert np.array_equal(z['row_ids'], row_ids)
        assert np.array_equal(z['y_true'], y)
        float_pred = z['float_pred'].astype(np.int8)
        audit['floating_score'] = z['float_score']
        audit['floating_prediction'] = float_pred
        result['floating_source'] = {'path_name': a.floating_predictions.name,
                                     'sha256': sha(float_data),
                                     'scope': 'See accompanying floating recovery provenance; original fitted transform was not preserved.'}
        for subgroup, mask in subgroup_masks.items():
            m = metrics(y[mask], float_pred[mask])
            result['subgroups'][subgroup]['floating'] = m
            result['subgroups'][subgroup]['floating_coefficient_decision_disagreements'] = int((float_pred[mask] != (score[mask] >= 0)).sum())
            metric_rows.append(dict(subgroup=subgroup, representation='floating', **m))
    a.out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(a.out / 'canonical_test_row_audit.csv.gz', index=False, lineterminator="\n", compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(metric_rows).to_csv(a.out / 'subgroup_metrics.csv', index=False, lineterminator="\n")
    pd.DataFrame(certificate_rows).to_csv(a.out / 'subgroup_certificates.csv', index=False, lineterminator="\n")
    pd.DataFrame({'canonical_row_id': row_ids[duplicate], 'label': y[duplicate]}).to_csv(a.out / 'exact_duplicate_test_row_ids.csv', index=False, lineterminator="\n")
    result['input_sha256'] = hashes
    result['output_sha256'] = {p.name: sha(p.read_bytes()) for p in sorted(a.out.iterdir()) if p.suffix in ['.gz', '.csv']}
    result['analysis_script_sha256'] = sha(Path(__file__).read_bytes())
    (a.out / 'overlap_subgroup_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding="utf-8", newline="\n")
    print(json.dumps({'status': 'PASS', 'subgroups': result['subgroups'], 'certificates': certificate_rows}, indent=2))


if __name__ == '__main__':
    main()

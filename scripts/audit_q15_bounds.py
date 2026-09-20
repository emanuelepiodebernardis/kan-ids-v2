#!/usr/bin/env python3
"""Exact finite basis audit and conservative signed int32 bounds; no fitting.

The full t=0..32768 grid contains every first/second-layer basis input.
Arithmetic-right-shift semantics and valid categorical indices are assumed.
Frozen firmware snapshots remain historical; this audits current headers.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INT32_MAX = (1 << 31) - 1


def bases(t):
    om = 32768 - t
    t2 = t*t >> 15
    t3 = t2*t >> 15
    return [(om*om >> 15)*om >> 15,
            3*t3 - 6*t2 + (4 << 15),
            -3*t3 + 3*t2 + 3*t + (1 << 15), t3]


def integers(text, name):
    start = text.index('= {', text.index(' '+name+'[')) + 2
    level = 0
    for end in range(start, len(text)):
        level += (text[end] == '{') - (text[end] == '}')
        if level == 0:
            return [int(v) for v in re.findall(r'-?\d+', text[start:end+1])]
    raise ValueError(name)


def q15_bounds(abs_a, abs_m):
    """Ceil handles the negative arithmetic-shift endpoint, including -abs_a."""
    high_abs = ((abs_a + 32767)//32768)*abs_m
    low_product_abs = 32767*abs_m
    # This bound protects addition even before any cancellation of terms.
    sum_abs = high_abs + (low_product_abs + 32767)//32768
    for value in (abs_a, high_abs, low_product_abs, sum_abs):
        assert value <= INT32_MAX, (abs_a, abs_m, value)
    return dict(acc_abs=abs_a, multiplier_abs=abs_m,
                high_product_abs=high_abs, low_product_abs=low_product_abs,
                sum_abs=sum_abs)


def analyze():
    grid = [bases(t) for t in range(32769)]
    assert min(min(b) for b in grid) >= 0
    sums = [sum(b) for b in grid]
    # All basis evaluation intermediates are bounded by 2^30 (square/product)
    # or 6*32768 (linear combination); both are below signed int32 maximum.
    basis_product_max = max(max(t*t, (32768-t)**2,
        (t*t >> 15)*t, ((32768-t)**2 >> 15)*(32768-t))
        for t in range(32769))
    assert basis_product_max <= INT32_MAX
    report = {'scope': 'current frozen coefficient headers; finite t grid and conservative bounds, no training',
              'basis_t_count': len(grid), 'basis_sum_min': min(sums),
              'basis_sum_max': max(sums), 'basis_product_max': basis_product_max,
              'admissible_witness': {'q12': -4094, 'nseg': 16, 't': 128,
                                     'basis': bases(128), 'sum': sum(bases(128))},
              'assumptions': ['arithmetic signed right shift', 'valid categorical indices'],
              'headers': []}
    for filename, prefix in [('kan14_coeff_int8.h','KC'),
                             ('kan14_ml_coeff_int8.h','KML'),
                             ('kan14_mc_coeff_int8.h','KMC')]:
        path = REPO/'mcu_pio/include'/filename
        text = path.read_text(encoding='utf-8')
        coef_names = ['COEF'] if prefix == 'KC' else ['C1','C2']
        mult_names = ['MULT','CAT_MULT'] if prefix == 'KC' else ['M1','M2','CAT_MULT']
        coefs = [c for n in coef_names for c in integers(text, prefix+'_'+n)]
        multipliers = [m for n in mult_names for m in integers(text, prefix+'_'+n)]
        acc_abs = max(sums)*max(map(abs, coefs))
        edge = q15_bounds(acc_abs, max(map(abs, multipliers)))
        cat_abs = max(map(abs, integers(text, prefix+'_CAT')))
        cat_mult_abs = max(map(abs, integers(text, prefix+'_CAT_MULT')))
        categorical_edge_abs = cat_abs*cat_mult_abs*6
        first_layer_acc_abs = 10*edge['sum_abs'] + 4*categorical_edge_abs
        assert first_layer_acc_abs <= INT32_MAX
        row = {'path': str(path.relative_to(REPO)),
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'coefficient_abs': max(map(abs, coefs)), 'edge_q15': edge,
               'categorical_product_abs': categorical_edge_abs,
               'first_layer_acc_abs': first_layer_acc_abs}
        if prefix != 'KC':
            idx = int(re.search(r'#define\s+'+prefix+r'_IDX_MULT\s+(\d+)',text)[1])
            row['hidden_index_q15'] = q15_bounds(first_layer_acc_abs, idx)
            hid = int(re.search(r'#define\s+'+prefix+r'_HID\s+(\d+)',text)[1])
            row['second_layer_acc_abs'] = hid*edge['sum_abs']
            assert row['second_layer_acc_abs'] <= INT32_MAX
        report['headers'].append(row)
    report['status'] = 'PASS'
    return report


if __name__ == '__main__':
    out = REPO/'evidence/review_v012/q15_bounds.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analyze(), indent=2)+'\n', encoding='utf-8', newline="\n")
    print(out.relative_to(REPO))

#!/usr/bin/env python3
"""Verify review receipts and current source bindings without external raw data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
EVIDENCE=REPO/'evidence/review_v012'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    saved=read(EVIDENCE/'saved_results_summary.json')
    for item in saved['sources']:
        assert sha(REPO/item['path'])==item['sha256'], item['path']
    assert saved['unsw']['counts_by_space']=={
        'rich':{'n':60,'below_0_5':57}, 'reduced':{'n':60,'below_0_5':41}}
    assert saved['ratio_selection']['ratio5_seed_wins']==10
    bounds=read(EVIDENCE/'q15_bounds.json')
    for item in bounds['headers']:
        assert sha(REPO/item['path'])==item['sha256'], item['path']
    assert (bounds['basis_sum_min'],bounds['basis_sum_max'])==(196607,196609)
    overlap=read(EVIDENCE/'overlap/overlap_subgroup_results.json')
    assert sha(REPO/'scripts/analyze_overlap_subgroups.py')==overlap['analysis_script_sha256']
    for path,digest in overlap['output_sha256'].items():
        assert sha(EVIDENCE/'overlap'/path)==digest,path
    assert overlap['overlap']['duplicate_test_rows']==5201
    assert overlap['subgroups']['non_overlapping']['coefficient']['n']==37008
    for c in overlap['candidates']:
        assert c['subgroups']['all']['coefficient_lut_disagreements']==0
    floats=read(EVIDENCE/'float_recovery_report.json')
    for path,digest in floats['input_sha256'].items():
        if path.startswith('repository/'):
            relative=path.removeprefix('repository/')
            assert sha(REPO/relative)==digest,relative
    print('PASS: review receipts, source hashes, AUROC counts, ratio selection, Q15, overlap and frozen floating model')


if __name__=='__main__': main()

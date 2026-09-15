#!/usr/bin/env python3
"""Verify both manuscripts use the exact accepted hardware means and photographs."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
with (ROOT.parent/'experiments/hardware_energy_20260915/results/board_model_means.csv').open(encoding='utf-8',newline='') as stream:
    source={(r['board'],r['model']):r for r in csv.DictReader(stream)}
for folder,evidence in [('ieee_access','evidence/energy_pilot_results.json'),('ru','verification/ENERGY_RESULTS_V011.json')]:
    paper=ROOT/folder
    means=json.loads((paper/evidence).read_text(encoding='utf-8'))['means']
    assert len(means)==len(source)==10
    assert len({(r['board'],r['model']) for r in means})==10
    for row in means:
        reference=source[row['board'],row['model']]
        for key in ['mean_call_us','mean_power_W','energy_uJ_per_call']:
            assert math.isclose(row[key],float(reference[key]),rel_tol=1e-12),(folder,key)
    authors=json.loads((paper/'AUTHOR_METADATA.json').read_text(encoding='utf-8'))
    assert [a['name'] for a in authors['authors']]==['Oleksandr Kuznetsov','Emanuele Pio De Bernardis','Emanuele Frontoni']
for name in ['C3_measurement.jpg','Mega_measurement.jpg','energy_pilot_comparison.pdf']:
    left=(ROOT/'ieee_access/figures'/name).read_bytes()
    right=(ROOT/'ru/figures'/name).read_bytes()
    assert hashlib.sha256(left).digest()==hashlib.sha256(right).digest(),name
print('PAPER_HARDWARE_LINKS_PASS: two languages, ten means each, shared authentic photographs and comparison figure')

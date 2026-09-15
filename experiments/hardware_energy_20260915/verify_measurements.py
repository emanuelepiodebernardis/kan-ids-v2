#!/usr/bin/env python3
"""Verify the compact physical-evidence export and recompute its scalar tables.

Python >=3.11, standard library only. Default is read-only. Accepted marker
fits are retained, not refitted here. The complete checkpoint has fit scripts,
plots, actual binaries, and the independent binary/provenance audits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'analysis'))
from cfn_decoder import decode


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_files(base, entries):
    seen = set()
    for entry in entries:
        path = (base / entry['path']).resolve()
        check(path.is_relative_to(base.resolve()), 'Path escapes evidence root')
        check(entry['path'] not in seen, 'Duplicate manifest path')
        seen.add(entry['path'])
        data = path.read_bytes()
        check(len(data) == entry['bytes'], 'Size mismatch: ' + entry['path'])
        check(hashlib.sha256(data).hexdigest() == entry['sha256'],
              'SHA-256 mismatch: ' + entry['path'])


def clipped_mean_power(rows, indices, bounds):
    lo, hi = bounds
    check(rows[0][0] <= lo < hi <= rows[-1][0], 'Window outside recording')
    areas = []
    for left, right in zip(rows, rows[1:]):
        check(right[0] > left[0], 'Nonincreasing recorded time')
        a, b = max(lo, left[0]), min(hi, right[0])
        if b <= a:
            continue
        p0 = left[indices[0]] * left[indices[1]]
        p1 = right[indices[0]] * right[indices[1]]
        pa = p0 + (p1-p0) * (a-left[0]) / (right[0]-left[0])
        pb = p0 + (p1-p0) * (b-left[0]) / (right[0]-left[0])
        areas.append((pa+pb) * (b-a) / 2)
    return math.fsum(areas) / (hi-lo)


def calculate():
    result = []
    index = read_json(ROOT/'RUN_INDEX.json')
    check(len(index['runs']) == 20, 'Expected twenty acquisitions')
    identities = set()
    for run in index['runs']:
        identity = (run['board'], run['model'], run['repeat'])
        check(identity not in identities, 'Repeated acquisition identity')
        identities.add(identity)
        folder = ROOT / run['directory']
        record = read_json(folder/'RUN_RECORD.json')
        accepted = read_json(folder/'accepted_analysis.json')
        done = record['result']['done']
        check(record['run_id'] == run['run_id'], 'Run identity mismatch')
        check(done['correct'] == done['timing_ok'] == 1, 'Failed serial gate')
        check(done['checksum'] == done['expected'], 'Checksum mismatch')
        check(done['count'] > 0 and done['count'] % 20 == 0, 'Invalid batch size')
        events = {e['name']:e['us'] for e in record['result']['events']}
        check(len(events) == 14, 'Expected fourteen firmware events')
        check(events['active_end']-events['active_begin'] == done['active_us'],
              'Firmware duration differs from event duration')
        metadata, rows = decode(folder/'recording.cfn')
        indices = {c['code']:i+1 for i,c in enumerate(metadata['channels'])}
        check(metadata['sample_rate_header_sps'] == 10, 'Unexpected recording rate')
        bounds = run['cfn_bounds_s']
        check(bounds == accepted['central_window']['cfn_bounds_s'], 'Window changed')
        power = clipped_mean_power(rows, indices, bounds)
        check(math.isclose(power, accepted['central_window']['mean_power_W'],
                           rel_tol=1e-11), 'Power recomputation mismatch')
        duration = done['active_us'] / 1e6
        latency = done['active_us'] / done['count']
        energy = power * latency
        check(math.isclose(latency, accepted['protocol']['mean_call_us'],
                           rel_tol=1e-11), 'Latency mismatch')
        check(math.isclose(energy / 1000, accepted['energy_estimate']['mJ_per_call'],
                           rel_tol=1e-11), 'Energy mismatch')
        result.append(dict(board=run['board'], model=run['model'], repeat=run['repeat'],
                           count=done['count'], active_seconds_MCU=duration,
                           mean_call_us=latency, central_mean_power_W=power,
                           energy_uJ_per_call=energy))
    means = []
    for board in ['Mega 2560','ESP32-C3']:
        for model in ['coeff','lut','mlp','kanml','dt5']:
            pair = [r for r in result if r['board']==board and r['model']==model]
            check(len(pair)==2 and {r['repeat'] for r in pair}=={1,2}, 'Incomplete pair')
            means.append(dict(board=board,model=model,acquisitions=2,
                              mean_call_us=statistics.mean(r['mean_call_us'] for r in pair),
                              mean_power_W=statistics.mean(r['central_mean_power_W'] for r in pair),
                              energy_uJ_per_call=statistics.mean(r['energy_uJ_per_call'] for r in pair)))
    return result, means


def verify_table(name, expected):
    with (ROOT/'results'/name).open(encoding='utf-8', newline='') as stream:
        actual = list(csv.DictReader(stream))
    check(len(actual)==len(expected), 'Unexpected table length: '+name)
    for a,b in zip(actual,expected):
        check(set(a)==set(b), 'Unexpected columns: '+name)
        for key, value in b.items():
            check(a[key] == value if isinstance(value,str)
                  else math.isclose(float(a[key]),value,rel_tol=1e-11),
                  'Table differs: '+name+' '+key)


def write_table(name, rows):
    with (ROOT/'results'/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-results',action='store_true',help='Rewrite only the two scalar CSVs; default validates without writes')
    parser.add_argument('--checkpoint',type=Path,help='Optional full checkpoint ZIP to verify against its external reference')
    args=parser.parse_args()
    check_files(ROOT,read_json(ROOT/'IMPORTED_FILES.json')['files'])
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists() and not args.write_results:
        check_files(ROOT,read_json(manifest)['files'])
    for firmware in [ROOT/'firmware/c3',*(ROOT/'firmware/mega').iterdir()]:
        if firmware.is_dir():
            check_files(firmware,read_json(firmware/'SOURCE_MANIFEST.json')['files'])
    rows,means=calculate()
    for name,values in [('all_20_acquisitions.csv',rows),('board_model_means.csv',means)]:
        if args.write_results:
            write_table(name,values)
        else:
            verify_table(name,values)
    if args.checkpoint:
        reference=read_json(ROOT/'CHECKPOINT_REFERENCE.json')
        data=args.checkpoint.read_bytes()
        check(len(data)==reference['bytes'] and hashlib.sha256(data).hexdigest()==reference['sha256'], 'Full checkpoint identity mismatch')
    print('HARDWARE_ENERGY_PASS: 20 CFNs, 20 serial records, six source packages, ten means')
    print('Accepted marker fits and central 60-MCU-second windows retained; no new measurement or instrument calibration.')

if __name__ == '__main__':
    main()

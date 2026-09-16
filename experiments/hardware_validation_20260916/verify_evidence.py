#!/usr/bin/env python3
"""Verify retained v0.15 numerical evidence without training or a connected board.

Default mode recomputes summaries from committed per-run/per-seed records.
It does not claim to replay omitted raw ZIPs, fit models or certify instruments.
Use the experiment-specific replay scripts for the external original archives.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def require(ok, message):
    if not ok: raise ValueError(message)
def close(a, b): return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
def confusion(row):
    tn, fp, fn, tp = [row[k] for k in ('tn', 'fp', 'fn', 'tp')]
    require(tn + fp == row['normal'] and fn + tp == row['attack'], 'Class support')
    require(tn + fp + fn + tp == row['n'], 'Row support')
    return {'f1': 2*tp/(2*tp+fp+fn), 'balanced_accuracy': .5*(tn/(tn+fp)+tp/(tp+fn)), 'fpr': fp/(tn+fp), 'tpr': tp/(tp+fn)}

def verify_pair():
    data = read(ROOT/'paired_software/PAIR_STAGE2_SUMMARY.json')
    require(data['test_rows_per_seed'] == 38087 and data['independent_test_datasets'] == 1, 'Pair split scope')
    require(data['frozen_hardware_models_changed'] is False and data['integer_certificate_applies_to_new_models'] is False, 'New fits must not inherit hardware/certificate')
    for name, model in data['models'].items():
        rows = model['per_seed']; require(len(rows) == 5, name+' seed count')
        for row in rows:
            for metric, value in confusion(row).items(): require(close(value, row[metric]), name+' confusion '+metric)
        for metric, recorded in model['metrics'].items():
            values = [r[metric] for r in rows]
            require(close(statistics.mean(values), recorded['mean']), name+' mean '+metric)
            require(close(statistics.stdev(values), recorded['sample_sd_ddof1']), name+' SD '+metric)
        for missing in ('backdoor', 'ransomware', 'xss'):
            require(all(v['n_defined'] == 0 for v in model['by_type'][missing].values()), 'Absent attack type undefined')
    for filename in ('PAIR_INTEGRITY_AUDIT.json','SAVED_PREDICTION_AUDIT.json'):
        require(read(ROOT/'paired_software'/filename)['status'] == 'PASS', filename)
    return {'models':4,'fits':20,'test_rows':38087,'independent_splits':1,'mode':'retained_per_seed_counts_and_metrics; AUROC summarised, not reranked'}

def verify_hw500():
    data = read(ROOT/'hw500/HW500_PUBLICATION_DATA.json')
    with (ROOT/'hw500/HW500_RUNS.csv').open(newline='') as f: rows = list(csv.DictReader(f))
    require(len(rows) == 52 and data['cohort']['distinct_flows'] == 500, 'HW500 support')
    require(data['primary_idle_subtraction'] is False and data['calibrated'] is False and data['NRG_used'] is False, 'HW500 energy scope')
    for board, summary in data['boards'].items():
        require(summary['included_runs'] == 25 and summary['excluded_pilots'] == 1, 'HW500 pilot exclusion')
        for mean in summary['means']:
            group = [r for r in rows if r['board'] == board and r['model'] == mean['model'] and r['phase'] == 'campaign']
            require(len(group) == 5, 'HW500 five technical repeats')
            for metric in ('latency_us', 'USB_power_W', 'USB_energy_uJ'):
                values = [float(r[metric]) for r in group]
                require(close(statistics.mean(values), mean[metric+'_mean']), board+' mean '+metric)
                require(close(statistics.stdev(values), mean[metric+'_sample_SD']), board+' SD '+metric)
            for row in group:
                require(int(row['N']) % 500 == 0, 'Whole cohort traversals')
                require(close(float(row['active_MCU_s'])*1e6/int(row['N']), float(row['latency_us'])), 'Batch timing normalization')
    return {'raw_intervals':52,'publication_intervals':50,'separate_pilots':2,'mode':'retained_per_run_numbers; CFN not replayed'}

def verify_ram():
    spec = importlib.util.spec_from_file_location('observed_ram', ROOT/'ram500/verify_ram.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    table = module.recompute(); require(table == read(ROOT/'ram500/OBSERVED_RAM_TABLE.json'), 'RAM table')
    return {'uploads':30,'reference_comparisons':45000,'distinct_flows':500,'total_peak_claim':False,'mode':'retained_records_and_acceptance_receipts; raw ZIPs not replayed'}

def verify_manifest():
    path = ROOT/'EVIDENCE_MANIFEST.json'
    for item in read(path)['files']:
        p=ROOT/item['path']; require(p.is_file(), 'Missing '+item['path'])
        require(hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'] and p.stat().st_size==item['bytes'], 'Identity '+item['path'])
    return len(read(path)['files'])

def verify_source_origins():
    checked = {}
    for kind, script in (("ram500", "run_ram.py"), ("hw500", "run_suite.py")):
        directory = ROOT/kind/"runner"
        origin = read(directory/"SOURCE_ORIGIN.json")
        require(hashlib.sha256((directory/"project/src/main.cpp").read_bytes()).hexdigest() == origin["accepted_firmware_main"]["sha256"], kind+" accepted firmware identity")
        require(hashlib.sha256((directory/"UPSTREAM_MANIFEST.json").read_bytes()).hexdigest() == origin["upstream_manifest"]["sha256"], kind+" original release manifest identity")
        spec = importlib.util.spec_from_file_location("verify_"+kind, directory/script)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        checked[kind] = len(module.verify_sources(directory))
    return checked

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ram-artifacts-dir',type=Path);a=p.parse_args()
    result={'status':'PASS','physical_experiments_executed':False,'training_executed':False,
            'manifest_files_checked':verify_manifest(),'source_manifest_files':verify_source_origins(),'paired_software':verify_pair(),'hw500':verify_hw500(),'ram500':verify_ram()}
    if a.ram_artifacts_dir:
        subprocess.run([sys.executable,str(ROOT/'ram500/verify_ram.py'),'--artifacts-dir',str(a.ram_artifacts_dir)],check=True)
        result['ram500']['external_RAM_UART_replay']='PASS'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

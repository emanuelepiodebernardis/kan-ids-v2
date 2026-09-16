#!/usr/bin/env python3
"""Aggregate accepted run-level observations, never individual meter samples."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import statistics
from run_suite import ACCEPTED, MODELS, digest, plan_for, read_json, verify_packet, write_json

FIELDS = ['board', 'model', 'phase', 'block', 'run_id', 'N', 'full_traversals',
          'active_MCU_s', 'latency_us', 'USB_power_W', 'USB_energy_uJ',
          'idle_before_W', 'idle_after_W', 'incremental_uJ_diagnostic',
          'registration_max_residual_s', 'registration_shift_sensitivity_percent']
METRICS = ['latency_us', 'USB_power_W', 'USB_energy_uJ']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize_session(directory):
    directory = Path(directory).resolve()
    session = read_json(directory / 'SUITE_RECORD.json')
    require(session.get('schema') == 'kanids-hw500-suite-v1', 'Wrong session schema')
    board = session['board_kind']
    require(board in ('mega', 'c3'), 'Wrong board')
    plan = plan_for(board)
    require(session['plan'] == plan, 'Changed fixed acquisition plan')
    rows, seen, failed = [], set(), []
    for entry in session['attempts']:
        attempt = (directory / entry['relative_path']).resolve()
        require(attempt.is_relative_to(directory), 'Attempt outside session')
        verify_packet(attempt)
        require(digest(attempt / 'RUN_RECORD.json') == entry['record_file'], 'Changed run receipt')
        record = read_json(attempt / 'RUN_RECORD.json')
        item = record['plan_item']
        require(item in plan, 'Unknown planned acquisition')
        if record['status'] != ACCEPTED:
            failed.append({'run_id': record.get('run_id'), 'plan_item': item,
                           'error': record.get('error', 'Incomplete attempt')})
            continue
        require(item['key'] not in seen, 'Duplicate accepted planned acquisition')
        seen.add(item['key'])
        a = read_json(attempt / 'TRACE_ANALYSIS.json')
        require(a['status'] == 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_ESTIMATE', 'Unaccepted trace')
        require(a['board_kind'] == board and a['model'] == item['model'], 'Mismatched trace identity')
        require(a['run_id'] == record['run_id'], 'Trace from another run')
        require(a['source_sha256'] == record['cfn']['file']['sha256'], 'Mismatched trace hash')
        if session.get('acquisition_protocol') == 'kanids-hw500-continuous-v1':
            source = (directory / record['cfn']['session_relative_path']).resolve()
            require(source.is_relative_to(directory), 'Continuous trace outside session')
            require(source.is_file() and digest(source) == record['cfn']['file'],
                    'Original continuous CFN is missing or changed')
        work, power, diag, reg = a['workload'], a['primary_energy'], a['diagnostic_only'], a['registration']
        require(work['distinct_frozen_flows'] == 500, 'Wrong cohort count')
        n = work['completed_inferences']
        require(n == record['result']['done']['count'] and n > 0 and n % 500 == 0, 'Wrong measured count')
        row = dict(board=board, model=item['model'], phase=item['phase'], block=item['repeat'],
                   run_id=record['run_id'], N=n, full_traversals=n // 500,
                   active_MCU_s=work['active_MCU_seconds'], latency_us=work['mean_replay_us_per_inference'],
                   USB_power_W=power['mean_power_W'], USB_energy_uJ=power['uJ_per_inference'],
                   idle_before_W=a['idle_before_guard']['mean_power_W'],
                   idle_after_W=a['idle_after_guard']['mean_power_W'],
                   incremental_uJ_diagnostic=diag['incremental_uJ_per_inference'],
                   registration_max_residual_s=reg['edge_residual_max_abs_s'],
                   registration_shift_sensitivity_percent=a['maximum_absolute_energy_shift_percent'])
        require(all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float))), 'Nonfinite metric')
        rows.append(row)
    require(seen == {x['key'] for x in plan}, 'Full campaign incomplete: summary withheld')
    campaign = [x for x in rows if x['phase'] == 'campaign']
    require(len(campaign) == 25 and len(rows) == 26, 'Wrong campaign/pilot count')
    means = []
    for model in MODELS:
        group = [x for x in campaign if x['model'] == model]
        require(len(group) == 5 and {x['block'] for x in group} == set(range(1, 6)), 'Incomplete model blocks')
        out = {'board': board, 'model': model, 'technical_repeats': 5, 'physical_boards': 1}
        for metric in METRICS:
            values = [x[metric] for x in group]
            out[metric + '_mean'] = statistics.mean(values)
            out[metric + '_sample_SD'] = statistics.stdev(values)
        means.append(out)
    report = {'schema': 'kanids-hw500-summary-v1', 'status': 'FULL_25_RUN_CAMPAIGN_SUMMARIZED',
              'board': board, 'protocol_file': session['protocol_file'],
              'confirmatory_runs': 25, 'separate_pilot_runs': 1, 'retained_failed_attempts': failed,
              'mean_rows': means,
              'interpretation': 'Unweighted mean and sample SD across five run-level technical repetitions on one physical board. Pilot excluded. Whole-board USB engineering estimates; no calibrated uncertainty or independent-sample confidence interval.',
              'boundary': 'Flash prepared-row load + prediction + checksum; same active batch for latency and energy'}
    write_csv(directory / 'results_all_runs.csv', rows, FIELDS)
    write_csv(directory / 'results_confirmatory.csv', campaign, FIELDS)
    write_csv(directory / 'results_model_means.csv', means, list(means[0]))
    write_json(directory / 'SUMMARY.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    result = summarize_session(args.session)
    print(result['status'])


if __name__ == '__main__':
    main()

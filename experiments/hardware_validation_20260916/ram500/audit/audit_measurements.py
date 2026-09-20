"""Independent, read-only C3 RAM500 numerical/transport audit; Python stdlib only."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.session.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise SystemExit('Output must be outside immutable input session')
    checks = []

    def check(name, value):
        checks.append({'check': name, 'pass': bool(value)})

    def read_json(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def fields(line):
        result = {}
        for token in line.split()[1:]:
            key, value = token.split('=', 1)
            if key in result:
                raise ValueError('Duplicate field: ' + key)
            result[key] = value
        return result

    attempts = sorted((root / 'attempts').iterdir())
    check('15 attempts', len(attempts) == 15)
    summary = read_json(root / 'RAM_SUMMARY.json')
    check('summary complete', summary['complete'] and summary['runs'] == 15)
    all_tokens = []
    runs = []
    common_ids = None
    expected_checksums = {'coeff': 256, 'lut': 256, 'mlp': 254, 'kanml': 250, 'dt5': 249}
    for attempt in attempts:
        run = read_json(attempt / 'RUN_RECORD.json')
        baseline = read_json(attempt / 'BASELINE_RECORD.json')
        captures = sorted(attempt.glob('serial_connection_*'))
        name = attempt.name
        check(name + ': one connection', len(captures) == 1)
        conn = captures[0]
        raw = (conn / 'serial_raw.bin').read_bytes()
        events = [json.loads(line) for line in (conn / 'serial.jsonl').read_text().splitlines()]
        check(name + ': raw RX equals JSONL', raw == b''.join(bytes.fromhex(e['hex']) for e in events if e['direction'] == 'RX'))
        check(name + ': monotonic event timestamps', all(a['host_monotonic_ns'] <= b['host_monotonic_ns'] for a, b in zip(events, events[1:])))
        lines = [line for line in raw.decode('ascii').splitlines() if line]
        check(name + ': serial text equals raw nonempty lines', lines == (conn / 'serial.log').read_text().splitlines())
        expected_markers = ['HELLO', 'MODEL', 'COHORT', 'RAW_IDS', 'SYSTEM', 'INFO_DONE', 'RUN_BEGIN', 'REPORT_BEGIN', 'BEGIN', 'RAM_RESULT', 'DONE', 'CONTROL_BEGIN', 'CONTROL']
        check(name + ': complete frame order', [line.split()[0] for line in lines] == expected_markers)
        parsed = {line.split()[0]: fields(line) for line in lines}
        token = run['run_token']
        all_tokens.append(token)
        check(name + ': nonce syntax', re.fullmatch('[0-9a-f]{16}', token) is not None)
        check(name + ': all six received nonces match', all(parsed[key]['run_token'] == token for key in ['RUN_BEGIN', 'REPORT_BEGIN', 'RAM_RESULT', 'DONE', 'CONTROL_BEGIN', 'CONTROL']))
        tx = [bytes.fromhex(e['hex']).decode('ascii') for e in events if e['direction'] == 'TX']
        check(name + ': exact INFO RUN CHECK sequence without retransmission', tx == ['INFO\n', f'RUN {token}\n', f'CHECK {token}\n'])
        check(name + ': accepted run and baseline snapshot', run['status'] == 'accepted_observed_ram_components' and baseline['status'] == 'incomplete' and 'control' not in baseline)
        check(name + ': immutable baseline', baseline['result'] == run['result'] and baseline['baseline_serial_lines'] == run['baseline_serial_lines'])
        check(name + ': parsed baseline equals records', parsed['BEGIN'] == run['result']['begin'] and parsed['RAM_RESULT'] == run['result']['ram'] and parsed['DONE'] == run['result']['done'])
        check(name + ': parsed control equals record', parsed['CONTROL'] == run['control'])
        check(name + ': all metadata equals record', all(parsed[key] == run['info'][key] for key in ['HELLO', 'MODEL', 'COHORT', 'RAW_IDS', 'SYSTEM', 'INFO_DONE']))
        state = read_json(conn / 'TRANSPORT_STATE.json')
        check(name + ': transport state agrees with record', state == run['transport_final_state'])
        check(name + ': no pending bytes or frames', not state['pending_partial_hex'] and not state['pending_partial_ascii'] and not state['pending_complete_lines_hex'])
        check(name + ': no recovery events', [e['event'] for e in state['recovery_events']] == ['run_acknowledged', 'terminal_quiet_verified'] and run['connection_retries'] == [])
        check(name + ': baseline/control timestamps correctly ordered at millisecond resolution', run['baseline_end_utc'] <= run['control_start_utc'] < run['control_end_utc'])
        check(name + ': fresh upload for every repeat', run['fresh_boot_method'] == 'upload_each_repeat' and len(run['commands']) == 1 and run['commands'][0]['returncode'] == 0)
        ids = [int(x) for x in parsed['RAW_IDS']['values'].split(',')]
        check(name + ': 500 distinct raw row IDs', len(ids) == len(set(ids)) == 500)
        if common_ids is None:
            common_ids = ids
        check(name + ': same ordered cohort', ids == common_ids)
        ram, control = parsed['RAM_RESULT'], parsed['CONTROL']
        n = {k: int(v) for k, v in ram.items() if v.isdecimal()}
        c = {k: int(v) for k, v in control.items() if v.isdecimal()}
        model = run['model']
        check(name + ': 1500 exact reference comparisons', n['rows'] == 500 and n['passes'] == 3 and n['checked'] == 1500 and n['mismatches'] == 0 and n['correct'] == n['ram_valid'] == 1)
        check(name + ': expected checksum', n['checksum_per500'] == expected_checksums[model] and n['checksum'] == 3 * n['checksum_per500'])
        check(name + ': stack accounting', n['loop_stack_reserved_bytes'] == 8192 and n['loop_stack_observed_used_bytes'] == 8192 - n['loop_stack_hwm_min_bytes'])
        check(name + ': stack baseline already reaches final watermark', n['loop_stack_hwm_before_bytes'] == n['loop_stack_hwm_after_bytes'] == n['loop_stack_hwm_min_bytes'])
        check(name + ': heap snapshots unchanged', all(n[f'heap_{key}_before_bytes'] == n[f'heap_{key}_after_bytes'] for key in ['free', 'allocated', 'largest', 'lifetime_min']))
        check(name + ': sampled heap unchanged', n['heap_sampled_min_free_bytes'] == n['heap_free_before_bytes'])
        check(name + ': explicit measurement scope', ram['heap_domain'] == 'internal_8bit' and ram['heap_min_scope'] == 'sum_region_lifetime_minima' and ram['stack_scope'] == 'loop_task_lifetime' and n['stack_inside_heap'] == 1 and n['total_peak_claim'] == 0)
        check(name + ': positive control flags', c['stack_pass'] == c['heap_pass'] == c['correct'] == c['separate_task'] == c['baseline_already_saved'] == 1)
        check(name + ': positive control stack detects probe', c['stack_probe_bytes'] == 768 and c['stack_before_bytes'] - c['stack_after_bytes'] >= 512 and c['stack_after_bytes'] > 0)
        check(name + ': positive control heap detects and releases allocation', c['heap_probe_bytes'] == 512 and c['heap_before'] - c['heap_during'] >= 512 and c['heap_before'] == c['heap_after'])
        runs.append({'attempt': name, 'model': model, 'repeat': run['repeat'], 'run_token': token, 'ram': ram, 'control': control, 'build_ram': run['build_ram'], 'raw_sha256': hashlib.sha256(raw).hexdigest()})
    check('15 independent nonces', len(set(all_tokens)) == 15)
    check('22500 reference comparisons', sum(int(r['ram']['checked']) for r in runs) == 22500)
    extrema = []
    corrections = []
    compiler = []
    for model in expected_checksums:
        group = sorted([r for r in runs if r['model'] == model], key=lambda r: r['repeat'])
        check(model + ': exactly three repeats', [r['repeat'] for r in group] == [1, 2, 3])
        stored = next(row for row in summary['rows'] if row['model'] == model)
        for key, value in stored.items():
            if key.endswith('_maximum_observed'):
                field = key.removesuffix('_maximum_observed')
                check(model + ': summary ' + key, value == max(int(r['ram'][field]) for r in group))
            elif key.endswith('_minimum_observed'):
                field = key.removesuffix('_minimum_observed')
                check(model + ': summary ' + key, value == min(int(r['ram'][field]) for r in group))
            elif key in ['static_sram_bytes', 'diagnostic_globals_bytes', 'iram_code_bytes']:
                check(model + ': summary ' + key, all(r['build_ram'][key] == value for r in group))
        audit = read_json(root / f'builds/{model}/binaries/firmware.audit.json')
        true_globals = [s for s in audit['diagnostic_globals'] if re.fullmatch(r'_ZL\d+ram_[A-Za-z0-9_]+', s['symbol'])]
        excluded = [s for s in audit['diagnostic_globals'] if s not in true_globals]
        corrections.append({'model': model, 'as_recorded_bytes': audit['diagnostic_globals_bytes'], 'corrected_ram_named_globals_bytes': sum(s['bytes'] for s in true_globals), 'excluded_sdk_symbols': excluded, 'impact': 'diagnostic subset only; static SRAM and physical observations unchanged'})
        stack_text = (root / f'builds/{model}/binaries/stack_usage/src/main.cpp.su').read_text()
        stack_sizes = {}
        for line in stack_text.splitlines():
            parts = line.split('\t')
            for symbol in ['hw500_predict_loaded()', 'ram_active_pass()', 'ram_stack_probe()', 'loop()']:
                if symbol in parts[0]:
                    stack_sizes[symbol] = int(parts[1])
        assembly = (root / f'builds/{model}/binaries/firmware.disassembly.txt').read_text()
        bodies = dict(re.findall(r'^[0-9a-f]+ <([^\n]+)>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)', assembly, re.M | re.S))
        check(model + ': actual pass calls predictor', '<hw500_predict_loaded>' in bodies['ram_active_pass'])
        for symbol in ['ram_active_pass', 'hw500_predict_loaded', 'ram_stack_probe']:
            frame = re.search(r'addi\s+sp,sp,-(\d+)', bodies[symbol])
            actual_frame = int(frame.group(1)) if frame else 0
            check(model + ': assembly frame agrees ' + symbol, actual_frame == stack_sizes[symbol + '()'])
        compiler.append({'model': model, 'static_frame_sizes_bytes': stack_sizes, 'interpretation': 'compiler frame sizes only, not a complete call-chain or runtime peak'})
        extrema.append({'model': model, 'stack_used_per_boot_bytes': [int(r['ram']['loop_stack_observed_used_bytes']) for r in group], 'stack_used_min_bytes': min(int(r['ram']['loop_stack_observed_used_bytes']) for r in group), 'stack_used_max_bytes': max(int(r['ram']['loop_stack_observed_used_bytes']) for r in group), 'stack_growth_beyond_preinference_hwm_bytes': [int(r['ram']['loop_stack_hwm_before_bytes']) - int(r['ram']['loop_stack_hwm_min_bytes']) for r in group], 'heap_free_bytes': 309800, 'heap_allocated_bytes': 20588, 'heap_lifetime_min_free_bytes': 304820, 'heap_largest_free_block_bytes': 286708, 'static_sram_bytes': audit['static_sram_bytes'], 'iram_code_bytes': audit['iram_code_bytes']})
    report = {'schema': 'independent-c3-ram142-measurement-review-v1', 'session': root.name, 'status': 'PASS_WITH_SCOPE_LIMITATIONS_AND_METADATA_CORRECTION' if all(c['pass'] for c in checks) else 'FAIL', 'check_count': len(checks), 'failed_checks': [c for c in checks if not c['pass']], 'accepted_physical_runs': 15 if all(c['pass'] for c in checks) else None, 'reference_comparisons': 22500, 'distinct_cohort_rows': len(common_ids), 'mismatches': sum(int(r['ram']['mismatches']) for r in runs), 'transport_retries': 0, 'extrema': extrema, 'diagnostic_global_metadata_correction': corrections, 'compiler_frame_evidence': compiler, 'limitations': ['All C3 loop-task lifetime stack maxima were already reached before inference. The experiment does not resolve inference-specific stack usage or establish relative model stack efficiency.', 'Heap samples are taken before and after each 500-row pass. Unchanged values do not alone rule out a transient allocate/free pair within a pass.', 'The SDK heap lifetime minimum is a sum of region minima, not necessarily a simultaneous whole-system minimum.', 'Allocated heap already includes dynamically created task stacks. Do not add the 8192-byte loop stack or its used portion to allocated heap.', 'Static SRAM includes diagnostic globals, row buffers and the statically reserved ISR stack. Those components must not be added again.', 'IRAM code, static DRAM and heap/task components are separately scoped. No exact whole-system peak RAM is established.', 'The workload checks predictions against frozen C references; zero mismatches is export agreement, not zero classification errors.', 'Reporting recovery was not exercised by this physical run: every report arrived on its first request.'], 'runs': runs, 'checks': checks}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'C3_MEASUREMENT_REVIEW.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'check_count', 'failed_checks', 'reference_comparisons', 'mismatches', 'transport_retries']}, ensure_ascii=False))
    if report['failed_checks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

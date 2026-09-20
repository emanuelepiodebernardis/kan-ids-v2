#!/usr/bin/env python3
"""Offline, read-only evidence audit of the received C3 RAM500 v0.14.2 series.

Requires the extracted session and the independently retained released kit.
Does not execute firmware, rebuild, access a serial port, or change raw evidence.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXPECTED_RUNNER = '692927d74f13d0f00067eda2133b0e74ba97cb769d449b5b1ec8edc907f3760f'
EXPECTED_MANIFEST = '5e5cb818e016504c65b63172f3f643ad864367149426b566797566c466a5ebd0'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    b = path.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    session, kit, out = (x.resolve() for x in (args.session, args.kit, args.output))
    if out.exists() or out.is_relative_to(session) or out.is_relative_to(kit):
        raise SystemExit('Output must be new and outside the input directories')
    checks = []
    def check(label, ok):
        if not ok:
            raise AssertionError(label)
        checks.append(label)
    check('trusted runner identity', digest(kit/'run_ram.py')['sha256'] == EXPECTED_RUNNER)
    check('trusted manifest identity', digest(kit/'KIT_MANIFEST.json')['sha256'] == EXPECTED_MANIFEST)
    spec = importlib.util.spec_from_file_location('trusted_ram_runner', kit/'run_ram.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.verify_sources(kit)
    check('trusted release complete', True)
    runner.verify_packet(session)
    packet = read(session/'PACKET_MANIFEST.json')['files']
    for entry in packet:
        check('packet '+entry['path'], digest(session/entry['path']) == {k:entry[k] for k in ('bytes','sha256')})
    source_files = sorted(p for p in (session/'sources').rglob('*') if p.is_file())
    for source in source_files:
        rel = source.relative_to(session/'sources')
        check('source '+str(rel), (kit/rel).is_file() and source.read_bytes() == (kit/rel).read_bytes())
    runner.verify_sources(session/'sources')
    suite = read(session/'SUITE_RECORD.json')
    check('complete suite', suite['status'] == 'all_15_runs_accepted')
    check('plan', suite['plan'] == runner.plan_for('c3'))
    check('attempt count', len(suite['attempts']) == 15)
    check('manifest receipt', suite['source_manifest'] == digest(kit/'KIT_MANIFEST.json'))
    check('protocol receipt', suite['protocol_file'] == digest(kit/'PROTOCOL.json'))
    check('dependency pins', suite['versions'] == runner.PINS)
    for bp in (session/'bootstrap').rglob('BOOTSTRAP_RECORD.json'):
        b = read(bp)
        check('bootstrap pass', b['status'] == 'bootstrap_and_tests_pass')
        check('bootstrap commands', all(c['returncode'] == 0 for c in b['commands']))
        check('bootstrap runner tests', 'Ran 47 tests' in (bp.parent/'setup_01.log').read_text() and '\nOK\n' in (bp.parent/'setup_01.log').read_text())
    for model, build in suite['builds'].items():
        folder = session/'builds'/model/'binaries'
        check('build success '+model, all(c['returncode'] == 0 for c in build['commands']))
        check('audit receipt '+model, build['audit'] == read(folder/'firmware.audit.json'))
        runner.validate_build_audit(build['audit'], 'c3', model)
        check('build gates '+model, True)
        for name, identity in build['binaries'].items():
            check('build identity '+model+'/'+name, digest(folder/name) == {k:identity[k] for k in ('bytes','sha256')})
    records, tokens, observations = [], set(), []
    for planned, entry in zip(suite['plan'], suite['attempts']):
        folder = session/entry['path']
        record = read(folder/'RUN_RECORD.json')
        baseline = read(folder/'BASELINE_RECORD.json')
        name, model, token = planned['key'], planned['model'], record['run_token']
        check(name+' plan and model', record['plan_item'] == planned and record['model'] == model)
        check(name+' record hash', digest(folder/'RUN_RECORD.json') == entry['record'])
        check(name+' accepted', record['status'] == entry['status'] == runner.ACCEPTED)
        check(name+' fresh upload', record['fresh_boot_method'] == 'upload_each_repeat' and len(record['commands']) == 1 and record['commands'][0]['returncode'] == 0)
        for field in ['source_manifest','build_identities']:
            target = suite['source_manifest'] if field == 'source_manifest' else suite['builds'][model]['binaries']
            check(name+' '+field, record[field] == target)
        for key, value in baseline.items():
            if key != 'status':
                check(name+' saved baseline '+key, record[key] == value)
        check(name+' baseline before control', record['baseline_end_utc'] <= record['control_start_utc'] <= record['control_end_utc'])
        check(name+' nonce fresh', re.fullmatch('[0-9a-f]{16}', token) is not None and token not in tokens)
        tokens.add(token)
        check(name+' one connection', record['connection_retries'] == [] and len(list(folder.glob('serial_connection_*'))) == 1)
        for key in ['board_before_upload', 'board_after_upload']:
            observed = record[key]
            check(name+' '+key, observed['serial_number'] == '10:00:3B:CB:8D:70' and observed['vid'] == 0x303a and observed['pid'] == 0x1001)
        upload = (folder/'03_upload.log').read_text(encoding='utf-8')
        check(name+' physical flash verification', all(s in upload for s in ['Chip is ESP32-C3', 'MAC: 10:00:3b:cb:8d:70', 'Hash of data verified.', 'Hard resetting via RTS pin', '[SUCCESS]']))
        check(name+' uploaded length', 'Wrote '+str(suite['builds'][model]['binaries']['firmware.bin']['bytes'])+' bytes' in upload)
        serial = folder/'serial_connection_00'
        events = [json.loads(s) for s in (serial/'serial.jsonl').read_text().splitlines()]
        check(name+' monotonic events', all(a['host_monotonic_ns'] <= b['host_monotonic_ns'] for a,b in zip(events,events[1:])))
        rx = b''.join(bytes.fromhex(e['hex']) for e in events if e['direction'] == 'RX')
        tx = [bytes.fromhex(e['hex']).decode('ascii').strip() for e in events if e['direction'] == 'TX']
        check(name+' raw bytes equal events', rx == (serial/'serial_raw.bin').read_bytes())
        check(name+' no measurement or control retries', tx == ['INFO','RUN '+token,'CHECK '+token])
        lines = [s for s in rx.decode('ascii').splitlines() if s]
        check(name+' raw equals text log', lines == (serial/'serial.log').read_text().splitlines())
        tags = [s.split()[0] for s in lines]
        expected_tags = ['HELLO','MODEL','COHORT','RAW_IDS','SYSTEM','INFO_DONE','RUN_BEGIN','REPORT_BEGIN','BEGIN','RAM_RESULT','DONE','CONTROL_BEGIN','CONTROL']
        check(name+' exact frame order', tags == expected_tags)
        for tag in ['RUN_BEGIN','REPORT_BEGIN','CONTROL_BEGIN']:
            check(name+' frame token '+tag, lines[tags.index(tag)] == tag+' run_token='+token)
        info = runner.validate_info(lines[:6], 'c3', model, kit)
        result = runner.validate_result(lines[8:11], 'c3', model, suite['builds'][model]['audit'], token)
        control = runner.validate_control([lines[12]], 'c3', result)
        for label, parsed in [('info',info),('result',result),('control',control)]:
            check(name+' parsed '+label, parsed == record[label])
        check(name+' baseline lines', lines[8:11] == record['baseline_serial_lines'])
        state = read(serial/'TRANSPORT_STATE.json')
        check(name+' transport saved', state == record['transport_final_state'])
        check(name+' no pending bytes', state['pending_partial_hex'] == '' and state['pending_complete_lines_hex'] == [])
        check(name+' terminal quiet no recovery', [e['event'] for e in state['recovery_events']] == ['run_acknowledged','terminal_quiet_verified'])
        ram = result['ram']
        observations.append({'key':name,'model':model,'repeat':planned['repeat'],'checked':int(ram['checked']),'mismatches':int(ram['mismatches']),'stack_before_bytes':int(ram['loop_stack_hwm_before_bytes']),'stack_after_bytes':int(ram['loop_stack_hwm_after_bytes']),'stack_used_lifetime_bytes':int(ram['loop_stack_observed_used_bytes']),'heap_free_before_bytes':int(ram['heap_free_before_bytes']),'heap_free_after_bytes':int(ram['heap_free_after_bytes'])})
        records.append(record)
    recomputed = runner.make_summary(records, 'c3')
    check('summary recomputed', recomputed == read(session/'RAM_SUMMARY.json'))
    with (session/'RAM_SUMMARY.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    check('CSV recomputed', rows == [{k:str(v) for k,v in row.items()} for row in recomputed['rows']])
    output = {'schema':'kanids-c3-ram142-receipt-audit-v1','status':'accepted_observed_ram_components','session':session.name,'checks_passed':len(checks),'packet_payload_files_verified':len(packet),'released_source_files_identical':len(source_files),'accepted_runs':len(records),'reference_comparisons':sum(o['checked'] for o in observations),'mismatches':sum(o['mismatches'] for o in observations),'measurement_retries':0,'control_retries':0,'observations':observations,'checks':checks,'limitations':['Recorded finite-workload RAM components, not global worst-case peak RAM','Loop-task stack high-water mark already reached before workload in all 15 runs','Heap snapshots do not measure inference allocation churn','Signature-free hashes establish artifact consistency, not an independent flash readback','Raw diagnostic_globals_bytes has a reporting-only SDK-symbol overcount; see independent binary review']}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['status','checks_passed','packet_payload_files_verified','released_source_files_identical','accepted_runs','reference_comparisons','mismatches']},indent=2))

if __name__ == '__main__':
    main()

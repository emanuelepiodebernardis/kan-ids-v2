"""Continuity/evidence gates; no test opens or flashes physical hardware."""
from __future__ import annotations
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_continuous as runner
from run_suite import GateError, digest, read_json, write_json, plan_for, make_archive, verify_packet


def record_for(item):
    index = item['index']
    return {'schema': 'kanids-hw500-run-v1', 'run_id': item['key'] + '_synthetic', 'suite_id': 'SYNTHETIC',
            'board_kind': 'mega', 'model': item['model'], 'phase': item['phase'], 'repeat': item['repeat'],
            'plan_item': item, 'status': runner.PENDING,
            'host_timing': {'recording_confirm_monotonic_ns': 1_000_000_000,
                            'run_send_monotonic_ns': (2 + index * 250) * 1_000_000_000,
                            'prep_receive_monotonic_ns': (2 + index * 250) * 1_000_000_000 + 100_000_000,
                            'done_receive_monotonic_ns': (170 + index * 250) * 1_000_000_000},
            'result': {'done': {'count': 500000}}}


def make_session(directory, count=2):
    session = {'schema': 'kanids-hw500-suite-v1', 'acquisition_protocol': runner.ACQUISITION,
               'suite_id': 'SYNTHETIC', 'board_kind': 'mega', 'plan': plan_for('mega'), 'attempts': [],
               'recording': {'confirmed_utc': '2026-09-16T00:00:00+00:00'}, 'status': 'incomplete'}
    for item in session['plan'][:count]:
        current = record_for(item)
        attempt = directory / 'attempts' / current['run_id']; attempt.mkdir(parents=True)
        (attempt / 'serial_raw.bin').write_bytes(b'synthetic original serial\n')
        (attempt / 'serial.jsonl').write_text('{}\n')
        runner.seal_attempt(directory, session, attempt, current)
    return session


def synthetic_analyzer(path, records):
    return {'status': runner.ANALYSIS_ACCEPTED, 'registration': {}, 'analyses': [
        {'status': 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_ESTIMATE', 'run_id': r['run_id'],
         'board_kind': r['board_kind'], 'model': r['model'], 'source_sha256': digest(path)['sha256'],
         'workload': {'completed_inferences': r['result']['done']['count']}}
        for r in records]}


class ContinuousRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.session = self.root / 'session'; self.session.mkdir()
        self.source = self.root / 'trace.cfn'; self.source.write_bytes(b'fake trace supplied for runner-only tests')

    def tearDown(self):
        self.temp.cleanup()

    def test_immutable_software_records_and_one_shared_raw_trace(self):
        make_session(self.session)
        before = {p: digest(p) for p in self.session.rglob('SOFTWARE_RECORD.json')}
        result = runner.analyze_saved_session(self.session, self.source, synthetic_analyzer)
        self.assertEqual(result['status'], 'partial_series_trace_accepted_not_complete_campaign')
        self.assertEqual(len(list((self.session / 'captures').glob('*.cfn'))), 1)
        self.assertFalse(any(p.suffix == '.cfn' for p in (self.session / 'attempts').rglob('*')))
        self.assertEqual(before, {p: digest(p) for p in before})
        for entry in result['attempts']:
            attempt = self.session / entry['relative_path']; verify_packet(attempt)
            r = read_json(attempt / 'RUN_RECORD.json')
            self.assertEqual(r['status'], runner.ACCEPTED)
            self.assertEqual(digest(self.session / r['cfn']['session_relative_path']), digest(self.source))
        verify_packet(self.session)
        self.assertFalse((self.session / 'SUMMARY.json').exists())

    def test_failed_analysis_preserves_raw_and_no_run_accepted_then_retry(self):
        make_session(self.session)
        def fail(path, records):
            raise ValueError('Missing second marker')
        with self.assertRaisesRegex(ValueError, 'Missing second'):
            runner.analyze_saved_session(self.session, self.source, fail)
        self.assertEqual(len(list((self.session / 'captures').glob('*.cfn'))), 1)
        s, records, paths = runner.load_pending_records(self.session)
        self.assertTrue(all(r['status'] == runner.PENDING for r in records))
        self.assertEqual(s['analysis_attempts'][0]['status'], 'failed_or_interrupted_analysis')
        runner.analyze_saved_session(self.session, self.source, synthetic_analyzer)
        s = read_json(self.session / 'SUITE_RECORD.json')
        self.assertEqual(len(s['analysis_attempts']), 2)
        self.assertEqual(len(list((self.session / 'captures').glob('*.cfn'))), 1)

    def test_all_result_identities_checked_before_accepting_any_run(self):
        make_session(self.session)
        def wrong(path, records):
            output = synthetic_analyzer(path, records)
            output['analyses'][1]['model'] = 'alien'
            return output
        with self.assertRaisesRegex(GateError, 'идентичность'):
            runner.analyze_saved_session(self.session, self.source, wrong)
        _, records, _ = runner.load_pending_records(self.session)
        self.assertTrue(all(r['status'] == runner.PENDING for r in records))
        self.assertFalse(list(self.session.rglob('TRACE_ANALYSIS.json')))

    def test_duplicate_analysis_id_rejected(self):
        make_session(self.session)
        def duplicate(path, records):
            output = synthetic_analyzer(path, records)
            output['analyses'][1] = copy.deepcopy(output['analyses'][0])
            return output
        with self.assertRaisesRegex(GateError, 'сопоставление'):
            runner.analyze_saved_session(self.session, self.source, duplicate)
        self.assertFalse(list(self.session.rglob('TRACE_ANALYSIS.json')))

    def test_stale_trace_is_retained_before_rejection(self):
        make_session(self.session)
        os.utime(self.source, (1, 1))
        with self.assertRaisesRegex(GateError, 'до начала'):
            runner.analyze_saved_session(self.session, self.source, synthetic_analyzer, enforce_fresh=True)
        self.assertEqual(len(list((self.session / 'captures').glob('*.cfn'))), 1)
        runner.load_pending_records(self.session)

    def test_accepted_results_cannot_be_reanalyzed_or_overwritten(self):
        make_session(self.session)
        runner.analyze_saved_session(self.session, self.source, synthetic_analyzer)
        original = {p: digest(p) for p in self.session.rglob('RUN_RECORD.json')}
        with self.assertRaisesRegex(GateError, 'перезапись запрещена'):
            runner.analyze_saved_session(self.session, self.source, synthetic_analyzer)
        self.assertEqual(original, {p: digest(p) for p in original})

    def test_serial_mutation_rejected_even_if_outer_packet_refreshed(self):
        s = make_session(self.session)
        attempt = self.session / s['attempts'][0]['relative_path']
        (attempt / 'serial_raw.bin').write_bytes(b'changed')
        make_archive(attempt)
        with self.assertRaisesRegex(GateError, 'software/serial'):
            runner.load_pending_records(self.session)

    def test_orphan_or_duplicate_attempt_rejected(self):
        make_session(self.session)
        (self.session / 'attempts' / 'orphan').mkdir()
        with self.assertRaisesRegex(GateError, 'Незарегистрированная'):
            runner.load_pending_records(self.session)

    def test_monotonic_gate_rejects_missing_reversed_and_delayed_prep(self):
        good = record_for(plan_for('mega')[0])['host_timing']
        runner.validate_host_timing(good)
        for key, value in [('done_receive_monotonic_ns', 1), ('prep_receive_monotonic_ns', 5_000_000_001),
                           ('recording_confirm_monotonic_ns', None), ('run_send_monotonic_ns', True)]:
            changed = dict(good); changed[key] = value
            with self.subTest(key=key), self.assertRaises(GateError):
                runner.validate_host_timing(changed)

    def test_chunk_timestamps_not_delayed_line_consumption_timestamps(self):
        class Device:
            in_waiting = 24
            chunks = [b'PREP sync=beg', b'in\nDONE count=1\n']
            def read(self, amount): return self.chunks.pop(0)
            def write(self, data): return len(data)
            def flush(self): pass
        d = self.root / 'capture'; d.mkdir()
        capture = runner.ContinuousCapture(Device(), d)
        with patch.object(runner.time, 'perf_counter_ns', side_effect=[100, 200, 300]):
            capture.send('RUN')
            self.assertEqual(capture.line(time.monotonic() + 1), 'PREP sync=begin')
            self.assertEqual(capture.line(time.monotonic() + 1), 'DONE count=1')
        self.assertEqual(capture.protocol_timing, {'run_send_monotonic_ns': 100,
                         'prep_receive_monotonic_ns': 300, 'done_receive_monotonic_ns': 300})
        capture.close()
        logs = [json.loads(x) for x in (d / 'serial.jsonl').read_text().splitlines()]
        self.assertEqual([x['host_monotonic_ns'] for x in logs], [100, 200, 300])

    def test_optional_partial_save_can_be_deferred_without_losing_software(self):
        make_session(self.session)
        with patch('sys.stdout', new=io.StringIO()):
            value = runner.request_saved_trace(self.root / 'not_yet.cfn', prompt=lambda _: 's', optional=True)
        self.assertIsNone(value)
        runner.load_pending_records(self.session)


if __name__ == '__main__':
    unittest.main()

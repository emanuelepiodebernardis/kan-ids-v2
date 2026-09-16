import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_suite import ACCEPTED, digest, make_archive, plan_for, write_json
from summarize_results import summarize_session


class SummaryTests(unittest.TestCase):
    def fixture(self, root, omit_last=False):
        plan = plan_for('mega')
        session = {'schema': 'kanids-hw500-suite-v1', 'board_kind': 'mega', 'plan': plan,
                   'protocol_file': {'sha256': 'fixture'}, 'attempts': []}
        for item in plan[:-1] if omit_last else plan:
            path = root / 'attempts' / item['key']
            path.mkdir(parents=True)
            value = 10000 if item['phase'] == 'pilot' else item['repeat']
            # Deliberately unequal inference counts: averaging must be by run.
            n = 500 * (item['index'] + 1)
            record = {'status': ACCEPTED, 'run_id': item['key'], 'plan_item': item,
                      'cfn': {'file': {'sha256': item['key']}}, 'result': {'done': {'count': n}}}
            analysis = {'status': 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_ESTIMATE',
                        'board_kind': 'mega', 'model': item['model'], 'run_id': item['key'],
                        'source_sha256': item['key'],
                        'workload': {'distinct_frozen_flows': 500, 'completed_inferences': n,
                                     'active_MCU_seconds': 120, 'mean_replay_us_per_inference': value},
                        'primary_energy': {'mean_power_W': value, 'uJ_per_inference': value},
                        'idle_before_guard': {'mean_power_W': .1},
                        'idle_after_guard': {'mean_power_W': .1},
                        'diagnostic_only': {'incremental_uJ_per_inference': -.5},
                        'registration': {'edge_residual_max_abs_s': .05},
                        'maximum_absolute_energy_shift_percent': 1.0}
            write_json(path / 'RUN_RECORD.json', record)
            write_json(path / 'TRACE_ANALYSIS.json', analysis)
            make_archive(path)
            session['attempts'].append({'relative_path': path.relative_to(root).as_posix(),
                                        'record_file': digest(path / 'RUN_RECORD.json')})
        write_json(root / 'SUITE_RECORD.json', session)
        return session

    def test_run_level_unweighted_pilot_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            result = summarize_session(root)
            self.assertEqual(result['confirmatory_runs'], 25)
            for row in result['mean_rows']:
                self.assertEqual(row['USB_energy_uJ_mean'], 3)
                self.assertAlmostEqual(row['USB_energy_uJ_sample_SD'], 2.5 ** .5)
            self.assertIn('-0.5', (root / 'results_confirmatory.csv').read_text(encoding='utf-8'))

    def test_partial_campaign_not_finalized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, omit_last=True)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                summarize_session(root)
            self.assertFalse((root / 'SUMMARY.json').exists())

    def test_duplicate_planned_acquisition_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = self.fixture(root)
            session['attempts'].append(session['attempts'][0])
            write_json(root / 'SUITE_RECORD.json', session)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize_session(root)

    def test_modified_accepted_analysis_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            path = root / 'attempts' / 'pilot_coeff' / 'TRACE_ANALYSIS.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['primary_energy']['uJ_per_inference'] = 0
            write_json(path, data)
            with self.assertRaises(RuntimeError):
                summarize_session(root)

    def test_shared_continuous_raw_file_is_required_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = self.fixture(root)
            session['acquisition_protocol'] = 'kanids-hw500-continuous-v1'
            raw = root / 'captures' / 'whole_series.cfn'
            raw.parent.mkdir()
            raw.write_bytes(b'raw-fixture-for-summary-integrity-only')
            identity = digest(raw)
            for entry in session['attempts']:
                attempt = root / entry['relative_path']
                record = json.loads((attempt / 'RUN_RECORD.json').read_text())
                record['cfn'] = {'file': identity, 'session_relative_path': 'captures/whole_series.cfn'}
                write_json(attempt / 'RUN_RECORD.json', record)
                analysis = json.loads((attempt / 'TRACE_ANALYSIS.json').read_text())
                analysis['source_sha256'] = identity['sha256']
                write_json(attempt / 'TRACE_ANALYSIS.json', analysis)
                make_archive(attempt)
                entry['record_file'] = digest(attempt / 'RUN_RECORD.json')
            write_json(root / 'SUITE_RECORD.json', session)
            self.assertEqual(summarize_session(root)['confirmatory_runs'], 25)
            raw.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                summarize_session(root)
            raw.unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                summarize_session(root)


if __name__ == '__main__':
    unittest.main()

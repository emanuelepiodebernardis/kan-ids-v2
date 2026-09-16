"""Synthetic acceptance/negative checks, never physical HW500 measurements."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests_measurement'))
from test_trace import make_record, write_cfn
from analyze_trace import wave, inspect_recording
from analyze_continuous import analyze_session_trace, validate_host_records, assign_pairs


def make_series(n=3, missing=None, clock_jump_at=None, voltage_dip='gap'):
    pre = np.array([0., 2, 4, 8, 10, 12])
    scale, offset, width = 1.001, 4., 1.05
    # Deliberately nonuniform upload+ARM+settling gaps.
    prep_host = np.r_[.1, .1+np.cumsum([224.+(i*17)%53 for i in range(n-1)])]
    origins = offset+scale*(prep_host+2.)
    end = float(origins[-1]+scale*164+4.)
    t = np.arange(0., end, .1)
    current = np.full(len(t), .073)
    current += np.random.default_rng(301).normal(0, .002, len(t))/1000
    records = []
    ns0 = 800000000000000000  # Large uptime must be subtracted before float.
    for k, (h, origin) in enumerate(zip(prep_host, origins)):
        if clock_jump_at is not None and k >= clock_jump_at:
            origin += 6.
        for group, edges in [('pre', pre), ('post', pre+152)]:
            if missing != (k, group):
                current += .0008*wave(t, origin+scale*edges, width)
        if k:
            current += .0008*wave(t, origin-45*scale+scale*pre, width)
        current += .0002*(np.clip((t-origin-scale*22+width/2)/width, 0, 1)-
                         np.clip((t-origin-scale*142+width/2)/width, 0, 1))
        record = make_record()
        record.update(run_id='SYNTHETIC_'+str(k), phase='pilot' if not k else 'confirmatory',
                      model=['coeff', 'coeff', 'lut', 'mlp', 'kanml', 'dt5'][(k-1)%5+1] if k else 'coeff')
        record['host_timing'] = {
            'recording_confirm_monotonic_ns': ns0,
            'run_send_monotonic_ns': ns0+round((h-.02)*1e9),
            'prep_receive_monotonic_ns': ns0+round(h*1e9),
            'done_receive_monotonic_ns': ns0+round((h+168)*1e9),
        }
        records.append(record)
    voltage = np.full(len(t), 5.)
    if voltage_dip == 'gap' and n > 1:
        voltage[(t > origins[1]-30) & (t < origins[1]-28)] = 0.
    elif voltage_dip == 'active':
        voltage[(t > origins[0]+50) & (t < origins[0]+51)] = 0.
    return np.column_stack([t, voltage, current, voltage*current, np.zeros(len(t))]), records


class ContinuousAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'whole.cfn'

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_26_run_long_trace_upload_LEDTEST_and_clock_drift(self):
        arr, records = make_series(26)
        self.assertGreater(len(arr), 20000)
        write_cfn(self.path, arr)
        raw = self.path.read_bytes()
        t0 = time.monotonic()
        result = analyze_session_trace(self.path, records)
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertTrue(result['full_series'])
        self.assertEqual(len(result['analyses']), 26)
        self.assertEqual(len(result['registration']['unassigned_candidates']), 25)
        self.assertEqual(result['registration']['accepted_joint_assignments'], 1)
        self.assertAlmostEqual(result['registration']['refined_host_fit']['CFN_seconds_per_host_second'], 1.001, delta=.00001)
        for record, analysis in zip(records, result['analyses']):
            self.assertEqual(analysis['run_id'], record['run_id'])
            self.assertEqual(analysis['source_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(analysis['recording_integrity']['records'], len(arr))
            self.assertAlmostEqual(analysis['primary_energy']['duration_CFN_s'], 120.12, delta=.02)
            self.assertAlmostEqual(analysis['primary_energy']['energy_J'], .366*120.12, delta=.02)
        json.dumps(result, allow_nan=False)
        print('SYNTHETIC_FULL26_ANALYSIS_SECONDS', round(time.monotonic()-t0, 3))

    def test_partial_completed_prefix_supported_but_not_full(self):
        arr, records = make_series(3)
        write_cfn(self.path, arr)
        result = analyze_session_trace(self.path, records[:2])
        self.assertFalse(result['full_series'])
        self.assertEqual(len(result['analyses']), 2)
        self.assertGreater(len(result['registration']['unassigned_candidates']), 0)

    def test_missing_post_marker_rejects_entire_assignment(self):
        arr, records = make_series(3, missing=(1, 'post'))
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'Missing.*pair|assignment'):
            analyze_session_trace(self.path, records)

    def test_voltage_dip_during_batch_rejected(self):
        arr, records = make_series(2, voltage_dip='active')
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'voltage below 4 V within'):
            analyze_session_trace(self.path, records)

    def test_waveform_clock_jump_rejected_even_with_regular_CFN_grid(self):
        # Add an unused fifth synthetic run to cover all markers of the
        # shifted fourth run; analyze only the first four completed records.
        arr, records = make_series(5, clock_jump_at=2)
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'assignment|pair|timing'):
            analyze_session_trace(self.path, records[:4])

    def test_ambiguous_joint_assignment_rejected_not_best_fit_selected(self):
        _, records = make_series(1)
        host, markers = validate_host_records(records)
        candidates = [{'center': c} for c in [4., 8., 156., 160.]]
        with self.assertRaisesRegex(ValueError, 'ambiguous joint LED assignment: 2'):
            assign_pairs(candidates, host, markers)

    def test_swapped_records_duplicate_ID_and_clock_reset_rejected(self):
        _, records = make_series(3)
        variants = []
        swapped = copy.deepcopy(records); swapped[1], swapped[2] = swapped[2], swapped[1]
        variants.append(swapped)
        duplicate = copy.deepcopy(records); duplicate[1]['run_id'] = duplicate[0]['run_id']
        variants.append(duplicate)
        reset = copy.deepcopy(records); reset[1]['host_timing']['prep_receive_monotonic_ns'] = 1
        variants.append(reset)
        different_anchor = copy.deepcopy(records)
        different_anchor[1]['host_timing']['recording_confirm_monotonic_ns'] -= 1000000000
        variants.append(different_anchor)
        for variant in variants:
            with self.subTest(variant=variants.index(variant)), self.assertRaises(ValueError):
                validate_host_records(variant)

    def test_delayed_serial_receipt_or_wrong_duration_rejected(self):
        _, records = make_series(1)
        slow = copy.deepcopy(records)
        slow[0]['host_timing']['prep_receive_monotonic_ns'] += 3000000000
        with self.assertRaisesRegex(ValueError, 'transport'):
            validate_host_records(slow)
        late = copy.deepcopy(records)
        late[0]['host_timing']['done_receive_monotonic_ns'] += 5000000000
        with self.assertRaisesRegex(ValueError, 'duration'):
            validate_host_records(late)

    def test_regular_grid_does_not_override_missing_markers(self):
        arr, records = make_series(1, missing=(0, 'pre'))
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'pair'):
            analyze_session_trace(self.path, records)

    def test_continuous_length_bound_and_corrupt_grid_rejected(self):
        arr, _ = make_series(1)
        t = np.arange(108002)*.1
        long_arr = np.column_stack([t, np.full(len(t), 5.), np.full(len(t), .073),
                                    np.full(len(t), .365), np.zeros(len(t))])
        write_cfn(self.path, long_arr)
        with self.assertRaisesRegex(ValueError, 'length'):
            inspect_recording(self.path, continuous=True)
        arr = np.delete(arr, 100, axis=0)
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'gap'):
            inspect_recording(self.path, continuous=True)


if __name__ == '__main__':
    unittest.main()

"""Synthetic electrical traces exercise acceptance and corruption gates.

Synthetic traces test algorithms; they are not hardware measurements.
"""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_trace import analyze, inspect_recording, integrate, validate_record, wave
from cfn_decoder import decode, analyse as raw_analyse


def make_record():
    pre = [0, 2, 4, 8, 10, 12]
    active_begin, active_end = 22, 142
    post = [152+x for x in pre]
    names = ['on_1', 'off_1', 'on_2', 'off_2', 'on_3', 'off_3']
    events = {group+'_led_'+name: round(value*1e6)
              for group, values in [('pre', pre), ('post', post)] for name, value in zip(names, values)}
    events.update(active_begin=22000000, active_end=142000000, idle_before_begin=12000000,
                  idle_before_end=22000000, idle_after_begin=142000000, idle_after_end=152000000)
    return {'schema': 'kanids-hw500-run-v1', 'board_kind': 'mega', 'model': 'dt5', 'run_id': 'SYNTHETIC',
            'ready': {'count': 500000, 'checksum_per500': 250, 'warmup_count': 500},
            'result': {'events': [{'name': k, 'us': v} for k, v in events.items()],
                       'done': {'count': 500000, 'active_us': 120000000, 'checksum': 250000,
                                'expected': 250000, 'correct': 1, 'timing_ok': 1, 'wdt_restored': 1}}}


def synthetic(rows_shift=0., flat_active=False, no_marker=False, doubled=False):
    t = np.arange(0., 172., .1)
    origin, scale, width = 4., 1.0004, 1.05
    pre = np.array([0, 2, 4, 8, 10, 12.])
    signal = wave(t, origin+scale*pre, width) + wave(t, origin+scale*(pre+152), width)
    noise = np.random.default_rng(43).normal(0, .003, len(t))
    current = .073 + (.0008*signal if not no_marker else 0.) + noise/1000
    if not flat_active:
        current += .0002*(np.clip((t-origin-scale*22+width/2)/width, 0, 1)-np.clip((t-origin-scale*142+width/2)/width, 0, 1))
    v = np.full(len(t), 5.)
    arr = np.column_stack([t+rows_shift, v, current, v*current, np.zeros(len(t))])
    if doubled:
        arr = np.vstack([arr, np.column_stack([arr[:, 0]+172., arr[:, 1:]])])
    return arr


def write_cfn(path, arr):
    # code0V/code1I/code4P/code6NRG. FlatNRG must never enterprimaryintegration.
    data = bytearray(struct.pack('<diiih', 10., 0, 0, 5, 4))
    for col, code in enumerate([0, 1, 4, 6], 1):
        data += struct.pack('<hIBdd', code, 0, 1, arr[:, col].max(), arr[:, col].min())
    data += struct.pack('<i', len(arr))
    data += np.asarray(arr, dtype='<f8').tobytes()
    path.write_bytes(data)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'trace.cfn'
        self.record = make_record()

    def tearDown(self):
        self.temp.cleanup()

    def test_good_trace_uses_full_batch_and_actual_count(self):
        write_cfn(self.path, synthetic())
        result = analyze(self.path, self.record)
        self.assertEqual(result['workload']['distinct_frozen_flows'], 500)
        self.assertEqual(result['workload']['completed_inferences'], 500000)
        self.assertAlmostEqual(result['registration']['CFN_seconds_per_MCU_second'], 1.0004, delta=.0001)
        self.assertAlmostEqual(result['primary_energy']['duration_CFN_s'], 120.048, delta=.02)
        self.assertAlmostEqual(result['primary_energy']['energy_J'], .366*120.048, delta=.02)
        self.assertFalse(result['primary_energy']['NRG_used'])
        self.assertEqual(len(result['registration_shift_sensitivity']), 7)
        json.dumps(result, allow_nan=False)

    def test_flat_DT_active_power_does_not_imply_frozen_trace(self):
        arr = synthetic(flat_active=True)
        arr[(arr[:, 0]>28)&(arr[:, 0]<144), 2] = .073
        arr[:, 3] = arr[:, 1]*arr[:, 2]
        write_cfn(self.path, arr)
        result = analyze(self.path, self.record)
        self.assertGreater(result['recording_integrity']['time_grid']['longest_identical_VI_P_hold_s'], 100)
        self.assertAlmostEqual(result['primary_energy']['energy_J'], .365*120.048, delta=.02)

    def test_missing_markers_rejected_even_if_timestamps_advance(self):
        write_cfn(self.path, synthetic(no_marker=True))
        with self.assertRaisesRegex(ValueError, 'marker pair'):
            analyze(self.path, self.record)

    def test_two_acquisitions_rejected_as_ambiguous(self):
        write_cfn(self.path, synthetic(doubled=True))
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            analyze(self.path, self.record)

    def test_clock_scale_mismatch_rejected(self):
        arr = synthetic(); arr[:, 0] *= 1.04
        # Restore a regular .1-s grid: mismatch is waveform timing, not a CFN gap.
        t = np.arange(0, arr[-1, 0], .1)
        stretched = np.column_stack([t] + [np.interp(t, arr[:, 0], arr[:, j]) for j in range(1, 5)])
        write_cfn(self.path, stretched)
        with self.assertRaisesRegex(ValueError, 'marker pair|clock scale'):
            analyze(self.path, self.record)

    def test_raw_decoder_does_not_divide_by_zero_NRG(self):
        write_cfn(self.path, synthetic())
        meta, rows = decode(self.path)
        _, stats = raw_analyse(meta, rows)
        self.assertIsNone(stats['NRG_crosscheck']['integrated_VI_energy_divided_by_NRG_delta'])
        arr = synthetic(); arr[:, 2:] = 0
        write_cfn(self.path, arr)
        meta, rows = decode(self.path)
        _, stats = raw_analyse(meta, rows)
        self.assertEqual(stats['NRG_crosscheck']['status'], 'UNDEFINED_ZERO_INTEGRATED_ENERGY')

    def test_legacy_twenty_flow_record_rejected(self):
        rec = copy.deepcopy(self.record); rec['schema'] = 'old-pilot'
        with self.assertRaisesRegex(ValueError, 'HW500'):
            validate_record(rec)

    def test_incomplete_traversal_rejected(self):
        self.record['result']['done']['count'] = 500001
        with self.assertRaisesRegex(ValueError, '500-flow'):
            validate_record(self.record)

    def test_wrong_checksum_rejected(self):
        self.record['result']['done']['checksum'] -= 1
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_record(self.record)

    def test_missing_idle_event_rejected(self):
        self.record['result']['events'].pop()
        with self.assertRaisesRegex(ValueError, 'events'):
            validate_record(self.record)

    def test_wrong_marker_schedule_rejected(self):
        self.record['result']['events'][2]['us'] = 3000000
        with self.assertRaisesRegex(ValueError, 'schedule'):
            validate_record(self.record)

    def test_gap_rejected(self):
        arr = np.delete(synthetic(), 400, axis=0)
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'gap'):
            inspect_recording(self.path)

    def test_nonmonotonic_time_rejected(self):
        arr = synthetic(); arr[300, 0] = arr[299, 0]
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'Non-increasing'):
            inspect_recording(self.path)

    def test_truncation_and_append_rejected(self):
        write_cfn(self.path, synthetic()); good = self.path.read_bytes()
        for malformed in [good[:-1], good+b'\0']:
            self.path.write_bytes(malformed)
            with self.assertRaisesRegex(ValueError, 'EOF'):
                decode(self.path)

    def test_bad_voltage_or_power_rejected(self):
        arr = synthetic(); arr[100, 1] = 9
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'voltage'):
            inspect_recording(self.path)
        arr = synthetic(); arr[100, 3] += .01
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            inspect_recording(self.path)

    def test_cropped_marker_rejected(self):
        arr = synthetic(); arr = arr[arr[:, 0] > 8]
        write_cfn(self.path, arr)
        with self.assertRaisesRegex(ValueError, 'marker'):
            analyze(self.path, self.record)

    def test_integral_interpolates_endpoints_without_extrapolation(self):
        value = integrate(np.array([0., 1., 2.]), np.array([0., 2., 4.]), .5, 1.5)
        self.assertAlmostEqual(value['energy_J'], 2.)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            integrate(np.array([0., 1.]), np.array([1., 1.]), -1., 1.)


if __name__ == '__main__':
    unittest.main()

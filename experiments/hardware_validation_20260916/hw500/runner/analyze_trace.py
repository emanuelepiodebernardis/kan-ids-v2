#!/usr/bin/env python3
"""Fail-closed analysis of the HW500 whole-board USB energy experiment.

CFN coordinates are stored times, not hardware acquisition timestamps. LED
registration estimates an affine map to those coordinates; it cannot calibrate
common filter/transport delay or absolute clock/power accuracy. No NRG channel
is used for energy. This module never accepts legacy twenty-flow records.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from cfn_decoder import decode

EDGE_SUFFIXES = ['on_1', 'off_1', 'on_2', 'off_2', 'on_3', 'off_3']
EDGE_NAMES = [g + '_led_' + e for g in ['pre', 'post'] for e in EDGE_SUFFIXES]
EXPECTED_MARKER_STEPS = np.array([2., 2., 4., 2., 2.])
LIMITS = [
    'Whole-board USB engineering estimate, including downstream wiring and replay/index/checksum overhead.',
    'Prepared Flash row -> RAM buffer -> prediction -> checksum; excludes feature extraction and network acquisition.',
    'The recorded 0.1-second grid does not establish 10-Hz physical bandwidth or independent observations.',
    'No calibrated voltage/current/clock uncertainty; no confidence intervals from correlated trace samples.',
    'LED-center registration includes an unidentified common filtering/transport delay; fitted scatter is not calibration.',
    'Regular synthetic CFN timestamps and valid bookend markers cannot prove that every intervening sample was freshly acquired.',
    'Idle subtraction is diagnostic only and is not the primary energy result.',
    'NRG and CAP channels are retained as raw evidence but never used to calculate energy.',
]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool), name + ': integer required')
    return value


def validate_record(record):
    require(record.get('schema') == 'kanids-hw500-run-v1', 'Not a HW500 acquisition record')
    require(record.get('board_kind') in ('mega', 'c3'), 'Unsupported board identity')
    require(record.get('model') in ('coeff', 'lut', 'mlp', 'kanml', 'dt5'), 'Unsupported model')
    event_list = record['result']['events']
    names = [x['name'] for x in event_list]
    expected_names = set(EDGE_NAMES + ['active_begin', 'active_end', 'idle_before_begin',
                                     'idle_before_end', 'idle_after_begin', 'idle_after_end'])
    require(len(names) == 18 and set(names) == expected_names, 'Missing/duplicate/unknown MCU events')
    events = {x['name']: integer(x['us'], 'event time') / 1e6 for x in event_list}
    require(all(v >= 0 for v in events.values()), 'Negative MCU event time')
    markers = np.array([[events[g + '_led_' + n] for n in EDGE_SUFFIXES] for g in ('pre', 'post')])
    require(np.all(np.abs(np.diff(markers, axis=1) - EXPECTED_MARKER_STEPS) < .05), 'Unexpected LED marker schedule')
    a, b = events['active_begin'], events['active_end']
    require(markers[0, -1] < a < b < markers[1, 0], 'Invalid marker/active ordering')
    for left, right in [('idle_before_begin', 'pre_led_off_3'), ('idle_before_end', 'active_begin'),
                        ('idle_after_begin', 'active_end'), ('idle_after_end', 'post_led_on_1')]:
        require(abs(events[left] - events[right]) <= .005, 'Idle event does not match boundary')
    require(9.9 <= a - markers[0, -1] <= 10.1 and 9.9 <= markers[1, 0] - b <= 10.1,
            'Idle guards are not ten MCU seconds')
    done, ready = record['result']['done'], record['ready']
    for key in ('correct', 'timing_ok', 'wdt_restored'):
        require(done.get(key) == 1, 'Firmware gate failed: ' + key)
    count = integer(done['count'], 'count')
    require(0 < count <= 2000000000 and count % 500 == 0, 'Count must contain complete 500-flow traversals')
    require(count == ready['count'] and ready['warmup_count'] == 500, 'ARM/RUN counts do not match')
    per500 = integer(ready['checksum_per500'], 'checksum_per500')
    require(0 <= per500 <= 500, 'Invalid reference checksum')
    require(done['checksum'] == done['expected'] == (count // 500) * per500, 'Incorrect batch checksum')
    duration = integer(done['active_us'], 'active_us') / 1e6
    require(60 <= duration <= 180 and abs((b-a)-duration) < 1e-7, 'MCU interval differs from completed batch')
    return events, markers, duration, count


def inspect_recording(path, *, continuous=False):
    metadata, rows = decode(Path(path))
    require(metadata['records'] <= (108001 if continuous else 20000), 'Recording exceeds bounded acquisition length')
    arr = np.asarray(rows, dtype=float)
    idx = {c['code']: k+1 for k, c in enumerate(metadata['channels'])}
    require({0, 1, 4}.issubset(idx), 'VBUS, IBUS and PBUS are required')
    require(metadata['sample_rate_header_sps'] == 10, 'Recording must use 10-sps storage grid')
    require(metadata['start_current_mA'] == metadata['stop_current_mA'] == 0,
            'Start/Stop current thresholds must both be zero')
    times = arr[:, 0]
    dt = np.diff(times)
    require(np.all(dt > 0), 'Non-increasing CFN times')
    require(np.all(np.abs(dt - .1) <= 1e-7), 'CFN time gap/dropout or unexpected stored grid')
    require(times[-1]-times[0] <= (10800 if continuous else 600), 'Recording contains excessive wait/multiple acquisitions')
    for k, channel in enumerate(metadata['channels'], 1):
        if channel['has_header_extrema']:
            require(channel['header_min'] == float(arr[:, k].min()) and
                    channel['header_max'] == float(arr[:, k].max()), 'CFN header extrema mismatch')
    v, current = arr[:, idx[0]], arr[:, idx[1]]
    require(np.all((v >= (0. if continuous else 4.)) & (v <= 5.5)), 'USB voltage outside expected 5-V setup')
    require(np.all((current >= 0.) & (current <= 1.)) and current.max() > 0,
            'Current outside expected board range')
    power = v * current
    error = float(np.max(np.abs(arr[:, idx[4]] - power)))
    require(error <= max(1e-8, float(np.max(power))*1e-5), 'PBUS and VBUS*IBUS inconsistent')
    changes = np.flatnonzero(np.any(arr[1:, [idx[0], idx[1], idx[4]]] != arr[:-1, [idx[0], idx[1], idx[4]]], axis=1)) + 1
    run_lengths = np.diff(np.r_[0, changes, len(times)])
    changes_dt = np.diff(times[changes]) if len(changes) > 1 else np.array([])
    info = dict(metadata, time_grid={
        'first_s': float(times[0]), 'last_s': float(times[-1]), 'span_s': float(times[-1]-times[0]),
        'stored_step_median_s': float(np.median(dt)), 'stored_step_min_s': float(dt.min()),
        'stored_step_max_s': float(dt.max()), 'missing_slots': 0, 'monotonic': True,
        'observed_value_change_count': int(len(changes)),
        'observed_change_interval_median_s': float(np.median(changes_dt)) if len(changes_dt) else None,
        'longest_identical_VI_P_hold_s': float(max(run_lengths)*.1),
        'note': 'Value-change timing is descriptive; repeated values alone are not a freeze diagnosis or physical sample rate.',
    }, PBUS_minus_VI_max_abs_W=error)
    if continuous:
        bad = np.flatnonzero(v < 4.)
        chunks = np.split(bad, np.flatnonzero(np.diff(bad) != 1)+1) if len(bad) else []
        info['voltage_below_4V_spans_CFN_s'] = [[float(times[z[0]]), float(times[z[-1]])] for z in chunks]
        info['voltage_gate_scope'] = '0..5.5 V globally; 4..5.5 V throughout every analyzed run and both markers'
    return info, times, current*1000., power


def wave(x, centers, width):
    return (np.clip((np.asarray(x)[:, None] - np.asarray(centers)[None, :] + width/2) / width, 0, 1)
            * np.array([1., -1., 1., -1., 1., -1.])).sum(axis=1)


def find_patterns(times, current_mA, relative_edges):
    """Exhaustive bounded sliding template search; no active-power threshold."""
    relative_edges = np.asarray(relative_edges) - relative_edges[0]
    candidates = []
    step = float(np.median(np.diff(times)))
    window = np.arange(-1.5, relative_edges[-1]+1.5001, step)
    baseline = np.column_stack([np.ones(len(window)), window-window.mean()])
    for width in (.5, 1., 1.5):
        design = np.column_stack([baseline, wave(window, relative_edges, width)])
        inverse = np.linalg.pinv(design)
        inverse_baseline = np.linalg.pinv(baseline)
        for origin in times[(times >= times[0]+1.5) & (times <= times[-1]-relative_edges[-1]-1.5)]:
            y = np.interp(origin+window, times, current_mA)
            coeff = inverse @ y
            amp = coeff[2]
            if amp < .05:
                continue
            residual = y-design@coeff
            rms = float(np.sqrt(np.mean(residual**2)))
            detrended = y-baseline@(inverse_baseline@y)
            total = float(detrended@detrended)
            r2 = 1.-float(residual@residual)/total if total > 0 else -1.
            if rms/amp <= .15 and r2 >= .90:
                candidates.append(dict(center=float(origin), width=width, amplitude_mA=float(amp),
                                       normalized_rms=rms/amp, r2=r2))
    # Each peak can pass at adjacent grid positions/widths. Retain its best fit.
    chosen = []
    for item in sorted(candidates, key=lambda d: d['normalized_rms']):
        if all(abs(item['center']-x['center']) > 2.5 for x in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda d: d['center'])


def fit_registration(times, current_mA, marker_times, *, selected_pair=None):
    candidates = (find_patterns(times, current_mA, marker_times[0])
                  if selected_pair is None else list(selected_pair))
    separation = marker_times[1, 0]-marker_times[0, 0]
    pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i+1:]
             if .99 <= (b['center']-a['center'])/separation <= 1.01]
    require(len(pairs) == 1, 'Missing/ambiguous LED marker pair: ' + str(len(pairs)))
    pair = pairs[0]
    centers = np.concatenate([d['center'] + edges-edges[0] for d, edges in zip(pair, marker_times)])
    windows = [(d['center']-1.4, d['center']+edges[-1]-edges[0]+1.4) for d, edges in zip(pair, marker_times)]
    selectors = [(times >= lo) & (times <= hi) for lo, hi in windows]
    local = []
    for sel, d in zip(selectors, pair):
        yy = current_mA[sel]
        local += [float(np.quantile(yy, .1)), 0., d['amplitude_mA']]
    p0 = np.r_[centers, np.mean([x['width'] for x in pair]), local]
    lows = np.r_[centers-.7, .2, 0., -1., .025, 0., -1., .025]
    highs = np.r_[centers+.7, 2., 1000., 1., 30., 1000., 1., 30.]
    def residual(p):
        output = []
        for k, sel in enumerate(selectors):
            xx, yy = times[sel], current_mA[sel]
            base, drift, amp = p[13+k*3:16+k*3]
            output.append(base + drift*(xx-xx.mean()) + amp*wave(xx, p[k*6:k*6+6], p[12]) - yy)
        return np.concatenate(output)
    fitted = least_squares(residual, p0, bounds=(lows, highs), loss='linear',
                           xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1200)
    require(fitted.success, 'LED edge fit did not converge')
    p = fitted.x
    require(not np.any(fitted.active_mask), 'LED edge fit reached search boundary')
    # Free edge locations are fitted first; only then compare them to MCU times.
    design = np.column_stack([np.ones(12), marker_times.ravel()])
    offset, scale = np.linalg.lstsq(design, p[:12], rcond=None)[0]
    errors = p[:12] - design @ np.array([offset, scale])
    require(.995 <= scale <= 1.005, 'Marker clock scale differs by more than 0.5 percent')
    require(np.max(np.abs(errors)) <= .40 and np.sqrt(np.mean(errors**2)) <= .20,
            'Individual LED edges inconsistent with MCU schedule')
    fit_stats = []
    start = 0
    all_residuals = residual(p)
    for k, sel in enumerate(selectors):
        size = int(sel.sum())
        rr = all_residuals[start:start+size]; start += size
        amp = float(p[15+k*3])
        normalized = float(np.sqrt(np.mean(rr**2))/amp)
        require(amp >= .05 and normalized <= .15, 'LED contrast/noise insufficient')
        fit_stats.append(dict(amplitude_mA=amp, rms_residual_mA=float(np.sqrt(np.mean(rr**2))),
                              normalized_rms=normalized, baseline_mA=float(p[13+k*3]),
                              drift_mA_per_s=float(p[14+k*3])))
    return {
        'method': 'Exhaustive 2-4-2-second positive LED-template search; twelve free edge centers; affine fit to MCU times',
        'candidates': candidates, 'accepted_pair_count': 1, 'fit_windows_CFN_s': windows,
        'affine_center_offset_s': float(offset), 'CFN_seconds_per_MCU_second': float(scale),
        'empirical_ramp_width_s': float(p[12]), 'edge_centers_CFN_s': p[:12].tolist(),
        'MCU_edge_times_s': marker_times.ravel().tolist(), 'edge_residuals_s': errors.tolist(),
        'edge_residual_rms_s': float(np.sqrt(np.mean(errors**2))),
        'edge_residual_max_abs_s': float(np.max(np.abs(errors))), 'local_fit': fit_stats,
        'optimizer_success': bool(fitted.success), 'optimizer_evaluations': int(fitted.nfev),
        'optimizer_active_bound_indices': np.flatnonzero(fitted.active_mask).tolist(),
        'acceptance_limits': {'minimum_LED_amplitude_mA': .05, 'maximum_normalized_RMS': .15,
            'maximum_edge_residual_s': .40, 'maximum_edge_RMS_s': .20,
            'clock_scale_range': [.995, 1.005], 'ramp_width_search_s': [.2, 2.]},
        'unknown_common_delay_calibrated': False, 'fit_residual_is_uncertainty_budget': False,
    }


def integrate(times, values, lo, hi):
    require(times[0] <= lo < hi <= times[-1], 'Requested integration exceeds recording coverage')
    mask = (times > lo) & (times < hi)
    x = np.r_[lo, times[mask], hi]
    y = np.r_[np.interp(lo, times, values), values[mask], np.interp(hi, times, values)]
    area = float(np.sum((y[:-1]+y[1:])*np.diff(x)*.5))
    return {'bounds_CFN_s': [float(lo), float(hi)], 'duration_CFN_s': float(hi-lo),
            'stored_points_inside': int(mask.sum()), 'energy_J': area, 'mean_power_W': area/(hi-lo)}


def analyze(cfn_path, record):
    _, markers, _, _ = validate_record(record)
    metadata, times, current, power = inspect_recording(cfn_path)
    sync = fit_registration(times, current, markers)
    result = analyze_loaded(metadata, times, current, power, record, sync)
    require(hashlib.sha256(Path(cfn_path).read_bytes()).hexdigest() == metadata['source_sha256'],
            'CFN changed during analysis')
    return result


def analyze_loaded(metadata, times, current, power, record, sync):
    """Integrate a registered run without cropping or manufacturing a new CFN.

    The caller must decode/validate the original file and provide a registration
    returned by fit_registration. Continuous acquisition validates run identity
    jointly before calling this calculation with the original full arrays.
    """
    events, markers, duration, count = validate_record(record)
    offset, scale = sync['affine_center_offset_s'], sync['CFN_seconds_per_MCU_second']
    mapping = lambda x: offset+scale*x
    lo, hi = mapping(events['active_begin']), mapping(events['active_end'])
    require(times[0] <= mapping(markers[0, 0])-1.4 and times[-1] >= mapping(markers[1, -1])+1.4,
            'Recording does not cover both complete marker groups')
    active = integrate(times, power, lo, hi)
    require(active['energy_J'] > 0., 'Zero active-batch energy is not a powered-board measurement')
    before = integrate(times, power, mapping(events['idle_before_begin']+3.), mapping(events['idle_before_end']-3.))
    after = integrate(times, power, mapping(events['idle_after_begin']+3.), mapping(events['idle_after_end']-3.))
    baseline = (before['mean_power_W']+after['mean_power_W'])/2
    shifts = []
    for delta in (-2., -1., -.5, 0., .5, 1., 2.):
        window = integrate(times, power, lo+delta, hi+delta)
        shifts.append({'shift_CFN_s': delta, 'energy_J': window['energy_J'],
                       'uJ_per_inference': window['energy_J']*1e6/count})
    # All shifts move both boundaries equally and keep the number of calls fixed.
    sensitivity = max(abs(x['energy_J']-active['energy_J']) for x in shifts)
    segments = [integrate(times, power, lo+(hi-lo)*k/3, lo+(hi-lo)*(k+1)/3) for k in range(3)]
    digest = metadata['source_sha256']
    return {
        'schema': 'kanids-hw500-trace-analysis-v1', 'status': 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_ESTIMATE',
        'run_id': record.get('run_id'), 'board_kind': record['board_kind'], 'model': record['model'],
        'phase': record.get('phase'), 'repeat': record.get('repeat'), 'source_sha256': digest,
        'recording_integrity': metadata, 'registration': sync,
        'workload': {'distinct_frozen_flows': 500, 'completed_inferences': count,
                     'full_traversals': count//500, 'active_MCU_seconds': duration,
                     'mean_replay_us_per_inference': duration*1e6/count,
                     'boundary': 'Flash prepared-row load + prediction + checksum; same batch for time and energy'},
        'primary_energy': dict(active, uJ_per_inference=active['energy_J']*1e6/count,
            method='Trapezoidal integral of saved VBUS*IBUS over LED-registered full active interval / actual completed count',
            exact_hardware_synchronization=False, idle_subtracted=False, NRG_used=False),
        'idle_before_guard': before, 'idle_after_guard': after,
        'diagnostic_only': {'mean_idle_power_W': baseline,
            'incremental_energy_J': active['energy_J']-baseline*(hi-lo),
            'incremental_uJ_per_inference': (active['energy_J']-baseline*(hi-lo))*1e6/count,
            'idle_after_minus_before_W': after['mean_power_W']-before['mean_power_W'],
            'note': 'Signed descriptive subtraction, never clamped or substituted for primary energy.'},
        'equal_duration_active_segments': segments,
        'registration_shift_sensitivity': shifts,
        'maximum_absolute_energy_shift_J': sensitivity,
        'maximum_absolute_energy_shift_percent': sensitivity/active['energy_J']*100.,
        'limits': LIMITS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfn', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.cfn, json.loads(args.record.read_text(encoding='utf-8')))
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(result['status'])


if __name__ == '__main__':
    main()

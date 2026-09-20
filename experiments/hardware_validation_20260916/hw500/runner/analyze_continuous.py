#!/usr/bin/env python3
"""Register one untouched FNB58 recording against an ordered HW500 session.

Host monotonic timestamps identify runs, including the intervening upload and
LEDTEST gaps. Only the independently fitted MCU/LED bookends set energy bounds.
No choice is made between multiple acceptable assignments, and no favorable
runs or time windows are selected. A regular CFN grid is not an ADC clock.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from analyze_trace import (require, integer, validate_record, inspect_recording,
                           find_patterns, fit_registration, analyze_loaded)

SCALE_MIN, SCALE_MAX = .995, 1.005
HOST_RESIDUAL_MAX = 1.5
# The operator confirms a NEW recording within ten seconds. An additional
# two seconds allows an uncalibrated common display/filter/transport delay.
OFFSET_MIN, OFFSET_MAX = -1.5, 12.
MAX_ASSIGNMENT_NODES = 10000


def validate_host_records(records):
    require(isinstance(records, list) and 1 <= len(records) <= 26,
            'One to 26 completed HW500 records required')
    require(len({r.get('run_id') for r in records}) == len(records) and
            all(isinstance(r.get('run_id'), str) and r['run_id'] for r in records),
            'Missing/duplicate run identities')
    require(len({r.get('board_kind') for r in records}) == 1, 'Multiple boards in one recording')
    common = None
    previous_done = None
    host_edges, marker_schedules = [], []
    for record in records:
        _, markers, _, _ = validate_record(record)
        timing = record.get('host_timing', {})
        names = ('recording_confirm_monotonic_ns', 'run_send_monotonic_ns',
                 'prep_receive_monotonic_ns', 'done_receive_monotonic_ns')
        require(set(names).issubset(timing), 'Missing host monotonic synchronization fields')
        confirm, send, prep, done = [integer(timing[n], n) for n in names]
        require(0 < confirm <= send <= prep < done, 'Host clock reset or invalid event order')
        require(prep-send <= 2000000000, 'RUN to PREP exceeds two-second transport gate')
        if common is None:
            common = confirm
        require(confirm == common, 'Records do not share one recording confirmation/host clock')
        require(previous_done is None or send > previous_done,
                'Overlapping/out-of-order runs or discontinuous host clock')
        previous_done = done
        # PREP is flushed two seconds before MCU event origin; DONE is emitted
        # two seconds after the final post LED edge. RX timing is only used for
        # identity association, never as the primary inference duration.
        expected = markers[1, -1] - markers[0, 0] + 4.
        observed = (done-prep)/1e9
        require(abs(observed-expected) <= 2.+.005*expected,
                'Host RUN duration inconsistent with MCU marker schedule')
        pre_x = (prep-common)/1e9 + 2.
        post_x = (done-common)/1e9 - (markers[1, -1]-markers[1, 0]+2.)
        require(pre_x < post_x, 'Invalid host marker timing')
        host_edges.append([pre_x, post_x])
        marker_schedules.append(markers)
    return np.asarray(host_edges), marker_schedules


def _prefix_feasible(xs, ys):
    """Conservative necessary constraints; final bounded regression is stricter."""
    lo, hi = SCALE_MIN, SCALE_MAX
    for j in range(len(xs)):
        for i in range(j):
            dx, dy = xs[j]-xs[i], ys[j]-ys[i]
            if dx <= 0:
                return False
            lo = max(lo, (dy-2*HOST_RESIDUAL_MAX)/dx)
            hi = min(hi, (dy+2*HOST_RESIDUAL_MAX)/dx)
            if lo > hi:
                return False
    possible_offset_lo = max(y-hi*x-HOST_RESIDUAL_MAX for x, y in zip(xs, ys))
    possible_offset_hi = min(y-lo*x+HOST_RESIDUAL_MAX for x, y in zip(xs, ys))
    return max(possible_offset_lo, OFFSET_MIN) <= min(possible_offset_hi, OFFSET_MAX)


def _host_fit(xs, ys):
    design = np.column_stack([np.ones(len(xs)), xs])
    # Scale the time coordinate to avoid poor conditioning for long sessions.
    span = max(float(max(xs)), 1.)
    scaled = np.column_stack([np.ones(len(xs)), np.asarray(xs)/span])
    fit = lsq_linear(scaled, ys, bounds=([OFFSET_MIN, SCALE_MIN*span],
                                      [OFFSET_MAX, SCALE_MAX*span]), tol=1e-12)
    offset, scale = fit.x[0], fit.x[1]/span
    residuals = np.asarray(ys)-design@np.array([offset, scale])
    if not fit.success or np.max(np.abs(residuals)) > HOST_RESIDUAL_MAX:
        return None
    return dict(offset_CFN_s=float(offset), CFN_seconds_per_host_second=float(scale),
                residuals_s=residuals.tolist(), max_abs_residual_s=float(np.max(np.abs(residuals))),
                rms_residual_s=float(np.sqrt(np.mean(residuals**2))))


def assign_pairs(candidates, host_edges, marker_schedules):
    """Unique joint assignment of all runs; LEDTEST groups remain unassigned."""
    options = []
    for host, markers in zip(host_edges, marker_schedules):
        separation = markers[1, 0]-markers[0, 0]
        choices = []
        for i, first in enumerate(candidates):
            y0 = first['center']
            if not SCALE_MIN*host[0]+OFFSET_MIN-HOST_RESIDUAL_MAX <= y0 <= SCALE_MAX*host[0]+OFFSET_MAX+HOST_RESIDUAL_MAX:
                continue
            for j in range(i+1, len(candidates)):
                y1 = candidates[j]['center']
                if y1-y0 > 1.01*separation:
                    break
                if .99 <= (y1-y0)/separation <= 1.01 and _prefix_feasible(host, [y0, y1]):
                    choices.append((i, j))
        require(choices, 'Missing host-compatible LED marker pair for run '+str(len(options)+1))
        options.append(choices)
    solutions = []
    nodes = 0

    def visit(k, path, xs, ys):
        nonlocal nodes
        nodes += 1
        require(nodes <= MAX_ASSIGNMENT_NODES, 'Ambiguous marker assignment exceeds bounded search')
        if len(solutions) > 1:
            return
        if k == len(options):
            fitted = _host_fit(xs, ys)
            if fitted is not None:
                solutions.append((list(path), fitted))
            return
        for pair in options[k]:
            # Marker groups must be unique and the preceding post group must
            # end before the next pre group begins; identical group order alone
            # is never sufficient for assigning scientific run identities.
            if path and candidates[pair[0]]['center'] <= candidates[path[-1][1]]['center']+12:
                continue
            new_xs = xs+list(host_edges[k])
            new_ys = ys+[candidates[q]['center'] for q in pair]
            if _prefix_feasible(new_xs, new_ys):
                visit(k+1, path+[pair], new_xs, new_ys)
                if len(solutions) > 1:
                    return

    visit(0, [], [], [])
    require(len(solutions) == 1, 'Missing/ambiguous joint LED assignment: '+str(len(solutions)))
    pairs, fitted = solutions[0]
    return pairs, fitted, {'candidate_pairs_per_run': [len(x) for x in options],
                          'assignment_search_nodes': nodes, 'accepted_joint_assignments': 1}


def analyze_session_trace(cfn_path, records):
    """Analyze all supplied completed records; raise ValueError on any gate.

    Partial sessions are allowed as a completed prefix. Completeness of the
    acquisition plan is enforced separately by the runner/summary generator.
    """
    host_edges, marker_schedules = validate_host_records(records)
    metadata, times, current, power = inspect_recording(cfn_path, continuous=True)
    require(0 <= times[0] <= .2, 'Continuous CFN must start as a new recording near zero')
    require(metadata['stop_time_s'] == 5, 'Stop time must be five seconds')
    candidates = find_patterns(times, current, marker_schedules[0][0])
    pairs, host_fit, search = assign_pairs(candidates, host_edges, marker_schedules)
    analyses, assigned, refined = [], [], []
    for record, markers, pair, host in zip(records, marker_schedules, pairs, host_edges):
        selected = [candidates[i] for i in pair]
        registration = fit_registration(times, current, markers, selected_pair=selected)
        # Refined independent marker centers must retain the host association;
        # do not silently substitute an alternative pair after a failed fit.
        pre_center, post_center = np.asarray(registration['edge_centers_CFN_s'])[[0, 6]]
        refined += [float(pre_center), float(post_center)]
        boundaries = [pre_center-1.4, registration['edge_centers_CFN_s'][-1]+1.4]
        for a, b in metadata['voltage_below_4V_spans_CFN_s']:
            require(b < boundaries[0] or a > boundaries[1],
                    'USB voltage below 4 V within run/marker window: '+record['run_id'])
        registration['continuous_assignment'] = {
            'run_id': record['run_id'], 'candidate_indices_zero_based': list(pair),
            'host_expected_marker_times_relative_to_confirmation_s': host.tolist(),
            'identity_method': 'Unique ordered joint host/CFN affine association; independent MCU edges for energy',
            'voltage_gate_CFN_bounds_s': boundaries,
        }
        analysis = analyze_loaded(metadata, times, current, power, record, registration)
        analysis['acquisition_mode'] = 'one_unmodified_continuous_CFN_per_board'
        analyses.append(analysis)
        assigned += list(pair)
    refined_fit = _host_fit(host_edges.ravel().tolist(), refined)
    require(refined_fit is not None, 'Refined LED edges disagree with joint host timing')
    # The original bytes are referenced for every run. No cropped CFN with
    # newly invented extrema/header/timestamps is written.
    require(hashlib.sha256(Path(cfn_path).read_bytes()).hexdigest() == metadata['source_sha256'],
            'CFN changed during continuous analysis')
    return {
        'schema': 'kanids-hw500-continuous-analysis-v1',
        'status': 'ACCEPTED_DESCRIPTIVE_WHOLE_BOARD_SERIES',
        'source_sha256': metadata['source_sha256'], 'completed_run_count': len(records),
        'full_series': len(records) == 26,
        'analyses': analyses,
        'registration': dict(search, coarse_host_fit=host_fit, refined_host_fit=refined_fit,
            candidates=candidates, assigned_candidate_indices_zero_based=assigned,
            unassigned_candidates=[dict(candidate_index_zero_based=i, **item) for i, item in enumerate(candidates) if i not in assigned],
            unassigned_note='Unassigned groups may include upload LEDTEST; they are retained, not interpreted as completed scientific runs.',
            host_scale_range=[SCALE_MIN, SCALE_MAX], maximum_host_residual_s=HOST_RESIDUAL_MAX,
            recording_start_to_confirmation_instruction_s=[0, 10],
            allowed_common_CFN_offset_s=[OFFSET_MIN, OFFSET_MAX],
            host_timestamps_used_for='Run identity only; not inference latency or energy integration endpoints',
            absolute_clock_or_common_filter_delay_calibrated=False),
        'limits': [
            'All run acceptance is deferred until the final continuous file is decoded and registered.',
            'One continuous file cannot prove fresh acquisition of every intervening sample; valid markers and regular grid are bounded consistency checks.',
            'Upload, calibration, LEDTEST and inter-run gaps are preserved in raw evidence and excluded from active energy windows.',
            'Host association bounds are acceptance gates, not a traceable clock or instrument uncertainty budget.',
            'Partial sessions do not produce a complete five-repeat/model summary.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfn', required=True, type=Path)
    parser.add_argument('--records', required=True, type=Path,
                        help='JSON list of immutable completed software run records')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = analyze_session_trace(args.cfn, json.loads(args.records.read_text(encoding='utf-8')))
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(result['status'])


if __name__ == '__main__':
    main()

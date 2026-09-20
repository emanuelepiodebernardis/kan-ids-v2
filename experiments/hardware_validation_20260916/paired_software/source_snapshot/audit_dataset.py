"""Streaming structural audit. No training, resampling, or model evaluation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_SALT = 'KAN_IDS_PAIR_DISJOINT_V013_20260916'
SOURCE_SHA256 = '26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974'
NUMERIC = ['src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
           'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
           'dns_qclass', 'dns_qtype', 'dns_rcode', 'http_request_body_len',
           'http_response_body_len', 'http_status_code']
CATEGORICAL = ['proto', 'service', 'conn_state', 'dns_rejected']
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def key_hash(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def pair_split(src, dst):
    key = json.dumps(sorted([src, dst]), ensure_ascii=False, separators=(',', ':'))
    digest = hashlib.sha256((SPLIT_SALT + '\0' + key).encode('utf-8')).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 100
    return ('train' if bucket < 60 else 'validation' if bucket < 80 else 'test'), digest.hex()


def audit(csv_path, out, require_canonical=True):
    csv_path, out = Path(csv_path), Path(out)
    digest = sha256(csv_path)
    if require_canonical and digest != SOURCE_SHA256:
        raise ValueError('SOURCE_SHA256_MISMATCH: a different dataset needs its own protocol.')
    out.mkdir(parents=True, exist_ok=False)
    counts = {s: Counter() for s in SPLITS}
    type_counts = {s: Counter() for s in SPLITS}
    groups = {s: set() for s in SPLITS}
    hosts = {s: set() for s in SPLITS}
    raw_sets = {s: set() for s in SPLITS}
    input_counts = {s: Counter() for s in SPLITS}
    raw_counts, pair_counts = Counter(), Counter()
    all_input_labels = defaultdict(set)
    all_input_types = defaultdict(set)
    parent = {}
    row_hashers = {s: hashlib.sha256() for s in SPLITS}
    all_types, n_rows = set(), 0

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    assignment_path = out / 'split_assignments.csv.gz'
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f, \
            assignment_path.open('wb') as raw_output, \
            gzip.GzipFile(filename='', mode='wb', fileobj=raw_output, mtime=0) as compressed, \
            io.TextIOWrapper(compressed, encoding='utf-8', newline='') as g:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = set(NUMERIC + CATEGORICAL + ['src_ip', 'dst_ip', 'label', 'type'])
        if len(header) != len(set(header)) or not required.issubset(header):
            raise ValueError('SCHEMA_ERROR: missing or repeated columns')
        writer = csv.writer(g, lineterminator='\n')
        writer.writerow(['row_id', 'split', 'group_sha256'])
        for row_id, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()):
                raise ValueError('MALFORMED_CSV_ROW: ' + str(row_id))
            if row['label'] not in ('0', '1'):
                raise ValueError('INVALID_BINARY_LABEL: ' + str(row_id))
            label, attack_type = int(row['label']), row['type']
            if bool(label) != (attack_type != 'normal'):
                raise ValueError('LABEL_TYPE_DISAGREEMENT: ' + str(row_id))
            src, dst = row['src_ip'], row['dst_ip']
            if not src or not dst:
                raise ValueError('MISSING_ENDPOINT: ' + str(row_id))
            split, group = pair_split(src, dst)
            writer.writerow([row_id, split, group])
            row_hashers[split].update(struct.pack('<q', row_id))
            counts[split][label] += 1
            type_counts[split][attack_type] += 1
            all_types.add(attack_type)
            groups[split].add(group)
            hosts[split].update((src, dst))
            raw_key = key_hash([row[c] for c in header])
            # Explicit pre-selection candidate view, not fitted/quantized inputs.
            input_key = key_hash([row[c] for c in NUMERIC + CATEGORICAL])
            raw_sets[split].add(raw_key)
            raw_counts[raw_key] += 1
            input_counts[split][input_key] += 1
            all_input_labels[input_key].add(label)
            all_input_types[input_key].add(attack_type)
            pair_counts[(src, dst)] += 1
            a, b = find(src), find(dst)
            if a != b:
                parent[b] = a
            n_rows += 1
            if n_rows % 50000 == 0:
                print('DATA_AUDIT_ROWS', n_rows, flush=True)
    component_rows = Counter()
    for (src, dst), count in pair_counts.items():
        component_rows[find(src)] += count
    overlaps = {}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        shared = set(input_counts[left]) & set(input_counts[right])
        overlaps[left + '_to_' + right] = {
            'shared_pair_groups': len(groups[left] & groups[right]),
            'shared_full_row_keys': len(raw_sets[left] & raw_sets[right]),
            'shared_hosts': len(hosts[left] & hosts[right]),
            'shared_candidate_input_keys': len(shared),
            'right_rows_matching_left_candidate_inputs': sum(input_counts[right][h] for h in shared),
        }
    report = {
        'status': 'STRUCTURAL_AUDIT_COMPLETE', 'classifier_fits': 0,
        'source_csv_sha256': digest, 'source_bytes': csv_path.stat().st_size,
        'source_rows': n_rows, 'source_columns': header,
        'row_id_definition': 'Zero-based data-row position; BOM/header excluded; no row filtering.',
        'matching_definition': 'SHA256 over JSON arrays of decoded CSV field strings; numerical formatting is preserved.',
        'candidate_input_columns': NUMERIC + CATEGORICAL,
        'candidate_input_warning': 'Before MI, imputation, quantile transform, and Q12; not a feature-disjoint claim.',
        'split_rule': {'salt': SPLIT_SALT, 'key': 'compact UTF-8 JSON sorted [src_ip,dst_ip]',
                       'bucket': 'big-endian first 8 SHA256 bytes of salt+NUL+key, modulo 100',
                       'train': '[0,60)', 'validation': '[60,80)', 'test': '[80,100)',
                       'uses_labels_or_model_scores': False},
        'split_assignments_sha256': sha256(assignment_path),
        'row_ids_hash_encoding': 'concatenated little-endian signed int64',
        'splits': {s: {'rows': sum(counts[s].values()), 'normal': counts[s][0],
                       'attack': counts[s][1], 'pair_groups': len(groups[s]),
                       'types': dict(sorted(type_counts[s].items())),
                       'absent_types': sorted(all_types - set(type_counts[s])),
                       'row_ids_sha256': row_hashers[s].hexdigest()} for s in SPLITS},
        'overlap': overlaps,
        'full_row_unique': len(raw_counts), 'full_row_duplicate_excess': n_rows - len(raw_counts),
        'input_binary_conflicting_groups': sum(len(v) > 1 for v in all_input_labels.values()),
        'input_type_conflicting_groups': sum(len(v) > 1 for v in all_input_types.values()),
        'host_graph_component_rows': sorted(component_rows.values(), reverse=True),
        'time_or_capture_fields_present': [c for c in header if c.lower() in {
            'ts', 'timestamp', 'time', 'date', 'stime', 'ltime', 'session_id', 'uid', 'capture_id'}],
        'claims': {'pair_disjoint': True, 'host_disjoint': False,
                   'time_disjoint': False, 'feature_disjoint': False,
                   'previously_unseen_dataset': False},
    }
    checks = {
        'source_rows_match': n_rows == 211043 if require_canonical else True,
        'both_binary_classes_in_every_split': all(counts[s][0] and counts[s][1] for s in SPLITS),
        'pair_groups_do_not_cross_splits': all(v['shared_pair_groups'] == 0 for v in overlaps.values()),
        'full_rows_do_not_cross_splits': all(v['shared_full_row_keys'] == 0 for v in overlaps.values()),
    }
    report['checks'] = {k: bool(v) for k, v in checks.items()}
    report['pilot_allowed'] = all(checks.values())
    write_json(out / 'DATA_AUDIT.json', report)
    if not report['pilot_allowed']:
        raise ValueError('DATA_AUDIT_BLOCKED: see DATA_AUDIT.json')
    return report

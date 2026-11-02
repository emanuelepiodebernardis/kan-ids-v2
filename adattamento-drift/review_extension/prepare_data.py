"""Rebuild the original TON partitions and a chronological UNSW-NB15 check.

No source data are redistributed. TON's parser is the original versioned parser.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ton_temporal'))
import costruisci_flusso as original


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_partition(out, name, arrays, metadata):
    out.mkdir(parents=True, exist_ok=True)
    # Retain exactly the original npz key order for reproducibility checks.
    path = out / (name + '.npz')
    np.savez_compressed(path, **arrays)
    y = arrays['y']
    metadata.update(rows=len(y), normal=int((y == 0).sum()), attack=int(y.sum()),
                    sha256=sha(path),
                    array_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                  for k, v in arrays.items()})
    (out / (name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print(name, metadata['rows'], metadata['normal'], metadata['sha256'], flush=True)


def ton(raw, out):
    manifest = json.loads((HERE / 'vendor/ton_iot_mirror_v1_manifest.json').read_text())
    source = json.loads((HERE / 'vendor/source_manifest.json').read_text())
    for entry in source['files']:
        actual = sha(raw / entry['file'])
        if actual != entry['local_sha256']:
            raise ValueError('Source hash mismatch: ' + entry['file'])
    print('All 23 TON source hashes match the original campaign.', flush=True)
    audit = original.carica('audit_ton_full', HERE / 'vendor/audit_ton_full.py')
    days = {'A': ['2019-04-02', '2019-04-03', '2019-04-04', '2019-04-23'],
            'B': ['2019-04-24'], 'C': ['2019-04-25', '2019-04-26'],
            'D': ['2019-04-27', '2019-04-28', '2019-04-29']}
    X, y, tp, rid, frt, ts, te, names, counts = original.estrai(
        raw, audit, {x['name']: x for x in manifest['files']},
        sum(days.values(), []))
    assert counts['lette'] == 22339021 and len(y) == 22338152
    day = (frt * 1e6).astype('datetime64[us]').astype('datetime64[D]')
    inverse = {v: k for k, v in names.items()}
    for name, dates in days.items():
        ids = np.flatnonzero(np.isin(day, np.array(dates, dtype='datetime64[D]')))
        # Original extractor numbers type names in raw-file arrival order.
        types, first = np.unique(tp[ids], return_index=True)
        types = types[np.argsort(first)]
        mapping = {int(t): i for i, t in enumerate(types)}
        order = ids[np.lexsort((rid[ids], frt[ids]))]
        arrays = dict(X=X[order], y=y[order],
                      tipo=np.array([mapping[int(t)] for t in tp[order]], dtype=np.int16),
                      row_id=rid[order], frt=frt[order], t_start=ts[order], t_end=te[order],
                      tipi=np.array([inverse[int(t)] for t in types]))
        save_partition(out, name, arrays, {'source': 'TON_IoT registered mirror v1',
                                          'days': dates, 'source_hashes_verified': True,
                                          'parser_counts': counts})


def unsw(raw, out):
    import pandas as pd
    # Full raw files, not the publisher's preselected train/test tables.
    names = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
             'sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts',
             'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len',
             'Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
             'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
             'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
             'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']
    features = ['dur','sbytes','dbytes','sloss','dloss','Spkts','Dpkts','tcprtt']
    frames, source = [], {}
    previous_last = None
    removed_boundary_rows = []
    for i in range(1, 5):
        p = raw / f'UNSW-NB15_{i}.csv'
        source[p.name] = sha(p)
        with p.open('rb') as handle:
            first_line = handle.readline().decode('utf-8-sig').strip()
            handle.seek(max(0, p.stat().st_size - 5000))
            last_line = handle.readlines()[-1].decode('utf-8').strip()
        frame = pd.read_csv(p, names=names, low_memory=False,
                           usecols=features + ['Stime','Ltime','attack_cat','Label'])
        if first_line == previous_last:
            frame = frame.iloc[1:]
            removed_boundary_rows.append({'file':p.name, 'row':1, 'reason':'exact previous-file final row'})
        frames.append(frame)
        previous_last = last_line
    f = pd.concat(frames, ignore_index=True)
    assert len(f) == 2540044 and len(removed_boundary_rows) == 3
    X = f[features].to_numpy(dtype=np.float32)
    y = f.Label.to_numpy(dtype=np.uint8)
    start = f.Stime.to_numpy(dtype=np.float64)
    end = f.Ltime.to_numpy(dtype=np.float64)
    # Ltime is integer seconds in this archive. The max respects both end-time
    # fields without implying subsecond accuracy in the original timestamps.
    ready = np.maximum(end, start + X[:, 0].astype(np.float64))
    valid = np.isfinite(X).all(1) & (X >= 0).all(1) & np.isfinite(ready)
    valid &= np.isin(y, [0, 1]) & (end >= start)
    rid = np.arange(len(y), dtype=np.int64)[valid]
    order = np.lexsort((rid, ready[valid]))
    rid = rid[order]
    X, y, ready, start, end = [v[rid] for v in (X, y, ready, start, end)]
    kinds = f.attack_cat.fillna('Normal').str.strip().replace({'Backdoors':'Backdoor'}).to_numpy()[rid]
    tipi, tp = np.unique(kinds, return_inverse=True)
    # Fixed 40/20/40 chronological row fractions, specified before model results.
    n = len(y); cuts = [0, int(.4*n), int(.6*n), n]
    for name, a, b in zip(('A','B','D'), cuts[:-1], cuts[1:]):
        arrays = dict(X=X[a:b], y=y[a:b], tipo=tp[a:b].astype(np.int16),
                      row_id=rid[a:b], frt=ready[a:b], t_start=start[a:b],
                      t_end=end[a:b], tipi=tipi.astype(str))
        save_partition(out, name, arrays, {'source':'UNSW-NB15 full four CSVs, Kaggle mirror v1',
                                          'source_sha256':source, 'features':features,
                                          'raw_rows':2540047, 'publisher_reported_rows':2540044,
                                          'removed_boundary_duplicates':removed_boundary_rows,
                                          'excluded_rows':int((~valid).sum()),
                                          'split_rule':'chronological 40/20/40 by complete-feature time',
                                          'start_row':a, 'stop_row':b})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['ton','unsw']); p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); globals()[a.dataset](a.raw, a.out)

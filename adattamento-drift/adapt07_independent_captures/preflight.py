"""Audit source structure and support without fitting or scoring a detector."""

import argparse
import hashlib
import json
import platform
import subprocess
from array import array
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np

SCENARIOS = ("1-1", "48-1")
FEATURES = (
    b"duration",
    b"orig_bytes",
    b"resp_bytes",
    b"missed_bytes",
    b"orig_pkts",
    b"orig_ip_bytes",
    b"resp_pkts",
    b"resp_ip_bytes",
)
LABELS = {b"benign": 0, b"malicious": 1}
WARMUP_ROWS = 200_000
MICROSECOND = 1_000_000
FLOAT32_MAX = float(np.finfo(np.float32).max)


def date(value):
    return datetime.fromtimestamp(value / MICROSECOND, tz=UTC).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_number(token):
    if token in (b"-", b"(empty)"):
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if isfinite(value) and 0 <= value <= FLOAT32_MAX else None


def ordered_support(ready, labels):
    order = np.argsort(ready, kind="stable")
    ready = np.asarray(ready)[order]
    labels = np.asarray(labels)[order]
    size = len(labels)
    end_a = group_end(ready, 2 * size // 5)
    end_b = max(end_a, group_end(ready, 3 * size // 5))
    ranges = {
        "A": (0, end_a),
        "B": (end_a, end_b),
        "D": (end_b, size),
        "D_evaluated": (min(end_b + WARMUP_ROWS, size), size),
    }
    support = {}
    for name, (start, stop) in ranges.items():
        attacks = int(labels[start:stop].sum())
        support[name] = {
            "rows": stop - start,
            "benign": stop - start - attacks,
            "malicious": attacks,
            "first_ready_proxy_utc": date(int(ready[start])) if start < stop else None,
            "last_ready_proxy_utc": date(int(ready[stop - 1]))
            if start < stop
            else None,
        }
    reasons = [
        f"{name} lacks at least one class with complete features"
        for name in ("A", "B", "D_evaluated")
        if not (support[name]["benign"] and support[name]["malicious"])
    ]
    if support["D"]["rows"] <= WARMUP_ROWS:
        reasons.append(
            f"D has {support['D']['rows']} complete rows; needs more than {WARMUP_ROWS}"
        )
    return support, reasons


def group_end(ready, cut):
    if not cut:
        return 0
    return int(np.searchsorted(ready, ready[cut - 1], side="right"))


def inspect_log(path):
    raw_counts = Counter()
    complete_counts = Counter()
    invalid_by_feature = {key: Counter() for key in FEATURES}
    complete_type_counts = Counter()
    unique_ids = set()
    ready = array("q")
    labels = array("B")
    schema = None
    first_ts = last_ts = last_complete_ts = None
    closed = False
    with path.open("rb") as stream:
        for line_num, line in enumerate(stream, start=1):
            if line.startswith(b"#fields"):
                if schema is not None:
                    raise ValueError(f"{path.name}:{line_num}: repeated schema")
                schema = tuple(line.split()[1:])
                required = (*FEATURES, b"uid", b"ts", b"label", b"detailed-label")
                if len(schema) != len(set(schema)) or not set(required) <= set(schema):
                    raise ValueError(
                        f"{path.name}:{line_num}: Zeek schema lacks required fields"
                    )
                indexes = {field: schema.index(field) for field in required}
                if schema[-1] != b"detailed-label":
                    raise ValueError(
                        f"{path.name}:{line_num}: detailed label must be the last field"
                    )
                continue
            if line.startswith(b"#close"):
                closed = True
            if line.startswith(b"#") or not line.strip():
                continue
            if schema is None or closed:
                raise ValueError(
                    f"{path.name}:{line_num}: data outside the declared log"
                )
            fields = line.rstrip().split(maxsplit=len(schema) - 1)
            if len(fields) != len(schema):
                raise ValueError(f"{path.name}:{line_num}: unexpected field count")
            uid = fields[indexes[b"uid"]]
            binary_label = fields[indexes[b"label"]].lower()
            label = LABELS.get(binary_label)
            ts = optional_number(fields[indexes[b"ts"]])
            if (
                uid in (b"-", b"(empty)")
                or uid in unique_ids
                or label is None
                or ts is None
            ):
                raise ValueError(
                    f"{path.name}:{line_num}: invalid or duplicate UID, label or timestamp"
                )
            unique_ids.add(uid)
            ts_us = round(ts * MICROSECOND)
            first_ts = ts_us if first_ts is None else min(first_ts, ts_us)
            last_ts = ts_us if last_ts is None else max(last_ts, ts_us)
            raw_counts[label] += 1
            values = [optional_number(fields[indexes[key]]) for key in FEATURES]
            invalid = [
                key
                for key, value in zip(FEATURES, values, strict=True)
                if value is None
            ]
            if invalid:
                for key in invalid:
                    invalid_by_feature[key][label] += 1
                continue
            ready_us = ts_us + round(values[0] * MICROSECOND)
            ready.append(ready_us)
            last_complete_ts = (
                ready_us
                if last_complete_ts is None
                else max(last_complete_ts, ready_us)
            )
            labels.append(label)
            complete_counts[label] += 1
            complete_type_counts[
                fields[indexes[b"detailed-label"]].decode("utf-8")
            ] += 1

    if not closed or not raw_counts:
        raise ValueError(f"{path.name}: log lacks observations or the closing marker")
    support, reasons = ordered_support(ready, labels)
    return {
        "schema": [field.decode("ascii") for field in schema],
        "first_flow_start_utc": date(first_ts),
        "last_flow_start_utc": date(last_ts),
        "last_complete_feature_ready_proxy_utc": date(last_complete_ts)
        if last_complete_ts is not None
        else None,
        "unique_uids": len(unique_ids),
        "rows_raw": {"benign": raw_counts[0], "malicious": raw_counts[1]},
        "rows_excluded": {
            "benign": raw_counts[0] - complete_counts[0],
            "malicious": raw_counts[1] - complete_counts[1],
        },
        "invalid_by_feature": {
            key.decode("ascii"): {"benign": counts[0], "malicious": counts[1]}
            for key, counts in invalid_by_feature.items()
        },
        "detailed_labels_complete": dict(sorted(complete_type_counts.items())),
        "support": support,
        "status": "BLOCKED" if reasons else "PASS",
        "reasons": reasons,
    }


def audit(root):
    captures = {}
    for scenario in SCENARIOS:
        started = perf_counter()
        path = root / f"iot23-{scenario}.log"
        record = {"file": path.name}
        try:
            record.update(bytes=path.stat().st_size, sha256=sha256(path))
            record.update(inspect_log(path))
        except (OSError, ValueError, OverflowError) as exc:
            record.update(status="BLOCKED", reasons=[str(exc)])
        record["seconds"] = round(perf_counter() - started, 3)
        captures[scenario] = record
        print(f"{scenario}: {record['status']}", flush=True)
    has_timestamps = all("first_flow_start_utc" in item for item in captures.values())
    periods_disjoint = False
    if has_timestamps:
        ordered = sorted(captures.values(), key=lambda row: row["first_flow_start_utc"])
        periods_disjoint = all(
            max(
                a["last_flow_start_utc"],
                a["last_complete_feature_ready_proxy_utc"] or "",
            )
            < b["first_flow_start_utc"]
            for a, b in pairwise(ordered)
        )
    return {
        "status": "PASS"
        if periods_disjoint
        and all(row["status"] == "PASS" for row in captures.values())
        else "BLOCKED",
        "periods_disjoint": periods_disjoint,
        "captures": captures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    here = Path(__file__).resolve().parent
    report = {
        "execution_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True
        ).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sources": {
            name: sha256(here / name)
            for name in ("PROTOCOL.md", "INVENTORY.md", "preflight.py")
        },
        **audit(args.root),
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

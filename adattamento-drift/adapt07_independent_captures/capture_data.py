"""Materialize the verified eight-feature stream in preregistered order."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from preflight import (
    FEATURES,
    LABELS,
    MICROSECOND,
    WARMUP_ROWS,
    date,
    group_end,
    optional_number,
)


@dataclass(frozen=True)
class Capture:
    features: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    types: np.ndarray
    names: tuple[str, ...]
    cuts: tuple[int, int]


def load_capture(path: Path, support: dict) -> Capture:
    size = sum(support[period]["rows"] for period in ("A", "B", "D"))
    features = np.empty((size, len(FEATURES)), dtype=np.float32)
    labels = np.empty(size, dtype=np.uint8)
    row_ids = np.empty(size, dtype=np.int64)
    ready = np.empty(size, dtype=np.int64)
    types = np.empty(size, dtype=np.int16)
    names = {}
    count = 0
    with path.open("rb") as stream:
        for line_num, line in enumerate(stream, start=1):
            if line.startswith(b"#fields"):
                schema = tuple(line.split()[1:])
                feature_columns = [schema.index(name) for name in FEATURES]
                ts, label, detail = (
                    schema.index(name) for name in (b"ts", b"label", b"detailed-label")
                )
                continue
            if line.startswith(b"#") or not line.strip():
                continue
            fields = line.rstrip().split(maxsplit=len(schema) - 1)
            values = [optional_number(fields[i]) for i in feature_columns]
            if None in values:
                continue
            if count == size:
                raise ValueError(
                    f"{path.name}: more complete rows than in the preflight"
                )
            features[count] = values
            labels[count] = LABELS[fields[label].lower()]
            row_ids[count] = line_num
            ready[count] = round(float(fields[ts]) * MICROSECOND) + round(
                values[0] * MICROSECOND
            )
            # Keep the publisher's binary label even if the detailed label is absent.
            kind = fields[detail].decode("utf-8")
            group = ("Malicious" if labels[count] else "Benign", kind)
            types[count] = names.setdefault(group, len(names))
            count += 1
    if count != size:
        raise ValueError(f"{path.name}: complete row count differs from the preflight")
    order = np.argsort(ready, kind="stable")
    ordered_ready, ordered_y = ready[order], labels[order]
    a = group_end(ordered_ready, 2 * size // 5)
    b = max(a, group_end(ordered_ready, 3 * size // 5))
    for period, start, stop in (
        ("A", 0, a),
        ("B", a, b),
        ("D", b, size),
        ("D_evaluated", b + WARMUP_ROWS, size),
    ):
        n_attacks = int(ordered_y[start:stop].sum())
        actual = {
            "rows": stop - start,
            "benign": stop - start - n_attacks,
            "malicious": n_attacks,
            "first_ready_proxy_utc": date(int(ordered_ready[start])),
            "last_ready_proxy_utc": date(int(ordered_ready[stop - 1])),
        }
        if actual != support[period]:
            raise ValueError(f"{path.name}: {period} differs from the preflight")
    return Capture(
        features[order],
        ordered_y,
        row_ids[order],
        types[order],
        tuple(f"{label}:{kind}" for label, kind in names),
        (a, b),
    )

#!/usr/bin/env python3
"""Strict, read-only CFN decoding and numerical audit (stdlib only).

Binary layout reference, not manufacturer documentation:
https://github.com/didim99/usbmeter-utils/blob/master/cfn2csv.py
https://github.com/didim99/usbmeter-utils/blob/master/common.py
The reference describes FNIRSI Toolbox v0.0.6 files. This decoder validates
this file's byte layout, channel extrema, time grid and VI/P consistency.
NRG/CAP units below follow that specification and remain explicitly labelled.
No inference is made about ADC rate, firmware workload or calibration accuracy.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

CHANNELS = {
    0: "VBUS_V", 1: "IBUS_A", 2: "Dplus_V", 3: "Dminus_V",
    4: "PBUS_W", 5: "CAP_Ah_as_documented", 6: "NRG_Wh_as_documented",
}


def decode(path):
    raw = path.read_bytes()
    offset = 0

    def read(fmt):
        nonlocal offset
        fmt = "<" + fmt
        size = struct.calcsize(fmt)
        if offset + size > len(raw):
            raise ValueError(f"Truncated CFN at byte {offset}")
        result = struct.unpack_from(fmt, raw, offset)
        offset += size
        return result

    rate, start, stop, wait, count = read("diiih")
    if not math.isfinite(rate) or rate <= 0 or not 1 <= count <= 7:
        raise ValueError("Invalid rate or channel count")
    channels = []
    seen = set()
    for _ in range(count):
        code, color, extrema = read("hIB")
        if code not in CHANNELS or code in seen or extrema not in (0, 1):
            raise ValueError("Unsupported/duplicate channel or invalid extrema flag")
        seen.add(code)
        desc = {"code": code, "name": CHANNELS[code], "argb": f"#{color:08x}",
                "has_header_extrema": bool(extrema)}
        if extrema:
            desc["header_max"], desc["header_min"] = read("dd")
        channels.append(desc)
    n, = read("i")
    data_offset = offset
    stride = (count + 1) * 8
    if n < 2 or len(raw) != offset + n * stride:
        raise ValueError("Point count does not account for exact EOF")
    rows = [read("d" * (count + 1)) for _ in range(n)]
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError("Non-finite data")
    metadata = {
        "source_name": path.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw), "sample_rate_header_sps": rate,
        "start_current_mA": start, "stop_current_mA": stop,
        "stop_time_s": wait, "channels": channels, "records": n,
        "data_offset_bytes": data_offset, "record_bytes": stride,
        "decoded_to_exact_eof": offset == len(raw),
    }
    return metadata, rows


def analyse(metadata, rows):
    names = ["time_s"] + [c["name"] for c in metadata["channels"]]
    indices = {c["code"]: i + 1 for i, c in enumerate(metadata["channels"])}
    time = [r[0] for r in rows]
    delta = [b-a for a, b in zip(time, time[1:])]
    nominal_step = 1 / metadata["sample_rate_header_sps"]
    tol = max(1e-10, nominal_step * 1e-8)
    if min(delta) <= 0:
        raise ValueError("Non-increasing recorded time")
    metadata["time"] = {
        "first_s": time[0], "last_s": time[-1], "span_s": time[-1]-time[0],
        "step_min_s": min(delta), "step_max_s": max(delta),
        "step_deviations_from_nominal": sum(abs(d-nominal_step)>tol for d in delta),
        "missing_nominal_time_slots": sum(max(0, round(d/nominal_step)-1) for d in delta),
        "strictly_increasing": True,
        "note": "Regular recorded time coordinates do not establish real acquisition jitter or ADC rate.",
    }
    stats = {}
    for i, c in enumerate(metadata["channels"], start=1):
        values = [r[i] for r in rows]
        stats[c["name"]] = {"min": min(values), "max": max(values),
                              "mean": statistics.fmean(values),
                              "population_sd_descriptive": statistics.pstdev(values)}
        if c["has_header_extrema"]:
            c["header_extrema_match_data_exactly"] = (
                c["header_min"] == min(values) and c["header_max"] == max(values))
    metadata["statistics"] = stats

    def trap(i):
        return math.fsum((a[i]+b[i])*0.5*d for a,b,d in zip(rows, rows[1:], delta))

    if 0 in indices and 1 in indices:
        v, i = indices[0], indices[1]
        derived_power = [r[v]*r[i] for r in rows]
        energy = math.fsum((a+b)*0.5*d for a,b,d in zip(derived_power, derived_power[1:], delta))
        metadata["derived_energy"] = {
            "method": "Trapezoidal integral of recorded VBUS*IBUS over recorded time; no endpoint extrapolation",
            "energy_J": energy, "energy_Wh": energy/3600,
            "time_weighted_mean_power_W": energy/(time[-1]-time[0]),
            "charge_Ah": trap(i)/3600,
            "interpretation": "Energy over the connection-test interval; workload and firmware are not identified, so this is not inference energy or a certified idle baseline.",
        }
        if 4 in indices:
            p = indices[4]
            metadata["derived_energy"]["PBUS_minus_VI_max_abs_W"] = max(
                abs(r[p]-power) for r,power in zip(rows, derived_power))
        if 5 in indices:
            cap = indices[5]
            cap_delta = rows[-1][cap]-rows[0][cap]
            metadata["CAP_crosscheck"] = {"delta_Ah_as_documented": cap_delta,
                "minus_integral_current_Ah": cap_delta-trap(i)/3600}
        if 6 in indices:
            nrg = indices[6]
            nrg_delta = rows[-1][nrg]-rows[0][nrg]
            metadata["NRG_crosscheck"] = {
                "delta_Wh_as_documented": nrg_delta,
                "delta_J_if_Wh": nrg_delta*3600,
                "ratio_to_integrated_VI_energy": nrg_delta*3600/energy if energy else None,
                "integrated_VI_energy_divided_by_NRG_delta": energy/(nrg_delta*3600) if nrg_delta else None,
                "status": "UNDEFINED_ZERO_INTEGRATED_ENERGY" if not energy else ("INCONSISTENT_WITH_RECORDED_POWER_AND_TIME" if abs(nrg_delta*3600/energy-1)>0.01 else "NUMERICALLY_CONSISTENT"),
                "note": "Unit interpretation follows the cited third-party CFN specification. Do not silently multiply NRG by a correction factor; retain raw channel and diagnose software/version behavior.",
            }
        triplets = [(r[v],r[i],r[indices[4]] if 4 in indices else r[v]*r[i]) for r in rows]
        boundaries = [j for j in range(1,len(rows)) if triplets[j] != triplets[j-1]]
        endpoints = [0] + boundaries + [len(rows)]
        runs = [b-a for a,b in zip(endpoints,endpoints[1:])]
        equal_pairs = len(rows)-1-len(boundaries)
        block4 = [triplets[j:j+4] for j in range(0,len(rows)-3,4)]
        metadata["repeated_values"] = {
            "adjacent_equal_VI_P_pairs": equal_pairs,
            "adjacent_pairs": len(rows)-1,
            "adjacent_equal_fraction": equal_pairs/(len(rows)-1),
            "value_change_count": len(boundaries),
            "unique_VI_P_triplets": len(set(triplets)),
            "run_length_counts_records": dict(sorted(collections.Counter(runs).items())),
            "value_change_index_mod_10_counts": dict(sorted(collections.Counter(j%10 for j in boundaries).items())),
            "first_20_change_indices": boundaries[:20],
            "zero_aligned_4_record_blocks": len(block4),
            "constant_zero_aligned_4_record_blocks": sum(len(set(b))==1 for b in block4),
            "note": "Repeated saved values do not establish the tester's internal ADC rate, statistical independence, or acquisition jitter. The companion comparison script derives the observed change grid separately for each file.",
        }
    return names, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    metadata, rows = decode(args.source)
    names, metadata = analyse(metadata, rows)
    metadata["format_sources"] = [
        "https://github.com/didim99/usbmeter-utils/blob/master/cfn2csv.py",
        "https://github.com/didim99/usbmeter-utils/blob/master/common.py",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{args.source.stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names)
        writer.writerows(rows)
    with (args.out / f"{args.source.stem}_summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

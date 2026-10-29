"""Generate the two tables and two figures for the first article draft.

The script reads only versioned manifests and the verified final analysis. It
does not load A/B/C/D, fit a model, replay a stream, or tune a parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = HERE / "articolo_02" / "validazione_finale_D" / "analisi_finale.json"
DEFAULT_SPLIT = HERE / "split_manifest.json"
DEFAULT_OUTPUT = HERE / "articolo_02" / "manuscript_assets"
BASE_MERGE = "200a7985eb99d09926aa2e0c5b4606d0fa30ff4a"
SEEDS = [42, 43, 44, 45, 46]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_path(path: Path) -> str:
    """Store repository-relative paths when the source belongs to this tree."""
    try:
        return path.resolve().relative_to(HERE).as_posix()
    except ValueError:
        return str(path.resolve())


def _fmt_count(mean: float, sd: float) -> str:
    return f"{mean:,.1f} ± {sd:,.1f}"


def _fmt_percent(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def _fmt_auc(mean: float, sd: float) -> str:
    return f"{mean:.6f} ± {sd:.6f}"


def _table1_rows(split: dict) -> list[list[str]]:
    intervals = split["intervalli"]
    return [
        ["Data", "A", "initial transformation and model fitting", f'{intervals["A"]["righe_valide"]:,} rows'],
        ["Data", "B", "per-seed initial threshold calibration", f'{intervals["B"]["righe_valide"]:,} rows'],
        ["Data", "C", "development of ADAPT-01--04; not replayed in the final evaluation", f'{intervals["C"]["righe_valide"]:,} rows'],
        ["Data", "D", "single final chronological evaluation", f'{intervals["D"]["righe_valide"]:,} rows; 355 blocks'],
        ["Trace", "frozen+B", "initial weights", "fixed threshold calibrated on B"],
        ["Trace", "frozen+CAL64", "initial weights", "threshold recalibrated on recent CAL64"],
        ["Trace", "FIT192+B", "head updated on FIT192", "fixed threshold calibrated on B"],
        ["Trace", "FIT192+CAL64", "head updated on FIT192", "threshold recalibrated on recent CAL64"],
    ]


def _table1_markdown(rows: list[list[str]]) -> str:
    lines = [
        "| Type | Item | Role or weights | Size or threshold |",
        "|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _table2_rows(analysis: dict) -> list[list[str]]:
    rows = []
    for item in analysis["risultati_assoluti"]:
        metrics = item["metriche"]
        scope = "Full D" if item["ambito"] == "D_completo" else "A/B/C-unseen vectors"
        rows.append(
            [
                scope,
                item["etichetta"],
                _fmt_count(**_mean_sd(metrics["fn"])),
                _fmt_count(**_mean_sd(metrics["fp"])),
                _fmt_percent(**_mean_sd(metrics["recall_pp"])),
                _fmt_percent(**_mean_sd(metrics["fpr_pp"])),
                _fmt_percent(**_mean_sd(metrics["balanced_accuracy_pp"])),
                _fmt_auc(**_mean_sd(metrics["auroc_media_blocco"])),
            ]
        )
    return rows


def _mean_sd(metric: dict) -> dict[str, float]:
    return {"mean": float(metric["media"]), "sd": float(metric["deviazione_standard"])}


def _table2_markdown(rows: list[list[str]]) -> str:
    lines = [
        "| Scope | Trace | FN | FP | Recall (%) | FPR (%) | Global BA (%) | Mean block AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _causal_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850" role="img" aria-labelledby="title desc">
  <title id="title">Causal online evaluation protocol</title>
  <desc id="desc">A and B prepare the initial model. Each D block is predicted before labels are requested. Delayed labels enter separate FIT192 and CAL64 memories. If all requirements pass, the candidate state is applied atomically from block k plus 2; otherwise a skip keeps the previous state. Both outcomes continue with the next D block.</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#34495e"/></marker>
    <style>
      .box{fill:#f7f9fb;stroke:#34495e;stroke-width:2;rx:12;ry:12}
      .data{fill:#eaf3ff}.memory{fill:#fff4df}.state{fill:#e9f7ef}.note{fill:#f4ecf7}
      .label{font:600 18px Arial,sans-serif;fill:#1f2d3d}.small{font:15px Arial,sans-serif;fill:#34495e}
      .branch{font:600 14px Arial,sans-serif;fill:#34495e}
      .arrow{stroke:#34495e;stroke-width:2.2;fill:none;marker-end:url(#arrow)}
      .dash{stroke-dasharray:8 6}
    </style>
  </defs>
  <rect x="40" y="45" width="210" height="100" class="box data"/>
  <text x="145" y="78" text-anchor="middle" class="label">A: initial fit</text>
  <text x="145" y="105" text-anchor="middle" class="small">transform + KAN</text>
  <text x="145" y="128" text-anchor="middle" class="small">per seed</text>
  <rect x="40" y="190" width="210" height="100" class="box data"/>
  <text x="145" y="225" text-anchor="middle" class="label">B: calibration</text>
  <text x="145" y="252" text-anchor="middle" class="small">fixed BA threshold</text>
  <text x="145" y="275" text-anchor="middle" class="small">per initial model</text>
  <path d="M250 95 L315 95" class="arrow"/><path d="M250 240 L285 240 L285 125 L315 125" class="arrow"/>
  <rect x="315" y="70" width="230" height="100" class="box state"/>
  <text x="430" y="105" text-anchor="middle" class="label">Four paired traces</text>
  <text x="430" y="132" text-anchor="middle" class="small">identical initial state</text>
  <text x="430" y="155" text-anchor="middle" class="small">common calendar and queries</text>
  <path d="M545 120 L625 120" class="arrow"/>
  <rect x="625" y="55" width="245" height="130" class="box data"/>
  <text x="748" y="88" text-anchor="middle" class="label">Predict D block k</text>
  <text x="748" y="115" text-anchor="middle" class="small">scores and predictions first</text>
  <text x="748" y="140" text-anchor="middle" class="small">then request 1% labels</text>
  <text x="748" y="165" text-anchor="middle" class="small">no past prediction is revised</text>
  <path d="M748 185 L748 245" class="arrow"/>
  <rect x="610" y="245" width="275" height="90" class="box note"/>
  <text x="748" y="278" text-anchor="middle" class="label">Labels become available</text>
  <text x="748" y="305" text-anchor="middle" class="small">at the end of block k+1</text>
  <path d="M748 335 L748 360 L505 390" class="arrow"/>
  <path d="M748 360 L910 390" class="arrow"/>
  <rect x="390" y="390" width="230" height="105" class="box memory"/>
  <text x="505" y="425" text-anchor="middle" class="label">FIT192</text>
  <text x="505" y="452" text-anchor="middle" class="small">candidate head</text>
  <text x="505" y="477" text-anchor="middle" class="small">both classes required</text>
  <rect x="795" y="390" width="230" height="105" class="box memory"/>
  <text x="910" y="425" text-anchor="middle" class="label">CAL64</text>
  <text x="910" y="452" text-anchor="middle" class="small">two BA thresholds</text>
  <text x="910" y="477" text-anchor="middle" class="small">age &lt;= 50; min 5+5</text>
  <path d="M505 495 L670 535" class="arrow"/><path d="M910 495 L820 535" class="arrow"/>
  <rect x="595" y="535" width="300" height="75" class="box note"/>
  <text x="745" y="565" text-anchor="middle" class="label">Requirements satisfied?</text>
  <text x="745" y="590" text-anchor="middle" class="small">valid head and both BA thresholds</text>
  <path d="M665 610 L465 650" class="arrow"/>
  <text x="555" y="625" text-anchor="middle" class="branch">YES</text>
  <path d="M825 610 L905 650" class="arrow"/>
  <text x="875" y="625" text-anchor="middle" class="branch">NO</text>
  <rect x="300" y="650" width="290" height="75" class="box state"/>
  <text x="445" y="680" text-anchor="middle" class="label">Apply candidate state</text>
  <text x="445" y="705" text-anchor="middle" class="small">effective from block k+2 only</text>
  <rect x="760" y="650" width="290" height="75" class="box"/>
  <text x="905" y="680" text-anchor="middle" class="label">Keep previous state</text>
  <text x="905" y="705" text-anchor="middle" class="small">record skip</text>
  <path d="M445 725 L570 765" class="arrow"/><path d="M905 725 L680 765" class="arrow"/>
  <rect x="515" y="765" width="220" height="65" class="box data"/>
  <text x="625" y="792" text-anchor="middle" class="label">Next D block</text>
  <text x="625" y="816" text-anchor="middle" class="small">same fixed procedure</text>
  <path d="M735 798 C1135 798 1135 120 870 120" class="arrow dash"/>
</svg>
"""


def _paired_figure(analysis: dict, output_dir: Path) -> None:
    matplotlib.rcParams["svg.hashsalt"] = "article-02-paired-deltas-v1"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    scopes = [
        ("D_completo", "Full D"),
        ("D_non_visto_ABC", "A/B/C-unseen vectors"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for ax, (key, title) in zip(axes, scopes):
        comparison = analysis["confronti"][key]["principale_pesi"]
        xs = comparison["delta_fp"]["per_seme"]
        ys = comparison["delta_fn"]["per_seme"]
        x_min0, x_max0 = min(min(xs), 0), max(max(xs), 0)
        y_min0, y_max0 = min(min(ys), 0), max(max(ys), 0)
        x_pad = max((x_max0 - x_min0) * 0.08, 1.0)
        y_pad = max((y_max0 - y_min0) * 0.08, 1.0)
        x_min, x_max = x_min0 - x_pad, x_max0 + x_pad
        y_min, y_max = y_min0 - y_pad, y_max0 + y_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                0 - x_min,
                0 - y_min,
                facecolor="#e8f5e9",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.axvline(0, color="#4d4d4d", linewidth=1)
        ax.axhline(0, color="#4d4d4d", linewidth=1)
        for seed, x, y, color in zip(SEEDS, xs, ys, colors):
            ax.scatter(x, y, s=64, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(
                str(seed), (x, y), xytext=(5, 5), textcoords="offset points",
                fontsize=9, clip_on=False,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(r"$\Delta$FP (FIT192+CAL64 - frozen+CAL64)")
        ax.set_ylabel(r"$\Delta$FN (FIT192+CAL64 - frozen+CAL64)")
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.ticklabel_format(style="plain", axis="both")
    fig.suptitle(
        "Paired error differences across seeds (negative/negative is joint improvement)",
        y=0.97,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.82, wspace=0.28)
    fig.savefig(
        output_dir / "figure_2_paired_fp_fn.png",
        dpi=180,
        metadata={"Software": "matplotlib"},
    )
    svg_path = output_dir / "figure_2_paired_fp_fn.svg"
    fig.savefig(
        svg_path,
        metadata={"Date": None, "Creator": "genera_materiali_manoscritto.py"},
    )
    plt.close(fig)
    # Matplotlib follows os.linesep and leaves trailing spaces in path data.
    # The repository contract is LF with no trailing whitespace on every
    # platform, and the manifest hashes the bytes actually stored.
    svg = svg_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    _write_text(svg_path, "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def generate(analysis_path: Path, split_path: Path, output_dir: Path) -> dict:
    analysis = _read_json(analysis_path)
    split = _read_json(split_path)
    if analysis.get("commit_eseguito") != "ea216914f756433fe69fd825146de5c1e325cbd8":
        raise ValueError("unexpected final-campaign commit")
    if split["intervalli"]["D"]["righe_valide"] != 3_549_973:
        raise ValueError("unexpected D row count")

    output_dir.mkdir(parents=True, exist_ok=True)
    table1 = _table1_rows(split)
    table2 = _table2_rows(analysis)
    _write_csv(output_dir / "table_1_protocol.csv", ["type", "item", "role_or_weights", "size_or_threshold"], table1)
    _write_csv(
        output_dir / "table_2_results.csv",
        ["scope", "trace", "fn", "fp", "recall_percent", "fpr_percent", "global_ba_percent", "mean_block_auroc"],
        table2,
    )
    _write_text(output_dir / "table_1_protocol.md", _table1_markdown(table1))
    _write_text(output_dir / "table_2_results.md", _table2_markdown(table2))
    _write_text(output_dir / "figure_1_causal_protocol.svg", _causal_svg())
    _paired_figure(analysis, output_dir)

    products = [
        "table_1_protocol.csv",
        "table_1_protocol.md",
        "table_2_results.csv",
        "table_2_results.md",
        "figure_1_causal_protocol.svg",
        "figure_2_paired_fp_fn.png",
        "figure_2_paired_fp_fn.svg",
    ]
    manifest = {
        "schema": 1,
        "purpose": "first Materials and Methods and Results draft; no fit, replay, or tuning",
        "base_merge": BASE_MERGE,
        "sources": {
            "analysis": {"path": _source_path(analysis_path), "sha256": _sha256(analysis_path)},
            "split_manifest": {"path": _source_path(split_path), "sha256": _sha256(split_path)},
        },
        "products": [
            {"path": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)}
            for name in products
        ],
    }
    _write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.analysis.resolve(), args.split_manifest.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publication figure from verified, equal-weight pilot means. No fitted values."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

ROOT = Path(__file__).resolve().parent
MODELS = ["coeff", "lut", "kanml", "mlp", "dt5"]

def main():
    with (ROOT / "board_model_means.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    for ax, field, xlabel, title in zip(axes,
        ["mean_call_us", "energy_uJ_per_call"],
        ["Time per call (µs)", "Estimated energy per call (µJ)"],
        ["(a) Active batch timing", "(b) Whole-board USB energy"]):
        for board, color, marker, delta in [
            ("Mega 2560", "#176C8D", "o", -.13),
            ("ESP32-C3", "#C76527", "s", .13)]:
            values = [float(next(r[field] for r in rows
                          if r["board"] == board and r["model"] == model)) for model in MODELS]
            ax.scatter(values, [i + delta for i in range(5)], s=33,
                color=color, marker=marker, label=board, zorder=3, linewidths=.4, edgecolors="white")
        ax.set_xscale("log")
        ax.set_xlim(.08, 22000)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", which="major", color="#D5DDE2", linewidth=.55)
        ax.grid(axis="y", color="#EAEDEF", linewidth=.55)
        ax.set_ylim(4.6, -.65)
        ax.set_yticks(range(5), ["KAN coefficients", "KAN sampled LUT", "Multilayer KAN", "MLP16", "DT5"])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=8)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.60, .015),
               ncol=2, frameon=False, handletextpad=.4, columnspacing=1.5)
    fig.subplots_adjust(left=.165, right=.985, top=.89, bottom=.26, wspace=.22)
    fig.savefig(ROOT / "energy_pilot_comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(ROOT / "energy_pilot_comparison.png", dpi=240)
    plt.close(fig)
    provenance = {"input": "board_model_means.csv", "sha256": hashlib.sha256((ROOT / "board_model_means.csv").read_bytes()).hexdigest(),
      "aggregation": "Equal arithmetic weight for each of two acquisitions per model and board", "independent_error_bars": False,
      "scope": "Prepared-input batch timing and estimated whole-board USB energy; model quality not ranked; logarithmic axes"}
    (ROOT / "ENERGY_FIGURE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")

if __name__ == "__main__":
    main()

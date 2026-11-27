"""Plot memory coverage over time from the delivered ADAPT-07 blocks."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with args.blocks.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.4), layout="constrained")
    for ax, scenario in zip(axes, ("35-1", "48-1"), strict=True):
        selected = [row for row in rows if row["dataset"] == f"iot23-{scenario}"]
        if not selected:
            raise ValueError(f"{scenario}: no evaluated blocks")
        blocks = [int(row["block"]) for row in selected]
        peak = 0.0
        for arm, name, color, style in (
            ("fixed", "Memoria fissata", "#a14b1b", "--"),
            ("live", "FIT192 aggiornata", "#1a5573", "-"),
        ):
            fractions = [
                100 * int(row[f"{arm}_eligible_rows"]) / int(row["rows"])
                for row in selected
            ]
            peak = max(peak, max(fractions))
            ax.plot(
                blocks,
                fractions,
                label=name,
                color=color,
                linestyle=style,
                linewidth=1.6,
            )
        changed = sum(int(row["live_vs_base_changed"]) for row in selected)
        ax.set_title(
            f"{scenario} · {changed:,} decisioni cambiate rispetto a Frozen/B",
            loc="left",
        )
        ax.set_xlabel("Blocco di D (indice da zero)")
        ax.set_ylabel("Record ammissibili")
        lower = min(peak * 0.04, 4)
        ax.set_ylim(-lower, max(0.04, min(104, peak * 1.08)))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=2 if peak < 1 else 0))
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        if scenario == "35-1":
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle(
        "ADAPT-07 · copertura della memoria nel segmento valutato", fontsize=13
    )
    fig.savefig(
        args.out, dpi=170, metadata={"Title": "ADAPT-07 · copertura della memoria"}
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

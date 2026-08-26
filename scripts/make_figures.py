#!/usr/bin/env python3
"""
Generate publication-quality figures for the KAN-IDS report.

Reads existing CSVs from results/ and writes 150-dpi, tight-bbox PNGs into
figures/. No numeric values are invented: everything plotted is read
straight out of the CSV files. If an expected CSV or column is missing,
the corresponding figure is skipped with a clear warning (no fabricated
data is substituted).
"""

import csv
import os
from pathlib import Path
import sys

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = str(Path(__file__).resolve().parents[1])
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")

os.makedirs(FIGURES, exist_ok=True)


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def read_csv_rows(path):
    """Return list of dict rows for a CSV, or None (with a warning) if missing."""
    if not os.path.isfile(path):
        warn(f"missing CSV: {path}")
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        warn(f"CSV has no rows: {path}")
        return None
    return rows


def require_columns(rows, cols, path):
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        warn(f"missing column(s) {missing} in {path}")
        return False
    return True


def savefig_tight(fig, out_path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    size = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
    print(f"wrote {out_path} ({size} bytes)")


# ---------------------------------------------------------------------------
# FIGURE 1 — fig_indomain_comparison.png
# ---------------------------------------------------------------------------
def fig1():
    path = os.path.join(RESULTS, "cv_leakagefree_summary_binary_ALL.csv")
    rows = read_csv_rows(path)
    if rows is None:
        warn("skipping FIGURE 1 (fig_indomain_comparison.png): source CSV missing/empty")
        return
    needed = ["model", "f1_mean", "f1_std", "n_runs"]
    if not require_columns(rows, needed, path):
        warn("skipping FIGURE 1 (fig_indomain_comparison.png): required column(s) missing")
        return

    data = []
    for r in rows:
        try:
            data.append((r["model"], float(r["f1_mean"]), float(r["f1_std"])))
        except (TypeError, ValueError):
            warn(f"skipping malformed row in {path}: {r}")

    if not data:
        warn("skipping FIGURE 1 (fig_indomain_comparison.png): no usable rows")
        return

    # sort descending by f1_mean
    data.sort(key=lambda t: t[1], reverse=True)
    models = [d[0] for d in data]
    means = [d[1] for d in data]
    stds = [d[2] for d in data]

    print("FIGURE 1 data (model, f1_mean, f1_std):")
    for m, mu, sd in data:
        print(f"  {m}: {mu:.4f} +/- {sd:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(models))
    # reverse so the best model appears at the top of the horizontal chart
    y_pos_rev = list(reversed(list(y_pos)))
    ax.barh(y_pos_rev, means, xerr=stds, color="#4C72B0", edgecolor="black",
            height=0.6, capsize=4, zorder=3)
    ax.set_yticks(y_pos_rev)
    ax.set_yticklabels(models)
    ax.set_xlim(0.975, 1.0)
    ax.set_xlabel("F1")
    ax.set_title("TON_IoT in-domain, binary task (5-fold x 3 seed)")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # etichetta dentro la barra: con xlim stretto (0.975-1.0) un'etichetta
    # esterna uscirebbe dall'asse per il modello migliore
    for yp, mu, sd in zip(y_pos_rev, means, stds):
        ax.annotate(f"{mu:.4f} ± {sd:.4f}", xy=(mu, yp),
                    xytext=(-6, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=9,
                    color="white", fontweight="bold")

    savefig_tight(fig, os.path.join(FIGURES, "fig_indomain_comparison.png"))


# ---------------------------------------------------------------------------
# FIGURE 2 — fig_crossdomain_degradation.png
# ---------------------------------------------------------------------------
def fig2():
    path = os.path.join(RESULTS, "crossdomain_degradation.csv")
    rows = read_csv_rows(path)
    if rows is None:
        warn("skipping FIGURE 2 (fig_crossdomain_degradation.png): source CSV missing/empty")
        return
    needed = ["model", "ton_in_domain", "ton->bot"]
    if not require_columns(rows, needed, path):
        warn("skipping FIGURE 2 (fig_crossdomain_degradation.png): required column(s) missing")
        return

    data = []
    for r in rows:
        model = r["model"]
        indom_raw = r["ton_in_domain"]
        cross_raw = r["ton->bot"]
        if indom_raw in (None, "", "NA", "nan") or cross_raw in (None, "", "NA", "nan"):
            warn(f"skipping model '{model}' in {path}: missing ton_in_domain/ton->bot value")
            continue
        try:
            indom = float(indom_raw)
            cross = float(cross_raw)
        except ValueError:
            warn(f"skipping model '{model}' in {path}: non-numeric value")
            continue
        data.append((model, indom, cross))

    if not data:
        warn("skipping FIGURE 2 (fig_crossdomain_degradation.png): no usable rows")
        return

    # sort by cross-domain value ascending (so chart reads worst->best top->bottom
    # once reversed for horizontal display)
    data.sort(key=lambda t: t[2])

    print("FIGURE 2 data (model, ton_in_domain, ton->bot):")
    for m, i, c in data:
        print(f"  {m}: in-domain={i:.4f}, cross-domain={c:.4f}")

    models = [d[0] for d in data]
    indom_vals = [d[1] for d in data]
    cross_vals = [d[2] for d in data]

    n = len(models)
    y = list(range(n))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    # in-domain bars sit above, cross-domain below, for each model group
    y_indom = [yy + bar_h / 2 for yy in y]
    y_cross = [yy - bar_h / 2 for yy in y]

    ax.barh(y_indom, indom_vals, height=bar_h, color="#4C72B0",
            edgecolor="black", label="in-domain (TON_IoT)", zorder=3)
    ax.barh(y_cross, cross_vals, height=bar_h, color="#DD8452",
            edgecolor="black", label="cross-domain (TON_IoT -> BoT-IoT)", zorder=3)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, zorder=4)
    ax.text(0.505, -0.75, "chance", va="bottom", ha="left",
            fontsize=9, color="black")
    for yy, v in zip(y_cross, cross_vals):
        ax.annotate(f"{v:.3f}", xy=(v, yy), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Balanced accuracy: in-domain vs cross-domain (TON_IoT -> BoT-IoT)")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=9, frameon=False)

    savefig_tight(fig, os.path.join(FIGURES, "fig_crossdomain_degradation.png"))


# ---------------------------------------------------------------------------
# FIGURE 3 — fig_distribution_overlap.png
# ---------------------------------------------------------------------------
def fig3():
    path = os.path.join(RESULTS, "crossdomain_shift.csv")
    rows = read_csv_rows(path)
    if rows is None:
        warn("skipping FIGURE 3 (fig_distribution_overlap.png): source CSV missing/empty")
        return
    needed = ["feature", "sovrapposizione"]
    if not require_columns(rows, needed, path):
        warn("skipping FIGURE 3 (fig_distribution_overlap.png): required column(s) missing")
        return

    data = []
    for r in rows:
        try:
            data.append((r["feature"], float(r["sovrapposizione"])))
        except (TypeError, ValueError):
            warn(f"skipping malformed row in {path}: {r}")

    if not data:
        warn("skipping FIGURE 3 (fig_distribution_overlap.png): no usable rows")
        return

    data.sort(key=lambda t: t[1])  # ascending

    print("FIGURE 3 data (feature, sovrapposizione):")
    for f, v in data:
        print(f"  {f}: {v:.3f}")

    features = [d[0] for d in data]
    values = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(features))
    ax.barh(list(y_pos), values, color="#55A868", edgecolor="black",
            height=0.6, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(features)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Overlap (sovrapposizione)")
    ax.set_title("Overlap of marginal distributions, TON_IoT vs BoT-IoT (0 = disjoint)")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    for yp, v in zip(y_pos, values):
        ax.annotate(f"{v:.3f}", xy=(v, yp), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9)

    savefig_tight(fig, os.path.join(FIGURES, "fig_distribution_overlap.png"))


# ---------------------------------------------------------------------------
# FIGURE 4 — fig_confusion_crossdomain.png
# ---------------------------------------------------------------------------
def fig4():
    models = [
        ("KAN_cat_1L", "KAN(cat,1L)"),
        ("LightGBM", "LightGBM"),
        ("DecisionTree_d=5", "DecisionTree(d=5)"),
        ("XGBoost", "XGBoost"),
    ]

    matrices = {}
    labels = None
    for fname_key, title in models:
        path = os.path.join(RESULTS, f"confusion_crossdomain_cat_ton_bot_{fname_key}.csv")
        if not os.path.isfile(path):
            warn(f"missing confusion CSV for {title}: {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_rows = list(reader)
        if len(file_rows) < 3:
            warn(f"confusion CSV malformed (too few rows) for {title}: {path}")
            continue
        header = file_rows[0]
        col_labels = header[1:]
        row_labels = []
        mat = []
        for r in file_rows[1:]:
            row_labels.append(r[0])
            try:
                mat.append([float(x) for x in r[1:]])
            except ValueError:
                warn(f"non-numeric value in confusion CSV for {title}: {path}")
                mat = None
                break
        if mat is None:
            continue
        if labels is None:
            labels = (row_labels, col_labels)
        matrices[title] = mat

    if len(matrices) < 4:
        missing_titles = [t for _, t in models if t not in matrices]
        warn(
            "skipping FIGURE 4 (fig_confusion_crossdomain.png): missing confusion "
            f"data for model(s): {missing_titles}"
        )
        return

    row_labels, col_labels = labels

    print("FIGURE 4 data (row-normalised confusion matrices):")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes_flat = axes.flatten()

    im = None
    for ax, (_, title) in zip(axes_flat, models):
        mat = matrices[title]
        norm = []
        for row in mat:
            s = sum(row)
            if s > 0:
                norm.append([v / s for v in row])
            else:
                norm.append([0.0 for _ in row])

        print(f"  {title}:")
        for rl, row_raw, row_norm in zip(row_labels, mat, norm):
            raw_str = ", ".join(f"{v:.0f}" for v in row_raw)
            norm_str = ", ".join(f"{v:.2f}" for v in row_norm)
            print(f"    {rl}: raw=[{raw_str}] normalised=[{norm_str}]")

        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title, fontsize=11)

        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = norm[i][j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=11)

    fig.suptitle("Confusion matrices, TON_IoT -> BoT-IoT (row-normalised)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="row-normalised fraction")

    savefig_tight(fig, os.path.join(FIGURES, "fig_confusion_crossdomain.png"))


def main():
    fig1()
    fig2()
    fig3()
    fig4()


if __name__ == "__main__":
    main()


# ── FIGURA 5: frontiera di Pareto dimensione / accuratezza ────────────
def fig_pareto():
    """Due pannelli: in-domain e cross-domain, stessi modelli e stessi byte.

    Il pannello in-domain e' quello che decide se la tesi "accuratezza per
    byte" regge; quello cross-domain e' dove la stessa classifica si
    ribalta. Vanno letti insieme: separati, ciascuno dei due racconta meta'
    della storia.
    """
    fp = os.path.join(RESULTS, "footprint.csv")
    if not os.path.exists(fp):
        print("[skip] footprint.csv assente"); return
    d = pd.read_csv(fp)
    d = d[d["byte_parametri"].notna() & d["f1_cv"].notna()]

    cd_path = os.path.join(RESULTS, "crossdomain_degradation.csv")
    cd = pd.read_csv(cd_path) if os.path.exists(cd_path) else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, (vals, ylab, title) in zip(axes, [
        (dict(zip(d.modello, d.f1_cv)), "F1 (TON_IoT in-domain, 5-fold x 3 seed)",
         "In-domain: il Pareto"),
        ({} if cd is None else dict(zip(cd.model, cd["ton->bot"])),
         "Balanced accuracy (TON_IoT -> BoT-IoT)", "Cross-domain: la classifica si ribalta"),
    ]):
        pts = [(m, b, vals[m]) for m, b in zip(d.modello, d.byte_parametri) if m in vals]
        if not pts:
            ax.text(0.5, 0.5, "dati non disponibili", ha="center"); continue
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.scatter(xs, ys, s=70, zorder=3, color="#1F3864")

        # frontiera: non dominati (piu' piccoli e piu' accurati)
        order = sorted(pts, key=lambda p: p[1])
        front, best = [], -1e9
        for m, b, v in order:
            if v > best:
                front.append((b, v)); best = v
        ax.plot([f[0] for f in front], [f[1] for f in front],
                color="#DD8452", lw=1.6, ls="--", zorder=2, label="frontiera")

        for m, b, v in pts:
            dominated = any((bb <= b and vv >= v and (bb, vv) != (b, v)) for _, bb, vv in pts)
            ax.annotate(m + ("  (dominato)" if dominated else ""),
                        xy=(b, v), xytext=(6, -4), textcoords="offset points",
                        fontsize=8.5, color="#B03A2E" if dominated else "black")
        ax.set_xscale("log")
        ax.set_xlabel("byte dei parametri deployati (scala log)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right")

    if cd is not None:
        axes[1].axhline(0.5, color="black", ls=":", lw=1.2)
        axes[1].text(axes[1].get_xlim()[0] * 1.2, 0.505, "caso", fontsize=8)

    savefig_tight(fig, os.path.join(FIGURES, "fig_pareto_size_accuracy.png"))


fig_pareto()

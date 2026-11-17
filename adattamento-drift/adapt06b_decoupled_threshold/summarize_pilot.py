"""Verify and summarize the authorized ADAPT-06B seed-42 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
REPOSITORY = HERE.parents[1]

from adapt06b_decoupled_threshold.verify_pair import verify_reports  # noqa: E402

DEFAULT_ROOT = HERE / "pilot_seed42_02b82af"
EXECUTION_COMMIT = "02b82af36ae8151e4ac83d806f434704e4557abd"
DATASETS = ("ton", "unsw")
CONTROLS = (
    "frozen_b",
    "threshold_only_autonomous",
    "gated_legacy",
    "gated_decoupled",
    "threshold_only_matched",
)
CONTROL_TRACKS = {
    "frozen_b": ("decoupled", "frozen_b"),
    "threshold_only_autonomous": ("legacy", "threshold_only"),
    "gated_legacy": ("legacy", "gated"),
    "gated_decoupled": ("decoupled", "gated"),
    "threshold_only_matched": ("decoupled", "threshold_matched"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset(root: Path, dataset: str) -> dict:
    directory = root / f"{dataset}_seed42"
    paths = {
        "legacy": directory / f"{dataset}_seed42_legacy.json",
        "decoupled": directory / f"{dataset}_seed42_decoupled.json",
        "pair": directory / f"{dataset}_seed42_pair.json",
        "manifest": directory / "MANIFEST.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AssertionError(f"missing pilot files: {missing}")
    legacy = load_json(paths["legacy"])
    decoupled = load_json(paths["decoupled"])
    stored_pair = load_json(paths["pair"])
    manifest = load_json(paths["manifest"])
    if manifest["status"] != "PASS" or manifest["execution_commit"] != EXECUTION_COMMIT:
        raise AssertionError(f"{dataset}: invalid execution manifest")
    if not manifest["full_stream"] or manifest["initialization"] != "archived A/B":
        raise AssertionError(f"{dataset}: invalid scope or initialization")
    for name in ("legacy", "decoupled"):
        report = locals()[name]
        if report["commit"] != EXECUTION_COMMIT or report["scope"] != "full":
            raise AssertionError(f"{dataset}: {name} report has invalid provenance")
    for filename, record in manifest["files"].items():
        path = directory / filename
        if path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"{dataset}: byte count differs for {filename}")
        if sha256(path) != record["sha256"]:
            raise AssertionError(f"{dataset}: SHA-256 differs for {filename}")
    recomputed = verify_reports(legacy, decoupled)
    comparable = {key: value for key, value in stored_pair.items() if key != "sources"}
    if recomputed != comparable:
        raise AssertionError(f"{dataset}: stored paired comparison is not reproducible")
    for source, filename in (
        ("legacy", paths["legacy"].name),
        ("variant", paths["decoupled"].name),
    ):
        if stored_pair["sources"][source]["file"] != filename:
            raise AssertionError(f"{dataset}: invalid paired source name")
        if stored_pair["sources"][source]["sha256"] != sha256(directory / filename):
            raise AssertionError(f"{dataset}: invalid paired source digest")
    return {
        "legacy": legacy,
        "decoupled": decoupled,
        "pair": stored_pair,
        "manifest": manifest,
        "paths": paths,
    }


def verify_delivery(root: Path) -> dict[str, dict]:
    results = {dataset: verify_dataset(root, dataset) for dataset in DATASETS}
    if results["ton"]["pair"]["blocks"] != 355:
        raise AssertionError("TON must contain 355 complete blocks")
    if results["unsw"]["pair"]["blocks"] != 102:
        raise AssertionError("UNSW must contain 102 complete blocks")
    return results


def summary_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        pair = results[dataset]["pair"]
        for control in CONTROLS:
            metrics = pair["controls"][control]
            source, track = CONTROL_TRACKS[control]
            blocks = results[dataset][source]["per_block"]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "control": control,
                    "recall": metrics["recall"],
                    "fpr": metrics["fpr"],
                    "recall_delta_vs_threshold_only": metrics[
                        "recall_delta_vs_threshold_only"
                    ],
                    "fpr_delta_vs_threshold_only": metrics[
                        "fpr_delta_vs_threshold_only"
                    ],
                    "weight_promotions": metrics["weight_promotions"],
                    "threshold_changes": metrics["threshold_changes"],
                    "initial_threshold": blocks[0]["thresholds_used"][track],
                    "final_threshold": blocks[-1]["thresholds_used"][track],
                }
            )
    return rows


def render_csv(results: dict[str, dict]) -> str:
    fields = (
        "dataset",
        "seed",
        "control",
        "recall",
        "fpr",
        "recall_delta_vs_threshold_only",
        "fpr_delta_vs_threshold_only",
        "weight_promotions",
        "threshold_changes",
        "initial_threshold",
        "final_threshold",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_rows(results))
    return stream.getvalue()


def percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def points(value: float) -> str:
    return f"{100.0 * value:+.3f} pp"


def threshold(value: float) -> str:
    return f"{value:.6f}"


def render_report(results: dict[str, dict]) -> str:
    lines = [
        "# ADAPT-06B — pilot completo GAM/RANDOM, seed 42",
        "",
        "## Perimetro e provenienza",
        "",
        f"I quattro replay sono stati eseguiti dal commit `{EXECUTION_COMMIT}` "
        "sugli stream completi TON e UNSW, partendo dagli stati A/B archiviati. "
        "Legacy e decoupled sono stati avviati in processi distinti con BLAS a un "
        "thread. Non sono stati eseguiti nuovi fit iniziali, tuning, ACTIVE, altri "
        "seed o ADAPT-07.",
        "",
        "| Dataset | Blocchi | Righe | CONTROL | RANDOM_FIT | Label totali | Prima CAL eleggibile | Prima soglia applicata |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        pair = results[dataset]["pair"]
        lines.append(
            f"| {dataset.upper()} | {pair['blocks']} | {pair['rows']:,} | "
            f"{pair['roles']['CONTROL']:,} | {pair['roles']['RANDOM_FIT']:,} | "
            f"{pair['unique_labels_requested']:,} | {pair['first_cal_eligible_block']} | "
            f"{pair['first_threshold_application_block']} |"
        )
    lines.extend(
        [
            "",
            "## Confronto principale con threshold-only autonoma",
            "",
            "| Dataset | Politica | Recall | FPR | Δ recall | Δ FPR | Promozioni pesi | Variazioni soglia | Soglia iniziale | Soglia finale |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        pair = results[dataset]["pair"]
        for control in ("threshold_only_autonomous", "gated_decoupled"):
            item = pair["controls"][control]
            source, track = CONTROL_TRACKS[control]
            blocks = results[dataset][source]["per_block"]
            lines.append(
                f"| {dataset.upper()} | {control} | {percent(item['recall'])} | "
                f"{percent(item['fpr'])} | "
                f"{points(item['recall_delta_vs_threshold_only'])} | "
                f"{points(item['fpr_delta_vs_threshold_only'])} | "
                f"{item['weight_promotions']} | {item['threshold_changes']} | "
                f"{threshold(blocks[0]['thresholds_used'][track])} | "
                f"{threshold(blocks[-1]['thresholds_used'][track])} |"
            )
    lines.extend(
        [
            "",
            "## Decisioni e costi osservati",
            "",
            "| Dataset | Decisioni decoupled | Replay legacy | Replay decoupled | RSS legacy | RSS decoupled | Payload legacy | Payload decoupled |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        pair = results[dataset]["pair"]
        decisions = ", ".join(
            f"{name}={count}" for name, count in pair["decisions"]["decoupled"].items()
        )
        costs = pair["costs"]
        lines.append(
            f"| {dataset.upper()} | {decisions} | {costs['legacy_replay_seconds']:.3f} s | "
            f"{costs['decoupled_replay_seconds']:.3f} s | "
            f"{costs['legacy_peak_rss_mib']:.1f} MiB | "
            f"{costs['decoupled_peak_rss_mib']:.1f} MiB | "
            f"{costs['legacy_peak_online_logical_payload_bytes']:,} B | "
            f"{costs['decoupled_peak_online_logical_payload_bytes']:,} B |"
        )
    ton = results["ton"]["pair"]
    unsw = results["unsw"]["pair"]
    ton_dec = ton["controls"]["gated_decoupled"]
    unsw_dec = unsw["controls"]["gated_decoupled"]
    unsw_matched = unsw["controls"]["threshold_only_matched"]
    lines.extend(
        [
            "",
            "## Sintesi descrittiva",
            "",
            "- **TON:** non è stato promosso alcun candidato. GATED decoupled e "
            "threshold-only matched coincidono; rispetto alla threshold-only autonoma "
            f"il recall è invariato e l'FPR differisce di {points(ton_dec['fpr_delta_vs_threshold_only'])}. "
            "Il forte calo dell'FPR rispetto a Frozen/B deriva quindi dalla manutenzione "
            "della soglia, non dall'adattamento dei pesi.",
            "- **UNSW:** è stato promosso un candidato. Rispetto alla threshold-only autonoma "
            f"GATED decoupled mostra {points(unsw_dec['recall_delta_vs_threshold_only'])} di recall "
            f"e {points(unsw_dec['fpr_delta_vs_threshold_only'])} di FPR. Rispetto al controllo "
            f"threshold-only matched, il guadagno di recall è {points(unsw_dec['recall'] - unsw_matched['recall'])} "
            f"con un aumento dell'FPR di {points(unsw_dec['fpr'] - unsw_matched['fpr'])}.",
            "- Il pilot contiene un solo seed per dataset: il risultato UNSW è favorevole "
            "nel confronto primario, ma non dimostra stabilità né superiorità generale. "
            "TON costituisce un risultato nullo per il contributo dei pesi.",
            "- Le soglie sono espresse nella scala degli score del relativo stato del modello. "
            "Dopo una promozione, la soglia operativa finale si riferisce quindi ai pesi "
            "promossi e non va interpretata come una variazione isolata dello stesso score.",
            "",
            "Tutti i cicli risultano chiusi alla fine degli stream: non rimangono candidati "
            "o aggiornamenti di soglia pendenti.",
            "",
            "I tempi riportati sono tempi di replay per singolo processo; caricamento degli "
            "input e del modello, scrittura e verifica dei file restano distinti nei JSON. "
            "I valori RSS sono indipendenti perché ogni replay è stato eseguito in un processo nuovo.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(included: list[Path]) -> dict:
    return {
        "schema": "adapt06b-pilot-delivery-manifest-v1",
        "execution_commit": EXECUTION_COMMIT,
        "delivery_commit_identification": "the PR head containing this manifest",
        "datasets": list(DATASETS),
        "seed": 42,
        "scope": "full streams",
        "family": "spline-GAM",
        "acquisition": "RANDOM",
        "files": {
            path.relative_to(REPOSITORY).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(included)
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    results = verify_delivery(root)
    csv_path = root / "PILOT_SUMMARY.csv"
    report_path = root / "PILOT_REPORT.md"
    csv_path.write_text(render_csv(results), encoding="utf-8", newline="\n")
    report_path.write_text(render_report(results), encoding="utf-8", newline="\n")
    included = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "DELIVERY_MANIFEST.json"
    ]
    included.extend((Path(__file__).resolve(), HERE / "test_pilot_delivery.py"))
    manifest = build_manifest(included)
    (root / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS",
        "datasets=2",
        "replay_reports=4",
        "paired_comparisons=2",
        f"files={len(included)}",
    )


if __name__ == "__main__":
    main()

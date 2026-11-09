"""Execute the frozen ADAPT-06 full matrix from archived A/B states."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"
DATASETS = ("ton", "unsw")
FAMILIES = ("spline", "lr", "arf")
ACQUISITIONS = ("random", "active")
SEEDS = tuple(range(42, 47))
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True, order=True)
class Cell:
    dataset: str
    seed: int
    family: str
    acquisition: str

    @property
    def filename(self) -> str:
        return f"{self.dataset}_{self.family}_{self.acquisition}_seed{self.seed}.json"

    @property
    def adaptive_trajectories(self) -> int:
        return 1 if self.family == "arf" else 2


def matrix_cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(dataset, seed, family, acquisition)
        for dataset in DATASETS
        for seed in SEEDS
        for family in FAMILIES
        for acquisition in ACQUISITIONS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def command_for(
    cell: Cell, data_root: Path, model_root: Path, reports: Path
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--data",
        str(data_root / cell.dataset),
        "--model-root",
        str(model_root),
        "--dataset",
        cell.dataset,
        "--family",
        cell.family,
        "--acquisition",
        cell.acquisition,
        "--seed",
        str(cell.seed),
        "--out",
        str(reports / cell.filename),
    ]


def quoted_command(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def file_record(path: Path) -> dict[str, object]:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def tree_inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): file_record(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def package_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "scikit-learn", "matplotlib", "psutil")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def validate_inputs(data_root: Path, model_root: Path) -> None:
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    for dataset in DATASETS:
        dataset_root = data_root / dataset
        for name in ("D.npz", "D.json"):
            path = dataset_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
    if not model_root.is_dir() or not any(model_root.rglob("*")):
        raise FileNotFoundError(f"empty or missing model root: {model_root}")


def prepare_output(out: Path) -> tuple[Path, Path]:
    if out.exists():
        raise FileExistsError(
            f"output already exists; use a new directory as preregistered: {out}"
        )
    out.mkdir(parents=True)
    reports = out / "reports"
    reports.mkdir()
    return out, reports


def execute_cell(
    cell: Cell,
    command: list[str],
    environment: dict[str, str],
) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=HERE.parent.parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    return {
        **asdict(cell),
        "report": cell.filename,
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def validate_report(path: Path, cell: Cell, commit: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "adapt06-prospective-run-v1",
        "dataset": cell.dataset,
        "family": cell.family,
        "acquisition": cell.acquisition,
        "seed": cell.seed,
        "scope": "full",
        "commit": commit,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise AssertionError(
                f"{path.name}: {key}={report.get(key)!r}, expected {value!r}"
            )
    if report["blocks_evaluated"] != report["blocks_available"]:
        raise AssertionError(f"{path.name}: incomplete block evaluation")
    if report["rows_evaluated"] != report["rows_available"]:
        raise AssertionError(f"{path.name}: incomplete row evaluation")
    command = report["command"]
    parts = command if isinstance(command, list) else command.splitlines()
    if "--max-blocks" in parts:
        raise AssertionError(f"{path.name}: full run contains --max-blocks")
    identity = report["provenance"]["input_identity"]
    if identity["status"] != "PASS" or not identity["row_id_unique"]:
        raise AssertionError(f"{path.name}: input provenance failed")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    model_root = args.model_root.resolve()
    out, reports = prepare_output(args.out.resolve())
    validate_inputs(data_root, model_root)
    cells = matrix_cells()
    if len(cells) != 60 or sum(cell.adaptive_trajectories for cell in cells) != 100:
        raise AssertionError("the frozen matrix is not 60 reports / 100 trajectories")

    commit = git_head()
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    commands = {
        cell: command_for(cell, data_root, model_root, reports) for cell in cells
    }
    (out / "COMMANDS.txt").write_text(
        "\n".join(quoted_command(commands[cell]) for cell in cells) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    run_configuration = {
        "schema": "adapt06-full-matrix-execution-v1",
        "commit": commit,
        "runner_sha256": sha256(RUNNER),
        "workers": args.workers,
        "thread_environment": THREAD_ENVIRONMENT,
        "datasets": list(DATASETS),
        "families": list(FAMILIES),
        "acquisitions": list(ACQUISITIONS),
        "seeds": list(SEEDS),
        "reports": len(cells),
        "adaptive_trajectories": sum(cell.adaptive_trajectories for cell in cells),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "data": {
            dataset: {
                name: file_record(data_root / dataset / name)
                for name in ("D.npz", "D.json")
            }
            for dataset in DATASETS
        },
        "archived_models": tree_inventory(model_root),
    }
    (out / "RUN_CONFIGURATION.json").write_text(
        json.dumps(run_configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute_cell, cell, commands[cell], environment): cell
            for cell in cells
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "PASS" if result["returncode"] == 0 else "FAIL"
            print(
                status,
                result["report"],
                f"wall={float(result['wall_seconds']):.3f}s",
                flush=True,
            )
    overall_wall = time.perf_counter() - started
    results.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["seed"]),
            str(row["family"]),
            str(row["acquisition"]),
        )
    )
    failures = [row for row in results if int(row["returncode"]) != 0]
    (out / "RUN_LOG.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
        newline="\n",
    )
    timing_rows = [
        {
            "report": row["report"],
            "dataset": row["dataset"],
            "seed": row["seed"],
            "family": row["family"],
            "acquisition": row["acquisition"],
            "wall_seconds": row["wall_seconds"],
        }
        for row in results
    ]
    write_csv(out / "EXECUTION_TIMES.csv", timing_rows)
    if failures:
        names = ", ".join(str(row["report"]) for row in failures)
        raise RuntimeError(f"{len(failures)} matrix cells failed: {names}")

    for cell in cells:
        validate_report(reports / cell.filename, cell, commit)
    manifest = {
        **run_configuration,
        "overall_wall_seconds": overall_wall,
        "serial_cell_wall_seconds": sum(float(row["wall_seconds"]) for row in results),
        "completed_reports": len(results),
        "report_files": {
            cell.filename: file_record(reports / cell.filename) for cell in cells
        },
    }
    (out / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS",
        f"reports={len(results)}",
        "adaptive_trajectories=100",
        f"overall_wall_seconds={overall_wall:.3f}",
    )


if __name__ == "__main__":
    main()

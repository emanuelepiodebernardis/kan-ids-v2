"""Extract the Windows UNSW/45 promotion boundary from the original report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

FREEZE_BLOCK = 41
ACCEPT_BLOCK = 63
FIRST_EFFECT_BLOCK = 64
WINDOWS_TP = 201_179
WINDOWS_FP = 18_590
LINUX_TP = 201_178
LINUX_FP = 18_631
LINUX_THRESHOLD = -0.09784439655829225
LINUX_FIT_SHA256 = "aa4ee1a84dba5e98b5e0e546f7a2b880da18aaf60edc4373cf888d72f7381f9e"
LINUX_CAL_SHA256 = "5377b1482b89accf5c1dcb3d4ca531c57c837f02b461f7a4df4aaad869956666"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(values: list[int]) -> str:
    """Match the engine digest: recorded signed 64-bit IDs, little endian."""
    payload = b"".join(
        int(value).to_bytes(8, "little", signed=True) for value in values
    )
    return hashlib.sha256(payload).hexdigest()


def _single(items: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(f"expected one {label}, found {len(items)}")
    return items[0]


def extract_audit(
    report: dict[str, Any], report_sha256: str
) -> tuple[dict, list[dict]]:
    """Validate and extract the exact state that becomes operational at block 64."""
    if report.get("dataset") != "unsw" or report.get("seed") != 45:
        raise ValueError("expected the UNSW seed-45 report")
    if report.get("planned_only") or not report.get("executed"):
        raise ValueError("expected an executed report")

    freeze = _single(
        [
            item
            for item in report["decisions"]
            if item.get("status") == "FREEZE" and item.get("block") == FREEZE_BLOCK
        ],
        label="FREEZE event at block 41",
    )
    accept = _single(
        [
            item
            for item in report["decisions"]
            if item.get("status") == "ACCEPT"
            and item.get("freeze_block") == FREEZE_BLOCK
        ],
        label="ACCEPT event for the block-41 candidate",
    )
    if accept.get("block") != ACCEPT_BLOCK:
        raise ValueError("the block-41 candidate was not accepted at block 63")

    blocks = {item["block"]: item for item in report["per_block"]}
    freeze_record = blocks[FREEZE_BLOCK]
    effect_record = blocks[FIRST_EFFECT_BLOCK]
    ids_by_role = freeze_record["memory_observation_ids"]
    recorded_digests = freeze_record["memory_observation_ids_sha256"]

    id_rows: list[dict[str, int | str]] = []
    roles: dict[str, dict[str, Any]] = {}
    for role in ("fit", "cal"):
        values = [int(value) for value in ids_by_role[role]]
        digest = ids_sha256(values)
        if digest != recorded_digests[role]:
            raise ValueError(f"{role.upper()} ID digest differs from the report")
        roles[role] = {
            "count": len(values),
            "observation_ids_sha256": digest,
            "first_observation_id": values[0],
            "last_observation_id": values[-1],
        }
        id_rows.extend(
            {"role": role.upper(), "position": position, "observation_id": value}
            for position, value in enumerate(values)
        )

    candidate_parameters = [float(value) for value in freeze["candidate_parameters"]]
    operational_parameters = [
        float(value) for value in effect_record["state_parameters_used"]["gated"]
    ]
    candidate_threshold = float(freeze["candidate_threshold"])
    operational_threshold = float(effect_record["thresholds_used"]["gated"])
    candidate_digest = freeze["candidate_digest"]
    operational_digest = effect_record["state_digests_used"]["gated"]
    if operational_parameters != candidate_parameters:
        raise ValueError("block-64 weights differ from the frozen candidate")
    if operational_threshold != candidate_threshold:
        raise ValueError("block-64 threshold differs from the frozen candidate")
    if operational_digest != candidate_digest:
        raise ValueError("block-64 state digest differs from the frozen candidate")

    windows_metrics = report["summary"]["gated"]
    if (windows_metrics["tp"], windows_metrics["fp"]) != (WINDOWS_TP, WINDOWS_FP):
        raise ValueError("Windows full-stream confusion differs from the delivery")

    audit = {
        "schema": "adapt06e-unsw45-platform-audit-v1",
        "status": "PASS",
        "source": {
            "report": "unsw_seed45_verify50.json",
            "report_sha256": report_sha256,
            "execution_commit": report["commit"],
            "environment": report["environment"],
        },
        "candidate": {
            "freeze_block": FREEZE_BLOCK,
            "accept_block": ACCEPT_BLOCK,
            "first_effect_block": FIRST_EFFECT_BLOCK,
            "state_digest": candidate_digest,
            "parameters": candidate_parameters,
            "threshold": candidate_threshold,
            "fit": roles["fit"],
            "cal": roles["cal"],
        },
        "first_operational_use": {
            "block": FIRST_EFFECT_BLOCK,
            "state_digest": operational_digest,
            "parameters": operational_parameters,
            "threshold": operational_threshold,
            "matches_frozen_candidate": True,
        },
        "full_stream_confusion": {
            "windows": {"tp": WINDOWS_TP, "fp": WINDOWS_FP},
            "linux_reported": {"tp": LINUX_TP, "fp": LINUX_FP},
            "windows_minus_linux": {
                "tp": WINDOWS_TP - LINUX_TP,
                "fp": WINDOWS_FP - LINUX_FP,
            },
        },
        "linux_artifact_comparison": {
            "status": "PARTIAL_LINUX_VALUES_REPORTED",
            "environment": {
                "python": "3.13.5",
                "numpy": "2.3.5",
                "scikit_learn": "1.7.2",
                "scipy": "1.16.2",
                "blas_threads": 1,
            },
            "fit_observation_ids_sha256": {
                "linux": LINUX_FIT_SHA256,
                "windows": roles["fit"]["observation_ids_sha256"],
                "equal": roles["fit"]["observation_ids_sha256"] == LINUX_FIT_SHA256,
            },
            "cal_observation_ids_sha256": {
                "linux": LINUX_CAL_SHA256,
                "windows": roles["cal"]["observation_ids_sha256"],
                "equal": roles["cal"]["observation_ids_sha256"] == LINUX_CAL_SHA256,
            },
            "operational_threshold_from_block_64": {
                "linux": LINUX_THRESHOLD,
                "windows": operational_threshold,
                "equal": operational_threshold == LINUX_THRESHOLD,
                "windows_minus_linux": operational_threshold - LINUX_THRESHOLD,
            },
            "parameters": {
                "windows": operational_parameters,
                "linux": None,
                "status": "LINUX_VECTOR_NOT_REPORTED_IN_TRELLO_CARD",
            },
            "interpretation": (
                "FIT and CAL membership digests coincide, while the applied threshold "
                "already differs at the first operational block. The Linux parameter "
                "vector was not reported, so the divergence cannot be attributed "
                "specifically to fitting or calibration from the available logs."
            ),
        },
    }
    return audit, id_rows


def render_note(audit: dict[str, Any]) -> str:
    candidate = audit["candidate"]
    confusion = audit["full_stream_confusion"]
    platform = audit["linux_artifact_comparison"]
    windows_environment = audit["source"]["environment"]
    windows = confusion["windows"]
    linux = confusion["linux_reported"]
    delta = confusion["windows_minus_linux"]
    parameters = ", ".join(format(value, ".17g") for value in candidate["parameters"])
    return f"""# UNSW/45 — riscontro Windows sul candidato del blocco 41

Il controllo usa esclusivamente il report originale
`unsw_seed45_verify50.json` (SHA-256
`{audit["source"]["report_sha256"]}`), eseguito al commit
`{audit["source"]["execution_commit"]}`. Non sono stati ripetuti fit o replay.

## Frontiera causale

Il candidato viene congelato al blocco 41 e accettato al blocco 63 (indici
0-based). I nuovi pesi e la nuova soglia entrano nelle predizioni dal blocco
64. Nel report Windows, stato, nove parametri e soglia usati al blocco 64
coincidono esattamente con il candidato congelato.

- digest dello stato: `{candidate["state_digest"]}`;
- pesi: `[{parameters}]`;
- soglia: `{format(candidate["threshold"], ".17g")}`;
- FIT: {candidate["fit"]["count"]} ID, digest
  `{candidate["fit"]["observation_ids_sha256"]}`;
- CAL: {candidate["cal"]["count"]} ID, digest
  `{candidate["cal"]["observation_ids_sha256"]}`.

Gli ID completi, nel loro ordine registrato, sono in
`UNSW45_FREEZE41_IDS.csv`.

## Confronto degli artefatti Windows/Linux

I digest FIT e CAL coincidono fra i due ambienti:

| Oggetto | Windows | Linux | Esito |
|---|---|---|---|
| FIT IDs | `{platform["fit_observation_ids_sha256"]["windows"]}` | `{platform["fit_observation_ids_sha256"]["linux"]}` | uguale |
| CAL IDs | `{platform["cal_observation_ids_sha256"]["windows"]}` | `{platform["cal_observation_ids_sha256"]["linux"]}` | uguale |
| Soglia dal blocco 64 | `{format(platform["operational_threshold_from_block_64"]["windows"], ".17g")}` | `{format(platform["operational_threshold_from_block_64"]["linux"], ".17g")}` | diversa |

La scheda riporta per Linux Python 3.13.5, NumPy 2.3.5,
scikit-learn 1.7.2, SciPy 1.16.2 e BLAS a un thread. Non riporta il vettore
dei nove pesi Linux; il vettore Windows è pubblicato sopra per consentire il
confronto diretto, ma non viene dichiarata un'identità o una differenza dei
pesi senza il secondo vettore. Dai dati disponibili la prima divergenza
osservabile è quindi la soglia applicata dal blocco 64, non la selezione degli
esempi FIT/CAL.

Il report Windows registra {windows_environment["platform"]}, Python
{windows_environment["python"]}, NumPy {windows_environment["numpy"]},
scikit-learn {windows_environment["sklearn"]} e River
{windows_environment["river"]}; i quattro limiti dei thread sono impostati a
uno.

## Confusione sull'intero stream

| Ambiente | TP | FP |
|---|---:|---:|
| Windows consegnato | {windows["tp"]} | {windows["fp"]} |
| Linux riportato dal relatore | {linux["tp"]} | {linux["fp"]} |
| Windows − Linux | {delta["tp"]:+d} | {delta["fp"]:+d} |

La differenza finale è coerente con una divergenza numerica della traiettoria,
ma questi dati non bastano ad attribuirla al fit oppure alla calibrazione: per
questa distinzione serve il vettore dei pesi Linux.
"""


def write_outputs(report_path: Path, out: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit, id_rows = extract_audit(report, file_sha256(report_path))
    out.mkdir(parents=True, exist_ok=True)
    (out / "UNSW45_WINDOWS_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (out / "UNSW45_FREEZE41_IDS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "position", "observation_id"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(id_rows)
    (out / "UNSW45_PLATFORM_NOTE.md").write_text(
        render_note(audit), encoding="utf-8", newline="\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_outputs(args.report, args.out)
    print("PASS UNSW/45 Windows audit")


if __name__ == "__main__":
    main()

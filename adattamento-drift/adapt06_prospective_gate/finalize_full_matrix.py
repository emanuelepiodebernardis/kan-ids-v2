"""Produce the requested ADAPT-06 finalization from the recorded reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


DATASETS = ("ton", "unsw")
SEEDS = tuple(range(42, 47))
TRACKS = ("frozen_b", "threshold_only", "always", "gated")
METRIC_FIELDS = ("tn", "fp", "fn", "tp", "normal", "attack", "recall", "fpr")
EXECUTION_COMMIT = "2b2ec4d02672be8432f9e462e7011359bf5c66d8"
ANALYSIS_PARENT_COMMIT = "f9fc725974698a62c989a4ceceab9a92f26d977b"
MINIMUM_PER_CLASS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report_name(dataset: str, seed: int) -> str:
    return f"{dataset}_spline_active_seed{seed}.json"


def load_primary_reports(root: Path) -> list[tuple[Path, dict]]:
    loaded = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / "reports" / report_name(dataset, seed)
            if not path.is_file():
                raise AssertionError(f"missing primary report: {path.name}")
            report = json.loads(path.read_text(encoding="utf-8"))
            expected = (dataset, seed, "spline", "active", "full")
            observed = (
                report["dataset"],
                int(report["seed"]),
                report["family"],
                report["acquisition"],
                report["scope"],
            )
            if observed != expected:
                raise AssertionError(f"unexpected report identity: {path.name}")
            if report["commit"] != EXECUTION_COMMIT:
                raise AssertionError(f"unexpected execution commit: {path.name}")
            loaded.append((path, report))
    return loaded


def same_confusion(first: dict, second: dict) -> bool:
    return all(first[field] == second[field] for field in METRIC_FIELDS)


def first_cal_eligible_block(report: dict) -> int | None:
    eligible = [
        int(block["block"])
        for block in report["per_block"]
        if int(block["cal_memory"]["normal"]) >= MINIMUM_PER_CLASS
        and int(block["cal_memory"]["attack"]) >= MINIMUM_PER_CLASS
    ]
    return min(eligible) if eligible else None


def harm_boundaries(decision: dict) -> tuple[bool, bool, bool]:
    if decision["status"] != "REJECT_HARM":
        return False, False, False
    candidate = decision["metrics"]["candidate"]
    incumbent = decision["metrics"]["operational_incumbent"]
    attack = int(candidate["attack"])
    normal = int(candidate["normal"])
    recall_harm = 100 * (int(incumbent["tp"]) - int(candidate["tp"])) >= 2 * attack
    candidate_fpr_harm = 100 * int(candidate["fp"]) > 6 * normal
    fpr_delta_harm = 100 * (int(candidate["fp"]) - int(incumbent["fp"])) > normal
    return recall_harm, candidate_fpr_harm, fpr_delta_harm


def reference_row(report: dict) -> dict[str, object]:
    decisions = report["decisions"]
    accepts = sum(event["status"] == "ACCEPT" for event in decisions)
    row: dict[str, object] = {
        "dataset": report["dataset"],
        "seed": int(report["seed"]),
    }
    for track in TRACKS:
        metrics = report["summary"][track]
        row[f"{track}_recall"] = metrics["recall"]
        row[f"{track}_fpr"] = metrics["fpr"]
    row["frozen_b_promotions"] = 0
    row["threshold_only_promotions"] = 0
    row["threshold_only_effective_changes"] = int(
        report["threshold_only"]["effective_changes"]
    )
    row["always_promotions"] = int(report["updates"]["candidate_freezes"])
    row["always_promotions_with_evaluated_effect"] = int(
        report["updates"]["always_promotions_with_evaluated_effect"]
    )
    row["always_promotions_after_last_prediction"] = int(
        report["updates"].get("always_promotions_after_last_prediction", 0)
    )
    row["gated_promotions"] = accepts
    return row


def diagnostic_row(report: dict) -> dict[str, object]:
    decisions = report["decisions"]
    statuses = [event["status"] for event in decisions]
    freezes = [event for event in decisions if event["status"] == "FREEZE"]
    evaluated = [event for event in decisions if event["status"] != "FREEZE"]
    boundaries = [harm_boundaries(event) for event in evaluated]
    first_block = report["per_block"][0]
    last_block = report["per_block"][-1]
    return {
        "dataset": report["dataset"],
        "seed": int(report["seed"]),
        "first_cal_eligible_block": first_cal_eligible_block(report),
        "first_candidate_freeze_block": int(freezes[0]["block"]) if freezes else None,
        "candidate_freezes": len(freezes),
        "accept": statuses.count("ACCEPT"),
        "reject_harm": statuses.count("REJECT_HARM"),
        "wait": statuses.count("WAIT"),
        "withhold_insufficient": sum("INSUFFICIENT" in status for status in statuses),
        "withhold_ambiguous": sum("AMBIGUOUS" in status for status in statuses),
        "reject_recall_harm": sum(item[0] for item in boundaries),
        "reject_candidate_fpr_above_6pp": sum(item[1] for item in boundaries),
        "reject_fpr_delta_above_1pp": sum(item[2] for item in boundaries),
        "initial_gated_threshold": first_block["thresholds_used"]["gated"],
        "final_gated_threshold": last_block["thresholds_used"]["gated"],
        "threshold_only_effective_changes": int(
            report["threshold_only"]["effective_changes"]
        ),
        "final_threshold_only_threshold": report["threshold_only"]["final_threshold"],
        "gated_equals_frozen_b": same_confusion(
            report["summary"]["gated"], report["summary"]["frozen_b"]
        ),
        "candidate_pending_after_stream": bool(report["candidate_pending_at_end"]),
    }


def mean(rows: list[dict[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def pct(value: float) -> str:
    return f"{100 * value:.2f}".replace(".", ",")


def blocks(values: list[int | None]) -> str:
    return ", ".join("n.d." if value is None else str(value) for value in values)


def quantity(value: int, singular: str, plural: str) -> str:
    return f"{value} {singular if value == 1 else plural}"


def build_note(
    reference: list[dict[str, object]], diagnostics: list[dict[str, object]]
) -> str:
    by_dataset = {
        dataset: [row for row in reference if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    diag_by_dataset = {
        dataset: [row for row in diagnostics if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    ton = by_dataset["ton"]
    unsw = by_dataset["unsw"]
    ton_diag = diag_by_dataset["ton"]
    unsw_diag = diag_by_dataset["unsw"]

    def total(dataset: str, field: str) -> int:
        return sum(int(row[field]) for row in diag_by_dataset[dataset])

    lines = [
        "# ADAPT-06 — eleggibilità CAL e diagnosi del gate",
        "",
        "## Perimetro",
        "",
        "Questa nota usa esclusivamente i dieci report della configurazione primaria GAM ACTIVE della matrice già eseguita: TON e UNSW, seed 42–46. Non introduce fit, replay, tuning, soglie o criteri nuovi. Il commit eseguito resta `2b2ec4d02672be8432f9e462e7011359bf5c66d8`; il commit della consegna è quello che contiene questa nota e deve essere riportato nella PR. L'esito preregistrato resta **NO-GO** per entrambi i dataset.",
        "",
        "`PRIMARY_REFERENCE_COMPARISON.csv` riporta per ogni dataset e seed recall e FPR globali di Frozen/B, threshold-only, ALWAYS e GATED. Le promozioni riguardano i candidati di modello: Frozen/B e threshold-only ne hanno zero. Per threshold-only è riportato separatamente il numero di variazioni effettive della soglia, evitando di chiamarle promozioni.",
        "",
        "## Quando CAL diventa eleggibile",
        "",
        f"CAL256 è eleggibile quando la memoria causale, dopo scadenze ed espulsioni, contiene almeno 100 normali e 100 attacchi. Nei seed 42–46 il primo blocco eleggibile è rispettivamente **{blocks([row['first_cal_eligible_block'] for row in ton_diag])}** su TON e **{blocks([row['first_cal_eligible_block'] for row in unsw_diag])}** su UNSW. Il primo congelamento del candidato avviene agli stessi blocchi: **{blocks([row['first_candidate_freeze_block'] for row in ton_diag])}** su TON e **{blocks([row['first_candidate_freeze_block'] for row in unsw_diag])}** su UNSW. In questi dieci report eleggibilità e congelamento coincidono; concettualmente restano distinti, perché il congelamento richiede anche FIT192 eleggibile e nessun candidato già aperto.",
        "",
        "## Perché i candidati non vengono promossi",
        "",
        f"Su TON i cinque seed producono {quantity(total('ton', 'candidate_freezes'), 'candidato', 'candidati')}: nessuna promozione, {quantity(total('ton', 'reject_harm'), 'rifiuto', 'rifiuti')} per una frontiera di danno preregistrata e {quantity(total('ton', 'withhold_insufficient'), 'chiusura', 'chiusure')} per supporto insufficiente. Su UNSW i candidati sono {total('unsw', 'candidate_freezes')}: {quantity(total('unsw', 'accept'), 'promozione', 'promozioni')}, {quantity(total('unsw', 'reject_harm'), 'rifiuto', 'rifiuti')} per danno, {quantity(total('unsw', 'withhold_insufficient'), 'chiusura', 'chiusure')} per supporto insufficiente e {quantity(total('unsw', 'withhold_ambiguous'), 'chiusura ambigua', 'chiusure ambigue')}. I conteggi completi per seed sono in `PRIMARY_GATE_DIAGNOSTICS.csv`.",
        "",
        f"Le cause possono sovrapporsi nello stesso rifiuto. Su TON la perdita di recall oltre la frontiera compare {total('ton', 'reject_recall_harm')} volte, l'FPR candidato sopra il 6% {total('ton', 'reject_candidate_fpr_above_6pp')} volte e l'incremento di FPR oltre 1 punto {total('ton', 'reject_fpr_delta_above_1pp')} volte. Su UNSW i conteggi sono rispettivamente {total('unsw', 'reject_recall_harm')}, {total('unsw', 'reject_candidate_fpr_above_6pp')} e {total('unsw', 'reject_fpr_delta_above_1pp')}. Questi sono motivi diagnostici derivati dagli stessi conteggi usati dal gate, non nuovi criteri decisionali.",
        "",
        "Un candidato non valutabile non è trattato come dato mancante favorevole: rimane un withholding. Analogamente, un look intermedio che non soddisfa né accettazione né danno resta WAIT e, se lo stream termina, viene chiuso secondo la regola terminale già preregistrata.",
        "",
        "## Protezione e miglioramento sono domande diverse",
        "",
        f"Su TON GATED coincide esattamente con Frozen/B in {sum(bool(row['gated_equals_frozen_b']) for row in ton_diag)}/5 seed e non promuove alcun candidato. Il gate evita i {total('ton', 'reject_harm')} aggiornamenti giudicati dannosi e non promuove gli altri {total('ton', 'withhold_insufficient')} candidati, chiusi per supporto insufficiente; conserva però anche la soglia B iniziale. In media, GATED/Frozen-B ha FPR **{pct(mean(ton, 'gated_fpr'))}%** e recall **{pct(mean(ton, 'gated_recall'))}%**; threshold-only porta l'FPR a **{pct(mean(ton, 'threshold_only_fpr'))}%** con recall **{pct(mean(ton, 'threshold_only_recall'))}%**. Il calo dell'FPR non deriva dall'apprendimento dei pesi: è l'effetto descrittivo della ricalibrazione causale della soglia sugli stessi score congelati.",
        "",
        f"Su UNSW il gate rifiuta {quantity(total('unsw', 'reject_harm'), 'candidato', 'candidati')} che oltrepassano almeno una frontiera di danno e accetta soltanto {quantity(total('unsw', 'accept'), 'candidato', 'candidati')}. I risultati documentano una protezione rispetto ad ALWAYS, non una superiorità stabile rispetto a Frozen/B. Le medie descrittive GATED sono recall **{pct(mean(unsw, 'gated_recall'))}%** e FPR **{pct(mean(unsw, 'gated_fpr'))}%**, contro **{pct(mean(unsw, 'frozen_b_recall'))}%** e **{pct(mean(unsw, 'frozen_b_fpr'))}%** per Frozen/B; le differenze per seed restano visibili nella tabella e non vengono compresse in un'affermazione di vittoria generale.",
        "",
        "## Conclusione operativa",
        "",
        "Il gate svolge il compito conservativo per cui è stato progettato: blocca candidati con evidenza prospettica sfavorevole. Il NO-GO mostra però che questa protezione non basta a produrre un beneficio stabile rispetto allo stato congelato e può mantenere una soglia iniziale inadatta al nuovo regime. Prima di ADAPT-07 occorre quindi discutere separatamente come trattare la soglia operativa e come ottenere supporto CAL/VERIFY sufficiente, senza modificare retroattivamente il protocollo di questa campagna.",
    ]
    return "\n".join(lines) + "\n"


def parse_archive_pointer(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    file_match = re.search(r"- file: `([^`]+)`", text)
    size_match = re.search(r"- dimensione: ([0-9.]+) byte", text)
    sha_match = re.search(r"- SHA-256: `([0-9a-f]{64})`", text)
    url_match = re.search(
        r"\[Google Drive\]\((https://drive\.google\.com/[^)]+)\)", text
    )
    if not all((file_match, size_match, sha_match, url_match)):
        raise AssertionError("incomplete archive pointer")
    return {
        "file": file_match.group(1),
        "bytes": int(size_match.group(1).replace(".", "")),
        "sha256": sha_match.group(1),
        "url": url_match.group(1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    reports = load_primary_reports(root)
    reference = [reference_row(report) for _, report in reports]
    diagnostics = [diagnostic_row(report) for _, report in reports]
    write_csv(out / "PRIMARY_REFERENCE_COMPARISON.csv", reference)
    write_csv(out / "PRIMARY_GATE_DIAGNOSTICS.csv", diagnostics)
    (out / "GATE_FINALIZATION_NOTE.md").write_text(
        build_note(reference, diagnostics), encoding="utf-8", newline="\n"
    )

    archive = parse_archive_pointer(out / "ARCHIVE.md")
    local_archive = root.parent / str(archive["file"])
    if not local_archive.is_file():
        raise AssertionError(f"missing local archive: {local_archive}")
    if local_archive.stat().st_size != archive["bytes"]:
        raise AssertionError("archive size does not match ARCHIVE.md")
    if sha256(local_archive) != archive["sha256"]:
        raise AssertionError("archive SHA-256 does not match ARCHIVE.md")

    generated = (
        "PRIMARY_REFERENCE_COMPARISON.csv",
        "PRIMARY_GATE_DIAGNOSTICS.csv",
        "GATE_FINALIZATION_NOTE.md",
        "FINALIZATION_TEST_LOG.txt",
    )
    manifest = {
        "schema": "adapt06-gate-finalization-v1",
        "execution_commit": EXECUTION_COMMIT,
        "analysis_parent_commit": ANALYSIS_PARENT_COMMIT,
        "delivery_commit_identification": "the commit containing this manifest; record its SHA in the PR",
        "scope": {
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "family": "spline",
            "acquisition": "active",
            "tracks": list(TRACKS),
            "new_fit_or_replay": False,
            "preregistered_outcome": "NO-GO",
        },
        "source_package": archive,
        "source_reports": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path, _ in reports
        },
        "generator": {
            "path": "adattamento-drift/adapt06_prospective_gate/finalize_full_matrix.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)}
            for name in generated
        },
    }
    (out / "FINALIZATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "PASS",
        "primary_reports=10",
        "reference_rows=10",
        "diagnostic_rows=10",
        "outcome=NO-GO",
    )


if __name__ == "__main__":
    main()

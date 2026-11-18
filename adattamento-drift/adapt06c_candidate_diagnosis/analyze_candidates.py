"""Riproduce la diagnosi ADAPT-06C dai dieci report decoupled esistenti."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
SOURCE_ROOT = REPOSITORY / "adattamento-drift" / "adapt06b_decoupled_threshold"
DATASETS = ("ton", "unsw")
SEEDS = (42, 43, 44, 45, 46)
VERIFY_LOOKS = ((100, 100), (200, 200), (400, 400))
BASE_MERGE = "24e25723ffac292b419adf2cff4378bc8d0b2c4a"

CYCLE_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "candidate_digest",
    "close_block",
    "duration_blocks",
    "outcome",
    "outcome_class",
    "reason",
    "fit_normal_at_freeze",
    "fit_attack_at_freeze",
    "cal_normal_at_freeze",
    "cal_attack_at_freeze",
    "verify_normal_at_close",
    "verify_attack_at_close",
    "complete_looks",
    "first_complete_look_block",
    "last_complete_look_block",
    "terminal_partial_diagnostic",
    "next_required_normal",
    "next_required_attack",
    "normal_shortfall",
    "attack_shortfall",
    "operational_incumbent_digest",
    "operational_incumbent_threshold",
    "candidate_threshold",
)

LOOK_FIELDS = (
    "dataset",
    "seed",
    "freeze_block",
    "candidate_digest",
    "look_sequence",
    "event_block",
    "event_status",
    "event_reason",
    "look_index",
    "look_kind",
    "decision_eligible",
    "normal",
    "attack",
    "candidate_tn",
    "candidate_fp",
    "candidate_fn",
    "candidate_tp",
    "incumbent_tn",
    "incumbent_fp",
    "incumbent_fn",
    "incumbent_tp",
    "delta_tp",
    "delta_fp",
    "delta_recall_pp",
    "delta_fpr_pp",
    "candidate_fpr_pp",
    "recall_gain_pass",
    "fpr_delta_pass",
    "candidate_fpr_pass",
    "recall_harm_triggered",
    "acceptance_failures",
    "harm_triggers",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(dataset: str, seed: int) -> Path:
    if seed == 42:
        root = SOURCE_ROOT / "pilot_seed42_02b82af"
    else:
        root = SOURCE_ROOT / "matrix_seed42_46_ed50e86" / "new_replays"
    return root / f"{dataset}_seed{seed}" / f"{dataset}_seed{seed}_decoupled.json"


def load_reports() -> dict[tuple[str, int], tuple[Path, dict]]:
    reports = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            path = source_path(dataset, seed)
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["dataset"] != dataset or int(report["seed"]) != seed:
                raise AssertionError(f"source identity mismatch: {path}")
            if report["method"] != "gated_decoupled_operational_threshold":
                raise AssertionError(f"unexpected method: {path}")
            reports[(dataset, seed)] = (path, report)
    return reports


def _outcome_class(status: str) -> str:
    if status == "ACCEPT":
        return "accepted"
    if status == "REJECT_HARM":
        return "harmful_by_fixed_margin"
    if "INSUFFICIENT" in status:
        return "insufficient_support"
    if status.startswith("WITHHOLD"):
        return "ambiguous"
    raise AssertionError(f"unexpected terminal status: {status}")


def _metrics(event: dict, name: str) -> dict:
    metric = event["metrics"][name]
    required = {"tn", "fp", "fn", "tp", "normal", "attack", "recall", "fpr"}
    if not required.issubset(metric):
        raise AssertionError(f"incomplete {name} metrics")
    if int(metric["tn"]) + int(metric["fp"]) != int(event["normal"]):
        raise AssertionError(f"{name} normal confusion does not close")
    if int(metric["fn"]) + int(metric["tp"]) != int(event["attack"]):
        raise AssertionError(f"{name} attack confusion does not close")
    return metric


def _complete_look_conditions(event: dict) -> dict:
    normal = int(event["normal"])
    attack = int(event["attack"])
    candidate = _metrics(event, "candidate")
    incumbent = _metrics(event, "operational_incumbent")
    delta_tp = int(candidate["tp"]) - int(incumbent["tp"])
    delta_fp = int(candidate["fp"]) - int(incumbent["fp"])
    recall_gain_pass = delta_tp * 50 >= attack
    fpr_delta_pass = delta_fp * 100 <= normal
    candidate_fpr_pass = int(candidate["fp"]) * 50 <= normal * 3
    recall_harm_triggered = -delta_tp * 50 >= attack
    acceptance_failures = []
    if not recall_gain_pass:
        acceptance_failures.append("RECALL_GAIN_LT_0.02")
    if not fpr_delta_pass:
        acceptance_failures.append("FPR_DELTA_GT_0.01")
    if not candidate_fpr_pass:
        acceptance_failures.append("CANDIDATE_FPR_GT_0.06")
    harm_triggers = []
    if not fpr_delta_pass:
        harm_triggers.append("FPR_DELTA_GT_0.01")
    if not candidate_fpr_pass:
        harm_triggers.append("CANDIDATE_FPR_GT_0.06")
    if recall_harm_triggered:
        harm_triggers.append("RECALL_HARM_GE_0.02")
    return {
        "delta_tp": delta_tp,
        "delta_fp": delta_fp,
        "delta_recall_pp": 100.0 * delta_tp / attack,
        "delta_fpr_pp": 100.0 * delta_fp / normal,
        "candidate_fpr_pp": 100.0 * int(candidate["fp"]) / normal,
        "recall_gain_pass": recall_gain_pass,
        "fpr_delta_pass": fpr_delta_pass,
        "candidate_fpr_pass": candidate_fpr_pass,
        "recall_harm_triggered": recall_harm_triggered,
        "acceptance_failures": ";".join(acceptance_failures) or "NONE",
        "harm_triggers": ";".join(harm_triggers) or "NONE",
    }


def _expected_complete_status(event: dict) -> str:
    conditions = _complete_look_conditions(event)
    if (
        conditions["recall_gain_pass"]
        and conditions["fpr_delta_pass"]
        and conditions["candidate_fpr_pass"]
    ):
        return "ACCEPT"
    if conditions["harm_triggers"] != "NONE":
        return "REJECT_HARM"
    if int(event["look_index"]) == len(VERIFY_LOOKS) - 1:
        return "WITHHOLD_AMBIGUOUS"
    return "WAIT"


def _look_row(
    *, dataset: str, seed: int, freeze: dict, event: dict, sequence: int
) -> dict:
    candidate = _metrics(event, "candidate")
    incumbent = _metrics(event, "operational_incumbent")
    complete = event.get("look_index") is not None
    if complete:
        kind = "complete_decision"
        conditions = _complete_look_conditions(event)
        if _expected_complete_status(event) != event["status"]:
            raise AssertionError("recorded gate status does not match exact counts")
    else:
        kind = (
            "terminal_partial_diagnostic"
            if sequence > 1
            else "terminal_before_first_look"
        )
        conditions = {
            "delta_tp": "",
            "delta_fp": "",
            "delta_recall_pp": "",
            "delta_fpr_pp": "",
            "candidate_fpr_pp": "",
            "recall_gain_pass": "",
            "fpr_delta_pass": "",
            "candidate_fpr_pass": "",
            "recall_harm_triggered": "",
            "acceptance_failures": "NOT_EVALUATED",
            "harm_triggers": "NOT_EVALUATED",
        }
    row = {
        "dataset": dataset,
        "seed": seed,
        "freeze_block": int(freeze["block"]),
        "candidate_digest": freeze["candidate_digest"],
        "look_sequence": sequence,
        "event_block": int(event["block"]),
        "event_status": event["status"],
        "event_reason": event["reason"],
        "look_index": int(event["look_index"]) if complete else "",
        "look_kind": kind,
        "decision_eligible": complete,
        "normal": int(event["normal"]),
        "attack": int(event["attack"]),
        "candidate_tn": int(candidate["tn"]),
        "candidate_fp": int(candidate["fp"]),
        "candidate_fn": int(candidate["fn"]),
        "candidate_tp": int(candidate["tp"]),
        "incumbent_tn": int(incumbent["tn"]),
        "incumbent_fp": int(incumbent["fp"]),
        "incumbent_fn": int(incumbent["fn"]),
        "incumbent_tp": int(incumbent["tp"]),
    }
    row.update(conditions)
    return row


def extract(
    reports: dict[tuple[str, int], tuple[Path, dict]],
) -> tuple[list[dict], list[dict]]:
    cycles = []
    looks = []
    for dataset in DATASETS:
        for seed in SEEDS:
            report = reports[(dataset, seed)][1]
            active: dict[str, dict] = {}
            for event in report["decisions"]:
                status = event["status"]
                digest = event["candidate_digest"]
                if status == "FREEZE":
                    if digest in active:
                        raise AssertionError("candidate digest frozen twice")
                    active[digest] = {"freeze": event, "events": []}
                    continue
                if digest not in active:
                    raise AssertionError("gate event without matching FREEZE")
                active[digest]["events"].append(event)
                if status == "WAIT":
                    continue
                item = active.pop(digest)
                freeze = item["freeze"]
                terminal = event
                complete_events = [
                    value
                    for value in item["events"]
                    if value.get("look_index") is not None
                ]
                for sequence, value in enumerate(item["events"], start=1):
                    looks.append(
                        _look_row(
                            dataset=dataset,
                            seed=seed,
                            freeze=freeze,
                            event=value,
                            sequence=sequence,
                        )
                    )
                next_look = len(complete_events)
                is_withhold = terminal["status"].startswith("WITHHOLD")
                if is_withhold and next_look < len(VERIFY_LOOKS):
                    required_normal, required_attack = VERIFY_LOOKS[next_look]
                    normal_shortfall = max(0, required_normal - int(terminal["normal"]))
                    attack_shortfall = max(0, required_attack - int(terminal["attack"]))
                else:
                    required_normal = required_attack = ""
                    normal_shortfall = attack_shortfall = ""
                cycles.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "freeze_block": int(freeze["block"]),
                        "candidate_digest": digest,
                        "close_block": int(terminal["block"]),
                        "duration_blocks": int(terminal["block"])
                        - int(freeze["block"]),
                        "outcome": terminal["status"],
                        "outcome_class": _outcome_class(terminal["status"]),
                        "reason": terminal["reason"],
                        "fit_normal_at_freeze": int(freeze["fit_counts"][0]),
                        "fit_attack_at_freeze": int(freeze["fit_counts"][1]),
                        "cal_normal_at_freeze": int(freeze["cal_counts"][0]),
                        "cal_attack_at_freeze": int(freeze["cal_counts"][1]),
                        "verify_normal_at_close": int(terminal["normal"]),
                        "verify_attack_at_close": int(terminal["attack"]),
                        "complete_looks": len(complete_events),
                        "first_complete_look_block": (
                            int(complete_events[0]["block"]) if complete_events else ""
                        ),
                        "last_complete_look_block": (
                            int(complete_events[-1]["block"]) if complete_events else ""
                        ),
                        "terminal_partial_diagnostic": terminal.get("look_index")
                        is None,
                        "next_required_normal": required_normal,
                        "next_required_attack": required_attack,
                        "normal_shortfall": normal_shortfall,
                        "attack_shortfall": attack_shortfall,
                        "operational_incumbent_digest": freeze[
                            "operational_incumbent_digest"
                        ],
                        "operational_incumbent_threshold": float(
                            freeze["operational_incumbent_threshold"]
                        ),
                        "candidate_threshold": float(freeze["candidate_threshold"]),
                    }
                )
            if active:
                raise AssertionError(f"unclosed candidates in {dataset}/seed{seed}")
    cycles.sort(key=lambda row: (row["dataset"], row["seed"], row["freeze_block"]))
    looks.sort(
        key=lambda row: (
            row["dataset"],
            row["seed"],
            row["freeze_block"],
            row["look_sequence"],
        )
    )
    return cycles, looks


def _csv(rows: list[dict], fields: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _first_cal_frontiers(
    reports: dict[tuple[str, int], tuple[Path, dict]],
) -> dict[str, list[tuple[int, int]]]:
    result = {dataset: [] for dataset in DATASETS}
    for (dataset, _seed), (_path, report) in reports.items():
        proposed = [
            event
            for event in report["decoupled_threshold"]["history"]
            if event["status"] == "PROPOSED"
        ]
        if not proposed:
            raise AssertionError("report has no CAL proposal")
        result[dataset].append(
            (int(proposed[0]["calibration_block"]), int(proposed[0]["effective_block"]))
        )
    return result


def render_note(
    reports: dict[tuple[str, int], tuple[Path, dict]],
    cycles: list[dict],
    looks: list[dict],
) -> str:
    outcomes = Counter(row["outcome"] for row in cycles)
    complete = [row for row in looks if row["decision_eligible"]]
    partial = [row for row in looks if not row["decision_eligible"]]
    quadrants = Counter()
    for row in complete:
        delta_tp = int(row["delta_tp"])
        delta_fp = int(row["delta_fp"])
        if delta_tp > 0 and delta_fp > 0:
            quadrants["more_tp_more_fp"] += 1
        elif delta_tp > 0 and delta_fp <= 0:
            quadrants["more_tp_not_more_fp"] += 1
        elif delta_tp <= 0 and delta_fp > 0:
            quadrants["not_more_tp_more_fp"] += 1
        else:
            quadrants["not_more_tp_not_more_fp"] += 1
    by_dataset = {
        dataset: Counter(
            row["outcome_class"] for row in cycles if row["dataset"] == dataset
        )
        for dataset in DATASETS
    }
    harmful = [row for row in complete if row["event_status"] == "REJECT_HARM"]
    triggers = Counter()
    for row in harmful:
        for trigger in str(row["harm_triggers"]).split(";"):
            if trigger != "NONE":
                triggers[trigger] += 1
    accepted = next(row for row in complete if row["event_status"] == "ACCEPT")
    cal_frontiers = _first_cal_frontiers(reports)
    ton_calibration = [item[0] for item in cal_frontiers["ton"]]
    ton_application = [item[1] for item in cal_frontiers["ton"]]
    unsw_candidate_seconds = float(
        reports[("unsw", 42)][1]["timings_seconds"]["candidate_fit_calibration_seconds"]
    )
    unsw_replay_seconds = float(reports[("unsw", 42)][1]["elapsed_replay_seconds"])
    return f"""# ADAPT-06C — qualità dei candidati

## Perimetro e metodo

Questa è una diagnosi retrospettiva sui dati di sviluppo. Legge i dieci JSON
decoupled già verificati, senza nuovi fit, replay o tuning. La chiave di un
ciclo è dataset/seed/blocco di `FREEZE`/digest. Il confronto usa esclusivamente
l'incumbent operativo congelato all'apertura e le stesse osservazioni VERIFY.
Le condizioni sono ricalcolate con rapporti esatti sui conteggi: guadagno di
recall almeno 2%, aumento FPR non oltre 1 punto e FPR candidato non oltre 6%.

I 28 `FREEZE` producono {outcomes["ACCEPT"]} `ACCEPT`,
{outcomes["REJECT_HARM"]} `REJECT_HARM` e {sum(value for key, value in outcomes.items() if key.startswith("WITHHOLD"))} `WITHHOLD`;
i dieci `WAIT` sono look intermedi. Ogni ciclo è chiuso. La tabella dei look
contiene {len(complete)} decisioni complete e {len(partial)} chiusure parziali
diagnostiche. Per queste ultime confusioni e supporti sono conservati, ma delta
e condizioni decisionali restano non valutati.

## TON e UNSW

Su TON si osservano {by_dataset["ton"]["harmful_by_fixed_margin"]} rifiuti per
margine dannoso, {by_dataset["ton"]["ambiguous"]} esito ambiguo e
{by_dataset["ton"]["insufficient_support"]} chiusure per supporto insufficiente;
non vi sono promozioni. La prima calibrazione eleggibile viene proposta a fine
blocco {min(ton_calibration)}–{max(ton_calibration)} e applicata dal blocco successivo
{min(ton_application)}–{max(ton_application)}. Questa attesa iniziale è
precedente ai cicli e non è tempo trascorso in VERIFY. Dopo ogni `FREEZE`, la
durata riportata parte invece dall'apertura del candidato.

Su UNSW si osservano {by_dataset["unsw"]["harmful_by_fixed_margin"]} rifiuti,
{by_dataset["unsw"]["ambiguous"]} esiti ambigui,
{by_dataset["unsw"]["insufficient_support"]} casi insufficienti e l'unico
`ACCEPT`. CAL diventa disponibile molto prima che su TON e consente più
osservazioni VERIFY nel prefisso, ma questo non rende automaticamente migliori
i pesi. I log non separano causalmente qualità del FIT, composizione temporale
di VERIFY e drift dello stream: mostrano l'esito congiunto del protocollo.

Nei look completi, {quadrants["more_tp_more_fp"]} confronti aumentano sia TP sia
FP: sono compromessi empirici, non automaticamente esiti favorevoli. Altri
{quadrants["more_tp_not_more_fp"]} aumentano i TP senza aumentare gli FP,
{quadrants["not_more_tp_more_fp"]} non aumentano i TP ma aumentano gli FP e
{quadrants["not_more_tp_not_more_fp"]} non aumentano né TP né FP. Questa lettura
per quadrante è distinta dall'esito del gate. Fra i 15 candidati rifiutati, le
violazioni simultanee restano tutte visibili: FPR candidato oltre 6% in
{triggers["CANDIDATE_FPR_GT_0.06"]} look, delta FPR oltre 1 punto in
{triggers["FPR_DELTA_GT_0.01"]} e danno di recall almeno 2% in
{triggers["RECALL_HARM_GE_0.02"]}. Questi conteggi non sono cause indipendenti:
più condizioni possono fallire nello stesso look.

## Il caso accettato

L'unico candidato promosso è UNSW/seed42, congelato al blocco 18 e deciso al
blocco {accepted["event_block"]} (indici 0-based). Su {accepted["normal"]}
normali e {accepted["attack"]} attacchi registra {accepted["candidate_tp"]} TP
e {accepted["candidate_fp"]} FP, contro {accepted["incumbent_tp"]} TP e
{accepted["incumbent_fp"]} FP dell'incumbent: delta recall
{float(accepted["delta_recall_pp"]):+.3f} punti, delta FPR
{float(accepted["delta_fpr_pp"]):+.3f} e FPR candidato
{float(accepted["candidate_fpr_pp"]):.3f}%. Il guadagno di recall supera il
margine; entrambi i vincoli FPR restano entro i limiti. È un'accettazione
locale su VERIFY, non la prestazione controfattuale dell'intero stream.

## Ipotesi e decisione operativa

**Ipotesi 1 — preferita.** Il limite dominante è la qualità instabile dei
candidati prodotti dal FIT192: molti look attraversano almeno un margine di
danno e la sola maggiore disponibilità di VERIFY non garantisce una
promozione utile. I log sostengono la diagnosi del gate, ma non distinguono se
l'instabilità provenga da composizione, età o capacità di FIT.

**Ipotesi 2.** In una parte dei cicli la potenza di verifica è il limite:
supporto insufficiente o chiusure ambigue impediscono una decisione anche
senza un danno già dimostrato. I log non consentono di sapere se ulteriori
etichette avrebbero confermato o smentito quei candidati.

Propongo un solo esperimento minimo, subordinato a un protocollo separato:
**confronto shadow FIT192/FIT256 nei tre `FREEZE` già osservati su UNSW/seed42**.
Si riusano richieste, label disponibili, blocchi di congelamento, incumbent e
osservazioni VERIFY; i candidati shadow non vengono promossi e non cambiano la
traiettoria. La metrica primaria è il numero di candidati che superano
congiuntamente i tre criteri sul medesimo look; controllo FIT192 registrato,
secondarie confusioni appaiate e costo. Prima dei risultati va fissato GO
soltanto se FIT256 aggiunge almeno un candidato che passa tutti i margini e non
introduce nuovi attraversamenti di danno nei tre confronti. Servono tre fit
shadow e scoring sugli intervalli già fissati, non due replay completi: i tre
fit/calibrazioni candidati originali hanno richiesto complessivamente
{unsw_candidate_seconds:.3f} s e l'intero replay decoupled {unsw_replay_seconds:.1f}
s. Con caricamento, ricostruzione, scoring e controlli si stima quindi un wall
time conservativo di 1–2 minuti sullo stesso dispositivo. Un esito nullo
smentirebbe l'ipotesi che la sola capacità FIT sia sufficiente. Non è ancora
giustificata una griglia di tuning né ADAPT-07.
"""


def validate_delivery(cycles: list[dict], looks: list[dict]) -> None:
    if len(cycles) != 28:
        raise AssertionError("the diagnosis must contain exactly 28 candidate cycles")
    keys = {
        (row["dataset"], row["seed"], row["freeze_block"], row["candidate_digest"])
        for row in cycles
    }
    if len(keys) != len(cycles):
        raise AssertionError("candidate cycle keys are not unique")
    outcomes = Counter(row["outcome"] for row in cycles)
    if outcomes["ACCEPT"] != 1 or outcomes["REJECT_HARM"] != 15:
        raise AssertionError("accepted/rejected candidate totals changed")
    if (
        sum(value for key, value in outcomes.items() if key.startswith("WITHHOLD"))
        != 12
    ):
        raise AssertionError("withheld candidate total changed")
    if sum(row["event_status"] == "WAIT" for row in looks) != 10:
        raise AssertionError("WAIT look total changed")
    if len(looks) != 38:
        raise AssertionError("available look/terminal record total changed")
    cycle_keys = {
        (row["dataset"], row["seed"], row["freeze_block"], row["candidate_digest"])
        for row in looks
    }
    if cycle_keys != keys:
        raise AssertionError("look table does not cover every candidate cycle")


def build_manifest(
    reports: dict[tuple[str, int], tuple[Path, dict]], out: Path
) -> dict:
    sources = {}
    source_commits = set()
    for path, report in reports.values():
        relative = path.relative_to(REPOSITORY).as_posix()
        sources[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        source_commits.add(report["commit"])
    output_names = (
        "CANDIDATE_DIAGNOSIS.csv",
        "CANDIDATE_LOOKS.csv",
        "CANDIDATE_NOTE.md",
    )
    outputs = {
        name: {
            "bytes": (out / name).stat().st_size,
            "sha256": file_sha256(out / name),
        }
        for name in output_names
    }
    provenance = {}
    for name in (
        "__init__.py",
        "README.md",
        "PROTOCOL.md",
        "TEST_LOG.txt",
        "analyze_candidates.py",
        "test_candidate_diagnosis.py",
    ):
        path = HERE / name
        if path.is_file():
            provenance[name] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return {
        "schema": "adapt06c-candidate-diagnosis-manifest-v1",
        "base_merge": BASE_MERGE,
        "delivery_commit_identification": "the PR head containing this manifest",
        "source_commits": sorted(source_commits),
        "source_reports": sources,
        "outputs": outputs,
        "provenance": provenance,
    }


def write_outputs(out: Path) -> tuple[list[dict], list[dict]]:
    reports = load_reports()
    cycles, looks = extract(reports)
    validate_delivery(cycles, looks)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CANDIDATE_DIAGNOSIS.csv").write_text(
        _csv(cycles, CYCLE_FIELDS), encoding="utf-8", newline="\n"
    )
    (out / "CANDIDATE_LOOKS.csv").write_text(
        _csv(looks, LOOK_FIELDS), encoding="utf-8", newline="\n"
    )
    (out / "CANDIDATE_NOTE.md").write_text(
        render_note(reports, cycles, looks), encoding="utf-8", newline="\n"
    )
    manifest = build_manifest(reports, out)
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return cycles, looks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv)
    cycles, looks = write_outputs(args.out.resolve())
    outcomes = Counter(row["outcome"] for row in cycles)
    print(
        "PASS",
        f"cycles={len(cycles)}",
        f"looks={len(looks)}",
        f"accept={outcomes['ACCEPT']}",
        f"reject={outcomes['REJECT_HARM']}",
        f"withhold={sum(v for k, v in outcomes.items() if k.startswith('WITHHOLD'))}",
    )


if __name__ == "__main__":
    main()

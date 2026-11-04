import csv
from pathlib import Path
import sys
import zipfile

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from adapt05_occurrence import analyze, campaign, package_campaign


def test_expected_matrix_has_exactly_thirty_unique_cells():
    rows = campaign.read_matrix(HERE / "MATRIX.csv")
    assert len(rows) == 30
    assert len({(row["dataset"], row["seed_model_query"], row["role_salt"])
               for row in rows}) == 30
    assert len(analyze.expected_stems()) == 30


def test_matrix_reader_rejects_duplicate_or_missing_cell(tmp_path):
    source = list(csv.DictReader((HERE / "MATRIX.csv").open(encoding="utf-8")))
    source[-1] = dict(source[0])
    path = tmp_path / "matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source[0])
        writer.writeheader()
        writer.writerows(source)
    try:
        campaign.read_matrix(path)
    except ValueError as error:
        assert "30 combinations" in str(error)
    else:
        raise AssertionError("invalid matrix accepted")


def test_paired_differences_keep_direction_head_minus_frozen():
    rows = []
    for mode, offset in (("vector", 0.0), ("occurrence", 0.1)):
        for score, gain in (("frozen", 0.0), ("head", 0.02)):
            rows.append({"dataset": "ton", "seed": 42, "salt": "default",
                         "mode": mode, "score": score, "objective": "lambda1",
                         "recall": .7 + offset + gain, "fpr": .1 + gain,
                         "ba": .8 + gain, "fp": 10 + 100 * gain,
                         "fn": 20 - 100 * gain})
    paired, role = analyze.paired_tables(pd.DataFrame(rows))
    assert np.allclose(paired.delta_ba, 0.02)
    assert np.allclose(role.delta_recall, 0.1)


def test_package_is_deterministic_and_manifest_covers_files(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "b.txt").write_text("b\n", encoding="utf-8")
    manifest = root / "MANIFEST.sha256"
    package_campaign.write_manifest(root, manifest)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    package_campaign.build_archive(root, first)
    package_campaign.build_archive(root, second)
    assert package_campaign.sha256(first) == package_campaign.sha256(second)
    assert "a.txt" in manifest.read_text(encoding="utf-8")
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["MANIFEST.sha256", "a.txt", "nested/b.txt"]


def test_cal64_memory_rows_keep_online_state_separate_from_coverage():
    report = {
        "dataset": "ton", "seed_model_query": 42, "role_salt_name": "default",
        "per_block": [
            {"block": 0},
            {
                "block": 1,
                "feedback": {
                    "from_block": 0,
                    "modes": {
                        mode: {
                            "status": "applied",
                            "memory": {"cal64": {
                                "size": 8, "normal": 3, "attack": 5,
                                "recent_normal": 2, "recent_attack": 5,
                                "age_min": 1, "age_max": 4,
                            }},
                        }
                        for mode in analyze.MODES
                    },
                },
            },
        ],
    }
    rows = analyze.calibration_memory_rows([report])
    assert len(rows) == 2
    assert set(rows["mode"]) == set(analyze.MODES)
    assert set(rows.normal) == {3}
    assert set(rows.recent_normal) == {2}
    assert set(rows.eligible_age_max) == {4}
    assert not rows.no_normal.any()


def test_adherence_table_declares_missing_analyses_explicitly():
    rows = analyze.adherence_rows()
    assert {"complete", "partial", "not_estimable"}.issubset(set(rows.status))
    common_fpr = rows[rows.planned_analysis.str.contains("FPR raggiunto comune")]
    assert common_fpr.iloc[0].status == "not_estimable"
    assert "none" == common_fpr.iloc[0].outputs


def test_type_coverage_is_per_normal_vector_digest(tmp_path, monkeypatch):
    dataset = tmp_path / "toy"
    dataset.mkdir()
    np.savez(
        dataset / "D.npz",
        X=np.asarray([[1.0] * 8, [1.0] * 8, [2.0] * 8, [9.0] * 8]),
        y=np.asarray([0, 0, 0, 1]),
        row_id=np.arange(4),
    )
    monkeypatch.setattr(analyze, "DATASETS", ("toy",))
    monkeypatch.setattr(
        analyze.runner.experiment, "query_indices",
        lambda rows, _seed, _block, _budget: np.arange(rows),
    )
    coverage, detail, checks = analyze.coverage_rows(tmp_path)
    assert len(coverage) == 5 * 3 * 2
    assert checks == len(coverage) * 4
    assert "vector_digest" in detail
    assert "type_id" not in detail
    assert set(detail.total_normal_requests) == {15, 30}
    assert (detail.calibration_assignments + detail.fit_assignments ==
            detail.total_normal_requests).all()
    assert set(detail.seed_salt_cells) == {15}
    assert detail[detail.total_normal_requests == 30].repeated_within_requested_D.all()

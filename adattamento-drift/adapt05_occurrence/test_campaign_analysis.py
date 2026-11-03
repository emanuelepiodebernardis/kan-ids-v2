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

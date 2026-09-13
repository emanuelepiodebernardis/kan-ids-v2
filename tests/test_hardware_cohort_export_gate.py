"""Synthetic provenance negatives only; these fixtures are not benchmark inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location("hw_export",Path(__file__).resolve().parents[1]/"scripts/export_hardware_cohort.py")
EXPORT=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXPORT)

def fixture(tmp_path):
    # Tiny synthetic files exercise integrity gates without source redistribution.
    source=tmp_path/"source.csv"; source.write_text("synthetic,test\n", encoding="utf-8", newline="\n")
    cohort=tmp_path/"hardware_cohort.npz"
    ids=np.arange(500,dtype=np.int64)
    data={"row_ids":ids}
    np.savez(cohort,**data)
    raw=tmp_path/"hardware_raw_flows.csv"; raw.write_text("synthetic,fixture\n", encoding="utf-8", newline="\n")
    pre=tmp_path/"preprocessor.npz"; pre.write_bytes(b"synthetic preprocessing checksum fixture")
    meta={"npz_sha256":EXPORT.sha(cohort),"row_ids_sha256":hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
          "n_unique_flows":500,"source_rows":1000,"source_csv_sha256":EXPORT.sha(source),
          "raw_flows_sha256":EXPORT.sha(raw),"preprocessing_hashes":{pre.name:EXPORT.sha(pre)}}
    path=tmp_path/"hardware_cohort.json"; path.write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    return source,cohort,path,data

def test_valid_manifest_binding(tmp_path):
    source,cohort,path,data=fixture(tmp_path)
    assert EXPORT.validate_provenance(cohort,path,source,data)["n_unique_flows"]==500

def test_stale_npz_cannot_acquire_new_c_reference(tmp_path):
    source,cohort,path,data=fixture(tmp_path)
    np.savez(cohort,**data,changed_prepared_features=np.ones((500,10),dtype=np.int16))
    with pytest.raises(ValueError,match="NPZ differs"):
        EXPORT.validate_provenance(cohort,path,source,data)

def test_source_csv_must_be_the_bound_file(tmp_path):
    source,cohort,path,data=fixture(tmp_path)
    source.write_text("different synthetic source\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError,match="Source CSV differs"):
        EXPORT.validate_provenance(cohort,path,source,data)

def test_reordered_raw_ids_are_rejected(tmp_path):
    source,cohort,path,data=fixture(tmp_path)
    data["row_ids"]=data["row_ids"][::-1]
    with pytest.raises(ValueError,match="Ordered raw row IDs"):
        EXPORT.validate_provenance(cohort,path,source,data)

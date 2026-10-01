from __future__ import annotations

import h5py
import numpy as np
import pytest

from g4utils.HDF5.shared import RUN_LOG_COLUMNS, _read_run_log
from g4utils.HDF5.vox_file_3d import G4VoxFile3D

from .conftest import write_snapshot3d

DEFAULT = ["unix_timestamp", "primaries", "runtime_s", "subrun_id"]


def test_names_from_attr(tmp_path):
    p = write_snapshot3d(tmp_path / "a.h5", subrun_ids=(3, 4))
    sim = G4VoxFile3D(p)
    assert sim.run_log.columns.tolist() == DEFAULT
    assert sim.run_log["subrun_id"].tolist() == [3, 4]
    assert sim.total_primaries() == 200


def test_names_without_attr(tmp_path):
    p = write_snapshot3d(tmp_path / "a.h5", run_log_columns_attr=False)
    sim = G4VoxFile3D(p)
    assert sim.run_log.columns.tolist() == list(RUN_LOG_COLUMNS)


def test_dtypes_and_values(tmp_path):
    p = write_snapshot3d(tmp_path / "a.h5", subrun_ids=(0, 5), primaries=7)
    df = G4VoxFile3D(p).run_log
    assert df["subrun_id"].dtype == np.int64
    assert df["primaries"].dtype == np.int64
    assert df["unix_timestamp"].dtype == np.float64
    assert df["runtime_s"].dtype == np.float64
    assert df["primaries"].tolist() == [7, 7]
    assert df["runtime_s"].tolist() == [1.5, 1.5]


def test_single_row(tmp_path):
    p = write_snapshot3d(tmp_path / "a.h5", subrun_ids=(9,))
    df = G4VoxFile3D(p).run_log
    assert df.shape == (1, 4)
    assert df["subrun_id"].tolist() == [9]


def test_missing_run_log(tmp_path):
    p = write_snapshot3d(tmp_path / "a.h5", with_run_log=False)
    sim = G4VoxFile3D(p)
    assert sim.run_log is None
    assert sim.total_primaries() == 0


@pytest.mark.parametrize("attr", ["a, b ,c,d", b"a, b ,c,d"])
def test_attr_str_or_bytes(tmp_path, attr):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        ds = f.create_dataset("run_log", data=np.ones((2, 4)))
        ds.attrs["columns"] = attr
    with h5py.File(p, "r") as f:
        df = _read_run_log(f)
    assert df.columns.tolist() == ["a", "b", "c", "d"]


def test_count_mismatch_warns(tmp_path):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        ds = f.create_dataset("run_log", data=np.ones((2, 6)))
        ds.attrs["columns"] = "x,y"
    with h5py.File(p, "r") as f:
        with pytest.warns(UserWarning):
            df = _read_run_log(f)
    assert df.columns.tolist() == [*DEFAULT, "col_4", "col_5"]

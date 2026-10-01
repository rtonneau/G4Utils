"""Subrun IDs for Extendable4D files come from ``run_log["subrun_id"]``."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from g4utils.HDF5 import G4VoxFile4D

from .conftest import expected_array, write_extendable4d

DOSE, EDEP = 0, 1


def _open_no_warning(path) -> G4VoxFile4D:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return G4VoxFile4D(path)


@pytest.fixture
def sparse_ids_file(tmp_path):
    return write_extendable4d(tmp_path / "ids579.h5", subrun_ids=(5, 7, 9))


def test_subrun_ids_from_run_log(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    assert sim.subrun_ids == [5, 7, 9]
    assert sim.n_subruns == 3


def test_iteration_yields_run_log_ids(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    seen = []
    for sid in sim:
        seen.append(sid)
        np.testing.assert_array_equal(sim.data["Dose"], expected_array(DOSE, sid))
        np.testing.assert_array_equal(sim.data["Edep"], expected_array(EDEP, sid))
    assert seen == [5, 7, 9]


def test_select_subrun_by_id_loads_matching_slice(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    sim.select_subrun([7])
    assert list(sim) == [7]
    sim.select_subrun([7])
    sid = next(iter(sim))
    assert sid == 7
    np.testing.assert_array_equal(sim.data["Dose"], expected_array(DOSE, 7))


def test_get_by_subrun_id(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    np.testing.assert_array_equal(sim.get("Dose", 9), expected_array(DOSE, 9))


def test_get_unknown_subrun_raises_keyerror(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    with pytest.raises(KeyError, match="Subrun '1' not found"):
        sim.get("Dose", 1)


def test_select_subrun_range_uses_ids(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    sim.select_subrun(start=6, stop=10)
    assert sim.selected_subrun_ids == [7, 9]


def test_sum_uses_ids(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    total = sim.sum("Dose", [5, 9])
    np.testing.assert_array_equal(total, expected_array(DOSE, 5) + expected_array(DOSE, 9))


def test_pvd_timesteps_are_subrun_ids(sparse_ids_file, tmp_path):
    sim = _open_no_warning(sparse_ids_file)
    pvd = sim.dump_selection_to_vti_timeseries(tmp_path / "out" / "series.pvd")
    root = ET.parse(pvd).getroot()
    timesteps = [float(ds.get("timestep")) for ds in root.iter("DataSet")]
    assert timesteps == [5.0, 7.0, 9.0]


def test_quantity_names_stable_during_iteration(sparse_ids_file):
    sim = _open_no_warning(sparse_ids_file)
    sim.select_quantity("Dose")
    for _ in sim:
        assert sim.quantity_names == ["Dose", "Edep"]
        assert list(sim.data) == ["Dose"]
    assert sim.quantity_names == ["Dose", "Edep"]


# ── Fallback to slice indices ────────────────────────────────────────────────


def _assert_slices_by_index(sim, slice_ids):
    """Fallback IDs are slice indices; slice ``i`` holds data for ``slice_ids[i]``."""
    assert sim.subrun_ids == list(range(len(slice_ids)))
    for i, written_id in enumerate(slice_ids):
        np.testing.assert_array_equal(sim.get("Dose", i), expected_array(DOSE, written_id))
    for i in sim:
        np.testing.assert_array_equal(sim.data["Edep"], expected_array(EDEP, slice_ids[i]))


def test_duplicate_ids_fall_back(tmp_path):
    path = write_extendable4d(tmp_path / "dup.h5", subrun_ids=(0, 1, 0))
    with pytest.warns(UserWarning, match="duplicate subrun_id"):
        sim = G4VoxFile4D(path)
    assert sim.subrun_ids == [0, 1, 2]
    _assert_slices_by_index(sim, (0, 1, 0))


def test_row_count_mismatch_falls_back(tmp_path):
    path = write_extendable4d(tmp_path / "rows.h5", subrun_ids=(4, 5, 6), run_log_ids=(0, 1))
    with pytest.warns(UserWarning, match="run_log has 2 rows but datasets have 3 slices"):
        sim = G4VoxFile4D(path)
    _assert_slices_by_index(sim, (4, 5, 6))


def test_missing_run_log_falls_back(tmp_path):
    path = write_extendable4d(tmp_path / "norl.h5", subrun_ids=(4, 5, 6), with_run_log=False)
    with pytest.warns(UserWarning, match="run_log missing"):
        sim = G4VoxFile4D(path)
    assert sim.run_log is None
    _assert_slices_by_index(sim, (4, 5, 6))


def test_no_warning_when_no_slices(tmp_path):
    path = write_extendable4d(tmp_path / "empty.h5", subrun_ids=(), with_run_log=False)
    sim = _open_no_warning(path)
    assert sim.subrun_ids == []
    assert sim.n_subruns == 0

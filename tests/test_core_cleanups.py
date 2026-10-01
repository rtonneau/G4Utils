"""Core cleanups: dtype-aware sums, metadata warning, removed legacy wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from g4utils.HDF5 import G4VoxFile3D
from g4utils.HDF5.shared import select_quantities

from .conftest import expected_array, write_snapshot3d

DOSE = 0


def test_sum_float32_accumulates_in_float64(tmp_path):
    path = write_snapshot3d(tmp_path / "f32.h5", dtype=np.float32)
    result = G4VoxFile3D(path).sum("Dose")
    expected = sum(expected_array(DOSE, sid, dtype=np.float64) for sid in (0, 1, 2))
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


def test_sum_int32_accumulates_in_int64(tmp_path):
    path = write_snapshot3d(tmp_path / "i32.h5", dtype=np.int32)
    result = G4VoxFile3D(path).sum("Dose")
    assert result.dtype == np.int64
    expected = sum(expected_array(DOSE, sid, dtype=np.int64) for sid in (0, 1, 2))
    np.testing.assert_array_equal(result, expected)


def test_missing_metadata_warns_and_infers_dims(tmp_path):
    path = write_snapshot3d(tmp_path / "nometa.h5", with_metadata=False)
    with pytest.warns(UserWarning, match="metadata absent"):
        sim = G4VoxFile3D(path)
    assert tuple(int(d) for d in sim.geometry.dims_xyz) == (4, 3, 2)


def test_read_g4vox_hdf5_3d_is_removed():
    with pytest.raises(ImportError):
        from g4utils.HDF5 import read_g4vox_hdf5_3d  # noqa: F401


def test_select_quantities_lives_in_shared():
    assert select_quantities(["a", "b"], ["b"]) == ["b"]
    with pytest.raises(KeyError):
        select_quantities(["a"], ["z"])

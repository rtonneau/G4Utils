"""Baseline tests for G4VoxFile3D / G4VoxFile4D on synthetic G4Vox files."""

from __future__ import annotations

import numpy as np
import pytest

from g4utils.HDF5 import G4VoxFile3D, G4VoxFile4D

from .conftest import (
    DIMS_XYZ,
    ORIGIN_MM,
    SPACING_MM,
    expected_array,
)

SUBRUNS = [0, 1, 2]


@pytest.fixture(params=["3d", "4d"])
def vox_file(request, snapshot3d_file, extendable4d_file):
    if request.param == "3d":
        return G4VoxFile3D(snapshot3d_file)
    return G4VoxFile4D(extendable4d_file)


def test_geometry(vox_file):
    geo = vox_file.geometry
    assert geo is not None
    assert (geo.nx, geo.ny, geo.nz) == DIMS_XYZ
    np.testing.assert_allclose(geo.spacing_mm, SPACING_MM)
    np.testing.assert_allclose(geo.origin_mm, ORIGIN_MM)
    assert geo.shape == (DIMS_XYZ[2], DIMS_XYZ[1], DIMS_XYZ[0])


def test_quantity_names(vox_file):
    assert vox_file.quantity_names == ["Dose", "Edep"]


def test_iteration_yields_subruns_with_data(vox_file):
    seen = []
    for sid in vox_file:
        seen.append(sid)
        np.testing.assert_array_equal(vox_file.data["Dose"], expected_array(0, sid))
        np.testing.assert_array_equal(vox_file.data["Edep"], expected_array(1, sid))
    assert seen == SUBRUNS


def test_sum(vox_file):
    expected = sum(expected_array(0, sid) for sid in SUBRUNS)
    np.testing.assert_array_equal(vox_file.sum("Dose"), expected)


def test_get(vox_file):
    np.testing.assert_array_equal(vox_file.get("Edep", 1), expected_array(1, 1))

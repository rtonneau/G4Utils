"""Layout detection and ``open_vox_file``."""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from g4utils.HDF5 import G4VoxFile3D, G4VoxFile4D, detect_layout, open_vox_file

from .conftest import write_extendable4d, write_snapshot3d


@pytest.mark.parametrize(
    ("writer", "layout", "cls"),
    [
        (write_snapshot3d, "Snapshot3D", G4VoxFile3D),
        (write_extendable4d, "Extendable4D", G4VoxFile4D),
    ],
)
@pytest.mark.parametrize("with_metadata", [True, False])
def test_detect_and_open(tmp_path, writer, layout, cls, with_metadata):
    path = writer(tmp_path / "f.h5", with_metadata=with_metadata)
    assert detect_layout(path) == layout
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sim = open_vox_file(path)
    assert isinstance(sim, cls)
    assert isinstance(sim, G4VoxFile4D) == (layout == "Extendable4D")


def test_detect_accepts_str_path(tmp_path):
    path = write_snapshot3d(tmp_path / "f.h5")
    assert detect_layout(str(path)) == "Snapshot3D"


def test_unknown_mode_raises(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["mode"] = "Weird"
    with pytest.raises(ValueError, match="Weird"):
        detect_layout(path)


def test_bytes_mode(tmp_path):
    path = tmp_path / "bytes.h5"
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["mode"] = np.bytes_(b"Extendable4D")
    assert detect_layout(path) == "Extendable4D"


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="Cannot determine layout of"):
        detect_layout(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_layout(tmp_path / "nope.h5")
    with pytest.raises(FileNotFoundError):
        open_vox_file(tmp_path / "nope.h5")

"""Shared test helpers: synthetic G4Vox HDF5 files.

The file layout mirrors what the C++ writer produces; the format source is
``G4Vox/src/HDF5Writer.cc``:

- ``/metadata`` group with attrs ``dims_xyz``, ``spacing_mm``, ``origin_mm``
  (float64 arrays), ``mode`` and ``quantities`` (variable-length strings).
- Snapshot3D: ``/subrun_NNNN/<qty>`` datasets of shape ``(nZ, nY, nX)``.
- Extendable4D: ``/<qty>`` datasets of shape ``(N, nZ, nY, nX)``.
- ``/run_log``: float64 ``(N, 4)``, rows ``[unix_timestamp, primaries, runtime_s, subrun_id]``,
  with a ``columns`` string attribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pytest

DIMS_XYZ = (4, 3, 2)
SPACING_MM = (0.5, 1.0, 2.0)
ORIGIN_MM = (-1.0, -1.5, -2.0)
QUANTITIES = ("Dose", "Edep")

_RUN_LOG_COLUMNS = "unix_timestamp, primaries, runtime_s, subrun_id"


def expected_array(
    qty_index: int,
    subrun_id: int,
    dims_xyz: Sequence[int] = DIMS_XYZ,
    dtype=np.float64,
) -> np.ndarray:
    """Return the deterministic ``(nz, ny, nx)`` array for a quantity and subrun."""
    nx, ny, nz = dims_xyz
    base = np.arange(nz * ny * nx).reshape(nz, ny, nx)
    return (base + 1000 * qty_index + 100 * subrun_id).astype(dtype)


def _write_str_attr(obj, name: str, value: str) -> None:
    obj.attrs.create(name, value, dtype=h5py.string_dtype())


def _write_metadata(f, mode, dims_xyz, quantities) -> None:
    meta = f.create_group("metadata")
    meta.attrs["dims_xyz"] = np.asarray(dims_xyz, dtype=np.float64)
    meta.attrs["spacing_mm"] = np.asarray(SPACING_MM, dtype=np.float64)
    meta.attrs["origin_mm"] = np.asarray(ORIGIN_MM, dtype=np.float64)
    _write_str_attr(meta, "mode", mode)
    _write_str_attr(meta, "quantities", ",".join(quantities))


def _write_run_log(f, ids: Sequence[int], primaries: int, columns_attr: bool) -> None:
    rows = np.array(
        [[1.7e9 + i, primaries, 1.5, sid] for i, sid in enumerate(ids)],
        dtype=np.float64,
    ).reshape(len(ids), 4)
    ds = f.create_dataset("run_log", data=rows, maxshape=(None, 4), chunks=(1, 4))
    if columns_attr:
        _write_str_attr(ds, "columns", _RUN_LOG_COLUMNS)


def write_snapshot3d(
    path,
    *,
    subrun_ids: Sequence[int] = (0, 1, 2),
    dims_xyz: Sequence[int] = DIMS_XYZ,
    quantities: Sequence[str] = QUANTITIES,
    dtype=np.float64,
    primaries: int = 100,
    with_metadata: bool = True,
    with_run_log: bool = True,
    run_log_columns_attr: bool = True,
) -> Path:
    """Write a Snapshot3D file matching ``HDF5Writer.cc``."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        if with_metadata:
            _write_metadata(f, "Snapshot3D", dims_xyz, quantities)
        for sid in subrun_ids:
            group = f.create_group(f"subrun_{sid:04d}")
            for qi, qty in enumerate(quantities):
                group.create_dataset(qty, data=expected_array(qi, sid, dims_xyz, dtype))
        if with_run_log:
            _write_run_log(f, list(subrun_ids), primaries, run_log_columns_attr)
    return path


def write_extendable4d(
    path,
    *,
    subrun_ids: Sequence[int] = (0, 1, 2),
    n_slices: int | None = None,
    run_log_ids: Sequence[int] | None = None,
    dims_xyz: Sequence[int] = DIMS_XYZ,
    quantities: Sequence[str] = QUANTITIES,
    dtype=np.float64,
    primaries: int = 100,
    with_metadata: bool = True,
    with_run_log: bool = True,
    run_log_columns_attr: bool = True,
) -> Path:
    """Write an Extendable4D file matching ``HDF5Writer.cc``.

    Slice ``i`` holds ``expected_array(qi, subrun_ids[i])``. ``n_slices`` limits the
    number of slices written; ``run_log_ids`` overrides the run_log subrun_id column
    and may differ in length from the slices.
    """
    path = Path(path)
    nx, ny, nz = dims_xyz
    slice_ids = list(subrun_ids)[: n_slices if n_slices is not None else len(subrun_ids)]
    with h5py.File(path, "w") as f:
        if with_metadata:
            _write_metadata(f, "Extendable4D", dims_xyz, quantities)
        for qi, qty in enumerate(quantities):
            data = np.zeros((len(slice_ids), nz, ny, nx), dtype=dtype)
            for i, sid in enumerate(slice_ids):
                data[i] = expected_array(qi, sid, dims_xyz, dtype)
            f.create_dataset(
                qty,
                data=data,
                maxshape=(None, nz, ny, nx),
                chunks=(1, nz, ny, nx),
            )
        if with_run_log:
            ids = list(run_log_ids) if run_log_ids is not None else slice_ids
            _write_run_log(f, ids, primaries, run_log_columns_attr)
    return path


@pytest.fixture
def snapshot3d_file(tmp_path) -> Path:
    return write_snapshot3d(tmp_path / "snapshot3d.h5")


@pytest.fixture
def extendable4d_file(tmp_path) -> Path:
    return write_extendable4d(tmp_path / "extendable4d.h5")

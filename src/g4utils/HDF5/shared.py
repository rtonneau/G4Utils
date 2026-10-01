from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from g4utils.Vox.vox_geometry import VoxGeometry

if TYPE_CHECKING:
    from g4utils.HDF5.vox_file_3d import G4VoxFile3D

# ═════════════════════════════════════════════════════════════════════════════
#  Shared private helpers
# ═════════════════════════════════════════════════════════════════════════════

_SKIP_KEYS = {"metadata", "run_log"}

RUN_LOG_COLUMNS = ("unix_timestamp", "primaries", "runtime_s", "subrun_id")


def _read_geometry(f: h5py.File) -> VoxGeometry:
    if "metadata" in f:
        m = f["metadata"]
        xyz = np.asarray(m.attrs["dims_xyz"], dtype=float)
        sp = np.asarray(
            m.attrs.get("spacing_mm", [1.0, 1.0, 1.0]), dtype=float
        )
        ori = np.asarray(
            m.attrs.get("origin_mm", [0.0, 0.0, 0.0]), dtype=float
        )
        return VoxGeometry(dims_xyz=xyz, spacing_mm=sp, origin_mm=ori)

    # ── fallback: infer from data ────────────────────────────────────────────
    first = next(k for k in f if k not in _SKIP_KEYS)
    first_obj = f[first]
    if isinstance(first_obj, h5py.Group):
        first_key = next(iter(first_obj.keys()))
        first_dataset = first_obj[first_key]
        if not isinstance(first_dataset, h5py.Dataset):
            raise TypeError(f"Expected dataset for '{first_key}'")
        ndim = len(first_dataset.shape)
        shape = first_dataset.shape
    else:
        ndim = len(first_obj.shape)  # type: ignore
        shape = first_obj.shape  # type: ignore

    if ndim == 3:  # 3D layout: (nZ, nY, nX)
        nz, ny, nx = shape  # type: ignore[misc]
    elif ndim == 4:  # 4D layout: (N, nZ, nY, nX)
        _, nz, ny, nx = shape  # type: ignore[misc]
    else:
        raise ValueError(f"Unexpected dataset rank {ndim}")

    print("⚠  /metadata absent – spacing=1 mm, origin=0 mm")
    return VoxGeometry(dims_xyz=np.array([nx, ny, nz], dtype=float))


def _decode_columns_attr(value: object) -> list[str]:
    """Split a ``columns`` attribute (str, bytes or array of those) on commas."""
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else ",".join(
            v.decode() if isinstance(v, bytes) else str(v) for v in value.ravel()
        )
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return [c.strip() for c in str(value).split(",")]


def _read_run_log(f: h5py.File) -> pd.DataFrame | None:
    """
    Read ``/run_log`` into a DataFrame with named columns.

    Names come from the dataset ``columns`` attribute when present, else from
    ``RUN_LOG_COLUMNS``. If the number of names differs from the number of
    columns, a ``UserWarning`` is emitted and ``RUN_LOG_COLUMNS`` is used by
    position, extra columns being named ``col_<i>``. ``subrun_id`` and
    ``primaries`` are cast to ``int64``.
    """
    if "run_log" not in f:
        return None
    ds = f["run_log"]
    raw = np.asarray(ds[()])  # type: ignore
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    ncols = raw.shape[1]

    attr = ds.attrs.get("columns")  # type: ignore
    names = _decode_columns_attr(attr) if attr is not None else list(RUN_LOG_COLUMNS)
    if len(names) != ncols:
        warnings.warn(
            f"/run_log has {ncols} columns but {len(names)} names were found; "
            "falling back to default column names by position.",
            UserWarning,
            stacklevel=2,
        )
        names = [
            RUN_LOG_COLUMNS[i] if i < len(RUN_LOG_COLUMNS) else f"col_{i}"
            for i in range(ncols)
        ]

    df = pd.DataFrame(raw, columns=names)
    for col in ("subrun_id", "primaries"):
        if col in df.columns:
            df[col] = df[col].astype(np.int64)
    return df


def _qty_whitelist(
    available: list[str], requested: list[str] | None
) -> list[str]:
    if requested is None:
        return available
    missing = set(requested) - set(available)
    if missing:
        raise KeyError(f"Quantities not found in file: {missing}")
    return [q for q in available if q in requested]


def read_g4vox_hdf5_3d(
    filepath: str | Path,
    quantities: list[str] | None = None,
    subrun_ids: list[int] | None = None,
) -> G4VoxFile3D:
    """
    Compatibility wrapper for a G4Vox HDF5 file written in Snapshot3D mode.

    Returns a lazy G4VoxFile3D instance and applies any requested quantity or
    subrun selections without materializing voxel arrays immediately.
    """
    from g4utils.HDF5.vox_file_3d import G4VoxFile3D

    sim = G4VoxFile3D(filepath)
    if quantities is not None:
        sim.select_quantity(_qty_whitelist(sim.quantity_names, quantities))
    if subrun_ids is not None:
        sim.select_subrun(subrun_ids=subrun_ids)
    return sim

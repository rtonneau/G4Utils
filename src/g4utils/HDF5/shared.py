from __future__ import annotations

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


def _read_run_log(f: h5py.File) -> pd.DataFrame | None:
    if "run_log" not in f:
        return None
    raw = f["run_log"][()]  # type: ignore
    cols = ["subrun_id", "nPrimaries", "seed1", "seed2"]
    if raw.ndim == 1:  # type: ignore
        raw = raw.reshape(1, -1)  # type: ignore
    return pd.DataFrame(raw[:, : len(cols)], columns=cols[: raw.shape[1]])  # type: ignore


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

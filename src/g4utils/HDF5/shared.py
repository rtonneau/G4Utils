import typing as tp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from g4utils.HDF5.vox_file_3d import G4VoxFile3D, SubRun
from g4utils.HDF5.vox_file_4d import G4VoxFile4D
from g4utils.Vox.vox_geometry import VoxGeometry

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
    ndim = len(f[first].shape)  # type: ignore

    if ndim == 3:  # 3D layout: (nZ, nY, nX)
        nz, ny, nx = f[first].shape  # type: ignore
    elif ndim == 4:  # 4D layout: (N, nZ, nY, nX)
        _, nz, ny, nx = f[first].shape  # type: ignore
    else:
        raise ValueError(f"Unexpected dataset rank {ndim}")

    print("⚠  /metadata absent – spacing=1 mm, origin=0 mm")
    return VoxGeometry(dims_xyz=np.array([nx, ny, nz], dtype=float))


def _read_run_log(f: h5py.File) -> tp.Optional[pd.DataFrame]:
    if "run_log" not in f:
        return None
    raw = f["run_log"][()]  # type: ignore
    cols = ["subrun_id", "nPrimaries", "seed1", "seed2"]
    if raw.ndim == 1:  # type: ignore
        raw = raw.reshape(1, -1)  # type: ignore
    return pd.DataFrame(raw[:, : len(cols)], columns=cols[: raw.shape[1]])  # type: ignore


def _qty_whitelist(
    available: list[str], requested: tp.Optional[list[str]]
) -> list[str]:
    if requested is None:
        return available
    missing = set(requested) - set(available)
    if missing:
        raise KeyError(f"Quantities not found in file: {missing}")
    return [q for q in available if q in requested]


# ═════════════════════════════════════════════════════════════════════════════
#  Public readers
# ═════════════════════════════════════════════════════════════════════════════


def read_g4vox_hdf5_3d(
    filepath: str | Path,
    quantities: tp.Optional[list[str]] = None,
    subrun_ids: tp.Optional[list[int]] = None,
) -> G4VoxFile3D:
    """
    Read a G4Vox HDF5 file written in **Snapshot3D** mode.

    Layout expected
    ───────────────
    /metadata               group  attrs: dims_xyz, spacing_mm, origin_mm
    /subrun_XXXX/<qty>      dataset  shape (nZ, nY, nX)
    /run_log                dataset  shape (N, 4)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with h5py.File(filepath, "r") as f:
        geometry = _read_geometry(f)
        root_attrs = dict(f.attrs)
        run_log = _read_run_log(f)

        # ── discover subrun groups ────────────────────────────────────────────
        all_keys = sorted(
            k
            for k in f.keys()
            if k.startswith("subrun_") and isinstance(f[k], h5py.Group)
        )
        if subrun_ids is not None:
            all_keys = [
                k for k in all_keys if int(k.split("_")[1]) in subrun_ids
            ]

        subruns: dict[int, SubRun] = {}
        for key in all_keys:
            sid = int(key.split("_")[1])
            group = f[key]
            avail = list(group.keys())  # type: ignore
            load = _qty_whitelist(avail, quantities)
            qtys = {q: group[q][()] for q in load}  # type: ignore
            subruns[sid] = SubRun(subrun_id=sid, quantities=qtys)  # type: ignore

    return G4VoxFile3D(
        path=filepath,
        geometry=geometry,
        run_log=run_log,
        subruns=subruns,
        root_attrs=root_attrs,
    )


def read_g4vox_hdf5_4d(
    filepath: str | Path,
    quantities: tp.Optional[list[str]] = None,
    subrun_ids: tp.Optional[list[int]] = None,
) -> G4VoxFile4D:
    """
    Read a G4Vox HDF5 file written in **Extendable4D** mode.

    Layout expected
    ───────────────
    /metadata          group  attrs: dims_xyz, spacing_mm, origin_mm
    /<qty>             dataset  shape (N_subruns, nZ, nY, nX)
    /run_log           dataset  shape (N, 4)

    Parameters
    ----------
    subrun_ids : if given, only those indices along axis-0 are loaded
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with h5py.File(filepath, "r") as f:
        geometry = _read_geometry(f)
        root_attrs = dict(f.attrs)
        run_log = _read_run_log(f)

        # ── discover quantity datasets (rank-4 only) ──────────────────────────
        avail = [
            k
            for k in f.keys()
            if k not in _SKIP_KEYS
            and isinstance(f[k], h5py.Dataset)
            and f[k].ndim == 4  # type: ignore
        ]
        load = _qty_whitelist(avail, quantities)

        # ── load, optionally slicing axis-0 ──────────────────────────────────
        data: dict[str, np.ndarray] = {}
        for qty in load:
            ds = f[qty]
            if subrun_ids is None:
                data[qty] = ds[()]  # type: ignore
            else:
                data[qty] = ds[subrun_ids, ...]  # type: ignore

    return G4VoxFile4D(
        path=filepath,
        geometry=geometry,
        run_log=run_log,
        data=data,
        root_attrs=root_attrs,
    )

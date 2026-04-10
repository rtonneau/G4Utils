import h5py
import numpy as np
import pandas as pd

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

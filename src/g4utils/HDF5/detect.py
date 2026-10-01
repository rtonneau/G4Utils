"""Layout detection for G4Vox HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py

from g4utils.HDF5.vox_file_3d import G4VoxFile3D
from g4utils.HDF5.vox_file_4d import G4VoxFile4D
from g4utils.HDF5.vox_file_base import G4VoxFileBase

_LAYOUTS = ("Snapshot3D", "Extendable4D")
_NON_QUANTITY_KEYS = ("metadata", "run_log")


def detect_layout(path: str | Path) -> Literal["Snapshot3D", "Extendable4D"]:
    """Return the layout of a G4Vox HDF5 file.

    The ``mode`` attribute of ``/metadata`` is used when present. Otherwise the
    root content decides: a ``subrun_*`` group means ``"Snapshot3D"``, a 4D root
    dataset other than ``metadata``/``run_log`` means ``"Extendable4D"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``mode`` is unknown or the layout cannot be determined.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path}'")

    with h5py.File(path, "r") as f:
        meta = f.get("metadata")
        if meta is not None and "mode" in meta.attrs:
            mode = meta.attrs["mode"]
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8")
            mode = str(mode)
            if mode not in _LAYOUTS:
                raise ValueError(f"Unknown layout mode '{mode}' in '{path}'")
            return mode  # type: ignore[return-value]

        for key, item in f.items():
            if isinstance(item, h5py.Group) and key.startswith("subrun_"):
                return "Snapshot3D"
        for key, item in f.items():
            if (
                isinstance(item, h5py.Dataset)
                and key not in _NON_QUANTITY_KEYS
                and item.ndim == 4
            ):
                return "Extendable4D"

    raise ValueError(f"Cannot determine layout of '{path}'")


def open_vox_file(path: str | Path) -> G4VoxFileBase:
    """Open a G4Vox HDF5 file with the class matching its layout."""
    if detect_layout(path) == "Snapshot3D":
        return G4VoxFile3D(path)
    return G4VoxFile4D(path)

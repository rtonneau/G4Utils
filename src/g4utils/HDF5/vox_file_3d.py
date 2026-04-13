from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g4utils.HDF5.vox_file_base import (
    G4VoxFileBase,
    Snapshot3DBackend,
)

# ═════════════════════════════════════════════════════════════════════════════
#  3D layout containers
#
#  /metadata
#  /subrun_0000/Dose   (nZ, nY, nX)
#  /subrun_0001/Dose   (nZ, nY, nX)
#  /run_log
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class SubRun:
    subrun_id: int
    quantities: dict[str, np.ndarray]  # name → (nZ,nY,nX) array

    def __repr__(self) -> str:
        return f"SubRun(id={self.subrun_id}  qty={list(self.quantities)})"


class G4VoxFile3D(G4VoxFileBase):
    """
    Lightweight 3D HDF5 voxel container with lazy per-subrun loading.

    Examples
    --------
    sim = G4VoxFile3D("path/to/file.h5")
    sim.select_quantity(["Edep", "Dose"])
    sim.select_subrun(start=10, stop=20)

    for subrun_id in sim:
        print(subrun_id, sim.data["Edep"].shape)
        sim.to_vti(f"subrun_{subrun_id:04d}.vti")

    sim.select_subrun([0, 5, 9])
    sim.dump_selection_to_vti("selected_subruns_sum.vti")
    sim.dump_selection_to_vti_timeseries("selected_subruns.pvd")
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(
            path,
            backend=Snapshot3DBackend(path),
            label="G4VoxFile3D",
        )

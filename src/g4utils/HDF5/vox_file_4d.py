from pathlib import Path

from g4utils.HDF5.vox_file_base import (
    Dataset4DBackend,
    G4VoxFileBase,
)

# ═════════════════════════════════════════════════════════════════════════════
#  4D layout containers
#
#  /metadata
#  /Dose   (N_slices, nZ, nY, nX)
#  /Edep   (N_slices, nZ, nY, nX)
#  /run_log   row i describes slice i; subrun IDs come from run_log.subrun_id
# ═════════════════════════════════════════════════════════════════════════════


class G4VoxFile4D(G4VoxFileBase):
    """
    Lightweight 4D HDF5 voxel container with lazy per-subrun loading.

    Examples
    --------
    sim = G4VoxFile4D("path/to/file.h5")
    sim.select_quantity(["Edep", "Dose"])
    sim.select_subrun(start=10, stop=20)

    for subrun_id in sim:
        print(subrun_id, sim.data["Edep"].shape)
        sim.to_vti(f"subrun_{subrun_id:04d}.vti")

    sim.select_subrun([0, 5, 9])  # subrun IDs, not slice indices
    sim.dump_selection_to_vti("selected_subruns_sum.vti")
    sim.dump_selection_to_vti_timeseries("selected_subruns.pvd")
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(
            path,
            backend=Dataset4DBackend(path),
            label="G4VoxFile4D",
        )

from pathlib import Path

import numpy as np

from g4utils.HDF5.vox_file_base import (
    Dataset4DBackend,
    G4VoxFileBase,
)

# ═════════════════════════════════════════════════════════════════════════════
#  4D layout containers
#
#  /metadata
#  /Dose   (N_subruns, nZ, nY, nX)
#  /Edep   (N_subruns, nZ, nY, nX)
#  /run_log
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

    sim.select_subrun([0, 5, 9])
    sim.dump_selection_to_vti("selected_subruns_sum.vti")
    sim.dump_selection_to_vti_timeseries("selected_subruns.pvd")
    """

    def __init__(self, path: str | Path) -> None:
        self._iter_source_data: dict[str, np.ndarray] | None = None
        super().__init__(
            path,
            backend=Dataset4DBackend(path),
            label="G4VoxFile4D",
        )

    @property
    def n_subruns(self) -> int:
        if self.n_subruns_hint is not None:
            return int(self.n_subruns_hint)
        if self.data:
            first = next(iter(self.data.values()))
            if first.ndim == 4:
                return int(first.shape[0])
        return int(self.n_subruns_hint or 0)

    @property
    def quantity_names(self) -> list[str]:
        return list(self.data) if self.data else list(self.dataset_names)

    def _has_materialized_4d(self) -> bool:
        return bool(self.data) and all(v.ndim == 4 for v in self.data.values())

    def _materialized_axis_index(self, subrun_id: int) -> int:
        try:
            return self.available_subrun_ids.index(subrun_id)
        except ValueError as exc:
            raise KeyError(f"Subrun '{subrun_id}' not found") from exc

    def __iter__(self):
        if self._has_materialized_4d():
            self._iter_source_data = dict(self.data)
            self._iter_subrun_ids = self._resolved_subrun_ids()
            self._iter_pos = 0
            self.current_subrun_id = None
            self.data = {}
            return self

        self._iter_source_data = None
        self._iter_subrun_ids = self._resolved_subrun_ids()
        self._iter_pos = 0
        self.current_subrun_id = None
        self.data = {}
        return self

    def __next__(self) -> int:
        if self._iter_source_data is None:
            return super().__next__()

        source_data = self._iter_source_data

        if self._iter_pos >= len(self._iter_subrun_ids):
            self.data = {}
            self.current_subrun_id = None
            self._iter_source_data = None
            raise StopIteration

        sid = self._iter_subrun_ids[self._iter_pos]
        self._iter_pos += 1
        self.current_subrun_id = sid

        axis_index = self._materialized_axis_index(sid)
        self.data = {q: v[axis_index, ...] for q, v in source_data.items()}
        return sid

    def get(self, qty: str, subrun_id: int) -> np.ndarray:
        """Single subrun slice → (nZ, nY, nX)."""
        if qty in self.data and self.data[qty].ndim == 4:
            axis_index = self._materialized_axis_index(subrun_id)
            return self.data[qty][axis_index]
        return super().get(qty, subrun_id)

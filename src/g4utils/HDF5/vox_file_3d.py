import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from g4utils.HDF5.vox_file_4d import G4VoxFile4D
from g4utils.Vox.vox_geometry import VoxGeometry

# ══════════════════════════════════════════════════════════════════════════════
#  3D layout containers
#
#  /metadata
#  /subrun_0000/Dose   (nZ, nY, nX)
#  /subrun_0001/Dose   (nZ, nY, nX)
#  /run_log
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SubRun:
    subrun_id: int
    quantities: tp.Dict[str, np.ndarray]  # name → (nZ,nY,nX) array

    def __repr__(self) -> str:
        return f"SubRun(id={self.subrun_id}  qty={list(self.quantities)})"


@dataclass
class G4VoxFile3D:
    path: Path
    geometry: VoxGeometry
    run_log: tp.Optional[pd.DataFrame]
    subruns: tp.Dict[int, SubRun]
    root_attrs: tp.Dict = field(default_factory=dict)

    @property
    def subrun_ids(self) -> tp.List[int]:
        return sorted(self.subruns)

    @property
    def n_subruns(self) -> int:
        return len(self.subruns)

    @property
    def quantity_names(self) -> tp.List[str]:
        first = next(iter(self.subruns.values()))
        return list(first.quantities)

    def total_primaries(self) -> int:
        if self.run_log is None or "nPrimaries" not in self.run_log.columns:
            return 0
        return int(self.run_log["nPrimaries"].sum())

    def get(self, qty: str, subrun_id: int) -> np.ndarray:
        return self.subruns[subrun_id].quantities[qty]

    def sum(self, qty: str) -> np.ndarray:
        """Sum quantity over all loaded subruns → (nZ,nY,nX)."""
        arrays = [sr.quantities[qty] for sr in self.subruns.values()]
        return np.add.reduce(arrays) if arrays else np.array([])

    def to_4d(
        self,
        quantities: tp.Optional[tp.List[str]] = None,
        subrun_ids: tp.Optional[tp.List[int]] = None,
    ) -> "G4VoxFile4D":
        """
        Convert this 3D file into a G4VoxFile4D container.

        subrun axis order follows subrun_ids (or sorted IDs by default).

        Diagram
        ───────
        G4VoxFile3D                        G4VoxFile4D
        ├─ subruns[0]["Dose"] (nZ,nY,nX)  ├─ data["Dose"]  (N,nZ,nY,nX)
        ├─ subruns[1]["Dose"] (nZ,nY,nX)  ├─ data["LET"]   (N,nZ,nY,nX)
        └─ subruns[2]["Dose"] (nZ,nY,nX)  └─ geometry, run_log  (shared)
        """
        ids = subrun_ids if subrun_ids is not None else self.subrun_ids
        qtys = quantities if quantities is not None else self.quantity_names

        data = {
            qty: np.stack(
                [self.subruns[i].quantities[qty] for i in ids], axis=0
            )
            for qty in qtys
        }

        return G4VoxFile4D(
            path=self.path,
            geometry=self.geometry,
            run_log=self.run_log,
            data=data,
            root_attrs=self.root_attrs,
        )

    def __repr__(self) -> str:
        return (
            f"G4VoxFile3D '{self.path.name}'\n"
            f"  {self.geometry}\n"
            f"  subruns   : {self.n_subruns}  ids={self.subrun_ids}\n"
            f"  quantities: {self.quantity_names}\n"
            f"  primaries : {self.total_primaries():,}"
        )

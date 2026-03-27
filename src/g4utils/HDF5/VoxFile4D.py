import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from g4utils.Vox.voxgeometry import VoxGeometry

# ══════════════════════════════════════════════════════════════════════════════
#  4D layout containers
#
#  /metadata
#  /Dose   (N_subruns, nZ, nY, nX)
#  /Edep   (N_subruns, nZ, nY, nX)
#  /run_log
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class G4VoxFile4D:
    path: Path
    geometry: VoxGeometry
    run_log: tp.Optional[pd.DataFrame]
    data: tp.Dict[str, np.ndarray]  # name → (N, nZ, nY, nX) array
    root_attrs: tp.Dict = field(default_factory=dict)

    @property
    def n_subruns(self) -> int:
        return next(iter(self.data.values())).shape[0]

    @property
    def quantity_names(self) -> tp.List[str]:
        return list(self.data)

    def total_primaries(self) -> int:
        if self.run_log is None or "nPrimaries" not in self.run_log.columns:
            return 0
        return int(self.run_log["nPrimaries"].sum())

    def get(self, qty: str, subrun_id: int) -> np.ndarray:
        """Single subrun slice → (nZ, nY, nX)."""
        return self.data[qty][subrun_id]

    def get_4d(self, qty: str) -> np.ndarray:
        """Return full (N_subruns, nZ, nY, nX) array."""
        return self.data[qty]

    def sum(self, qty: str) -> np.ndarray:
        """Sum over all subruns → (nZ, nY, nX)."""
        return self.data[qty].sum(axis=0)

    def __repr__(self) -> str:
        return (
            f"G4VoxFile4D '{self.path.name}'\n"
            f"  {self.geometry}\n"
            f"  subruns   : {self.n_subruns}\n"
            f"  quantities: {self.quantity_names}\n"
            f"  primaries : {self.total_primaries():,}"
        )

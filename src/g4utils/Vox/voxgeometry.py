from dataclasses import dataclass, field

import numpy as np


@dataclass
class VoxGeometry:
    """Mirrors the /metadata attributes written by WriteGeometryMetadata()."""

    dims_xyz: np.ndarray  # [nx, ny, nz]
    spacing_mm: np.ndarray = field(default_factory=lambda: np.ones(3))
    origin_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def nx(self) -> int:
        return int(self.dims_xyz[0])

    @property
    def ny(self) -> int:
        return int(self.dims_xyz[1])

    @property
    def nz(self) -> int:
        return int(self.dims_xyz[2])

    @property
    def shape(self) -> tuple:  # (nZ, nY, nX)
        return (self.nz, self.ny, self.nx)

    @property
    def extent_mm(self) -> np.ndarray:
        """Full physical size of the voxel grid in mm, shape (3,)."""
        return self.dims_xyz * self.spacing_mm

    def voxel_centers(self, axis: str) -> np.ndarray:
        """
        1-D array of voxel-centre coordinates along 'x', 'y', or 'z'.

        The C++ writer stores data as [nZ][nY][nX], i.e. numpy axis 0 = Z.
        """
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        n = int(self.dims_xyz[ax])
        return self.origin_mm[ax] + (np.arange(n) + 0.5) * self.spacing_mm[ax]

    def __repr__(self) -> str:
        return (
            f"VoxGeometry  (nX={self.nx}, nY={self.ny}, nZ={self.nz})  "
            f"spacing={np.round(self.spacing_mm,3)} mm  "
            f"origin={np.round(self.origin_mm,3)} mm"
        )

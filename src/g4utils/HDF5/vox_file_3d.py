from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from g4utils.HDF5.vox_file_4d import G4VoxFile4D
from g4utils.HDF5.vti_export import (
    select_quantities,
    write_pvd_collection,
    write_vti,
)
from g4utils.Vox.vox_geometry import VoxGeometry

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


class G4VoxFile3D:
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
    sim4d = sim.to_4d()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.geometry: VoxGeometry | None = None
        self.run_log: pd.DataFrame | None = None
        self.data: dict[str, np.ndarray] = {}
        self.root_attrs: dict = {}
        self.dataset_names: list[str] = []
        self.available_subrun_ids: list[int] = []
        self.selected_quantities: list[str] = []
        self.selected_subrun_ids: list[int] = []
        self.current_subrun_id: int | None = None
        self._iter_subrun_ids: list[int] = []
        self._iter_pos: int = 0

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self._load_metadata_only()

    @property
    def subrun_ids(self) -> list[int]:
        return list(self.available_subrun_ids)

    @property
    def n_subruns(self) -> int:
        return len(self.available_subrun_ids)

    @property
    def quantity_names(self) -> list[str]:
        return list(self.dataset_names)

    @property
    def is_lazy(self) -> bool:
        return not bool(self.data) and self.path.exists()

    def _load_metadata_only(self) -> None:
        with h5py.File(self.path, "r") as f:
            if "metadata" in f:
                m = f["metadata"]
                xyz = np.asarray(m.attrs["dims_xyz"], dtype=float)
                sp = np.asarray(
                    m.attrs.get("spacing_mm", [1.0, 1.0, 1.0]), dtype=float
                )
                ori = np.asarray(
                    m.attrs.get("origin_mm", [0.0, 0.0, 0.0]), dtype=float
                )
                self.geometry = VoxGeometry(
                    dims_xyz=xyz, spacing_mm=sp, origin_mm=ori
                )

            self.root_attrs = dict(f.attrs)

            if "run_log" in f:
                run_log_obj = f["run_log"]
                if isinstance(run_log_obj, h5py.Dataset):
                    raw = run_log_obj[()]
                    cols = ["subrun_id", "nPrimaries", "seed1", "seed2"]
                    if raw.ndim == 1:  # type: ignore
                        raw = raw.reshape(1, -1)  # type: ignore
                    self.run_log = pd.DataFrame(
                        raw[:, : len(cols)],  # type: ignore
                        columns=cols[: raw.shape[1]],  # type: ignore
                    )

            subrun_names = sorted(
                key
                for key in f
                if isinstance(key, str)
                and key.startswith("subrun_")
                and isinstance(f[key], h5py.Group)
            )
            self.available_subrun_ids = [
                int(name.split("_")[1]) for name in subrun_names
            ]

            if subrun_names:
                first_group = f[subrun_names[0]]
                if not isinstance(first_group, h5py.Group):
                    raise TypeError(f"Expected group for '{subrun_names[0]}'")
                dataset_names: list[str] = []
                for key in first_group:
                    if not isinstance(key, str):
                        continue
                    obj = first_group[key]
                    if isinstance(obj, h5py.Dataset):
                        dataset_names.append(key)
                self.dataset_names = dataset_names
            self.selected_quantities = list(self.dataset_names)

        if (
            self.geometry is None
            and self.available_subrun_ids
            and self.dataset_names
        ):
            first_subrun = self.available_subrun_ids[0]
            first_qty = self.dataset_names[0]
            with h5py.File(self.path, "r") as f:
                group = f[f"subrun_{first_subrun:04d}"]
                if not isinstance(group, h5py.Group):
                    raise TypeError(
                        f"Expected group for 'subrun_{first_subrun:04d}'"
                    )
                ds_obj = group[first_qty]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise TypeError(
                        "Expected dataset for "
                        f"'{first_qty}' in subrun_{first_subrun:04d}"
                    )
                nz, ny, nx = ds_obj.shape
            self.geometry = VoxGeometry(
                dims_xyz=np.array([nx, ny, nz], dtype=float)
            )

    def select_quantity(self, quantities: str | list[str]) -> "G4VoxFile3D":
        wanted = [quantities] if isinstance(quantities, str) else quantities
        self.selected_quantities = select_quantities(
            self.dataset_names, wanted
        )
        return self

    def select_subrun(
        self,
        subrun_ids: list[int] | None = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> "G4VoxFile3D":
        if subrun_ids is not None and (start is not None or stop is not None):
            raise ValueError("Use either subrun_ids or start/stop, not both")

        if subrun_ids is not None:
            ids = [int(sid) for sid in subrun_ids]
        elif start is not None or stop is not None:
            s0 = (
                min(self.available_subrun_ids) if start is None else int(start)
            )
            s1 = (
                max(self.available_subrun_ids) + 1
                if stop is None
                else int(stop)
            )
            ids = [sid for sid in self.available_subrun_ids if s0 <= sid < s1]
        else:
            self.selected_subrun_ids = []
            return self

        valid = set(self.available_subrun_ids)
        invalid_ids = [sid for sid in ids if sid not in valid]
        if invalid_ids:
            raise IndexError(
                f"Some subrun ids are not available: {invalid_ids}"
            )

        self.selected_subrun_ids = ids
        return self

    def _resolved_subrun_ids(self) -> list[int]:
        return (
            self.selected_subrun_ids
            if self.selected_subrun_ids
            else list(self.available_subrun_ids)
        )

    def __iter__(self):
        self._iter_subrun_ids = self._resolved_subrun_ids()
        self._iter_pos = 0
        self.current_subrun_id = None
        self.data = {}
        return self

    def __next__(self) -> int:
        if self._iter_pos >= len(self._iter_subrun_ids):
            self.data = {}
            self.current_subrun_id = None
            raise StopIteration

        sid = self._iter_subrun_ids[self._iter_pos]
        self._iter_pos += 1
        self.current_subrun_id = sid

        qtys = (
            self.selected_quantities
            if self.selected_quantities
            else self.dataset_names
        )
        group_name = f"subrun_{sid:04d}"

        with h5py.File(self.path, "r") as f:
            group = f[group_name]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"Expected group for '{group_name}'")
            loaded: dict[str, np.ndarray] = {}
            for q in qtys:
                ds_obj = group[q]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise TypeError(f"Expected dataset for '{q}'")
                loaded[q] = np.asarray(ds_obj[()])
            self.data = loaded

        return sid

    def total_primaries(self) -> int:
        if self.run_log is None or "nPrimaries" not in self.run_log.columns:
            return 0
        return int(self.run_log["nPrimaries"].sum())

    def get(self, qty: str, subrun_id: int) -> np.ndarray:
        if self.current_subrun_id == subrun_id and qty in self.data:
            return self.data[qty]

        group_name = f"subrun_{subrun_id:04d}"
        with h5py.File(self.path, "r") as f:
            if group_name not in f:
                raise KeyError(f"Subrun '{subrun_id}' not found")
            group = f[group_name]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"Expected group for '{group_name}'")
            if qty not in group:
                raise KeyError(
                    f"Quantity '{qty}' not found in subrun {subrun_id}"
                )
            ds_obj = group[qty]
            if not isinstance(ds_obj, h5py.Dataset):
                raise TypeError(f"Expected dataset for '{qty}'")
            return np.asarray(ds_obj[()])

    def sum(self, qty: str, subrun_ids: list[int] | None = None) -> np.ndarray:
        """Sum quantity over selected or provided subruns → (nZ,nY,nX)."""
        ids = (
            subrun_ids
            if subrun_ids is not None
            else self._resolved_subrun_ids()
        )
        if not ids:
            return np.array([])

        acc: np.ndarray | None = None
        for sid in ids:
            arr = self.get(qty, sid)
            if acc is None:
                acc = np.zeros_like(arr)
            np.add(acc, arr, out=acc)
        return acc if acc is not None else np.array([])

    def to_4d(
        self,
        quantities: list[str] | None = None,
        subrun_ids: list[int] | None = None,
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
        ids = (
            subrun_ids
            if subrun_ids is not None
            else self._resolved_subrun_ids()
        )
        qtys = select_quantities(self.quantity_names, quantities)

        data = {
            qty: np.stack([self.get(qty, i) for i in ids], axis=0)
            for qty in qtys
        }

        out = G4VoxFile4D(path=self.path)
        out.geometry = self._require_geometry()
        out.run_log = self.run_log
        out.data = data
        out.root_attrs = self.root_attrs
        out.dataset_names = list(data)
        out.n_subruns_hint = next(iter(data.values())).shape[0] if data else 0
        out.selected_quantities = list(out.dataset_names)
        out.selected_subrun_ids = list(ids)
        return out

    def dump_selection_to_vti(
        self,
        filepath: str | Path,
        dtype: npt.DTypeLike = np.float32,
    ) -> Path:
        """
        Export all selected quantities and subruns into one VTI file.

        Selected subruns are aggregated by sum per quantity.
        """
        qtys = (
            self.selected_quantities
            if self.selected_quantities
            else self.dataset_names
        )
        ids = self._resolved_subrun_ids()

        if not qtys:
            raise ValueError("No quantities selected")
        if not ids:
            raise ValueError("No subruns selected")

        arrays = {q: self.sum(q, ids) for q in qtys}

        return write_vti(
            filepath=filepath,
            geometry=self._require_geometry(),
            cell_arrays=arrays,
            dtype=dtype,
        )

    def dump_selection_to_vti_timeseries(
        self,
        filepath: str | Path,
        dtype: npt.DTypeLike = np.float32,
    ) -> Path:
        """
        Export selected subruns as a ParaView time series.

        This writes one .vti file per selected subrun plus one .pvd collection
        file that ParaView recognizes as a time sequence.
        """
        pvd_path = Path(filepath)
        if pvd_path.suffix.lower() != ".pvd":
            pvd_path = pvd_path.with_suffix(".pvd")

        qtys = (
            self.selected_quantities
            if self.selected_quantities
            else self.dataset_names
        )
        ids = self._resolved_subrun_ids()

        if not qtys:
            raise ValueError("No quantities selected")
        if not ids:
            raise ValueError("No subruns selected")

        datasets: list[tuple[float, str]] = []
        stem = pvd_path.stem
        for sid in ids:
            arrays = {q: self.get(q, sid) for q in qtys}
            frame_name = f"{stem}_{sid:04d}.vti"
            frame_path = pvd_path.parent / frame_name
            write_vti(
                filepath=frame_path,
                geometry=self._require_geometry(),
                cell_arrays=arrays,
                dtype=dtype,
            )
            datasets.append((float(sid), frame_name))

        return write_pvd_collection(filepath=pvd_path, datasets=datasets)

    def to_vti(
        self,
        filepath: str | Path,
        dtype: npt.DTypeLike = np.float32,
    ) -> Path:
        """
        Export this container to a ParaView-compatible .vti file.

        Exports only currently loaded data (self.data), typically after one
        iteration step in "for subrun in sim".
        """
        if not self.data:
            raise ValueError(
                "No subrun data loaded. Iterate once or call next(sim) first."
            )
        return write_vti(
            filepath=filepath,
            geometry=self._require_geometry(),
            cell_arrays=self.data,
            dtype=dtype,
        )

    def _require_geometry(self) -> VoxGeometry:
        if self.geometry is None:
            raise ValueError("Geometry is unavailable; cannot export to VTI")
        return self.geometry

    def __repr__(self) -> str:
        return (
            f"G4VoxFile3D '{self.path.name}'\n"
            f"  {self.geometry}\n"
            f"  subruns   : {self.n_subruns}  ids={self.subrun_ids}\n"
            f"  quantities: {self.quantity_names}\n"
            f"  primaries : {self.total_primaries():,}"
        )

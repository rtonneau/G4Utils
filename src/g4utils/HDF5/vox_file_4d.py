from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from g4utils.HDF5.vti_export import (
    select_quantities,
    write_pvd_collection,
    write_vti,
)
from g4utils.Vox.vox_geometry import VoxGeometry

# ═════════════════════════════════════════════════════════════════════════════
#  4D layout containers
#
#  /metadata
#  /Dose   (N_subruns, nZ, nY, nX)
#  /Edep   (N_subruns, nZ, nY, nX)
#  /run_log
# ═════════════════════════════════════════════════════════════════════════════


class G4VoxFile4D:
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
        self.path = Path(path)
        self.geometry: VoxGeometry | None = None
        self.run_log: pd.DataFrame | None = None
        self.data: dict[str, np.ndarray] = {}
        self.root_attrs: dict = {}
        self.dataset_names: list[str] = []
        self.selected_quantities: list[str] = []
        self.selected_subrun_ids: list[int] = []
        self.n_subruns_hint: int | None = None
        self.current_subrun_id: int | None = None
        self._iter_subrun_ids: list[int] = []
        self._iter_pos: int = 0
        self._iter_source_data: dict[str, np.ndarray] | None = None

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # Lightweight constructor: discover metadata and dataset layout only.
        self._load_metadata_only()

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

            names: list[str] = []
            for k in f:
                if not isinstance(k, str) or k in {"metadata", "run_log"}:
                    continue
                obj = f[k]
                if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
                    names.append(k)
            self.dataset_names = names
            self.selected_quantities = list(names)

            if self.dataset_names:
                ds_obj = f[self.dataset_names[0]]
                if isinstance(ds_obj, h5py.Dataset):
                    self.n_subruns_hint = int(ds_obj.shape[0])

        if self.geometry is None and self.dataset_names:
            # Fallback when /metadata is absent.
            with h5py.File(self.path, "r") as f:
                ds_obj = f[self.dataset_names[0]]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise TypeError(
                        f"Expected dataset for '{self.dataset_names[0]}'"
                    )
                _, nz, ny, nx = ds_obj.shape
            self.geometry = VoxGeometry(
                dims_xyz=np.array([nx, ny, nz], dtype=float)
            )

    def select_quantity(self, quantities: str | list[str]) -> "G4VoxFile4D":
        """
        Select one or more quantities to load at iteration time.

        Examples
        --------
        sim.select_quantity("Edep")
        sim.select_quantity(["Edep", "Dose"])
        """
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
    ) -> "G4VoxFile4D":
        """
        Select subruns either with explicit IDs or a start/stop slice.

        Examples
        --------
        sim.select_subrun([0, 3, 7])
        sim.select_subrun(start=10, stop=20)
        """
        if subrun_ids is not None and (start is not None or stop is not None):
            raise ValueError("Use either subrun_ids or start/stop, not both")

        if subrun_ids is not None:
            ids = [int(sid) for sid in subrun_ids]
        elif start is not None or stop is not None:
            s0 = 0 if start is None else int(start)
            s1 = self.n_subruns if stop is None else int(stop)
            ids = list(range(s0, s1))
        else:
            # Empty selection means "all subruns".
            self.selected_subrun_ids = []
            return self

        valid_ids = [sid for sid in ids if 0 <= sid < self.n_subruns]
        if len(valid_ids) != len(ids):
            raise IndexError(
                f"Some subrun ids are out of bounds [0, {self.n_subruns - 1}]"
            )

        self.selected_subrun_ids = valid_ids
        return self

    def _resolved_subrun_ids(self) -> list[int]:
        return (
            self.selected_subrun_ids
            if self.selected_subrun_ids
            else list(range(self.n_subruns))
        )

    def _has_materialized_4d(self) -> bool:
        return bool(self.data) and all(v.ndim == 4 for v in self.data.values())

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
        if self._iter_pos >= len(self._iter_subrun_ids):
            self.data = {}
            self.current_subrun_id = None
            self._iter_source_data = None
            raise StopIteration

        sid = self._iter_subrun_ids[self._iter_pos]
        self._iter_pos += 1
        self.current_subrun_id = sid

        if self._iter_source_data is not None:
            self.data = {
                q: v[sid, ...] for q, v in self._iter_source_data.items()
            }
            return sid

        qtys = (
            self.selected_quantities
            if self.selected_quantities
            else self.dataset_names
        )

        with h5py.File(self.path, "r") as f:
            loaded: dict[str, np.ndarray] = {}
            for q in qtys:
                ds = f[q]
                if not isinstance(ds, h5py.Dataset):
                    raise TypeError(f"Expected dataset for '{q}'")
                loaded[q] = np.asarray(ds[sid, ...])

        self.data = loaded
        return sid

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

        with h5py.File(self.path, "r") as f:
            summed: dict[str, np.ndarray] = {}
            for q in qtys:
                ds = f[q]
                if not isinstance(ds, h5py.Dataset):
                    raise TypeError(f"Expected dataset for '{q}'")

                acc = np.zeros(ds.shape[1:], dtype=ds.dtype)
                for sid in ids:
                    np.add(acc, ds[sid, ...], out=acc)
                summed[q] = acc

        return write_vti(
            filepath=filepath,
            geometry=self._require_geometry(),
            cell_arrays=summed,
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
        with h5py.File(self.path, "r") as f:
            for sid in ids:
                arrays: dict[str, np.ndarray] = {}
                for q in qtys:
                    ds = f[q]
                    if not isinstance(ds, h5py.Dataset):
                        raise TypeError(f"Expected dataset for '{q}'")
                    arrays[q] = np.asarray(ds[sid, ...])

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

    def total_primaries(self) -> int:
        if self.run_log is None or "nPrimaries" not in self.run_log.columns:
            return 0
        return int(self.run_log["nPrimaries"].sum())

    def get(self, qty: str, subrun_id: int) -> np.ndarray:
        """Single subrun slice → (nZ, nY, nX)."""
        if qty in self.data and self.data[qty].ndim == 4:
            return self.data[qty][subrun_id]
        if self.current_subrun_id == subrun_id and qty in self.data:
            return self.data[qty]
        with h5py.File(self.path, "r") as f:
            if qty not in f:
                raise KeyError(f"Quantity '{qty}' not found")
            ds = f[qty]
            if not isinstance(ds, h5py.Dataset):
                raise TypeError(f"Expected dataset for '{qty}'")
            return np.asarray(ds[subrun_id, ...])

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
            f"G4VoxFile4D '{self.path.name}'\n"
            f"  {self.geometry}\n"
            f"  subruns   : {self.n_subruns}\n"
            f"  quantities: {self.quantity_names}\n"
            f"  primaries : {self.total_primaries():,}"
        )

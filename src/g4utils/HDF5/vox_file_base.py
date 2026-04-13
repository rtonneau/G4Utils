from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from g4utils.HDF5.shared import _read_geometry, _read_run_log
from g4utils.HDF5.vti_export import (
    select_quantities,
    write_pvd_collection,
    write_vti,
)
from g4utils.Vox.vox_geometry import VoxGeometry


@dataclass
class BackendDiscovery:
    """
    Metadata bundle returned by a layout backend discovery pass.

    Parameters
    ----------
    geometry : VoxGeometry
        Geometry information inferred from ``/metadata`` or from dataset
        dimensions when metadata is absent.
    run_log : pandas.DataFrame or None
        Run-level table containing per-subrun information such as number of
        primaries and RNG seeds.
    root_attrs : dict
        Top-level HDF5 attributes copied from the file root.
    dataset_names : list of str
        Available quantity names (for example ``Dose`` or ``Edep``).
    available_subrun_ids : list of int
        Logical subrun identifiers available through the selected layout.
    n_subruns_hint : int
        Fast-access subrun count used by façade properties.

    Notes
    -----
    This object is intentionally lightweight and immutable-in-practice: it is
    created by the backend and consumed once by the façade to initialize its
    public state.
    """

    geometry: VoxGeometry
    run_log: pd.DataFrame | None
    root_attrs: dict
    dataset_names: list[str]
    available_subrun_ids: list[int]
    n_subruns_hint: int


class HDF5LayoutBackend(ABC):
    """
    Storage adapter contract for one concrete HDF5 layout.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the HDF5 file to serve.

    Notes
    -----
    Backends isolate all layout-specific concerns. The façade does not know
    whether data is stored as ``/subrun_XXXX/<qty>`` groups or as
    ``/<qty>[subrun, z, y, x]`` datasets; it only calls this interface.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def discover(self) -> BackendDiscovery:
        """Return metadata needed to initialize the shared façade."""

    @abstractmethod
    def load_subrun(
        self, subrun_id: int, quantities: list[str]
    ) -> dict[str, np.ndarray]:
        """Load one logical subrun worth of arrays."""

    def sum_subruns(
        self, subrun_ids: list[int], quantities: list[str]
    ) -> dict[str, np.ndarray]:
        summed: dict[str, np.ndarray] = {}
        for sid in subrun_ids:
            arrays = self.load_subrun(sid, quantities)
            if not summed:
                summed = {
                    name: np.zeros_like(array)
                    for name, array in arrays.items()
                }
            for name, array in arrays.items():
                np.add(summed[name], array, out=summed[name])
        return summed


class Snapshot3DBackend(HDF5LayoutBackend):
    """
    Backend for Snapshot3D files using one group per subrun.

    Notes
    -----
    Expected layout:

    - ``/metadata``
    - ``/run_log``
    - ``/subrun_0000/<quantity>`` with 3D arrays of shape ``(nZ, nY, nX)``
    """

    def discover(self) -> BackendDiscovery:
        with h5py.File(self.path, "r") as f:
            geometry = _read_geometry(f)
            root_attrs = dict(f.attrs)
            run_log = _read_run_log(f)
            subrun_names = sorted(
                key
                for key in f
                if isinstance(key, str)
                and key.startswith("subrun_")
                and isinstance(f[key], h5py.Group)
            )
            subrun_ids = [int(name.split("_")[1]) for name in subrun_names]

            dataset_names: list[str] = []
            if subrun_names:
                first_group = f[subrun_names[0]]
                if not isinstance(first_group, h5py.Group):
                    raise TypeError(f"Expected group for '{subrun_names[0]}'")
                for key in first_group:
                    if not isinstance(key, str):
                        continue
                    obj = first_group[key]
                    if isinstance(obj, h5py.Dataset):
                        dataset_names.append(key)

        return BackendDiscovery(
            geometry=geometry,
            run_log=run_log,
            root_attrs=root_attrs,
            dataset_names=dataset_names,
            available_subrun_ids=subrun_ids,
            n_subruns_hint=len(subrun_ids),
        )

    def load_subrun(
        self, subrun_id: int, quantities: list[str]
    ) -> dict[str, np.ndarray]:
        group_name = f"subrun_{subrun_id:04d}"
        with h5py.File(self.path, "r") as f:
            if group_name not in f:
                raise KeyError(f"Subrun '{subrun_id}' not found")
            group = f[group_name]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"Expected group for '{group_name}'")

            loaded: dict[str, np.ndarray] = {}
            for quantity in quantities:
                if quantity not in group:
                    raise KeyError(
                        "Quantity "
                        f"'{quantity}' not found in subrun {subrun_id}"
                    )
                ds_obj = group[quantity]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise TypeError(f"Expected dataset for '{quantity}'")
                loaded[quantity] = np.asarray(ds_obj[()])
            return loaded


class Dataset4DBackend(HDF5LayoutBackend):
    """
    Backend for 4D dataset files using one dataset per quantity.

    Notes
    -----
    Expected layout:

    - ``/metadata``
    - ``/run_log``
    - ``/Dose`` (or other quantities) with shape ``(N, nZ, nY, nX)``

    where ``N`` is the subrun axis.
    """

    def discover(self) -> BackendDiscovery:
        with h5py.File(self.path, "r") as f:
            geometry = _read_geometry(f)
            root_attrs = dict(f.attrs)
            run_log = _read_run_log(f)

            dataset_names: list[str] = []
            for key in f:
                if not isinstance(key, str) or key in {"metadata", "run_log"}:
                    continue
                obj = f[key]
                if isinstance(obj, h5py.Dataset) and obj.ndim == 4:
                    dataset_names.append(key)

            n_subruns_hint = 0
            if dataset_names:
                ds_obj = f[dataset_names[0]]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise TypeError(
                        f"Expected dataset for '{dataset_names[0]}'"
                    )
                n_subruns_hint = int(ds_obj.shape[0])

        return BackendDiscovery(
            geometry=geometry,
            run_log=run_log,
            root_attrs=root_attrs,
            dataset_names=dataset_names,
            available_subrun_ids=list(range(n_subruns_hint)),
            n_subruns_hint=n_subruns_hint,
        )

    def load_subrun(
        self, subrun_id: int, quantities: list[str]
    ) -> dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as f:
            loaded: dict[str, np.ndarray] = {}
            for quantity in quantities:
                if quantity not in f:
                    raise KeyError(f"Quantity '{quantity}' not found")
                ds_obj = f[quantity]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise TypeError(f"Expected dataset for '{quantity}'")
                loaded[quantity] = np.asarray(ds_obj[subrun_id, ...])
            return loaded


class G4VoxFileBase:
    """
    Common lazy façade shared by all voxel-file front-end classes.

    This class centralizes user-facing behavior (selection, iteration,
    extraction, and export) while delegating file-layout details to a backend
    object implementing :class:`HDF5LayoutBackend`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the source HDF5 file.
    backend : HDF5LayoutBackend
        Concrete storage adapter for the file layout.
    label : str
        Display label used in ``__repr__``.

    Attributes
    ----------
    path : pathlib.Path
        Source file location.
    geometry : VoxGeometry or None
        Geometry read at construction from backend discovery.
    run_log : pandas.DataFrame or None
        Optional run log table.
    data : dict[str, numpy.ndarray]
        Currently materialized subrun data for active iteration step.
    dataset_names : list[str]
        Available quantity names discovered from file.
    available_subrun_ids : list[int]
        Subrun IDs discovered from layout.
    selected_quantities : list[str]
        Active quantity filter.
    selected_subrun_ids : list[int]
        Active subrun filter. Empty means "all available".

    Notes
    -----
    High-level flow:

    1. Construction calls backend ``discover()`` once (metadata only).
    2. User sets filters with ``select_quantity`` and ``select_subrun``.
    3. Iteration lazily pulls one subrun at a time via backend ``load_subrun``.
    4. Export helpers consume the same selection state.

    Simple interaction diagram::

        user API (G4VoxFile3D / G4VoxFile4D)
                     |
                     v
               G4VoxFileBase
            (selection + iteration)
                     |
                     v
             HDF5LayoutBackend
          /                      \
    Snapshot3DBackend      Dataset4DBackend

    This design keeps API behavior consistent across formats while limiting
    future changes to small backend classes when a new storage layout appears.

    Examples
    --------
    ``G4VoxFileBase`` is intended to be subclassed, not instantiated directly.
    See concrete wrappers ``G4VoxFile3D`` and ``G4VoxFile4D`` for user code.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        backend: HDF5LayoutBackend,
        label: str,
    ) -> None:
        self.path = Path(path)
        self._backend = backend
        self._label = label
        self.geometry: VoxGeometry | None = None
        self.run_log: pd.DataFrame | None = None
        self.data: dict[str, np.ndarray] = {}
        self.root_attrs: dict = {}
        self.dataset_names: list[str] = []
        self.available_subrun_ids: list[int] = []
        self.selected_quantities: list[str] = []
        self.selected_subrun_ids: list[int] = []
        self.n_subruns_hint: int | None = None
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
        if self.n_subruns_hint is not None:
            return int(self.n_subruns_hint)
        return len(self.available_subrun_ids)

    @property
    def quantity_names(self) -> list[str]:
        return list(self.dataset_names)

    @property
    def is_lazy(self) -> bool:
        return not bool(self.data) and self.path.exists()

    def _load_metadata_only(self) -> None:
        discovered = self._backend.discover()
        self.geometry = discovered.geometry
        self.run_log = discovered.run_log
        self.root_attrs = dict(discovered.root_attrs)
        self.dataset_names = list(discovered.dataset_names)
        self.available_subrun_ids = list(discovered.available_subrun_ids)
        self.selected_quantities = list(self.dataset_names)
        self.n_subruns_hint = int(discovered.n_subruns_hint)

    def select_quantity(self, quantities: str | list[str]):
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
    ):
        if subrun_ids is not None and (start is not None or stop is not None):
            raise ValueError("Use either subrun_ids or start/stop, not both")

        if subrun_ids is not None:
            ids = [int(sid) for sid in subrun_ids]
        elif start is not None or stop is not None:
            if self.available_subrun_ids:
                default_start = min(self.available_subrun_ids)
                default_stop = max(self.available_subrun_ids) + 1
            else:
                default_start = 0
                default_stop = 0
            s0 = default_start if start is None else int(start)
            s1 = default_stop if stop is None else int(stop)
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
        self.data = self._backend.load_subrun(sid, qtys)
        return sid

    def total_primaries(self) -> int:
        if self.run_log is None or "nPrimaries" not in self.run_log.columns:
            return 0
        return int(self.run_log["nPrimaries"].sum())

    def get(self, qty: str, subrun_id: int) -> np.ndarray:
        if self.current_subrun_id == subrun_id and qty in self.data:
            return self.data[qty]
        return self._backend.load_subrun(subrun_id, [qty])[qty]

    def sum(self, qty: str, subrun_ids: list[int] | None = None) -> np.ndarray:
        ids = (
            subrun_ids
            if subrun_ids is not None
            else self._resolved_subrun_ids()
        )
        if not ids:
            return np.array([])
        return self._backend.sum_subruns(ids, [qty])[qty]

    def dump_selection_to_vti(
        self,
        filepath: str | Path,
        dtype: npt.DTypeLike = np.float32,
    ) -> Path:
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

        arrays = self._backend.sum_subruns(ids, qtys)
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
            arrays = self._backend.load_subrun(sid, qtys)
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
            f"{self._label} '{self.path.name}'\n"
            f"  {self.geometry}\n"
            f"  subruns   : {self.n_subruns}  ids={self.subrun_ids}\n"
            f"  quantities: {self.quantity_names}\n"
            f"  primaries : {self.total_primaries():,}"
        )

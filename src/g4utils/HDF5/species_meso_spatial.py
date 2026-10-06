from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import numpy.typing as npt

_RUN_RE = re.compile(r"^run(\d+)$")
_EVENT_RE = re.compile(r"^event(\d+)$")
_SNAP_RE = re.compile(r"^snapshot(\d+)$")

# Avogadro's number
_N_A = 6.02214076e23


def _attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def concentration_M(counts: npt.NDArray[np.uint32], cell_size_nm: float) -> npt.NDArray[np.float64]:
    """
    Convert molecule counts to molar concentration.

    Parameters
    ----------
    counts : numpy.ndarray
        Molecule counts, shape (N, S) or (N,), uint32.
    cell_size_nm : float
        Side of the cubic cell, in nm.

    Returns
    -------
    numpy.ndarray
        Concentration in molarity (M), same shape as counts, float64.
    """
    volume_L = (cell_size_nm * 1e-8) ** 3
    return np.asarray(counts, dtype=np.float64) / (_N_A * volume_L)


@dataclass(frozen=True)
class MesoGrid:
    """
    Geometry of a regular cubic grid in the world frame.

    The lower corner of cell ``i`` along an axis is ``origin_nm + i * cell_size_nm``
    and its centre is half a cell further.

    Parameters
    ----------
    origin_nm : tuple[float, float, float]
        Lower corner of cell (0, 0, 0), in nm.
    cell_size_nm : float
        Side of the cubic cells, in nm.
    shape : tuple[int, int, int]
        Number of cells along x, y and z.
    """

    origin_nm: tuple[float, float, float]
    cell_size_nm: float
    shape: tuple[int, int, int]

    def index_of(self, position_nm: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """
        Index of the cell containing each position.

        Parameters
        ----------
        position_nm : array_like
            Positions in nm, shape (..., 3).

        Returns
        -------
        numpy.ndarray
            Integer indices, shape (..., 3), int64. Positions outside the
            grid give indices that are negative or >= ``shape``; no check is
            made.
        """
        pos = np.asarray(position_nm, dtype=np.float64)
        origin = np.asarray(self.origin_nm, dtype=np.float64)
        return np.floor((pos - origin) / self.cell_size_nm).astype(np.int64)

    def centers(self, axis: int) -> npt.NDArray[np.float64]:
        """
        Cell centres along one axis, in nm.

        Parameters
        ----------
        axis : int
            0 (x), 1 (y) or 2 (z).

        Returns
        -------
        numpy.ndarray
            Shape (shape[axis],), float64.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
        i = np.arange(self.shape[axis], dtype=np.float64)
        return self.origin_nm[axis] + (i + 0.5) * self.cell_size_nm


@dataclass
class MesoSpatialSnapshot:
    """
    One mesoscopic spatial snapshot.

    Parameters
    ----------
    run, event, index : int
        Run number, event number within the run and snapshot index.
    time_ns : float
        Record time in ns.
    cell_size_nm : float
        Side of the cubic mesh cell at that time, in nm.
    position_nm : numpy.ndarray
        Cell centres, shape (N, 3), float64, in nm.
    counts : numpy.ndarray
        Molecules per cell and species, shape (N, S), uint32.
    species : list[str] or None
        Column names of counts (in order).
    """

    run: int
    event: int
    index: int
    time_ns: float
    cell_size_nm: float
    position_nm: npt.NDArray[np.float64]
    counts: npt.NDArray[np.uint32]
    species: list[str] | None = None

    def concentration_M(self, species: list[str] | None = None) -> npt.NDArray[np.float64]:
        """
        Compute molar concentration.

        Parameters
        ----------
        species : list[str] or None
            Species names to extract (in that order).
            If None, return all columns.

        Returns
        -------
        numpy.ndarray
            Concentration in molarity (M), shape (N, S) or (N, 0) if empty.
            If species is None, columns match snapshot.species;
            otherwise, columns match the requested species list in order.

        Raises
        ------
        KeyError
            If any species name is unknown.
        """
        if species is None:
            # Return all columns
            counts = self.counts
        else:
            # Select requested species columns
            if self.species is None:
                raise KeyError("Snapshot has no species information")
            cols = []
            for sp in species:
                try:
                    idx = self.species.index(sp)
                    cols.append(idx)
                except ValueError:
                    raise KeyError(f"Unknown species: {sp}") from None
            counts = self.counts[:, cols]

        return concentration_M(counts, self.cell_size_nm)

    def to_dense(
        self,
        species: str,
        bounds_nm: npt.ArrayLike | None = None,
        concentration: bool = False,
    ) -> tuple[npt.NDArray, MesoGrid]:
        """
        Rebuild a zero-filled 3D array of one species from the sparse rows.

        Parameters
        ----------
        species : str
            Species name (must be in ``self.species``).
        bounds_nm : array_like or None
            Extent to cover, ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` in
            nm (world frame). It is snapped outwards to the cell lattice
            (multiples of ``cell_size_nm``); cells outside it are dropped.
            If None, the bounding box of all occupied cells of the snapshot
            (whatever the species) is used.
        concentration : bool
            If True, return molar concentration (float64, mol/L) instead of
            counts (uint32).

        Returns
        -------
        (numpy.ndarray, MesoGrid)
            Array of shape ``grid.shape`` and its grid. If the snapshot is
            empty and ``bounds_nm`` is None, the shape is (0, 0, 0).

        Raises
        ------
        KeyError
            If the species is unknown.
        """
        if self.species is None:
            raise KeyError("Snapshot has no species information")
        try:
            col = self.species.index(species)
        except ValueError:
            raise KeyError(f"Unknown species: {species}") from None

        cell = self.cell_size_nm
        pos = np.asarray(self.position_nm, dtype=np.float64).reshape(-1, 3)
        # Cell index on the lattice of multiples of the cell size
        lattice = np.floor(pos / cell).astype(np.int64)

        if bounds_nm is None:
            if len(lattice) == 0:
                lo = np.zeros(3, dtype=np.int64)
                hi = lo.copy()
            else:
                lo = lattice.min(axis=0)
                hi = lattice.max(axis=0) + 1
        else:
            b = np.asarray(bounds_nm, dtype=np.float64)
            if b.shape != (3, 2):
                raise ValueError(f"bounds_nm must have shape (3, 2), got {b.shape}")
            lo = np.floor(b[:, 0] / cell + 1e-9).astype(np.int64)
            hi = np.maximum(np.ceil(b[:, 1] / cell - 1e-9).astype(np.int64), lo + 1)

        shape = tuple(int(n) for n in hi - lo)
        grid = MesoGrid(
            origin_nm=tuple(float(v) for v in lo * cell),
            cell_size_nm=cell,
            shape=shape,  # type: ignore[arg-type]
        )

        dense = np.zeros(shape, dtype=np.uint32)
        if len(lattice):
            idx = lattice - lo
            inside = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
            idx = idx[inside]
            np.add.at(
                dense, (idx[:, 0], idx[:, 1], idx[:, 2]), self.counts[inside, col]
            )

        if concentration:
            return concentration_M(dense, cell), grid
        return dense, grid


@dataclass
class DenseMesoPeriod:
    """
    Dense time series of one species over a run of snapshots with one cell size.

    Parameters
    ----------
    data : numpy.ndarray
        Shape (T, nx, ny, nz); uint32 counts, or float64 molar concentration.
    grid : MesoGrid
        Geometry shared by every time step; ``grid.shape == data.shape[1:]``.
    times_ns : numpy.ndarray
        Record time of each step, shape (T,), float64, in ns.
    snapshot_indices : list[int]
        Snapshot index of each step.
    species : str
        Species name.
    run, event : int
        Run and event numbers.
    """

    data: npt.NDArray
    grid: MesoGrid
    times_ns: npt.NDArray[np.float64]
    snapshot_indices: list[int]
    species: str
    run: int
    event: int


class SpeciesMesoSpatialFile:
    """
    Reader for ``SpeciesMesoSpatial.h5`` files written by dnachem-min.

    The constructor reads the index once (group names and snapshot attributes);
    array data are read on demand by :meth:`read_snapshot`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the HDF5 file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the root attribute ``species`` or ``formatVersion`` is missing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"File not found: {self.path}")

        # {run: {event: {index: (time_ns, cell_size_nm)}}}
        index: dict[int, dict[int, dict[int, tuple[float, float]]]] = {}
        with h5py.File(self.path, "r") as f:
            for key in ("species", "formatVersion"):
                if key not in f.attrs:
                    raise ValueError(f"Missing root attribute '{key}' in {self.path}")
            self._species = [_attr_str(s) for s in f.attrs["species"]]
            self._format_version = int(f.attrs["formatVersion"])
            self._units = _attr_str(f.attrs["units"]) if "units" in f.attrs else ""

            for run_name, run_obj in f.items():
                m_run = _RUN_RE.match(run_name)
                if m_run is None:
                    continue
                if not isinstance(run_obj, h5py.Group):
                    raise TypeError(f"Expected group for '{run_name}'")
                events: dict[int, dict[int, tuple[float, float]]] = {}
                for event_name, event_obj in run_obj.items():
                    m_ev = _EVENT_RE.match(event_name)
                    if m_ev is None:
                        continue
                    if not isinstance(event_obj, h5py.Group):
                        raise TypeError(f"Expected group for '{run_name}/{event_name}'")
                    snaps: dict[int, tuple[float, float]] = {}
                    for snap_name, snap_obj in event_obj.items():
                        m_sn = _SNAP_RE.match(snap_name)
                        if m_sn is None:
                            continue
                        if not isinstance(snap_obj, h5py.Group):
                            raise TypeError(
                                f"Expected group for "
                                f"'{run_name}/{event_name}/{snap_name}'"
                            )
                        snaps[int(m_sn.group(1))] = (
                            float(snap_obj.attrs["time_ns"]),
                            float(snap_obj.attrs["cellSize_nm"]),
                        )
                    events[int(m_ev.group(1))] = dict(sorted(snaps.items()))
                index[int(m_run.group(1))] = dict(sorted(events.items()))
        self._index = dict(sorted(index.items()))

    # ------------------------------------------------------------------
    # Index access
    # ------------------------------------------------------------------
    @property
    def species(self) -> list[str]:
        """Column names of ``counts``, in column order."""
        return list(self._species)

    @property
    def format_version(self) -> int:
        """File format version."""
        return self._format_version

    @property
    def units(self) -> str:
        """One-line units summary from the file (empty if absent)."""
        return self._units

    @property
    def runs(self) -> list[int]:
        """Run numbers, sorted numerically."""
        return list(self._index)

    def _events(self, run: int) -> dict[int, dict[int, tuple[float, float]]]:
        try:
            return self._index[run]
        except KeyError:
            raise KeyError(f"Unknown run {run}") from None

    def _snapshots(self, run: int, event: int) -> dict[int, tuple[float, float]]:
        try:
            return self._events(run)[event]
        except KeyError:
            raise KeyError(f"Unknown event {event} in run {run}") from None

    def _snapshot_meta(self, run: int, event: int, index: int) -> tuple[float, float]:
        try:
            return self._snapshots(run, event)[index]
        except KeyError:
            raise KeyError(
                f"Unknown snapshot {index} in run {run}, event {event}"
            ) from None

    def events(self, run: int) -> list[int]:
        """Event numbers of ``run``, sorted numerically."""
        return list(self._events(run))

    def snapshot_indices(self, run: int, event: int) -> list[int]:
        """Snapshot indices of an event, sorted numerically."""
        return list(self._snapshots(run, event))

    def snapshot_time_ns(self, run: int, event: int, index: int) -> float:
        """Record time of a snapshot, in ns."""
        return self._snapshot_meta(run, event, index)[0]

    def snapshot_cell_size_nm(self, run: int, event: int, index: int) -> float:
        """Cell size of a snapshot, in nm."""
        return self._snapshot_meta(run, event, index)[1]

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def read_snapshot(self, run: int, event: int, index: int) -> MesoSpatialSnapshot:
        """
        Read one snapshot from disk.

        Raises
        ------
        KeyError
            If the run, event or snapshot index is unknown.
        """
        time_ns, cell_size_nm = self._snapshot_meta(run, event, index)
        group_name = f"run{run}/event{event}/snapshot{index}"
        with h5py.File(self.path, "r") as f:
            group = f[group_name]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"Expected group for '{group_name}'")
            pos_ds = group["position_nm"]
            counts_ds = group["counts"]
            if not isinstance(pos_ds, h5py.Dataset):
                raise TypeError(f"Expected dataset for '{group_name}/position_nm'")
            if not isinstance(counts_ds, h5py.Dataset):
                raise TypeError(f"Expected dataset for '{group_name}/counts'")
            position = np.asarray(pos_ds[()], dtype=np.float64).reshape(-1, 3)
            counts = np.asarray(counts_ds[()], dtype=np.uint32).reshape(
                -1, len(self._species)
            )
        return MesoSpatialSnapshot(
            run=run,
            event=event,
            index=index,
            time_ns=time_ns,
            cell_size_nm=cell_size_nm,
            position_nm=position,
            counts=counts,
            species=list(self._species),
        )

    def iter_snapshots(
        self, run: int | None = None, event: int | None = None
    ) -> Iterator[MesoSpatialSnapshot]:
        """
        Lazily yield snapshots in (run, event, index) order.

        Parameters
        ----------
        run, event : int or None
            Optional filters; ``None`` means no filter.
        """
        for r, events in self._index.items():
            if run is not None and r != run:
                continue
            for e, snaps in events.items():
                if event is not None and e != event:
                    continue
                for k in snaps:
                    yield self.read_snapshot(r, e, k)

    def read_dense(
        self,
        run: int,
        event: int,
        species: str,
        bounds_nm: npt.ArrayLike | None = None,
        concentration: bool = False,
    ) -> list[DenseMesoPeriod]:
        """
        Read one species of an event as dense 4D arrays, one per cell-size period.

        Consecutive snapshots sharing the same ``cellSize_nm`` form a period; a
        new period starts whenever the cell size changes. Snapshots are read
        one at a time.

        Parameters
        ----------
        run, event : int
            Run and event numbers.
        species : str
            Species name (must be in :attr:`species`).
        bounds_nm : array_like or None
            Extent ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` in nm (world
            frame), snapped outwards to each period's cell lattice. If None,
            the bounding box of all occupied cells of the event (whatever the
            species, all snapshots) is used. Two files given the same bounds
            give identically shaped arrays for periods of equal cell size.
        concentration : bool
            If True, return molar concentration (float64) instead of counts
            (uint32).

        Returns
        -------
        list[DenseMesoPeriod]
            One period per run of consecutive equal cell sizes, in snapshot
            order; empty if the event has no snapshot.

        Raises
        ------
        KeyError
            If the run, event or species is unknown.
        """
        snaps = self._snapshots(run, event)
        if species not in self._species:
            raise KeyError(f"Unknown species: {species}")

        if bounds_nm is None:
            # First pass: bounding box of the occupied cells, in nm.
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for k, (_, cell) in snaps.items():
                pos = self.read_snapshot(run, event, k).position_nm
                if len(pos) == 0:
                    continue
                lattice = np.floor(pos / cell)
                lo = np.minimum(lo, lattice.min(axis=0) * cell)
                hi = np.maximum(hi, (lattice.max(axis=0) + 1) * cell)
            if np.isfinite(lo).all():
                bounds_nm = np.stack([lo, hi], axis=1)

        periods: list[DenseMesoPeriod] = []
        group: list[int] = []

        def flush() -> None:
            frames: list[npt.NDArray] = []
            grid: MesoGrid | None = None
            for k in group:
                frame, grid = self.read_snapshot(run, event, k).to_dense(
                    species, bounds_nm=bounds_nm, concentration=concentration
                )
                frames.append(frame)
            assert grid is not None
            periods.append(
                DenseMesoPeriod(
                    data=np.stack(frames),
                    grid=grid,
                    times_ns=np.array([snaps[k][0] for k in group], dtype=np.float64),
                    snapshot_indices=list(group),
                    species=species,
                    run=run,
                    event=event,
                )
            )

        current_cell: float | None = None
        for k, (_, cell) in snaps.items():
            if group and cell != current_cell:
                flush()
                group = []
            group.append(k)
            current_cell = cell
        if group:
            flush()
        return periods

    def __repr__(self) -> str:
        return (
            f"SpeciesMesoSpatialFile({str(self.path)!r}, "
            f"runs={len(self._index)}, species={len(self._species)})"
        )

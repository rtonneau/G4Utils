from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

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


# Off-lattice tolerance, as a fraction of a cell, and slack used when growing
# a lattice by whole cells so float noise does not add a spurious cell.
_LATTICE_TOL = 1e-3
_EDGE_EPS = 1e-6


class _DenseGrid(NamedTuple):
    """
    Dense lattice view of a sparse mesoscopic snapshot.

    Attributes
    ----------
    origin_nm : numpy.ndarray
        Lower corner (x, y, z) of the lattice, in nm, shape (3,).
    cell_size_nm : float
        Cell side, in nm.
    dims : tuple[int, int, int]
        Number of cells along (x, y, z), i.e. (nX, nY, nZ).
    index : numpy.ndarray
        Shape (nZ, nY, nX), int64: 1-based row number in the snapshot arrays
        for occupied cells, 0 for unoccupied ones.
    counts : numpy.ndarray
        Shape (S, nZ, nY, nX), uint32: counts per species, 0 in unoccupied cells.
    """

    origin_nm: npt.NDArray[np.float64]
    cell_size_nm: float
    dims: tuple[int, int, int]
    index: npt.NDArray[np.int64]
    counts: npt.NDArray[np.uint32]


def _densify_snapshot(
    snapshot: MesoSpatialSnapshot,
    extent: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
) -> _DenseGrid:
    """
    Place the cells of a sparse snapshot on a dense regular lattice.

    Without ``extent`` the lattice origin is ``min(centre) - cell_size / 2``
    per axis and spans the occupied cells only. With ``extent =
    (min_xyz_nm, max_xyz_nm)`` the lattice keeps the snapshot's anchor (the
    origin it would have without extent) and grows by whole cells, on either
    side, until it covers both the box and every cell. A snapshot without
    cells needs an extent; its lattice is then anchored on ``min_xyz_nm``.

    Cell ``i`` along an axis has its centre at ``origin + (i + 0.5) * cell_size``.

    Parameters
    ----------
    snapshot : MesoSpatialSnapshot
        Snapshot to densify.
    extent : (array-like, array-like) or None
        Physical box ``(min_xyz_nm, max_xyz_nm)`` in nm to cover.

    Returns
    -------
    _DenseGrid
        Origin (nm), cell size (nm), dims (nX, nY, nZ) and dense arrays in
        (nZ, nY, nX) axis order, with 0 in unoccupied cells.

    Raises
    ------
    ValueError
        If the snapshot has no cells and no extent is given, if the extent is
        malformed, if a centre is off the lattice by more than a small
        tolerance, or if two cells fall in the same lattice index.
    """
    cs = float(snapshot.cell_size_nm)
    if not cs > 0.0:
        raise ValueError(f"cell_size_nm must be positive, got {cs}")
    pos = np.asarray(snapshot.position_nm, dtype=np.float64).reshape(-1, 3)
    n_cells = pos.shape[0]
    counts = np.asarray(snapshot.counts)
    if counts.ndim != 2 or counts.shape[0] != n_cells:
        raise ValueError(
            f"counts must have shape (N, S) with N={n_cells}, got {counts.shape}"
        )
    n_species = counts.shape[1]

    box_min = box_max = None
    if extent is not None:
        box_min = np.asarray(extent[0], dtype=np.float64).reshape(-1)
        box_max = np.asarray(extent[1], dtype=np.float64).reshape(-1)
        if box_min.shape != (3,) or box_max.shape != (3,):
            raise ValueError("extent must be (min_xyz_nm, max_xyz_nm), each of length 3")
        if not (np.all(np.isfinite(box_min)) and np.all(np.isfinite(box_max))):
            raise ValueError("extent must be finite")
        if np.any(box_max < box_min):
            raise ValueError(f"extent max {box_max} is below min {box_min}")

    if n_cells == 0 and extent is None:
        raise ValueError("Snapshot has no cells and no extent was given")

    if n_cells:
        anchor = pos.min(axis=0) - 0.5 * cs
        frac = (pos - anchor) / cs - 0.5
        ijk_all = np.rint(frac)
        off = np.abs(frac - ijk_all)
        if np.any(off > _LATTICE_TOL):
            row = int(np.argmax(off.max(axis=1)))
            raise ValueError(
                f"Cell {row} at {pos[row].tolist()} nm is off the lattice "
                f"(cell size {cs} nm, anchor {anchor.tolist()} nm): "
                f"{off[row].max():.3g} cell(s) from the nearest site"
            )
        ijk_all = ijk_all.astype(np.int64)
        lo = np.zeros(3, dtype=np.int64)
        hi = ijk_all.max(axis=0) + 1
    else:
        assert box_min is not None
        anchor = box_min
        ijk_all = np.zeros((0, 3), dtype=np.int64)
        lo = np.zeros(3, dtype=np.int64)
        hi = np.zeros(3, dtype=np.int64)

    if box_min is not None and box_max is not None:
        lo = np.minimum(lo, np.floor((box_min - anchor) / cs + _EDGE_EPS).astype(np.int64))
        hi = np.maximum(hi, np.ceil((box_max - anchor) / cs - _EDGE_EPS).astype(np.int64))
    # A zero-width box still yields one cell per axis.
    hi = np.maximum(hi, lo + 1)

    origin = anchor + lo * cs
    ijk = ijk_all - lo
    dims = tuple(int(v) for v in (hi - lo))
    nx, ny, nz = dims

    index = np.zeros((nz, ny, nx), dtype=np.int64)
    dense = np.zeros((n_species, nz, ny, nx), dtype=np.uint32)
    if n_cells:
        flat = (ijk[:, 2] * ny + ijk[:, 1]) * nx + ijk[:, 0]
        order = np.argsort(flat, kind="stable")
        dup = np.nonzero(flat[order][1:] == flat[order][:-1])[0]
        if dup.size:
            a, b = int(order[dup[0]]), int(order[dup[0] + 1])
            raise ValueError(
                f"Cells {a} and {b} share lattice index "
                f"(x, y, z) = {tuple(int(v) for v in ijk_all[a])} "
                f"(centres {pos[a].tolist()} and {pos[b].tolist()} nm)"
            )
        index.reshape(-1)[flat] = np.arange(1, n_cells + 1, dtype=np.int64)
        for s in range(n_species):
            dense[s].reshape(-1)[flat] = counts[:, s]

    return _DenseGrid(origin, cs, dims, index, dense)


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

    def __repr__(self) -> str:
        return (
            f"SpeciesMesoSpatialFile({str(self.path)!r}, "
            f"runs={len(self._index)}, species={len(self._species)})"
        )

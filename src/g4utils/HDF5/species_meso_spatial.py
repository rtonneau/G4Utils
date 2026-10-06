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


def _attr_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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
    """

    run: int
    event: int
    index: int
    time_ns: float
    cell_size_nm: float
    position_nm: npt.NDArray[np.float64]
    counts: npt.NDArray[np.uint32]


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

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator

import pandas as pd

from g4utils.DnaChem.dump import Dump
from g4utils.DnaChem.dumps import discover_dumps


class _Subruns(Mapping[str, Dump]):
    """Canonical keys for iteration, unique folder aliases for legacy lookup."""

    def __init__(self, dumps: dict[str, Dump]):
        self._dumps = dumps

    def __getitem__(self, key: str) -> Dump:
        if key in self._dumps:
            return self._dumps[key]
        matches = [dump for dump in self._dumps.values() if dump.name == key]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise KeyError(f"Ambiguous subrun folder {key!r}; use {[d.key for d in matches]}")
        raise KeyError(f"No subrun {key!r}; available: {list(self._dumps)}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._dumps)

    def __len__(self) -> int:
        return len(self._dumps)


_EXTRA_COLUMNS = (
    "chemistryModel",
    "handOverTime_ns",
    "chemistryEndTime_ns",
    "voxelSize_nm",
    "mesoPixels",
    "mesoTimesPerDecade",
    "mesoSpatialOutput",
    "threads",
)


class Simulation:
    """A dnachem-min simulation folder: a set of Dump subruns.

    ``path`` may be a flat Dump folder, a ``results`` folder holding Dumps, or
    a simulation folder with a ``results/`` subfolder (used when it exists).
    Construction reads manifests for discovery; data files remain lazy.
    Subruns use root-relative manifest keys, with unique folder-name aliases.
    """

    def __init__(
        self, path: str | Path, name_pattern: str | re.Pattern | None = None
    ) -> None:
        root = Path(path)
        results = root / "results"
        self.path = results if results.is_dir() else root
        self.name_pattern = name_pattern
        dumps: dict[str, Dump] = {}
        for location in discover_dumps(self.path):
            dump = Dump.from_manifest(location.manifest_path, name_pattern)
            dump.key = location.key
            dumps[location.key] = dump
        self.subruns: Mapping[str, Dump] = _Subruns(dumps)

    def subrun(self, name: str) -> Dump:
        return self.subruns[name]

    def __iter__(self) -> Iterator[Dump]:
        return iter(self.subruns.values())

    def __len__(self) -> int:
        return len(self.subruns)

    def table(self) -> pd.DataFrame:
        """One row per Dump: name, labels and Manifest summary columns."""
        return pd.DataFrame([self._row(dump) for dump in self])

    @staticmethod
    def _row(dump: Dump) -> dict:
        m = dump.manifest
        base = dump._columns()
        row = {"dump": base["dump"], **dump.labels}
        row.update(base)
        for key in _EXTRA_COLUMNS:
            row[key] = getattr(m, key, None)
        wall = [r.wallTime_s for r in m.runs if r.wallTime_s is not None]
        row["wallTime_s"] = sum(wall) if wall else None
        return row

    def __repr__(self) -> str:
        return f"Simulation({str(self.path)!r}, {len(self)} subruns)"

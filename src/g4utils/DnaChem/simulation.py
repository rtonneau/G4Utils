from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from g4utils.DnaChem.dump import Dump
from g4utils.DnaChem.dumps import find_dumps


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
    Construction only lists folders; no data file is read.
    """

    def __init__(
        self, path: str | Path, name_pattern: str | re.Pattern | None = None
    ) -> None:
        root = Path(path)
        results = root / "results"
        self.path = results if results.is_dir() else root
        self.name_pattern = name_pattern
        self.subruns: dict[str, Dump] = {}
        for folder in find_dumps(self.path):
            dump = Dump(folder, name_pattern)
            self.subruns[dump.name] = dump

    def subrun(self, name: str) -> Dump:
        try:
            return self.subruns[name]
        except KeyError:
            raise KeyError(
                f"No subrun {name!r}; available: {list(self.subruns)}"
            ) from None

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

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from g4utils.DnaChem.dump import Dump
from g4utils.DnaChem.dumps import find_dumps


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
        return pd.DataFrame([dump._columns() for dump in self])

    def __repr__(self) -> str:
        return f"Simulation({str(self.path)!r}, {len(self)} subruns)"

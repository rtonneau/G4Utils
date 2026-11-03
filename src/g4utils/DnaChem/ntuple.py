from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

_DTYPES = {"int": "int64", "double": "float64", "string": str}


def read_ntuple(path: Path) -> pd.DataFrame:
    """Read a Geant4 tools ``wcsv`` ntuple CSV into a DataFrame.

    Column names come from the ``#column <type> <name>`` header lines; all
    other ``#`` header lines are skipped. ``#`` is not treated as a comment
    marker in the data rows.
    """
    names: list[str] = []
    types: list[str] = []
    n_header = 0
    with open(path, encoding="utf-8", newline=None) as f:
        for line in f:
            if not line.startswith("#"):
                break
            n_header += 1
            parts = line.strip().split(None, 2)
            if len(parts) == 3 and parts[0] == "#column":
                types.append(parts[1])
                names.append(parts[2])

    df = pd.read_csv(
        path,
        comment=None,
        skiprows=n_header,
        header=None,
        names=names,
        encoding="utf-8",
        keep_default_na=False,
        dtype={name: str for name, typ in zip(names, types) if typ == "int"},
    )
    for name, typ in zip(names, types):
        if typ == "int":
            values = []
            for value in df[name]:
                if not re.fullmatch(r"[+-]?[0-9]+", value):
                    raise ValueError(f"{path}: {name} must contain integer tokens, got {value!r}")
                number = int(value)
                if not -(2**63) <= number < 2**63:
                    raise ValueError(f"{path}: {name} is outside int64 range: {value!r}")
                values.append(number)
            df[name] = pd.Series(values, index=df.index, dtype="int64")
            continue
        if typ in _DTYPES:
            df[name] = df[name].astype(_DTYPES[typ])
    return df

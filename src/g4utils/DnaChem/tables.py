"""Format-agnostic reader for the standardized dnachem-min output tables.

Each product is written in one or more formats (csv, h5, root, txt) holding the
same table. ``read_table`` picks the best available format and returns a frame
with identical column order and dtypes whichever format was read:
``int64`` for integers, ``float64`` for doubles, ``string`` for text.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_PREFERENCE = ("csv", "h5", "root", "txt")

# Writer column order per table; h5 stores datasets alphabetically.
_COUNT_STATS = ("nEvents", "countSum", "countSumSq", "countMean", "countStdErr")
_G_STATS = ("gSum", "gSumSq", "gMean", "gStdErr")
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "species": ("time_ns", "species", *_COUNT_STATS, *_G_STATS),
    "speciesMeso": ("time_ns", "species", *_COUNT_STATS, *_G_STATS),
    "reactions": ("reactionId", "binStart_ns", "binEnd_ns", *_COUNT_STATS),
    "reactionCatalog": ("reactionId", "reaction"),
    "physicsInteractions": ("process", *_COUNT_STATS),
    "trackLengths": (
        "run", "event", "primaryLength_nm", "primaryEkinStart_keV",
        "primaryEkinEnd_keV", "primaryEnd", "secondaryFirstGenLength_nm",
        "nSecondaryFirstGen", "secondaryAllLength_nm", "nSecondaryAll",
    ),
    "preChemical": (
        "run", "event", "parentTrackId", "molecule", "modification", "level",
        "energy_eV", "parentX_nm", "parentY_nm", "parentZ_nm", "x_nm", "y_nm", "z_nm",
    ),
}
_INT_COLUMNS = frozenset({
    "reactionId", "nEvents", "run", "event", "parentTrackId", "level",
    "nSecondaryFirstGen", "nSecondaryAll",
})
_STRING_COLUMNS = frozenset({
    "species", "reaction", "process", "primaryEnd", "molecule", "modification",
})


def _normalize(frame: pd.DataFrame, table: str, source: Path) -> pd.DataFrame:
    """Order columns and force the canonical dtypes."""
    known = TABLE_COLUMNS.get(table)
    if known is not None:
        missing = [c for c in known if c not in frame.columns]
        if missing:
            raise ValueError(f"{source}: table {table!r} lacks columns {missing}")
        extra = sorted(c for c in frame.columns if c not in known)
        frame = frame[[*known, *extra]]
    out = {}
    for name in frame.columns:
        values = frame[name]
        if name in _STRING_COLUMNS:
            out[name] = values.astype("string")
        elif name in _INT_COLUMNS:
            out[name] = _to_int64(values, name, source)
        else:
            out[name] = pd.to_numeric(values).astype("float64")
    return pd.DataFrame(out, index=pd.RangeIndex(len(frame)))


def _to_int64(values: pd.Series, name: str, source: Path) -> pd.Series:
    numbers = pd.to_numeric(values)
    if numbers.dtype.kind == "f" and not (numbers == np.floor(numbers)).all():
        raise ValueError(f"{source}: column {name} holds non-integer values")
    return numbers.astype("int64")


def _read_csv(path: Path, table: str) -> pd.DataFrame:
    # Strings stay strings: no NA inference on labels such as "None" or "NA".
    return pd.read_csv(path, encoding="utf-8", keep_default_na=False, dtype=str)


_TOKEN = re.compile(r'"[^"]*"|\S+')


def _read_txt(path: Path, table: str) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.startswith("#") or not line.strip():
                continue
            tokens = [t[1:-1] if t.startswith('"') else t for t in _TOKEN.findall(line)]
            if header is None:
                header = tokens
                continue
            if len(tokens) != len(header):
                raise ValueError(
                    f"{path}:{number}: expected {len(header)} fields, got {len(tokens)}"
                )
            rows.append(tokens)
    if header is None:
        raise ValueError(f"{path}: no column header found")
    return pd.DataFrame(rows, columns=header, dtype=object)


def _read_h5(path: Path, table: str) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as handle:
        if table not in handle:
            raise ValueError(f"{path}: no table {table!r}; found {list(handle)}")
        group = handle[table]
        columns = {}
        for name, dataset in group.items():
            if dataset.dtype.kind in ("O", "S"):
                columns[name] = np.asarray(dataset.asstr()[()], dtype=object)
            else:
                columns[name] = dataset[()]
    return pd.DataFrame(columns)


def _read_root(path: Path, table: str) -> pd.DataFrame:
    try:
        import uproot
    except ImportError as exc:
        raise ImportError(
            f"Reading {path.name} needs uproot; install it with "
            "'pip install g4utils[root]'"
        ) from exc
    with uproot.open(path) as handle:
        if table not in handle:
            raise ValueError(f"{path}: no tree {table!r}")
        arrays = handle[table].arrays(library="np")
    return pd.DataFrame({name: np.asarray(values) for name, values in arrays.items()})


_READERS = {"csv": _read_csv, "h5": _read_h5, "root": _read_root, "txt": _read_txt}


def read_table_file(path: str | Path, fmt: str, table: str) -> pd.DataFrame:
    """Read one file of a known format into the canonical frame for ``table``."""
    path = Path(path)
    if fmt not in _READERS:
        raise ValueError(f"Unknown output format {fmt!r}; expected one of {list(_READERS)}")
    return _normalize(_READERS[fmt](path, table), table, path)


def pick_format(
    directory: str | Path,
    stem: str,
    formats: Iterable[str],
    files: Iterable[str] | None = None,
    *,
    container: str | None = None,
) -> tuple[str, Path]:
    """First available format in csv, h5, root, txt order, with its file.

    ``formats`` is the manifest list for the product; ``files`` (optional) is
    the manifest inventory. A format counts only if its file is listed (when an
    inventory is given) and exists. ``container`` is the file stem used by h5
    and root when it differs from ``stem`` (the reaction catalogue).
    """
    wanted = set(formats)
    listed = None if files is None else {f.replace("\\", "/") for f in files}
    directory = Path(directory)
    tried = []
    for fmt in FORMAT_PREFERENCE:
        if fmt not in wanted:
            continue
        base = (container or stem) if fmt in ("h5", "root") else stem
        name = f"{base}.{fmt}"
        tried.append(name)
        if listed is not None and name not in listed:
            continue
        if (directory / name).is_file():
            return fmt, directory / name
    raise FileNotFoundError(
        f"No readable file for {stem!r} in {directory}; looked for "
        f"{tried or 'no format'} (manifest formats: {sorted(wanted)})"
    )


def read_table(
    directory: str | Path,
    stem: str,
    table: str,
    formats: Iterable[str],
    files: Iterable[str] | None = None,
    *,
    container: str | None = None,
) -> pd.DataFrame:
    """Read ``table`` from the preferred available format of ``stem``."""
    fmt, path = pick_format(directory, stem, formats, files, container=container)
    return read_table_file(path, fmt, table)

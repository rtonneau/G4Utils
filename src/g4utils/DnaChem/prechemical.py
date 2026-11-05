"""Read the observed fixed-width pre-chemical event format without code labels."""

from __future__ import annotations

import math
import re
from itertools import accumulate
from pathlib import Path

import pandas as pd

from g4utils.DnaChem.dumps import _relative_file
from g4utils.DnaChem.readers import _integer


PRECHEMICAL_COLUMNS = (
    "parent_id", "molecule", "electron_modification", "energy_eV",
    "parent_x_nm", "parent_y_nm", "parent_z_nm", "x_nm", "y_nm", "z_nm",
)
PRECHEMICAL_WIDTHS = (11, 10, 14, 13, 22, 22, 22, 14, 14, 14)
_EDGES = tuple(accumulate((0, *PRECHEMICAL_WIDTHS)))
_EVENT_NAME = re.compile(r"PreChemical_run([0-9]+)_event([0-9]+)\.txt\Z")


def _event_files(
    manifest_path: Path, prefix: str, inventory: tuple[str, ...] | None,
) -> list[tuple[int, int, Path]]:
    """Select only this literal prefix, with inventory authoritative when present."""
    directory = manifest_path.parent
    names = inventory if inventory is not None else (p.name for p in directory.iterdir())
    selected = {}
    for name in names:
        portable = name.replace("\\", "/")
        basename = Path(portable).name
        if not basename.startswith(f"{prefix}PreChemical_"):
            continue
        match = _EVENT_NAME.fullmatch(basename[len(prefix):])
        if match is None:
            raise ValueError(f"{manifest_path}: ambiguous pre-chemical filename {name!r}")
        file = _relative_file(directory, portable, str(manifest_path))
        key = tuple(_integer(value, file, 0, column)
                    for value, column in zip(match.groups(), ("run", "event")))
        if key in selected:
            raise ValueError(
                f"{manifest_path}: duplicate pre-chemical (run,event) {key}: "
                f"{selected[key]} and {file}"
            )
        if not file.is_file():
            raise FileNotFoundError(f"Missing pre-chemical data file {file}")
        selected[key] = file
    return [(run, event, selected[(run, event)]) for run, event in sorted(selected)]


def _finite_float(value: str, file: Path, line: int, column: str) -> float:
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{file}:{line}: {column} must be finite numeric data: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{file}:{line}: {column} must be finite numeric data: {value!r}")
    return number


def read_prechemical_records(
    manifest_path: Path, prefix: str, inventory: tuple[str, ...] | None,
    *, allow_missing: bool = False,
) -> pd.DataFrame:
    """Read records in numeric run/event order; final positions remain nullable.

    If the manifest has no inventory (absent or null), scan only its immediate
    directory. An explicitly empty inventory does not enable this fallback.
    """
    files = _event_files(manifest_path, prefix, inventory)
    if not files and not allow_missing:
        raise FileNotFoundError(
            f"{manifest_path}: no {prefix}PreChemical_run<R>_event<E>.txt files selected. "
            "Regenerate the dump with pre-chemical event output enabled and flush "
            "it with /run/dumpDataAndReset, or use allow_missing=True for absent optional output."
        )
    records = []
    for run, event, file in files:
        with file.open(encoding="utf-8-sig") as stream:
            for line_number, text in enumerate(stream, start=1):
                text = text.partition("#")[0].rstrip("\r\n")
                if not text.strip():
                    continue
                if text[_EDGES[-1]:].strip():
                    raise ValueError(f"{file}:{line_number}: unexpected trailing pre-chemical data")
                values = [text[start:end].strip() for start, end in zip(_EDGES, _EDGES[1:])]
                row = dict(zip(PRECHEMICAL_COLUMNS, values))
                for column in PRECHEMICAL_COLUMNS[:7]:
                    if not row[column]:
                        raise ValueError(f"{file}:{line_number}: incomplete required field {column}")
                row["parent_id"] = _integer(row["parent_id"], file, line_number, "parent_id")
                for column in PRECHEMICAL_COLUMNS[3:]:
                    row[column] = (_finite_float(row[column], file, line_number, column)
                                   if row[column] else pd.NA)
                records.append({"run": run, "event": event, **row})
    frame = pd.DataFrame(records, columns=("run", "event", *PRECHEMICAL_COLUMNS))
    for column in ("run", "event", "parent_id"):
        frame[column] = frame[column].astype("int64")
    for column in ("molecule", "electron_modification"):
        frame[column] = frame[column].astype("string")
    for column in PRECHEMICAL_COLUMNS[3:7]:
        frame[column] = frame[column].astype("float64")
    for column in PRECHEMICAL_COLUMNS[7:]:
        frame[column] = frame[column].astype("Float64")
    return frame


STANDARDIZED_PRECHEMICAL_COLUMNS = (
    "run", "event", "parentTrackId", "molecule", "modification", "level",
    "energy_eV", "parentX_nm", "parentY_nm", "parentZ_nm", "x_nm", "y_nm", "z_nm",
)
_STD_EVENT_NAME = re.compile(r"PreChemical_run([0-9]+)_event([0-9]+)\.(csv|txt)\Z")


def _standardized_files(manifest_path, prefix, inventory, formats):
    """Pick the preferred format; return (fmt, [(run, event, file)] or [file])."""
    from g4utils.DnaChem.tables import FORMAT_PREFERENCE

    directory = manifest_path.parent
    names = inventory if inventory is not None else tuple(p.name for p in directory.iterdir())
    portable = [n.replace("\\", "/") for n in names]
    base = {Path(n).name: n for n in portable}
    wanted = set(formats)
    for fmt in FORMAT_PREFERENCE:
        if fmt not in wanted:
            continue
        if fmt in ("h5", "root"):
            name = f"{prefix}PreChemical.{fmt}"
            if name in base:
                file = _relative_file(directory, base[name], str(manifest_path))
                if not file.is_file():
                    raise FileNotFoundError(f"Missing pre-chemical data file {file}")
                return fmt, [file]
            continue
        selected = {}
        for basename, entry in base.items():
            if not basename.startswith(prefix):
                continue
            match = _STD_EVENT_NAME.fullmatch(basename[len(prefix):])
            if match is None or match.group(3) != fmt:
                continue
            file = _relative_file(directory, entry, str(manifest_path))
            key = (int(match.group(1)), int(match.group(2)))
            if key in selected:
                raise ValueError(
                    f"{manifest_path}: duplicate pre-chemical (run,event) {key}: "
                    f"{selected[key]} and {file}"
                )
            if not file.is_file():
                raise FileNotFoundError(f"Missing pre-chemical data file {file}")
            selected[key] = file
        if selected:
            return fmt, [selected[k] for k in sorted(selected)]
    return None, []


def read_standardized_prechemical(
    manifest_path: Path, prefix: str, inventory, formats, *, allow_missing: bool = False,
) -> pd.DataFrame:
    """Read standardized PreChemical records from the preferred format."""
    from g4utils.DnaChem.tables import read_table_file

    fmt, files = _standardized_files(manifest_path, prefix, inventory, formats)
    if not files:
        if not allow_missing:
            raise FileNotFoundError(
                f"{manifest_path}: no pre-chemical file found for formats {sorted(formats)}. "
                "Regenerate the dump with pre-chemical output enabled, "
                "or use allow_missing=True for absent optional output."
            )
        frame = pd.DataFrame({c: [] for c in STANDARDIZED_PRECHEMICAL_COLUMNS})
        for column in STANDARDIZED_PRECHEMICAL_COLUMNS:
            if column in ("run", "event", "parentTrackId", "level"):
                frame[column] = frame[column].astype("int64")
            elif column in ("molecule", "modification"):
                frame[column] = frame[column].astype("string")
            else:
                frame[column] = frame[column].astype("float64")
        return frame
    frames = [read_table_file(f, fmt, "preChemical") for f in files]
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["run", "event"], kind="stable", ignore_index=True)

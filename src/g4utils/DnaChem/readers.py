"""Strict readers for the plain CSV reports written by dnachem-min."""

from __future__ import annotations

import csv
import math
import re
import warnings
from pathlib import Path

import pandas as pd


PHYSICAL_COLUMNS = ("label", "count")
MESO_SPECIES_COLUMNS = ("time_ns", "species", "count")
TRACK_COLUMNS = (
    "run", "event", "primaryLength_nm", "primaryEkin0_keV",
    "primaryEkinEnd_keV", "primaryEnd", "secondaryFirstGen_nm",
    "nSecondaryFirstGen", "secondaryAll_nm", "nSecondaryAll",
)
TRACK_INTEGER_COLUMNS = ("run", "event", "nSecondaryFirstGen", "nSecondaryAll")
TRACK_FLOAT_COLUMNS = (
    "primaryLength_nm", "primaryEkin0_keV", "primaryEkinEnd_keV",
    "secondaryFirstGen_nm", "secondaryAll_nm",
)
TRACK_ENDS = ("stopped", "escaped", "killed")
_INTEGER = re.compile(r"[+]?[0-9]+\Z")


def _rows(file: Path, columns: tuple[str, ...]) -> list[tuple[int, list[str]]]:
    """Read the writer's exact header and field count, including empty tables."""
    with file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader, None)
            if header != list(columns):
                raise ValueError(f"{file}: expected CSV header {','.join(columns)!r}; got {header!r}")
            rows = []
            for line, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise ValueError(
                        f"{file}:{line}: expected {len(columns)} CSV fields, got {len(row)}"
                    )
                rows.append((line, row))
            return rows
        except csv.Error as exc:
            raise ValueError(f"{file}: malformed CSV: {exc}") from exc


def _integer(value: str, file: Path, line: int, column: str) -> int:
    if not _INTEGER.fullmatch(value):
        raise ValueError(f"{file}:{line}: {column} must be a nonnegative integer: {value!r}")
    number = int(value)
    if number > 2**63 - 1:
        raise ValueError(f"{file}:{line}: {column} exceeds the supported int64 range: {value!r}")
    return number


def _nonnegative_float(value: str, file: Path, line: int, column: str) -> float:
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{file}:{line}: {column} must be a finite nonnegative number: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{file}:{line}: {column} must be a finite nonnegative number: {value!r}")
    return number


def read_physical_interactions(file: Path) -> pd.DataFrame:
    """Read raw process labels and dimensionless aggregate firing counts."""
    records = []
    for line, row in _rows(file, PHYSICAL_COLUMNS):
        if not row[0]:
            raise ValueError(f"{file}:{line}: label must not be empty")
        records.append((row[0], _integer(row[1], file, line, "count")))
    return pd.DataFrame(
        {"label": pd.Series((r[0] for r in records), dtype="string"),
         "count": pd.Series((r[1] for r in records), dtype="int64")}
    )


def read_meso_species(file: Path) -> pd.DataFrame:
    """Read summed molecule counts at mesoscopic record times (ns)."""
    records = []
    for line, row in _rows(file, MESO_SPECIES_COLUMNS):
        time = _nonnegative_float(row[0], file, line, "time_ns")
        if not row[1].strip():
            raise ValueError(f"{file}:{line}: species must not be empty")
        records.append((time, row[1], _integer(row[2], file, line, "count")))
    return pd.DataFrame(
        {"time_ns": pd.Series((r[0] for r in records), dtype="float64"),
         "species": pd.Series((r[1] for r in records), dtype="string"),
         "count": pd.Series((r[2] for r in records), dtype="int64")}
    )


def read_track_lengths(file: Path) -> pd.DataFrame:
    """Read one event per row, retaining the writer's nm and keV units."""
    records = []
    seen: set[tuple[int, int]] = set()
    for line, values in _rows(file, TRACK_COLUMNS):
        row = dict(zip(TRACK_COLUMNS, values))
        for column in TRACK_INTEGER_COLUMNS:
            row[column] = _integer(row[column], file, line, column)
        for column in TRACK_FLOAT_COLUMNS:
            row[column] = _nonnegative_float(row[column], file, line, column)
        if row["primaryEnd"] not in TRACK_ENDS:
            raise ValueError(f"{file}:{line}: unknown primaryEnd {row['primaryEnd']!r}")
        key = (row["run"], row["event"])
        if key in seen:
            raise ValueError(f"{file}:{line}: duplicate (run,event) {key}")
        seen.add(key)
        records.append(row)
    frame = pd.DataFrame(records, columns=TRACK_COLUMNS)
    for column in TRACK_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("int64")
    for column in TRACK_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    frame["primaryEnd"] = frame["primaryEnd"].astype("string")
    return frame


def _summary_integer(value: object, file: Path, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{file}: trackLengths.{key} must be a nonnegative integer")
    return value


def _summary_number(value: object, file: Path, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{file}: trackLengths.{key} must be finite and nonnegative")
    return float(value)


def _matches_writer(actual: float, expected: float) -> bool:
    # CSV rows are fixed to six decimals; manifest JSON uses 12 significant digits.
    return math.isclose(actual, expected, rel_tol=6e-12, abs_tol=6e-7)


def validate_track_summary(frame: pd.DataFrame, summary: object, file: Path) -> None:
    """Check all available writer summary fields against the parsed CSV rows."""
    if summary is None:
        return
    if not isinstance(summary, dict):
        raise ValueError(f"{file}: trackLengths summary must be an object")
    size = len(frame)
    expected_counts = {
        "events": size,
        "stopped": int((frame["primaryEnd"] == "stopped").sum()),
        "escaped": int((frame["primaryEnd"] == "escaped").sum()),
        "killed": int((frame["primaryEnd"] == "killed").sum()),
    }
    for key, expected in expected_counts.items():
        if key in summary and _summary_integer(summary[key], file, key) != expected:
            raise ValueError(f"{file}: trackLengths.{key} disagrees with CSV ({expected})")
    for column in (*TRACK_FLOAT_COLUMNS, "nSecondaryFirstGen", "nSecondaryAll"):
        if column not in summary:
            continue
        stat = summary[column]
        if not isinstance(stat, dict):
            raise ValueError(f"{file}: trackLengths.{column} must be an object")
        values = frame[column].astype("float64")
        mean = float(values.mean()) if size else None
        sem = float(values.std(ddof=1) / math.sqrt(size)) if size > 1 else (0.0 if size else None)
        for key, expected in (("mean", mean), ("sem", sem)):
            if key not in stat:
                continue
            actual = stat[key]
            if expected is None:
                if actual is not None:
                    raise ValueError(f"{file}: trackLengths.{column}.{key} must be null for no events")
            elif actual is None or not _matches_writer(_summary_number(actual, file, f"{column}.{key}"), expected):
                raise ValueError(f"{file}: trackLengths.{column}.{key} disagrees with CSV ({expected})")


def warn_track_event_count(frame: pd.DataFrame, total_events: object, file: Path) -> None:
    if total_events is not None and len(frame) != total_events:
        warnings.warn(
            f"{file}: {len(frame)} track rows differ from manifest totalEvents={total_events}; "
            "events may have been aborted or otherwise omitted",
            UserWarning,
            stacklevel=2,
        )

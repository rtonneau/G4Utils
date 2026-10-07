from __future__ import annotations

import dataclasses
import html
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from g4utils.DnaChem.dumps import MANIFEST_NAME, find_dumps, parse_dump_name

SUPPORTED_SCHEMA_VERSION = 1


def _tuple_or_none(value):
    return None if value is None else tuple(value)


def _split(data: dict, cls) -> tuple[dict, dict]:
    names = {f.name for f in dataclasses.fields(cls) if f.name != "raw"}
    known = {k: v for k, v in data.items() if k in names}
    unknown = {k: v for k, v in data.items() if k not in names}
    return known, unknown


@dataclass(frozen=True)
class Scavenger:
    """One entry of a Manifest's ``scavengers`` list."""

    species: str | None = None
    molarity_M: float | None = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> Scavenger:
        known, unknown = _split(data, cls)
        return cls(**known, raw=unknown)


@dataclass(frozen=True)
class ManifestRun:
    """One entry of a Manifest's ``runs`` list (a Manifest run)."""

    run: int | None = None
    events: int | None = None
    particle: str | None = None
    beamEnergy_keV: float | None = None
    position_um: tuple | None = None
    direction: tuple | None = None
    energyDeposit_eV: float | None = None
    seed: int | None = None
    wallTime_s: float | None = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> ManifestRun:
        known, unknown = _split(data, cls)
        for key in ("position_um", "direction"):
            known[key] = _tuple_or_none(known.get(key))
        return cls(**known, raw=unknown)


@dataclass(frozen=True)
class Manifest:
    """A parsed ``Manifest.json``; fields are the JSON keys."""

    schemaVersion: int | None = None
    timestamp: str | None = None
    elapsedSinceStart_s: float | None = None
    elapsedSincePreviousDump_s: float | None = None
    geant4Version: str | None = None
    macro: str | None = None
    chemistry: str | None = None
    scavengers: tuple[Scavenger, ...] = ()
    pH: float | None = None
    halfBox_um: float | None = None
    chemistryEndTime_ns: float | None = None
    runMode: str | None = None
    threads: int | None = None
    outputDirAsConfigured: str | None = None
    outputDirAbsolute: str | None = None
    prefix: str | None = None
    subdir: str | None = None
    totalEvents: int | None = None
    totalEnergyDeposit_eV: float | None = None
    files: tuple[str, ...] | None = None
    runs: tuple[ManifestRun, ...] = ()
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        known, unknown = _split(data, cls)
        known["scavengers"] = tuple(
            Scavenger.from_dict(s) for s in data.get("scavengers") or []
        )
        known["runs"] = tuple(ManifestRun.from_dict(r) for r in data.get("runs") or [])
        known["files"] = _tuple_or_none(known.get("files"))
        return cls(**known, raw=unknown)

    def runs_table(self) -> pd.DataFrame:
        """One row per Manifest run; unknown run keys become extra columns."""
        columns = [f.name for f in dataclasses.fields(ManifestRun) if f.name != "raw"]
        rows = []
        for r in self.runs:
            row = {c: getattr(r, c) for c in columns}
            row.update(r.raw)
            rows.append(row)
        return pd.DataFrame(rows, columns=None if rows else columns)

    def scavenger_molarity(self, species: str) -> float | None:
        """Molarity (M) of the scavenger ``species``, or None if absent."""
        for sc in self.scavengers:
            if sc.species == species:
                return sc.molarity_M
        return None

    def __str__(self) -> str:
        return _render_text(self)

    def _repr_html_(self) -> str:
        return _render_html(self)

    def show(self) -> None:
        """Display this Manifest in a notebook (requires IPython)."""
        try:
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "Manifest.show() requires IPython; install it with "
                "'pip install ipython'"
            ) from exc
        display(self)


def _fmt(value) -> str:
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def _present(pairs) -> list[tuple[str, str]]:
    return [(k, _fmt(v)) for k, v in pairs if v is not None]


_RUN_COLUMNS = [f.name for f in dataclasses.fields(ManifestRun) if f.name != "raw"]


def _summary(m: Manifest) -> dict:
    """Display content: key/value sections, runs table, files, unknown keys."""
    overview = _present(
        [
            ("schemaVersion", m.schemaVersion),
            ("timestamp", m.timestamp),
            ("elapsedSinceStart_s", m.elapsedSinceStart_s),
            ("elapsedSincePreviousDump_s", m.elapsedSincePreviousDump_s),
            ("geant4Version", m.geant4Version),
            ("macro", m.macro),
            ("runMode", m.runMode),
            ("threads", m.threads),
            ("outputDirAsConfigured", m.outputDirAsConfigured),
            ("outputDirAbsolute", m.outputDirAbsolute),
            ("prefix", m.prefix),
            ("subdir", m.subdir),
        ]
    )
    chemistry = _present(
        [
            ("chemistry", m.chemistry),
            ("pH", m.pH),
            ("halfBox_um", m.halfBox_um),
            ("chemistryEndTime_ns", m.chemistryEndTime_ns),
        ]
    )
    for sc in m.scavengers:
        value = "-" if sc.molarity_M is None else _fmt(sc.molarity_M)
        chemistry.append((f"{sc.species} molarity_M", value))
    totals = _present(
        [
            ("totalEvents", m.totalEvents),
            ("totalEnergyDeposit_eV", m.totalEnergyDeposit_eV),
        ]
    )
    other = [(k, _fmt(v)) for k, v in m.raw.items()]
    run_rows = []
    for r in m.runs:
        row = {c: _fmt(getattr(r, c)) for c in _RUN_COLUMNS if getattr(r, c) is not None}
        row.update({k: _fmt(v) for k, v in r.raw.items()})
        run_rows.append(row)
    run_cols = [c for c in _RUN_COLUMNS if any(c in row for row in run_rows)]
    for row in run_rows:
        for k in row:
            if k not in run_cols:
                run_cols.append(k)
    return {
        "Overview": overview,
        "Chemistry": chemistry,
        "Totals": totals,
        "run_cols": run_cols,
        "run_rows": run_rows,
        "files": list(m.files) if m.files else [],
        "Other": other,
    }


def _render_text(m: Manifest) -> str:
    s = _summary(m)
    out: list[str] = []

    def section(title, lines):
        if lines:
            out.extend([title, "-" * len(title), *lines, ""])

    def aligned(pairs):
        width = max((len(k) for k, _ in pairs), default=0)
        return [f"  {k.ljust(width)}  {v}" for k, v in pairs]

    for title in ("Overview", "Chemistry", "Totals"):
        section(title, aligned(s[title]))
    cols, rows = s["run_cols"], s["run_rows"]
    if rows:
        widths = [max(len(c), *(len(r.get(c, "")) for r in rows)) for c in cols]
        lines = ["  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))]
        for r in rows:
            lines.append(
                "  " + "  ".join(r.get(c, "").ljust(w) for c, w in zip(cols, widths))
            )
        section("Runs", [ln.rstrip() for ln in lines])
    section("Files", [f"  {f}" for f in s["files"]])
    section("Other", aligned(s["Other"]))
    return "\n".join(out).rstrip() + "\n"


def _render_html(m: Manifest) -> str:
    s = _summary(m)
    e = html.escape
    parts: list[str] = []

    def kv_table(title, pairs):
        if not pairs:
            return
        rows = "".join(
            f"<tr><th style='text-align:left'>{e(k)}</th><td>{e(v)}</td></tr>"
            for k, v in pairs
        )
        parts.append(f"<h4>{e(title)}</h4><table>{rows}</table>")

    for title in ("Overview", "Chemistry", "Totals"):
        kv_table(title, s[title])
    if s["run_rows"]:
        head = "".join(f"<th>{e(c)}</th>" for c in s["run_cols"])
        body = "".join(
            "<tr>"
            + "".join(f"<td>{e(r.get(c, ''))}</td>" for c in s["run_cols"])
            + "</tr>"
            for r in s["run_rows"]
        )
        parts.append(f"<h4>Runs</h4><table><tr>{head}</tr>{body}</table>")
    if s["files"]:
        items = "".join(f"<tr><td>{e(f)}</td></tr>" for f in s["files"])
        parts.append(
            f"<details><summary><b>Files</b> ({len(s['files'])})</summary>"
            f"<table>{items}</table></details>"
        )
    kv_table("Other", s["Other"])
    return "<div>" + "".join(parts) + "</div>"


def _parse_manifest_file(file: Path) -> Manifest:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{file}: manifest must be a JSON object")
    if "schemaVersion" not in data:
        raise ValueError(f"{file}: missing schemaVersion")
    version = data["schemaVersion"]
    if isinstance(version, (int, float)) and version > SUPPORTED_SCHEMA_VERSION:
        warnings.warn(
            f"{file}: schemaVersion {version} is newer than the supported "
            f"{SUPPORTED_SCHEMA_VERSION}; parsing anyway",
            UserWarning,
            stacklevel=3,
        )
    return Manifest.from_dict(data)


def read_manifest(path: str | Path) -> Manifest:
    """Read one Manifest from a Dump folder or a ``*Manifest.json`` file.

    A folder holding exactly one Dump (or only one ``*Manifest.json``) is
    accepted too.
    """
    p = Path(path).resolve()
    if p.is_file():
        return _parse_manifest_file(p)
    if not p.is_dir():
        raise FileNotFoundError(f"No manifest found at {p}")
    try:
        dumps = find_dumps(p)
    except FileNotFoundError:
        files = sorted(p.glob("*Manifest.json"))
        if not files:
            raise FileNotFoundError(f"No manifest (*Manifest.json) found in {p}")
        if len(files) > 1:
            raise ValueError(
                f"{p} holds several manifests ({', '.join(f.name for f in files)}); "
                "pass one file explicitly"
            )
        return _parse_manifest_file(files[0])
    if len(dumps) > 1:
        raise ValueError(
            f"{p} holds {len(dumps)} Dumps; use load_manifests to read them all"
        )
    return _parse_manifest_file(dumps[0] / MANIFEST_NAME)


def _common(runs: tuple[ManifestRun, ...], key: str, dump_name: str):
    """Value of ``key`` shared by all runs, else NaN with a warning."""
    all_values = [getattr(r, key) for r in runs]
    values = {v for v in all_values if v is not None}
    if len(values) == 1 and None not in all_values:
        return values.pop()
    if values:
        warnings.warn(
            f"Dump {dump_name!r}: runs disagree on {key}; using NaN",
            UserWarning,
            stacklevel=3,
        )
    return math.nan


def dump_columns(manifest: dict | Manifest, dump_name: str) -> dict:
    """Dump-level columns derived from a parsed ``Manifest.json``."""
    if isinstance(manifest, dict):
        manifest = Manifest.from_dict(manifest)
    cols: dict = {
        "dump": dump_name,
        "chemistry": manifest.chemistry,
        "pH": manifest.pH,
        "totalEvents": manifest.totalEvents,
        "totalEnergyDeposit_eV": manifest.totalEnergyDeposit_eV,
    }
    for sc in manifest.scavengers:
        cols[f"{sc.species}_molarity_M"] = sc.molarity_M
    cols["particle"] = _common(manifest.runs, "particle", dump_name)
    cols["beamEnergy_keV"] = _common(manifest.runs, "beamEnergy_keV", dump_name)
    return cols


def load_manifests(
    path: str | Path, name_pattern: str | re.Pattern | None = None
) -> pd.DataFrame:
    """One row per run across all Dumps found at ``path``."""
    rows: list[dict] = []
    for dump in find_dumps(path):
        manifest = json.loads((dump / MANIFEST_NAME).read_text(encoding="utf-8"))
        base = dump_columns(manifest, dump.name)
        base.update(parse_dump_name(dump.name, name_pattern))
        runs = manifest.get("runs") or []
        if not runs:
            rows.append(base)
            continue
        for run in runs:
            row = dict(base)
            for key, value in run.items():
                row["run" if key == "run" else f"run_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)

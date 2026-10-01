from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import pandas as pd

from g4utils.DnaChem.dumps import MANIFEST_NAME, find_dumps, parse_dump_name


def _common(runs: list[dict], key: str, dump_name: str):
    """Value of ``key`` shared by all runs, else NaN with a warning."""
    values = {r[key] for r in runs if key in r}
    if len(values) == 1 and all(key in r for r in runs):
        return values.pop()
    if values:
        warnings.warn(
            f"Dump {dump_name!r}: runs disagree on {key}; using NaN",
            UserWarning,
            stacklevel=3,
        )
    return math.nan


def dump_columns(manifest: dict, dump_name: str) -> dict:
    """Dump-level columns derived from a parsed ``Manifest.json``."""
    cols: dict = {
        "dump": dump_name,
        "chemistry": manifest.get("chemistry"),
        "pH": manifest.get("pH"),
        "totalEvents": manifest.get("totalEvents"),
        "totalEnergyDeposit_eV": manifest.get("totalEnergyDeposit_eV"),
    }
    for sc in manifest.get("scavengers") or []:
        cols[f"{sc['species']}_molarity_M"] = sc["molarity_M"]
    runs = manifest.get("runs") or []
    cols["particle"] = _common(runs, "particle", dump_name)
    cols["beamEnergy_keV"] = _common(runs, "beamEnergy_keV", dump_name)
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

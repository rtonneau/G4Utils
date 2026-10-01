from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from g4utils.DnaChem.dumps import MANIFEST_NAME, find_dumps, parse_dump_name
from g4utils.DnaChem.manifest import dump_columns
from g4utils.DnaChem.ntuple import read_ntuple
from g4utils.DnaChem.species_names import short_name

SPECIES_FILE = "Species_nt_species.csv"
REACTIONS_FILE = "Reactions_nt_reactions.csv"
REACTIONS_METADATA_FILE = "ReactionsMetadata.csv"


def _require(file: Path) -> Path:
    if not file.is_file():
        raise FileNotFoundError(f"Missing data file {file.name} ({file})")
    return file


def _load_table(
    path: str | Path,
    filename: str,
    name_pattern: str | re.Pattern | None,
    process: Callable[[pd.DataFrame, Path], pd.DataFrame],
) -> pd.DataFrame:
    """Read ``filename`` in every Dump, post-process, add Dump columns, concat."""
    frames: list[pd.DataFrame] = []
    for dump in find_dumps(path):
        df = process(read_ntuple(_require(dump / filename)), dump)
        manifest = json.loads((dump / MANIFEST_NAME).read_text(encoding="utf-8"))
        cols = dump_columns(manifest, dump.name) | parse_dump_name(
            dump.name, name_pattern
        )
        frames.append(df.assign(**cols))
    return pd.concat(frames, ignore_index=True)


def load_species(
    path: str | Path, name_pattern: str | re.Pattern | None = None
) -> pd.DataFrame:
    """Species counts and G values (per 100 eV) for every Dump at ``path``."""

    def process(df: pd.DataFrame, dump: Path) -> pd.DataFrame:
        df["species"] = df["speciesName"].map(short_name)
        df["time_s"] = df.pop("time") * 1e-9
        return df

    df = _load_table(path, SPECIES_FILE, name_pattern, process)
    df["G"] = df["number"] / (df["totalEnergyDeposit_eV"] / 100)
    return df


def load_reactions(
    path: str | Path, name_pattern: str | re.Pattern | None = None
) -> pd.DataFrame:
    """Reaction counts with labels for every Dump at ``path``."""

    def process(df: pd.DataFrame, dump: Path) -> pd.DataFrame:
        meta = _read_metadata(_require(dump / REACTIONS_METADATA_FILE))
        df = df.merge(meta, on="reactionId", how="left")
        df["time_s"] = df.pop("time") * 1e-9
        return df

    return _load_table(path, REACTIONS_FILE, name_pattern, process)


def _read_metadata(file: Path) -> pd.DataFrame:
    """Read ``ReactionsMetadata.csv`` (``reactionId,reaction``)."""
    return pd.read_csv(file, encoding="utf-8", keep_default_na=False)

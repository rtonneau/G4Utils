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


def _resolve_metadata_file(path: str | Path) -> Path:
    """Return the ``ReactionsMetadata.csv`` for a file, a Dump or a lone-Dump root."""
    root = Path(path)
    if root.is_file():
        return root
    if (root / REACTIONS_METADATA_FILE).is_file():
        return root / REACTIONS_METADATA_FILE
    dumps = find_dumps(root)  # raises FileNotFoundError if none
    if len(dumps) > 1:
        raise ValueError(f"Expected a single Dump at {root}, found {len(dumps)}")
    return _require(dumps[0] / REACTIONS_METADATA_FILE)


def _split_side(side: str) -> tuple[str, ...]:
    side = side.strip()
    if side == "(no products)":
        return ()
    return tuple(short_name(s) for s in re.split(r"\s+\+\s+", side))


def load_reaction_table(path: str | Path) -> pd.DataFrame:
    """One row per reaction, with Short-name equation and species count columns."""
    meta = _read_metadata(_resolve_metadata_file(path))
    reactants: list[tuple[str, ...]] = []
    products: list[tuple[str, ...]] = []
    for raw in meta["reaction"]:
        if "->" not in raw:
            raise ValueError(f"Malformed reaction line (no '->'): {raw!r}")
        left, right = raw.split("->", 1)
        reactants.append(_split_side(left))
        products.append(_split_side(right))

    equation = [
        f"{' + '.join(r)} -> {' + '.join(p) if p else '(no products)'}"
        for r, p in zip(reactants, products)
    ]
    species = sorted({s for side in (*reactants, *products) for s in side})
    wide = pd.DataFrame(
        {
            **{f"reactant_{x}": [r.count(x) for r in reactants] for x in species},
            **{f"product_{x}": [p.count(x) for p in products] for x in species},
        },
        dtype=int,
    )
    head = pd.DataFrame(
        {
            "reactionId": meta["reactionId"].to_numpy(),
            "reaction": meta["reaction"].to_numpy(),
            "equation": equation,
            "reactants": pd.Series(reactants, dtype=object),
            "products": pd.Series(products, dtype=object),
        }
    )
    return pd.concat([head, wide], axis=1)

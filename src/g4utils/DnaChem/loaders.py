from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from g4utils.DnaChem.dump import (
    REACTIONS_METADATA_FILE,
    _require,
    build_reaction_table,
    read_reactions_metadata,
)
from g4utils.DnaChem.simulation import Simulation


def load_species(
    path: str | Path, name_pattern: str | re.Pattern | None = None
) -> pd.DataFrame:
    """Species counts and G values (per 100 eV) for every Dump at ``path``."""
    sim = Simulation(path, name_pattern)
    return pd.concat([dump.species() for dump in sim], ignore_index=True)


def load_reactions(
    path: str | Path, name_pattern: str | re.Pattern | None = None
) -> pd.DataFrame:
    """Reaction counts with labels for every Dump at ``path``."""
    sim = Simulation(path, name_pattern)
    return pd.concat([dump.reactions() for dump in sim], ignore_index=True)


def load_reaction_table(path: str | Path) -> pd.DataFrame:
    """One row per reaction, with Short-name equation and species count columns.

    ``path`` is a ``ReactionsMetadata.csv`` file, a Dump, or a folder holding
    exactly one Dump.
    """
    root = Path(path)
    if root.is_file():
        return build_reaction_table(read_reactions_metadata(root))
    if (root / REACTIONS_METADATA_FILE).is_file():
        return build_reaction_table(
            read_reactions_metadata(root / REACTIONS_METADATA_FILE)
        )
    sim = Simulation(root)  # raises FileNotFoundError if no Dump
    if len(sim) > 1:
        raise ValueError(f"Expected a single Dump at {root}, found {len(sim)}")
    dump = next(iter(sim))
    _require(dump.path / REACTIONS_METADATA_FILE)
    return dump.reaction_table()

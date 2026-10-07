from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from g4utils.DnaChem.dumps import MANIFEST_NAME, parse_dump_name
from g4utils.DnaChem.manifest import Manifest, _parse_manifest_file, dump_columns
from g4utils.DnaChem.ntuple import read_ntuple
from g4utils.DnaChem.species_names import short_name

if TYPE_CHECKING:
    from g4utils.HDF5 import SpeciesMesoSpatialFile

SPECIES_FILE = "Species_nt_species.csv"
REACTIONS_FILE = "Reactions_nt_reactions.csv"
REACTIONS_METADATA_FILE = "ReactionsMetadata.csv"
MESO_FILE = "SpeciesMesoSpatial.h5"


def _require(file: Path) -> Path:
    if not file.is_file():
        raise FileNotFoundError(f"Missing data file {file.name} ({file})")
    return file


class Dump:
    """One dnachem-min Dump folder, read lazily.

    Construction only checks that the folder holds a ``Manifest.json``;
    everything else loads on first access and is cached.
    """

    def __init__(
        self, path: str | Path, name_pattern: str | re.Pattern | None = None
    ) -> None:
        self.path = Path(path)
        if not (self.path / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"No Dump ({MANIFEST_NAME}) found at {self.path}"
            )
        self.name_pattern = name_pattern
        self._manifest: Manifest | None = None
        self._species: pd.DataFrame | None = None
        self._reactions: pd.DataFrame | None = None
        self._reaction_table: pd.DataFrame | None = None
        self._meso: SpeciesMesoSpatialFile | None = None

    @property
    def name(self) -> str:
        return self.path.resolve().name

    @property
    def labels(self) -> dict[str, float | str]:
        """Values extracted from the folder name by the Name pattern."""
        return parse_dump_name(self.name, self.name_pattern)

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = _parse_manifest_file(self.path / MANIFEST_NAME)
        return self._manifest

    def _columns(self) -> dict:
        return dump_columns(self.manifest, self.name) | self.labels

    def species(self) -> pd.DataFrame:
        """Species counts and G values (per 100 eV)."""
        if self._species is None:
            df = read_ntuple(_require(self.path / SPECIES_FILE))
            df["species"] = df["speciesName"].map(short_name)
            df["time_s"] = df.pop("time") * 1e-9
            df = df.assign(**self._columns())
            df["G"] = df["number"] / (df["totalEnergyDeposit_eV"] / 100)
            self._species = df
        return self._species

    def reactions(self) -> pd.DataFrame:
        """Reaction counts with labels."""
        if self._reactions is None:
            df = read_ntuple(_require(self.path / REACTIONS_FILE))
            meta = self._read_metadata()
            df = df.merge(meta, on="reactionId", how="left")
            df["time_s"] = df.pop("time") * 1e-9
            self._reactions = df.assign(**self._columns())
        return self._reactions

    def reaction_table(self) -> pd.DataFrame:
        """One row per reaction, with Short-name equation and species counts."""
        if self._reaction_table is None:
            from g4utils.DnaChem.loaders import load_reaction_table

            _require(self.path / REACTIONS_METADATA_FILE)
            self._reaction_table = load_reaction_table(self.path)
        return self._reaction_table

    @property
    def meso(self) -> SpeciesMesoSpatialFile:
        if self._meso is None:
            from g4utils.HDF5 import SpeciesMesoSpatialFile

            self._meso = SpeciesMesoSpatialFile(_require(self.path / MESO_FILE))
        return self._meso

    def _read_metadata(self) -> pd.DataFrame:
        file = _require(self.path / REACTIONS_METADATA_FILE)
        return pd.read_csv(file, encoding="utf-8", keep_default_na=False)

    def __repr__(self) -> str:
        return f"Dump({str(self.path)!r})"

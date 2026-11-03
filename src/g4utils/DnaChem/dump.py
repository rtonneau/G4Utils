from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from g4utils.DnaChem.dumps import MANIFEST_NAME, _relative_file, parse_dump_name
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


def read_reactions_metadata(file: Path) -> pd.DataFrame:
    """Read ``ReactionsMetadata.csv`` (``reactionId,reaction``)."""
    return pd.read_csv(file, encoding="utf-8", keep_default_na=False)


def _split_side(side: str) -> tuple[str, ...]:
    side = side.strip()
    if side == "(no products)":
        return ()
    return tuple(short_name(s) for s in re.split(r"\s+\+\s+", side))


def build_reaction_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per reaction, with Short-name equation and species count columns."""
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


class Dump:
    """One dnachem-min output flush, selected by its manifest and read lazily.

    Construction only checks that the folder holds a ``Manifest.json``;
    everything else loads on first access and is cached.
    """

    def __init__(
        self, path: str | Path, name_pattern: str | re.Pattern | None = None
    ) -> None:
        self._initialize(Path(path) / MANIFEST_NAME, name_pattern)

    @classmethod
    def from_manifest(
        cls, manifest_path: str | Path, name_pattern: str | re.Pattern | None = None
    ) -> Dump:
        """Select one manifest explicitly, including a prefixed manifest."""
        dump = cls.__new__(cls)
        dump._initialize(Path(manifest_path), name_pattern)
        return dump

    def _initialize(self, manifest_path: Path, name_pattern) -> None:
        self.path = manifest_path.parent
        self.manifest_path = manifest_path.resolve()
        self.key = self.manifest_path.name
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"No Dump manifest found at {self.manifest_path}"
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
            self._manifest = _parse_manifest_file(self.manifest_path)
        return self._manifest

    @property
    def files(self) -> tuple[str, ...]:
        """File names listed in the Manifest ``files`` (empty if absent)."""
        return tuple(self.manifest.files or ())

    @property
    def prefix(self) -> str:
        """Filename prefix of the Dump's data files (empty if absent or null)."""
        return self.manifest.prefix or ""

    def _data_file(self, filename: str) -> Path:
        """Resolve a listed file first, then the conventional prefixed filename.

        Listed paths are relative to the selected manifest, never outputDir.
        Their existence does not affect selection: a missing listed file must
        not fall back to a different dump's file.
        """
        portable = filename.replace("\\", "/")
        # Validate the request even when a basename might match the inventory.
        _relative_file(self.manifest_path.parent, portable, str(self.manifest_path))
        requested = Path(portable)
        name = requested.name
        # The argument is a conventional filename, not a prefix heuristic:
        # even prefix="S" must turn Species... into SSpecies.... Already
        # prefixed inventory entries are selected verbatim below.
        prefixed = f"{self.prefix}{name}"
        matches = []
        for listed in self.files:
            listed = listed.replace("\\", "/")
            if Path(listed).name in (name, prefixed):
                matches.append(listed)
        if len(matches) > 1:
            raise ValueError(
                f"{self.manifest_path}: ambiguous files for {filename!r}: {matches}"
            )
        selected = matches[0] if matches else requested.with_name(prefixed).as_posix()
        return self._listed_file(selected)

    def _listed_file(self, filename: str) -> Path:
        """Resolve a literal inventory path without adding any prefix."""
        return _relative_file(self.manifest_path.parent, filename, str(self.manifest_path))

    def _provenance(self) -> dict:
        """Columns shared by dump readers, retaining the legacy folder label."""
        return {
            "dump": self.name,
            "dump_key": self.key,
            "manifest_path": str(self.manifest_path),
            "prefix": self.prefix,
        }

    @property
    def has_meso(self) -> bool:
        """True if meso output was not disabled and the meso h5 file exists."""
        return (
            self.manifest.mesoSpatialOutput is not False
            and self._data_file(MESO_FILE).is_file()
        )

    def _columns(self) -> dict:
        return dump_columns(self.manifest, self.name) | self.labels | self._provenance()

    def species(self) -> pd.DataFrame:
        """Species counts and G values (per 100 eV)."""
        if self._species is None:
            df = read_ntuple(_require(self._data_file(SPECIES_FILE)))
            df["species"] = df["speciesName"].map(short_name)
            df["time_s"] = df.pop("time") * 1e-9
            df = df.assign(**self._columns())
            df["G"] = df["number"] / (df["totalEnergyDeposit_eV"] / 100)
            self._species = df
        return self._species

    def reactions(self) -> pd.DataFrame:
        """Reaction counts with labels."""
        if self._reactions is None:
            df = read_ntuple(_require(self._data_file(REACTIONS_FILE)))
            meta = self._read_metadata()
            df = df.merge(meta, on="reactionId", how="left")
            df["time_s"] = df.pop("time") * 1e-9
            self._reactions = df.assign(**self._columns())
        return self._reactions

    def reaction_table(self) -> pd.DataFrame:
        """One row per reaction, with Short-name equation and species counts."""
        if self._reaction_table is None:
            self._reaction_table = build_reaction_table(self._read_metadata())
        return self._reaction_table

    @property
    def meso(self) -> SpeciesMesoSpatialFile:
        if self._meso is None:
            from g4utils.HDF5 import SpeciesMesoSpatialFile

            if self.manifest.mesoSpatialOutput is False:
                raise FileNotFoundError(
                    f"Meso output was disabled in the Manifest "
                    f"(mesoSpatialOutput=false) for {self.path}"
                )
            file = self._data_file(MESO_FILE)
            if not file.is_file():
                raise FileNotFoundError(
                    f"Meso file {file.name} is missing ({file})"
                )
            self._meso = SpeciesMesoSpatialFile(file)
        return self._meso

    def _read_metadata(self) -> pd.DataFrame:
        return read_reactions_metadata(_require(self._data_file(REACTIONS_METADATA_FILE)))

    def __repr__(self) -> str:
        if self.manifest_path.name != MANIFEST_NAME:
            return f"Dump.from_manifest({str(self.manifest_path)!r})"
        return f"Dump({str(self.path)!r})"

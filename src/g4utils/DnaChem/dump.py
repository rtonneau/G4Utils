from __future__ import annotations

import re
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from g4utils.DnaChem.dumps import MANIFEST_NAME, _relative_file, parse_dump_name
from g4utils.DnaChem.estimates import (
    particle_g_estimates, standardized_g_estimates, validate_total_energy,
)
from g4utils.DnaChem.manifest import Manifest, _parse_manifest_file, dump_columns
from g4utils.DnaChem.ntuple import read_ntuple
from g4utils.DnaChem.prechemical import read_prechemical_records
from g4utils.DnaChem.readers import (
    read_meso_species, read_physical_interactions, read_track_lengths, validate_track_summary,
    warn_track_event_count,
)
from g4utils.DnaChem.species_names import short_name
from g4utils.DnaChem.tables import pick_format, read_table_file

if TYPE_CHECKING:
    from g4utils.HDF5 import SpeciesMesoSpatialFile

SPECIES_FILE = "Species_nt_species.csv"
REACTIONS_FILE = "Reactions_nt_reactions.csv"
REACTIONS_METADATA_FILE = "ReactionsMetadata.csv"
MESO_FILE = "SpeciesMesoSpatial.h5"
MESO_SPECIES_FILE = "SpeciesMeso.csv"
PHYSICAL_INTERACTIONS_FILE = "PhysicsInteractions.csv"
TRACK_LENGTHS_FILE = "TrackLengths.csv"


# Statistics the legacy files never stored; legacy frames carry them as NaN.
_STAT_COLUMNS = ("countSumSq", "countStdErr", "gSum", "gSumSq", "gMean", "gStdErr")


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

    @property
    def output_formats(self) -> dict[str, tuple[str, ...]]:
        """Manifest ``outputFormats`` per product (empty for a legacy dump)."""
        return dict(self.manifest.outputFormats or {})

    @property
    def _layout(self) -> str:
        """``"standardized"`` iff the manifest lists ``outputFormats``."""
        return "standardized" if self.manifest.outputFormats is not None else "legacy"

    def _product_formats(self, product: str) -> tuple[str, ...]:
        """Formats written for ``product``; raise when it was disabled."""
        if self._layout == "legacy":
            return ()
        formats = self.manifest.outputFormats.get(product)
        if not formats:
            raise FileNotFoundError(
                f"Product {product!r} was not written: outputFormats lists no "
                f"format for it in {self.manifest_path}"
            )
        return formats

    def _table_location(self, stem: str, product: str, container: str | None = None):
        """Format and path of the preferred file of a standardized product."""
        formats = self._product_formats(product)
        listed = [f.replace("\\", "/") for f in self.files]
        directory = self.manifest_path.parent
        prefixed = f"{self.prefix}{stem}"
        for entry in listed:
            if Path(entry).name.startswith(prefixed + "."):
                directory = self._listed_file(Path(entry).parent.as_posix())
                break
        inventory = [Path(f).name for f in listed] if listed else None
        return pick_format(
            directory, prefixed, formats, inventory,
            container=f"{self.prefix}{container}" if container else None,
        )

    def _read_standardized(
        self, stem: str, product: str, table: str, container: str | None = None
    ):
        fmt, path = self._table_location(stem, product, container)
        return read_table_file(path, fmt, table)

    def _standardized_available(self, stem: str, product: str) -> bool:
        try:
            self._table_location(stem, product)
        except FileNotFoundError:
            return False
        return True

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

    def _meso_species_enabled(self) -> bool:
        # Modern manifests supply a validated boolean. Historical manifests
        # qualify only by the writer's explicit combined model name.
        enabled = self.manifest.mesoEnabled
        return enabled if enabled is not None else (
            self.manifest.chemistryModel == "IRT_syn+mesoscopic"
        )

    @property
    def has_meso_species(self) -> bool:
        """True when the mesoscopic stage is enabled and its aggregate CSV exists."""
        if not self._meso_species_enabled():
            return False
        if self._layout == "standardized":
            return self._standardized_available("SpeciesMeso", "speciesMeso")
        return self._data_file(MESO_SPECIES_FILE).is_file()

    def meso_species(self) -> pd.DataFrame:
        """Summed molecule counts and mean counts per event; these are not G values.

        Preserve raw species names, ns record times and manifest stage times.
        Availability is independent of the spatial HDF5 output.
        """
        if not self._meso_species_enabled():
            raise FileNotFoundError(
                f"Mesoscopic species output is disabled or not enabled by "
                f"{self.manifest_path} (mesoEnabled or legacy chemistryModel)"
            )
        metadata = {
            "handOverTime_ns": self.manifest.handOverTime_ns,
            "chemistryEndTime_ns": self.manifest.chemistryEndTime_ns,
        }
        if self._layout == "standardized":
            frame = self._read_standardized("SpeciesMeso", "speciesMeso", "speciesMeso")
            frame["count"] = frame["countSum"].astype("int64")
            frame["mean_count_per_event"] = frame["countMean"]
            return self._with_columns(frame, metadata)
        file = self._data_file(MESO_SPECIES_FILE)
        if not file.is_file():
            raise FileNotFoundError(f"Missing expected mesoscopic species CSV {file}")
        total = self.manifest.totalEvents
        if (isinstance(total, bool) or not isinstance(total, (int, float))
                or not math.isfinite(total) or total <= 0):
            raise ValueError(
                f"{self.manifest_path}: totalEvents must be finite and positive "
                "for mean_count_per_event"
            )
        frame = read_meso_species(file)
        frame["mean_count_per_event"] = (frame["count"] / total).astype("float64")
        frame["nEvents"] = int(total)
        frame["countSum"] = frame["count"].astype("float64")
        frame["countMean"] = frame["mean_count_per_event"]
        for column in _STAT_COLUMNS:
            frame[column] = math.nan
        return self._with_columns(frame, metadata)

    def _columns(self, extra: dict | None = None) -> dict:
        metadata = dump_columns(self.manifest, self.name) | self._provenance() | (extra or {})
        labels = self.labels
        self._check_labels(labels.keys() & metadata.keys())
        return metadata | labels

    def _check_labels(self, collisions: set) -> None:
        if collisions:
            raise ValueError(
                f"{self.manifest_path}: name-pattern labels conflict with reader columns: "
                f"{sorted(collisions)}; choose distinct label names"
            )

    def _with_columns(self, frame: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
        columns = self._columns(extra)
        self._check_labels(self.labels.keys() & set(frame.columns))
        return frame.assign(**columns)

    def species(self) -> pd.DataFrame:
        """Species counts and legacy aggregate-energy ``G`` (per 100 eV)."""
        if self._species is None:
            if self._layout == "standardized":
                self._species = self._standardized_species()
            else:
                df = read_ntuple(_require(self._data_file(SPECIES_FILE)))
                df["species"] = df["speciesName"].map(short_name)
                df["time_ns"] = df["time"]
                df["time_s"] = df.pop("time") * 1e-9
                df = self._with_columns(df)
                df["G"] = df["number"] / (df["totalEnergyDeposit_eV"] / 100)
                self._species = self._map_legacy_species(df)
        return self._species

    def _standardized_species(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        if frame is None:
            frame = self._read_standardized("Species", "species", "species")
        frame["speciesName"] = frame["species"].astype(object)
        frame["species"] = frame["speciesName"].map(short_name)
        frame["time_s"] = frame["time_ns"] * 1e-9
        frame = self._with_columns(frame)
        frame["G"] = frame["gMean"]
        return frame

    @staticmethod
    def _map_legacy_species(df: pd.DataFrame) -> pd.DataFrame:
        """Add the standardized columns to a legacy frame; legacy ones stay."""
        df["nEvents"] = df["nEvent"]
        df["countSum"] = df["number"].astype("float64")
        df["countMean"] = df["countSum"] / df["nEvents"]
        df["gSum"] = df["sumG"]
        df["gSumSq"] = df["sumG2"]
        df["gMean"] = df["sumG"] / df["nEvents"]
        df["countSumSq"] = math.nan
        df["countStdErr"] = math.nan
        df["gStdErr"] = math.nan
        return df

    def particle_g_estimates(self, *, assume_valid_event_g: bool = False) -> pd.DataFrame:
        """Particle G estimators (molecules per 100 eV) with optional SEM.

        The opt-in asserts equal valid-event coverage of both moments,
        including zero values, and positive per-event energy deposits. These
        facts and event independence cannot be established from the CSV.
        """
        energy = validate_total_energy(
            self.manifest.totalEnergyDeposit_eV, self.manifest_path
        )
        if self._layout == "standardized":
            fmt, file = self._table_location("Species", "species")
            frame = standardized_g_estimates(
                read_table_file(file, fmt, "species"), energy, file
            )
            return self._standardized_species(frame)
        file = _require(self._data_file(SPECIES_FILE))
        try:
            frame = read_ntuple(file)
        except ValueError as exc:
            raise ValueError(f"{file}: {exc}") from exc
        frame = particle_g_estimates(
            frame, energy, file, assume_valid_event_g=assume_valid_event_g
        )
        frame["species"] = frame["speciesName"].map(short_name)
        frame["time_ns"] = frame["time"]
        frame["time_s"] = frame.pop("time") * 1e-9
        frame = self._with_columns(frame)
        frame["G"] = frame["aggregate_energy_G"]
        return self._map_legacy_species(frame)

    def reactions(self) -> pd.DataFrame:
        """Reaction counts with labels.

        New dumps give one row per reaction and time bin (``binStart_ns``,
        ``binEnd_ns`` and the per-event count statistics). Legacy dumps are
        mapped onto the same columns, with the single record time as
        ``binEnd_ns`` and statistics the old file lacked as NaN. ``time_s``
        is the bin end (the record time for legacy dumps) in seconds and
        ``count`` the total firings (``countSum``).
        """
        if self._reactions is None:
            if self._layout == "standardized":
                df = self._read_standardized("Reactions", "reactions", "reactions")
                df = df.merge(self._read_metadata(), on="reactionId", how="left")
                df["count"] = df["countSum"].astype("int64")
                df["time_s"] = df["binEnd_ns"] * 1e-9
            else:
                df = read_ntuple(_require(self._data_file(REACTIONS_FILE)))
                meta = self._read_metadata()
                df = df.merge(meta, on="reactionId", how="left")
                df["binStart_ns"] = math.nan
                df["binEnd_ns"] = df["time"].astype("float64")
                df["time_s"] = df.pop("time") * 1e-9
                total = self.manifest.totalEvents
                valid = (
                    not isinstance(total, bool) and isinstance(total, (int, float))
                    and math.isfinite(total) and total > 0
                )
                df["nEvents"] = int(total) if valid else math.nan
                df["countSum"] = df["count"].astype("float64")
                df["countMean"] = df["countSum"] / total if valid else math.nan
                df["countSumSq"] = math.nan
                df["countStdErr"] = math.nan
            self._reactions = self._with_columns(df)
        return self._reactions

    def physical_interactions(self, *, per_event: bool = False) -> pd.DataFrame:
        """Raw process firing totals; optionally add mean firings per event.

        ``count`` is dimensionless. ``count_per_event`` divides by the
        manifest's finite positive ``totalEvents`` only when requested.
        """
        file = _require(self._data_file(PHYSICAL_INTERACTIONS_FILE))
        if per_event:
            total = self.manifest.totalEvents
            if (isinstance(total, bool) or not isinstance(total, (int, float))
                    or not math.isfinite(total) or total <= 0):
                raise ValueError(
                    f"{self.manifest_path}: totalEvents must be finite and positive "
                    "for count_per_event"
                )
        frame = self._with_columns(read_physical_interactions(file))
        if per_event:
            frame["count_per_event"] = (frame["count"] / total).astype("float64")
        return frame

    def track_lengths(self) -> pd.DataFrame:
        """One non-aborted event per row; lengths in nm, energies in keV."""
        file = _require(self._data_file(TRACK_LENGTHS_FILE))
        frame = read_track_lengths(file)
        validate_track_summary(frame, self.manifest.raw.get("trackLengths"), file)
        warn_track_event_count(frame, self.manifest.totalEvents, file)
        return self._with_columns(frame)

    def prechemical_records(self, *, allow_missing: bool = False) -> pd.DataFrame:
        """Pre-chemical records in nm/eV, retaining raw codes and null positions.

        Use the manifest inventory when present; otherwise scan this directory
        for this literal prefix. Listed missing files always raise, even when
        absent optional output is allowed with ``allow_missing=True``.
        """
        frame = read_prechemical_records(
            self.manifest_path, self.prefix, self.manifest.files,
            allow_missing=allow_missing,
        )
        return self._with_columns(frame)

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
        if self._layout == "standardized":
            return self._read_standardized(
                "ReactionsCatalog", "reactions", "reactionCatalog", container="Reactions"
            )
        return read_reactions_metadata(_require(self._data_file(REACTIONS_METADATA_FILE)))

    def __repr__(self) -> str:
        if self.manifest_path.name != MANIFEST_NAME:
            return f"Dump.from_manifest({str(self.manifest_path)!r})"
        return f"Dump({str(self.path)!r})"

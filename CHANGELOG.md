# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 0.5.0 (2026-11-05)

**Breaking.** `g4utils.DnaChem` reads the standardized dnachem-min output products. Legacy dumps still load.

- A manifest with `outputFormats` selects the new layout; products are read from csv, h5, root or txt (preference csv > h5 > root > txt). `.root` needs the new optional extra `g4utils[root]` (`uproot`).
- File map: `Species_nt_species.csv` → `Species.<fmt>`; `Reactions_nt_reactions.csv` → `Reactions.<fmt>`; `ReactionsMetadata.csv` → `ReactionsCatalog.<fmt>` (`Reactions.h5`/`.root` hold both tables); `PhysicsInteractions`, `TrackLengths`, `SpeciesMeso` and `PreChemical` gain csv/h5/root/txt forms.
- Column map: `TrackLengths` `primaryEkin0_keV` → `primaryEkinStart_keV`, `secondaryFirstGen_nm` → `secondaryFirstGenLength_nm`, `secondaryAll_nm` → `secondaryAllLength_nm`; `PreChemical` `parent_id` → `parentTrackId`, `electron_modification` → `modification`, new `level`, `parent_x_nm` → `parentX_nm` (and y, z).
- `species()["G"]` is the stored per-event `gMean` for new dumps. New per-event statistics columns (NaN for legacy dumps); `reactions()` has `binStart_ns`/`binEnd_ns`.
- `Manifest` gains `hdf5` and `outputFormats`.

## 0.4.0 (2026-10-06)

- Add `SpeciesMesoSpatialFile` to `g4utils.HDF5`, a lazy reader for the `SpeciesMesoSpatial.h5` files dnachem-min writes: list runs, events and snapshots, and read a snapshot's cell positions and per-species counts on demand.
- Add `concentration_M` (function and `MesoSpatialSnapshot` method) to convert counts to mol/L, optionally for selected species.

## 0.3.0 (2026-10-06)

- `g4utils.DnaChem.load_reaction_table(path)`: one row per reaction from `ReactionsMetadata.csv`, with a Short-name `equation`, `reactants`/`products` tuples and `reactant_<X>`/`product_<X>` count columns for filtering (e.g. `df[df["product_H2O2"] > 0]`).
- Short names now cover `O^-1` → `O-` and `O_3^-1` → `O3-` (also in `load_species`).

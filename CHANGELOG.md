# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 0.4.0 (2026-10-06)

- Add `SpeciesMesoSpatialFile` to `g4utils.HDF5`, a lazy reader for the `SpeciesMesoSpatial.h5` files dnachem-min writes: list runs, events and snapshots, and read a snapshot's cell positions and per-species counts on demand.
- Add `concentration_M` (function and `MesoSpatialSnapshot` method) to convert counts to mol/L, optionally for selected species.

## 0.3.0 (2026-10-06)

- `g4utils.DnaChem.load_reaction_table(path)`: one row per reaction from `ReactionsMetadata.csv`, with a Short-name `equation`, `reactants`/`products` tuples and `reactant_<X>`/`product_<X>` count columns for filtering (e.g. `df[df["product_H2O2"] > 0]`).
- Short names now cover `O^-1` → `O-` and `O_3^-1` → `O3-` (also in `load_species`).

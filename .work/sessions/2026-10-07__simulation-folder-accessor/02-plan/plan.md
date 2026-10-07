# Implementation Plan

**Session:** simulation-folder-accessor
**Date:** 2026-10-07T20:48:37.806Z
**Estimated effort:** 3 hours

## Strategy

Build the lazy `Dump` first (one folder, cached accessors), then `Simulation` (folder discovery and the mapping of Dumps), then rebuild the `load_*` functions on them and export and document the new API. Each ticket is committable and keeps the existing tests green.

## Tickets Overview

- **Ticket 1:** `Dump` class: lazy, cached `.manifest`, `.species()`, `.reactions()`, `.reaction_table()`, `.meso`, `.labels`.
- **Ticket 2:** `Simulation` class: finds Dumps in `results/` (else the folder), `subruns` mapping, `subrun(name)`, iteration, `table()`.
- **Ticket 3:** Rebuild `load_species` / `load_reactions` on `Simulation`/`Dump` with identical output; export and document in the README.

## Sequencing Rationale

`Simulation` holds `Dump` objects; the loaders can only be rebuilt once both exist, and the export and docs come last.

## Risks & Mitigation

- **Risk:** rebuilding the loaders changes their DataFrames (column order, dtypes). → **Mitigation:** move the existing processing code unchanged into `Dump` and keep the current loader tests as the guard.
- **Risk:** circular import between `loaders.py` and the new module. → **Mitigation:** put the table-processing helpers in the new module and have `loaders.py` call into it.

## Assumptions

- A Dump folder always has its own `Manifest.json`; the main folder has none.
- `SpeciesMesoSpatialFile` is importable from `g4utils.HDF5` and opened only when `.meso` is accessed.

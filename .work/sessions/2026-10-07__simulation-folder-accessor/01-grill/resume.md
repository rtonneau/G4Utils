# Session: simulation-folder-accessor

**Date:** 2026-10-07T19:12:02.602Z
**Status:** Grill phase complete

## Problem Statement

A dnachem-min simulation folder (e.g. `DATA/Geant4/261007/`) holds a `results/` folder with several Dump subfolders (one per scan level, e.g. `run_0p5pO2/`), each with its own `Manifest.json`. Today users call `load_species(path)` / `load_reactions(path)` and get everything eagerly. There is no object for the folder that makes the Dumps and their data easy to reach one by one and loaded on demand, to start mining.

## Context & Constraints

- **Current behavior:** `find_dumps(path)` returns Dump folders; `load_species`, `load_reactions`, `load_reaction_table` read every Dump eagerly; `read_manifest` / `load_manifests` parse Manifests; `SpeciesMesoSpatialFile` reads the meso h5. No object ties these together.
- **Pain point:** no handle on the Simulation folder; users re-read tables and hand-pick Dump paths.
- **Dependencies:** keep `find_dumps`, `Manifest`, `read_manifest`, `read_ntuple`, `SpeciesMesoSpatialFile`, and the exact DataFrames returned by `load_species` / `load_reactions`; existing tests must pass unchanged. Flat Dump folders (Manifest alone, as in tests/data) must keep working.
- **Tech stack:** Python, pandas, h5py, pytest; package `g4utils.DnaChem`.

## Success Metrics

- `Simulation(path)` on a folder like `261007/` lists its Dumps without reading any data file.
- `sim.subruns[name]` / `sim.subrun(name)` returns a `Dump` whose `.manifest`, `.species()`, `.reactions()`, `.reaction_table()` and `.meso` load on first access and are cached.
- `load_species` / `load_reactions` are rebuilt on `Simulation`/`Dump` and return identical DataFrames; existing tests pass unchanged.
- Flat Dump folders and a results folder passed directly still work.

## Architecture & Approach

New module in `g4utils.DnaChem`:
- `Simulation(path, name_pattern=None)`: finds Dumps in `path/results/` if it exists, else the folder itself, via `find_dumps`. `sim.subruns` is an ordered mapping folder name to `Dump`; `sim.subrun(name)`; iteration yields Dumps; `sim.table()` returns a small DataFrame of Dump name, labels and Manifest summary.
- `Dump`: one Dump folder, lazy and cached: `.manifest`, `.species()`, `.reactions()`, `.reaction_table()`, `.meso` (a `SpeciesMesoSpatialFile`), `.labels` from the Name pattern.
- `load_species` / `load_reactions` are re-implemented on top of these with the same output. Both classes are exported from `g4utils.DnaChem` and documented in the README. DnaChem only; no Vox files.

## Assumptions & Trade-offs

- The main folder holds no Manifest; each Dump subfolder has its own. "Subrun" in the API means a Dump of the Simulation folder.
- Not doing: a separate Scan object, Vox HDF5 access, prefix-style dumps written into the results root.
- Glossary updated: **Simulation folder** and **Dump subrun** in `CONTEXT.md`.

## Open Questions

- None open.

## Notes

The user first described Manifest-in-main-folder with subrun directories; the example tree showed each Dump has its own Manifest, so the design follows the tree.

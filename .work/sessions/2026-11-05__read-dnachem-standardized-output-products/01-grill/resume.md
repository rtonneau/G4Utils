# Session: read-dnachem-standardized-output-products

**Date:** 2026-11-05T20:26:09.477Z
**Status:** Grill phase complete

## Problem Statement

dnachem-min (ADR 0009, commits c967635..1a2b58d, 2026-11-04) replaced its ad hoc output files with standardized Output products: one table schema per product, files named `<Product>.<ext>`, formats `txt`/`csv`/`h5`/`root` chosen per product (`/output/<product>/format`, built-in default `csv`). `g4utils.DnaChem` still reads only the legacy files (`Species_nt_species.csv` wcsv ntuple, `Reactions_nt_reactions.csv`, `ReactionsMetadata.csv`, old `PhysicsInteractions.csv`/`SpeciesMeso.csv`/`TrackLengths.csv` columns, fixed-width `PreChemical_*.txt`), so it cannot load any dump from the current sim.

## Context & Constraints

- **Current behavior:** `Dump.species()` calls `read_ntuple` on `Species_nt_species.csv` (columns `number, nEvent, speciesName, time, sumG, sumG2`); `Dump.reactions()` joins `Reactions_nt_reactions.csv` to `ReactionsMetadata.csv`; `physical_interactions` reads `label,count`; `meso_species` reads `time_ns,species,count`; `track_lengths` expects `primaryEkin0_keV`, `secondaryFirstGen_nm`, `secondaryAll_nm`; `prechemical_records` parses fixed-width `.txt`; `particle_g_estimates` rebuilds G and SEM from rounded moments.
- **Pain point:** all of the above fail on new dumps. New files have plain headers: Species/SpeciesMeso `time_ns,species,nEvents,countSum,countSumSq,countMean,countStdErr,gSum,gSumSq,gMean,gStdErr`; Reactions `reactionId,binStart_ns,binEnd_ns,nEvents,count*`; `ReactionsCatalog` `reactionId,reaction`; PhysicsInteractions `process,nEvents,count*`; TrackLengths with `primaryEkinStart_keV`, `secondaryFirstGenLength_nm`, `secondaryAllLength_nm`; PreChemical `run,event,parentTrackId,molecule,modification,level,energy_eV,parent*_nm,*_nm`. h5: group per table, dataset per column (int64, float64, vlen UTF-8 strings, `unit`/`description` attrs, root attr `formatVersion`=1); root: TTree per table, integers stored as doubles. The manifest gains `outputFormats` (per product list) and `hdf5`; `mesoSpatialOutput` now means the spatial format list is non-empty (`/chem/meso/spatialOutput` is gone).
- **Dependencies:** keep old dumps readable (committed validation references, past scans). SpeciesMesoSpatial h5 layout is unchanged (format version 2), so `g4utils.HDF5.SpeciesMesoSpatialFile` is untouched. Reference real output: dnachem-min `.scratch/tests/2026-11-04__data-output-should-be-standardized-each-type-of-data-should-have/smoke12/all` (all four formats, 2 events).
- **Tech stack:** Python >= 3.10, numpy, pandas, h5py; `uproot` new and optional (root files); pytest; CI matrix 3.10-3.13.

## Success Metrics

- `Dump(...).species()`, `.reactions()`, `.physical_interactions()`, `.track_lengths()`, `.meso_species()`, `.prechemical_records()`, `.reaction_table()`, `.particle_g_estimates()` and `load_species/load_reactions/load_reaction_table` work unchanged in signature on both legacy and new dumps.
- The same new dump read from csv, h5, root and txt (where supported) yields equal frames (a cross-format test), and the real smoke12 dump loads.
- Legacy test suite still passes; new-layout fixtures and tests cover each product.
- `Manifest` exposes `outputFormats` and `hdf5`; a product with `none` raises a clear error naming `outputFormats`.
- README, CONTEXT.md and CHANGELOG describe the new layout; version 0.5.0.

## Architecture & Approach

- Layout detection per Dump: new iff the manifest has `outputFormats`, else legacy. Both layouts sit behind the same `Dump` accessors and loaders.
- New `tables.py`: `read_table(...)` picks the first available format from the manifest list in order csv > h5 > root > txt, restricted to files actually listed. csv via pandas; h5 via h5py (decode strings, int64); root via optional `uproot` (clear ImportError message; cast integer columns back from double); txt as a minimal fallback used only when no other format exists (reaction catalogue labels contain spaces, so it uses a dedicated fixed parse). Result dtypes are identical across formats.
- Readers: new-layout versions of species, meso, reactions (time bins), catalogue, physics interactions, track lengths (renamed columns; `validate_track_summary` keys updated), pre-chemical (`.csv`/`.txt` per event, or `PreChemical.h5/.root`; new filename regex). Legacy readers kept and their output mapped onto the new column names.
- `species()["G"]` = `gMean` for new dumps (legacy keeps pooled `number/(E/100)`); `particle_g_estimates` for new dumps uses stored `gMean/gStdErr/nEvents` without the rounded-moment tolerance logic; legacy path unchanged. `ReactionsCatalog` replaces `ReactionsMetadata` in `load_reaction_table` (legacy file still accepted). `has_meso`/`meso` consult `outputFormats["speciesMesoSpatial"]`.
- `Manifest`: add `outputFormats` and `hdf5`; show in `_summary()`; add `outputFormats` to `Simulation.table()` fields. New `Dump.output_formats` property.
- Packaging/docs: `uproot` as optional extra `[root]` and in `[dev]`; version 0.5.0; README (PhysicsInteractions, TrackLengths, PreChemical, SpeciesMeso, G estimators, loaders, stale `/chem/meso/spatialOutput` at ~line 473 -> `/output/speciesMesoSpatial/format h5`), CONTEXT.md (ReactionsMetadata -> ReactionsCatalog; terms Output product, Output format, per-event statistics), CHANGELOG breaking-change entry.

## Assumptions & Trade-offs

- Decided with the user: legacy and new share the same entry points; format preference csv > h5 > root > txt; `species().G` becomes `gMean` for new dumps; new column names everywhere with legacy dumps mapped onto them (columns a legacy dump lacks, e.g. standard errors, are NaN); txt gets a minimal fallback only.
- `species().G` changes meaning for new dumps (per-event mean instead of pooled ratio); documented as a breaking change, hence 0.5.0.
- Not doing: a reader for `SpeciesMesoSpatial.root` (possible later), writing output, changing the HDF5 Vox code.

## Open Questions

- Exact unified column names for legacy-mapped frames (e.g. whether `label` is kept as an alias of `process`) are settled during implementation; default is new names only.
- Whether `HO3` (Tonneau2025 extra species) needs a short name in `species_names.py`.

## Notes

- Draft plan file: `C:\DEV\python\G4Utils\.claude\plans\standardized-output-products.md`.
- Source of truth for the new schemas: dnachem-min `.claude/skills/sim-output/SKILL.md`, `docs/output/README.md`, `docs/output/tables-h5-root.md`, `docs/adr/0009-standardized-output-products.md`.
- G4Utils was not run during analysis; the breakage list comes from reading code. First implementation step should confirm it by running pytest against the smoke12 dump.

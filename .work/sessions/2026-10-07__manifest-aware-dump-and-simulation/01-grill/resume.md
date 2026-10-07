# Session: manifest-aware-dump-and-simulation

**Date:** 2026-10-07T20:59:48.932Z
**Status:** Grill phase complete

## Problem Statement

`Dump` and `Simulation` (added in the previous session) do not yet use everything a dnachem-min `Manifest.json` records. `Manifest` does not type the mesoscopic and model fields, `Dump` ignores `files`, `prefix` and `mesoSpatialOutput`, and `sim.table()` is thin. This session reinforces them with what the Manifest contains.

## Context & Constraints

- **Current behavior:** `Manifest` has typed fields up to `chemistryEndTime_ns`; `chemistryModel`, `handOverTime_ns`, `voxelSize_nm`, `mesoPixels`, `mesoTimesPerDecade`, `mesoSpatialOutput` fall into `raw` and show under "Other". `Dump.meso` only reports a missing file. `Dump` looks up data files by fixed names. `sim.table()` returns only `dump_columns` plus labels.
- **Pain point:** the model and mesoscopic settings are not first-class; a Dump with no meso output gives a vague error; prefixed data files are not found.
- **Dependencies:** one `Manifest.json` per Dump (one flush, possibly several `/run/beamOn`, each a Manifest run); `Manifest.json` is never prefixed; `load_species` / `load_reactions` / `load_manifests` / `dump_columns` output must stay identical.
- **Tech stack:** Python, pandas, h5py, pytest; `g4utils.DnaChem`.

## Success Metrics

- `Manifest` exposes the six new typed fields and shows them in a "Chemistry model" display section; unknown keys still go to `raw`.
- `Dump.files`, `Dump.prefix`, `Dump.has_meso` exist; `Dump.meso` raises a clear error when `mesoSpatialOutput` is false or the h5 is absent.
- Data files of a Dump with a non-empty Manifest `prefix` are found as prefix + filename.
- `sim.table()` carries the richer columns; existing tests pass unchanged.

## Architecture & Approach

1. Add the six fields to `Manifest` (and the display, text and HTML).
2. In `Dump`: `.files` from the Manifest, `.prefix` from `Manifest.prefix` (empty if absent), data paths built as prefix + name, `.has_meso`, and a clear `.meso` error.
3. `Simulation.table()` builds rows from the Manifest: chemistry, chemistryModel, pH, scavenger molarities, hand-over and end times, voxelSize_nm, mesoPixels, mesoTimesPerDecade, mesoSpatialOutput, totalEvents, totalEnergyDeposit_eV, particle, beamEnergy_keV, threads, total wall time.
4. README and glossary: Dump = one flush; Manifest run = one `/run/beamOn` inside it.

## Assumptions & Trade-offs

- Prefix is read from the Manifest `prefix` field, not from the file name (assumption; the user dismissed the question twice, after saying the Manifest file has no prefix).
- Not changing discovery (`find_dumps`) or the `load_*` / `dump_columns` output.

## Open Questions

- How prefix dumps look exactly on disk is unconfirmed.

## Notes

Confirmed by the user: each Dump has one Manifest; in `261007` each level folder has a single Manifest run; a macro with two `/run/beamOn 16` before one dump gives one Manifest with two runs and `totalEvents` 32.

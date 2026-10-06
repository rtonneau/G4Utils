# Session: the new hdf5 reader for mesoscopic simulation data should be able to export 3D data to vti format

**Date:** 2026-10-06T18:08:01.274Z
**Status:** Grill phase complete

## Problem Statement

The new `SpeciesMesoSpatialFile` reader (dnachem-min `SpeciesMesoSpatial.h5`) can read snapshots but cannot export them to a format ParaView opens. Users need to export the 3D mesoscopic data to VTK ImageData (`.vti`), as the Vox side already does.

## Context & Constraints

- **Current behavior:** `SpeciesMesoSpatialFile.read_snapshot` returns a `MesoSpatialSnapshot` holding sparse data: `position_nm` (N, 3) cell centres and `counts` (N, S) for occupied cells only; `cell_size_nm` is stored per snapshot and can change over time. `write_vti(path, VoxGeometry, {name: (nZ,nY,nX)}, dtype, encoding, compress)` and `write_pvd_collection` exist in `vti_export.py` and serve the Vox exports (in mm).
- **Pain point:** no way to turn sparse meso cells into a dense grid or a time series for visualisation.
- **Dependencies:** reuse `write_vti` and `write_pvd_collection` unchanged; `concentration_M` for molarity.
- **Tech stack:** Python, numpy, h5py; hand-written VTK XML (no vtk dependency).

## Success Metrics

- `MesoSpatialSnapshot.to_vti(path, ...)` writes a valid `.vti` whose CellData holds `<species>_count` and `<species>_M` arrays matching the snapshot, with 0 in unoccupied cells.
- `SpeciesMesoSpatialFile.to_vti_timeseries(run, event, path.pvd, ...)` writes one `.vti` per snapshot plus a `.pvd` with timestep = `time_ns`, reading snapshots lazily one at a time.
- Off-lattice cell centres, colliding cells, or an event with no occupied cells raise `ValueError`; covered by tests, and documented in the README.

## Architecture & Approach

- Vocabulary (in `CONTEXT.md`): Meso file, Meso snapshot, Meso cell, Event series, Common box.
- One `.vti` per snapshot; one `.pvd` per (run, event) Event series. Runs and events are never merged.
- Sparse to dense: index = round((pos - origin)/cell_size), origin = min centre - cell_size/2; unoccupied cells are 0. Validate lattice alignment and collisions, raising `ValueError`.
- Default extent is the Common box over all snapshots of the event series, overridable via an `extent` parameter. Each frame uses its own cell size as spacing, so frames may differ in dimensions.
- `MesoSpatialSnapshot.to_vti(path, species=None, dtype=float32, extent=None, encoding, compress)` holds the logic; `SpeciesMesoSpatialFile.to_vti_timeseries(...)` loops lazily.
- Array names are `<raw species name>_count` and `<raw species name>_M` (raw Geant4 names, XML-escaped). Species selectable. One dtype for all arrays (default float32, overflow-checked via the existing writer).
- Origin and spacing are in nm (native unit), documented in the README.
- A `VoxGeometry` is built from the nm values to reuse `write_vti`.

## Assumptions & Trade-offs

- Assumes meso cell centres lie on a regular lattice; validated at export rather than trusted.
- Float32 default means counts above 2^24 lose exactness; callers can pass float64.
- Not doing: merging runs/events, resampling onto a common lattice, converting to mm, per-array dtypes, changes to `write_vti`.

## Open Questions

- None open. Lattice regularity was not verified against the dnachem-min format doc; the validation guards against it.

## Notes

Grill ran with `grilling` + `domain-modeling`. Glossary terms were added to `CONTEXT.md`; no ADR was warranted.

# Session: the HDF5 code loading data from mesoscopic geant4 data loads sparsed data. It should provides a way to reconstruct full dataframe (fileld with 0) to have correct sizes and compare easily one to another

**Date:** 2026-10-06T17:08:19.128Z
**Status:** Grill phase complete

## Problem Statement

`SpeciesMesoSpatialFile` returns sparse snapshots: only occupied cells, with a different N per snapshot and a cell size that grows with time. Arrays from different snapshots, events or files therefore have different shapes and cannot be compared directly. Add a way to rebuild full dense arrays (zero-filled) with consistent shapes and the grid geometry needed to compare them.

## Context & Constraints

- **Current behavior:** `read_snapshot` returns `MesoSpatialSnapshot` with `position_nm` (N, 3) cell centres and `counts` (N, S) uint32 for occupied cells only. The file stores no mesh extent (only cell size per snapshot, positions in the world frame).
- **Pain point:** no way to get a regular grid; users hand-roll scatter code and cannot compare snapshots, events or files.
- **Dependencies:** keep the existing reader API unchanged; h5py and numpy only (no new dependency).
- **Tech stack:** Python >= 3.10, numpy, h5py, pytest.

## Success Metrics

- `SpeciesMesoSpatialFile.read_dense(run, event, species, bounds_nm=None, concentration=False)` returns, for one event, a list of `DenseMesoPeriod`, a new one each time the cell size increases between consecutive snapshots.
- Each period holds a zero-filled 4D array (time, x, y, z) for the one selected species: uint32 counts by default, float64 mol/L when `concentration=True` (using that period's cell size).
- Each period carries a grid helper: `origin_nm`, `cell_size_nm`, `shape`, `times_ns`, snapshot indices, provenance (species, run, event), plus index <-> position helpers.
- Extent: explicit `bounds_nm=(min_xyz, max_xyz)` box, else the bounding box of the event's occupied cells shared by all periods; two files give comparable arrays when given the same `bounds_nm` and cell sizes.
- A snapshot-level helper (`MesoSpatialSnapshot.to_dense`) exists; empty (N=0) snapshots give all-zero slices; tests with the synthetic fixture pass; README documents it; exported from `g4utils.HDF5`.

## Architecture & Approach

Methods on existing classes in `src/g4utils/HDF5/species_meso_spatial.py`: a `MesoGrid`-style geometry helper plus `DenseMesoPeriod` dataclass, `MesoSpatialSnapshot.to_dense(...)` and `SpeciesMesoSpatialFile.read_dense(...)`. Snapshots of an event are read in time order and grouped into periods of constant `cellSize_nm`; each period's grid covers the common extent at that cell size and snapshots are scattered into it from cell-centre positions (index = round((pos - origin) / cell - 0.5)). Exports added to `g4utils.HDF5.__init__`.

## Assumptions & Trade-offs

- One species per call; one event per call (run-level = loop over events).
- No memory/size guard (user's decision); a huge extent will raise numpy's MemoryError.
- Out of scope: resampling/coarsening across cell sizes, sparse output, pandas/xarray output.
- Cross-file comparison requires the caller to pass identical `bounds_nm`; cell sizes must match for direct comparison.

## Open Questions

- None blocking. Exact grid-origin alignment convention (origin at the lower corner of the first cell, snapped to the cell size) is settled at implementation.

## Notes

- Design approved by the user in chat via the grill questions.

# Session: Add a Reader to G4Utils to help parse SpeciesMesoSpatial HDF5 files

**Date:** 2026-10-06T15:43:04.940Z
**Status:** Grill phase complete

## Problem Statement

dnachem-min writes `SpeciesMesoSpatial.h5` (format documented in `dnachem-min/docs/output/SpeciesMesoSpatial-h5.md`): the spatial state of the mesoscopic stage, as sparse per-snapshot cell positions and per-species molecule counts, organised run → event → snapshot. G4Utils has no reader for it; users must hand-roll h5py code. Add a reader to G4Utils.

## Context & Constraints

- **Current behavior:** `g4utils.HDF5` only reads G4Vox voxel files (`G4VoxFile3D/4D`); `g4utils.DnaChem` loads CSV tallies and Manifests into pandas. Nothing reads `SpeciesMesoSpatial.h5`.
- **Pain point:** The file is hierarchical and sparse (variable number of occupied cells per snapshot, N may be 0, consecutive snapshots may share hard-linked datasets, cell size grows with time), so each user re-implements indexing and the concentration formula.
- **Dependencies:** h5py, numpy, pandas already in `pyproject.toml`. The Vox backend infrastructure (`G4VoxFileBase`, `HDF5LayoutBackend`) is tied to geometry/quantities/subruns/VTI export and is not reused.
- **Tech stack:** Python >= 3.10, h5py, numpy, pytest.

## Success Metrics

- `SpeciesMesoSpatialFile(path)` exposes `species`, `format_version`, `units`, `runs`, `events(run)`, `snapshot_indices(run, event)`, and cached per-snapshot `time_ns` / `cellSize_nm` without reading position/counts arrays.
- `read_snapshot(run, event, index)` returns a `MesoSpatialSnapshot` with `position_nm` (N, 3) float64 and `counts` (N, S) uint32, including empty (N = 0) and hard-linked snapshots; `iter_snapshots(run=None, event=None)` yields them lazily.
- `MesoSpatialSnapshot.concentration_M(...)` returns mol/L using count / (N_A * (cellSize_nm * 1e-8)^3); missing file raises `FileNotFoundError`, missing run/event/snapshot raises `KeyError`, malformed root attrs raise `ValueError`.
- Tests with a synthetic-file fixture pass, and the README documents usage.

## Architecture & Approach

New module `src/g4utils/HDF5/species_meso_spatial.py` (location chosen by the user: `g4utils.HDF5`), exported from `g4utils.HDF5.__init__`. Lazy object reader (chosen by the user over an eager DataFrame loader): the constructor opens the file once to index runs/events/snapshot indices and their `time_ns`/`cellSize_nm` attributes plus root `species`/`formatVersion`/`formatDoc`/`units`; each `read_snapshot` reopens the file and reads `position_nm` and `counts`, as the Vox backends do. Group names are `run<R>`, `event<E>`, `snapshot<k>`, sorted numerically. Dataclass `MesoSpatialSnapshot` carries run, event, index, time_ns, cell_size_nm, position_nm, counts. Tests: `write_species_meso_spatial` helper in `tests/conftest.py` and `tests/test_species_meso_spatial.py`. Docs: a short README section.

## Assumptions & Trade-offs

- Reader is single-file only: no Dump discovery via `Manifest.json`, no prefix/subfolder resolution.
- No format-version-specific logic: v1 and v2 parse identically; `format_version` is exposed for information.
- Concentration is opt-in (a method), not computed on read.
- Species names stay as the raw Geant4 display names; no `short_name` mapping by default.
- No file handle is held open between calls.

## Open Questions

- None blocking. The design was presented in chat and the user proceeded with `/gps auto` without requesting changes, taken as approval.

## Notes

- Possible follow-ups, not in scope: Dump-aware discovery of the file, a pandas export of a snapshot, CONTEXT.md glossary entries for snapshot / occupied cell.

# Session: fix-review-findings

**Date:** 2026-10-01T08:27:50.660Z
**Status:** Grill phase complete

## Problem Statement

A code review of g4utils (HDF5 readers for G4Vox voxel output) found issues to fix before new features are added:

1. CI fails: matrix includes Python 3.9 while `requires-python >= 3.10`, and `tests/` is empty (pytest exit code 5).
2. Local pytest doesn't run: `addopts` passes `--cov` but pytest-cov / g4utils aren't installed in the local Python 3.13.
3. `G4VoxFile4D` contains unreachable "materialized 4D" code and overrides (`n_subruns`, `quantity_names`) that diverge from the 3D class.
4. `_qty_whitelist` (shared.py) duplicates `select_quantities` (vti_export.py).
5. `sum_subruns` accumulates in the file's dtype (precision loss for float32).
6. VTI export is ASCII only: slow and huge on real grids.
7. 4D subrun IDs are assumed to be 0..N-1 and never checked against the run log.
8. Minor: no automatic layout detection; `_read_geometry` uses `print`; README/pyproject description are template text; matplotlib not a dependency.

New bug found during the grill (by reading `G4Vox/src/HDF5Writer.cc`): the writer's run_log columns are `unix_timestamp, primaries, runtime_s, subrun_id` (stored in a `columns` attribute), but `_read_run_log` labels them `subrun_id, nPrimaries, seed1, seed2`, so its `subrun_id` column actually holds timestamps.

## Context & Constraints

- Package is 0.1.0 / Alpha; breaking API changes are explicitly allowed (bump to 0.2.0, no deprecation shims).
- Ground truth for the file format is the C++ writer `${DEV_DIR}/GEANT4/LIB/G4Vox/src/HDF5Writer.cc`:
  - `/metadata` attrs: `dims_xyz`, `spacing_mm`, `origin_mm` (float64 arrays), `mode` (`"Snapshot3D"` | `"Extendable4D"`), `quantities` (comma-separated string).
  - Snapshot3D: `/subrun_XXXX/<qty>` 3D float64 datasets `(nZ, nY, nX)`.
  - Extendable4D: `/<qty>` float64 `(N, nZ, nY, nX)`, one slice appended per `Export`, together with one run_log row.
  - `/run_log`: float64 `(N, 4)` with attr `columns = "unix_timestamp, primaries, runtime_s, subrun_id"`.
  - Files can be reopened in append mode, so 4D subrun IDs may repeat across runs.
- No new runtime dependencies (zlib/base64 from the standard library only).
- Glossary is in `CONTEXT.md` (vox file, layout, quantity, geometry, subrun, subrun ID vs slice index, run log).

## Success Metrics

- CI green on Python 3.10, 3.11, 3.12, 3.13.
- Plain `pytest` runs locally after `pip install -e ".[dev]"` (installed into the local Python 3.13), with tests covering both layouts: discovery, selection, iteration, `get`, `sum`, VTI/PVD export, layout detection, run_log parsing, 4D subrun-ID resolution and fallback.
- Exported binary/compressed `.vti` files decode back to the original arrays in tests.
- No unreachable code left in `G4VoxFile4D`; both front-end classes are thin backend selectors.

## Architecture & Approach

1. **Tests & CI:** CI matrix 3.10–3.13; run `pytest --cov=g4utils --cov-report=term-missing` in the CI step; remove `--cov` from `addopts`. Add `tests/conftest.py` fixtures that write small Snapshot3D and Extendable4D files exactly the way `HDF5Writer.cc` does (metadata attrs incl. `mode`/`quantities`, float64 data, 4-column run_log with `columns` attr). Add 3.13 to classifiers.
2. **Local env:** `pip install -e ".[dev]"` into the Python on PATH (3.13).
3. **4D cleanup:** `G4VoxFile4D` becomes a thin subclass choosing `Dataset4DBackend` (like `G4VoxFile3D`); delete `_iter_source_data`, `_has_materialized_4d`, `_materialized_axis_index`, and the `__iter__`/`__next__`/`get`/`n_subruns`/`quantity_names` overrides.
4. **Dedupe:** delete `read_g4vox_hdf5_3d` and `_qty_whitelist`; move `select_quantities` into `shared.py`; update `__init__` exports.
5. **Sum precision:** `sum_subruns` accumulates floats in float64 and integers in int64.
6. **VTI export:** default to inline base64 `binary` DataArrays (UInt64 header for block size); `compress=True` uses stdlib zlib (`compressor="vtkZLibDataCompressor"`); `format="ascii"` kept for debugging. The options flow through `to_vti`, `dump_selection_to_vti` and `dump_selection_to_vti_timeseries`.
7. **4D subrun IDs:** `Dataset4DBackend.discover` uses `run_log.subrun_id` as the subrun IDs when the run log has exactly one row per slice and the IDs are unique; otherwise it emits a warning and falls back to slice indices 0..N-1. The backend keeps the subrun ID → slice index mapping and uses it in `load_subrun`.
8. **run_log parsing:** read column names from the `columns` attribute (strip whitespace), falling back to the writer order; cast `subrun_id` to int; `total_primaries()` sums `primaries`.
9. **Layout detection:** `open_vox_file(path)` in `g4utils.HDF5` reads `/metadata` `mode` and returns `G4VoxFile3D` or `G4VoxFile4D`; falls back to structural detection (`subrun_*` groups vs 4D datasets); raises a clear error if undetermined.
10. **Minor:** `warnings.warn` instead of `print` in `_read_geometry`; real README (install, `open_vox_file`, selection/iteration/sum/export examples, layouts) and pyproject description; version 0.2.0.

## Assumptions & Trade-offs

- Breaking changes accepted over compatibility shims (Alpha package).
- 4D subrun IDs: lenient fallback (warn + slice index) chosen over strict errors, so appended or log-less files stay readable; IDs stay consistent with the 3D layout when the run log is sound.
- VTI: inline base64 chosen over raw appended data (keeps valid XML, easy to inspect) and over pyvista/vtk (no heavy dependency).
- matplotlib deliberately deferred to the upcoming plotting feature so no unused dependency ships.
- Float64 accumulation mostly matters for non-G4Vox files: the writer already stores float64.
- No ADR: all decisions are cheap to reverse.

## Open Questions

None. Every branch of the design was settled during the grill.

## Notes

- `CONTEXT.md` was created at the repo root during the grill (glossary only).
- The local Python is 3.13.9 (conda); g4utils was not installed there at grill time.

## Token Usage

- **Input:** 44
- **Output:** 13320
- **Cache read:** 1773185
- **Cache creation:** 26627
- **Total:** 1813176

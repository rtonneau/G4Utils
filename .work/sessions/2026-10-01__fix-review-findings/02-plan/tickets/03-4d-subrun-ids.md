# Ticket 03: 4d-subrun-ids

**Model:** opus

**Acceptance Criteria:**
- [ ] `Dataset4DBackend.discover()` uses `run_log["subrun_id"]` as `available_subrun_ids` when the run log exists, has exactly one row per slice, and the IDs are unique.
- [ ] Otherwise it warns (`UserWarning`) and falls back to slice indices `0..N-1`. Each reason has its own message containing `"run_log missing"`, `"run_log has <R> rows but datasets have <N> slices"`, or `"duplicate subrun_id"`. No warning when N == 0.
- [ ] The backend stores a `dict[int, int]` from subrun ID to slice index, filled in `discover()`. `load_subrun` reads `ds[slice_index, ...]` and raises `KeyError("Subrun '<id>' not found")` for unknown IDs.
- [ ] `G4VoxFile4D` is a thin subclass identical in shape to `G4VoxFile3D` (only `__init__` choosing `Dataset4DBackend`). `_iter_source_data`, `_has_materialized_4d`, `_materialized_axis_index`, and the `__iter__`/`__next__`/`get`/`n_subruns`/`quantity_names` overrides are deleted.
- [ ] Tests with `write_extendable4d(p, subrun_ids=(5, 7, 9))`: `subrun_ids == [5, 7, 9]`; iteration yields 5, 7, 9 with `expected_array(qi, sid)`; `select_subrun([7])` loads slice 1; `get("Dose", 9)` is correct; `select_subrun(start=6, stop=10)` gives `[7, 9]`; the `.pvd` timesteps are `5.0, 7.0, 9.0`.
- [ ] Fallback tests: duplicate IDs `(0, 1, 0)` warn and give `[0, 1, 2]`; `run_log_ids=(0, 1)` with 3 slices warns; `with_run_log=False` warns. Every fallback still loads the right slice by index.
- [ ] `quantity_names` stays `["Dose", "Edep"]` during iteration after `select_quantity("Dose")`.

**Files to Touch:**
- `src/g4utils/HDF5/vox_file_base.py`
- `src/g4utils/HDF5/vox_file_4d.py`
- `tests/test_subrun_ids_4d.py` (create)

**Verification Step:**

Run:
```bash
pytest tests/test_subrun_ids_4d.py -v && pytest
```

Expected:
All pass. `grep -n "_iter_source_data\|_materialized" src` finds nothing.

**Notes:**

Depends on ticket 02 (`run_log["subrun_id"]` is int64). The `.pvd` timestep is the subrun ID (`float(sid)`), as before. Snapshot3D needs no fallback: group names are unique.

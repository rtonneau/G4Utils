# Ticket 02: run-log-columns

**Model:** sonnet

**Acceptance Criteria:**
- [ ] `_read_run_log` names columns from the dataset's `columns` attr (str or bytes, split on `,`, whitespace stripped).
- [ ] Without the attr, it uses `RUN_LOG_COLUMNS = ("unix_timestamp", "primaries", "runtime_s", "subrun_id")` (module constant in `shared.py`).
- [ ] When the name count differs from the column count, it warns (`UserWarning`) and names the columns from `RUN_LOG_COLUMNS` by position, with any extra columns named `col_<i>`.
- [ ] `subrun_id` and `primaries` columns are `int64`; the others stay float64.
- [ ] `total_primaries()` sums `primaries` (returns 0 when run_log or the column is missing), and `__repr__` still prints it.
- [ ] Tests cover: names from the attr, names without the attr, dtypes, values, a single-row run_log, a missing run_log (`run_log is None`, `total_primaries() == 0`).

**Files to Touch:**
- `src/g4utils/HDF5/shared.py`
- `src/g4utils/HDF5/vox_file_base.py`
- `tests/test_run_log.py` (create)

**Verification Step:**

Run:
```bash
pytest tests/test_run_log.py -v && pytest
```

Expected:
All pass.

**Notes:**

Example: `write_snapshot3d(p, subrun_ids=(3, 4))` gives `sim.run_log.columns.tolist() == ["unix_timestamp", "primaries", "runtime_s", "subrun_id"]`, `sim.run_log["subrun_id"].tolist() == [3, 4]` and `sim.total_primaries() == 200`.

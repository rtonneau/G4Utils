# Ticket 01: test-infra-ci

**Model:** sonnet

**Acceptance Criteria:**
- [ ] `pip install -e ".[dev]"` has been run in the Python on PATH, and `python -c "import g4utils"` works outside `src/`.
- [ ] `pyproject.toml` `addopts` is `"-v"` (no `--cov`), and classifiers include Python 3.13.
- [ ] `.github/workflows/ci.yml` matrix is `["3.10", "3.11", "3.12", "3.13"]` and the test step runs `pytest --cov=g4utils --cov-report=term-missing`.
- [ ] `tests/conftest.py` provides writer helpers and fixtures (see Notes) that produce files matching `HDF5Writer.cc`.
- [ ] `tests/test_vox_file.py` holds baseline tests that pass on the current code for both layouts: geometry values, `quantity_names == ["Dose", "Edep"]`, iteration yields `[0, 1, 2]` with the expected array per subrun, `sum("Dose")` equals the sum of the expected arrays, and `get("Edep", 1)` equals the expected array.
- [ ] `pytest` passes locally.

**Files to Touch:**
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `tests/conftest.py` (create)
- `tests/test_vox_file.py` (create)

**Verification Step:**

Run:
```bash
pip install -e ".[dev]" && pytest
```

Expected:
All tests pass. No `--cov` error, no "no tests collected".

**Notes:**

Conftest API (later tickets rely on these exact names):
- `DIMS_XYZ = (4, 3, 2)`, `SPACING_MM = (0.5, 1.0, 2.0)`, `ORIGIN_MM = (-1.0, -1.5, -2.0)`, `QUANTITIES = ("Dose", "Edep")`.
- `expected_array(qty_index: int, subrun_id: int, dims_xyz=DIMS_XYZ, dtype=np.float64) -> np.ndarray`: shape `(nz, ny, nx)`, value `np.arange(nz*ny*nx).reshape(nz, ny, nx) + 1000*qty_index + 100*subrun_id`, cast to `dtype`.
- `write_snapshot3d(path, *, subrun_ids=(0, 1, 2), dims_xyz=DIMS_XYZ, quantities=QUANTITIES, dtype=np.float64, primaries=100, with_metadata=True, with_run_log=True, run_log_columns_attr=True) -> Path`
- `write_extendable4d(path, *, subrun_ids=(0, 1, 2), n_slices=None, run_log_ids=None, dims_xyz=DIMS_XYZ, quantities=QUANTITIES, dtype=np.float64, primaries=100, with_metadata=True, with_run_log=True, run_log_columns_attr=True) -> Path`. Slice *i* holds `expected_array(qi, subrun_ids[i])`. `run_log_ids` overrides the subrun_id column when given; it may differ in length from the slices (that is how row/slice mismatches are tested).
- Metadata attrs: `dims_xyz`, `spacing_mm` and `origin_mm` as float64 arrays, `mode` as a variable-length string, `quantities` as the comma-joined string.
- run_log: float64 `(N, 4)` rows `[1.7e9 + i, primaries, 1.5, subrun_id]`, attr `columns = "unix_timestamp, primaries, runtime_s, subrun_id"` when `run_log_columns_attr`.
- Fixtures: `snapshot3d_file(tmp_path)` and `extendable4d_file(tmp_path)` call the writers with defaults.
- Add a comment citing `G4Vox/src/HDF5Writer.cc` as the format source.

Baseline tests must not assert run_log column names or 4D-ID behavior: tickets 02 and 03 change those.

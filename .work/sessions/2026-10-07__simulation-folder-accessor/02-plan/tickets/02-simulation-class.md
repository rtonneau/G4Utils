# Ticket 02: simulation-class

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `Simulation(path, name_pattern=None)` uses `path/results/` when it exists, else `path` itself, with `find_dumps`; it reads no data file.
- [ ] `sim.subruns` is an ordered mapping of folder name to `Dump`; `sim.subrun(name)` looks one up and raises `KeyError` listing the available names.
- [ ] Iterating a `Simulation` yields its Dumps; `len(sim)` is the number of Dumps.
- [ ] `sim.table()` returns a DataFrame with one row per Dump: name, the labels, and Manifest summary columns.
- [ ] A flat Dump folder, a results folder, and a simulation folder with `results/` all work; a folder with no Dump raises `FileNotFoundError`.

**Files to Touch:**
- `src/g4utils/DnaChem/simulation.py`
- `tests/test_dnachem_simulation.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_simulation.py -q
```

Expected:
All tests pass.

**Notes:**

Build the tests from `make_dump` inside `tmp_path / "results"`. Use `dump_columns` for the Manifest summary in `table()`.

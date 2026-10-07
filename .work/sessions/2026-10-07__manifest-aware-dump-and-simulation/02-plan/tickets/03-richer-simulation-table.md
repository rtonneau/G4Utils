# Ticket 03: richer-simulation-table

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `sim.table()` has one row per Dump with: dump name, labels, chemistry, chemistryModel, pH, scavenger molarity columns, handOverTime_ns, chemistryEndTime_ns, voxelSize_nm, mesoPixels, mesoTimesPerDecade, mesoSpatialOutput, totalEvents, totalEnergyDeposit_eV, particle, beamEnergy_keV, threads, total wall time (s, sum of runs).
- [ ] Missing Manifest fields are `None` or NaN; no exception.
- [ ] `load_species` / `load_reactions` columns are unchanged.

**Files to Touch:**
- `src/g4utils/DnaChem/simulation.py`
- `tests/test_dnachem_simulation.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_simulation.py tests/test_dnachem_loaders.py -q
```

Expected:
All tests pass.

**Notes:**

Do not change `dump_columns`; build the extra columns in `table()` from `Dump.manifest`. Reuse `dump_columns` for the shared columns.

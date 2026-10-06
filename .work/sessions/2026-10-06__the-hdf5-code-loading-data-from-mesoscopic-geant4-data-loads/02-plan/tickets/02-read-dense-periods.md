# Ticket 02: read-dense-periods

**Model:** sonnet
**Effort:** high

**Acceptance Criteria:**
- [ ] `DenseMesoPeriod` holds `data` (time, x, y, z), `grid`, `times_ns`, `snapshot_indices`, `species`, `run`, `event`.
- [ ] `SpeciesMesoSpatialFile.read_dense(run, event, species, bounds_nm=None, concentration=False)` returns a list of periods, a new one each time `cellSize_nm` changes between consecutive snapshots.
- [ ] Without `bounds_nm`, all periods share the bounding box of the event's occupied cells; with it, two files given the same bounds give identically shaped arrays when their cell sizes match.
- [ ] Unknown run/event raises `KeyError`; unknown species raises `KeyError`.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `tests/test_species_meso_spatial.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_species_meso_spatial.py -q
```

Expected:
All tests pass, including period grouping, empty snapshots and cross-file shape comparison.

**Notes:**

Hard-linked snapshots read the same data; periods are defined by cell size only. Read snapshots lazily one at a time to limit memory.

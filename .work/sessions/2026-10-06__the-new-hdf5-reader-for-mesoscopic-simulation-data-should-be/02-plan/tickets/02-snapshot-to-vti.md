# Ticket 02: snapshot-to-vti

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `MesoSpatialSnapshot.to_vti(path, species=None, dtype=np.float32, extent=None, *, encoding="binary", compress=False)` writes a `.vti` via `write_vti` and returns the path.
- [ ] CellData has `<species>_count` and `<species>_M` for each selected species, using the raw species names.
- [ ] Origin and spacing are in nm; an unknown species raises `KeyError`.
- [ ] Concentrations match `concentration_M` for the occupied cells.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `tests/test_vti_export.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_vti_export.py tests/test_species_meso_spatial.py -q
```

Expected:
All tests pass, including one that parses the written `.vti` and compares arrays to the snapshot.

**Notes:**

Build a `VoxGeometry` from nm values (dims_xyz, spacing = cell size on each axis, origin). Import `write_vti` lazily or at module top without creating a cycle. See the existing tests in `tests/test_vti_export.py` for how a `.vti` is parsed back.

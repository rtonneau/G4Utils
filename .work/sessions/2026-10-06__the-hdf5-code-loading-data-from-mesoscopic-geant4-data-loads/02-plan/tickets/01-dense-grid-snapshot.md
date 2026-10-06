# Ticket 01: dense-grid-snapshot

**Model:** sonnet
**Effort:** high

**Acceptance Criteria:**
- [ ] `MesoGrid` exposes `origin_nm`, `cell_size_nm`, `shape`, `index_of(position_nm)` and `centers(axis)`.
- [ ] `MesoSpatialSnapshot.to_dense(species, bounds_nm=None, concentration=False)` returns a zero-filled 3D array plus its grid; uint32 counts by default, float64 mol/L when `concentration=True`.
- [ ] Extent defaults to the bounding box of the snapshot's occupied cells; explicit `bounds_nm` is honoured; empty (N=0) snapshots give all zeros.
- [ ] Unknown species raises `KeyError`.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `tests/test_species_meso_spatial.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_species_meso_spatial.py -q
```

Expected:
All tests pass, including the new to_dense tests.

**Notes:**

Cell centres are in the world frame; the lower corner of cell i is origin + i * cell. Sum counts if two rows land in the same cell (should not occur).

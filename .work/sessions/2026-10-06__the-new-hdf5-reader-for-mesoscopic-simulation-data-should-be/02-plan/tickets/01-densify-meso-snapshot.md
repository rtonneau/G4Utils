# Ticket 01: densify-meso-snapshot

**Model:** sonnet
**Effort:** high

**Acceptance Criteria:**
- [ ] A function in `species_meso_spatial.py` takes a `MesoSpatialSnapshot` and an optional extent `(min_xyz_nm, max_xyz_nm)` and returns the grid origin (nm), the cell size, the dims and the dense index arrays or value arrays of shape (nZ, nY, nX) with 0 in unoccupied cells.
- [ ] Grid origin without an extent is min centre - cell_size/2; cell index = round((pos - origin)/cell_size).
- [ ] With an extent, the lattice keeps the snapshot's anchor and grows by whole cells to cover the box.
- [ ] A centre more than a small tolerance off-lattice, or two cells in the same index, raises `ValueError` naming the problem.
- [ ] A snapshot with no cells and no extent raises `ValueError`.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `tests/test_species_meso_spatial.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_species_meso_spatial.py -q
```

Expected:
All tests pass, including new ones for dense placement, extent growth, off-lattice, collision and empty snapshot.

**Notes:**

Keep the helper private (underscore) unless ticket 2 needs it public. Axis order is (nZ, nY, nX), matching `write_vti`. Use `np.add.at` or direct assignment after the collision check.

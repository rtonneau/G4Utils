# Ticket 02: concentration

**Model:** haiku
**Effort:** low

**Acceptance Criteria:**
- [ ] A module-level `concentration_M(counts, cell_size_nm)` in `species_meso_spatial.py` returns `counts / (N_A * (cell_size_nm * 1e-8) ** 3)` as float64 with N_A = 6.02214076e23.
- [ ] `MesoSpatialSnapshot.concentration_M(species=None)` applies it to `counts` with the snapshot's `cell_size_nm`; with a species list (names from the file's species) it returns only those columns, in the requested order.
- [ ] An empty snapshot (N = 0) gives an empty (0, S) array; an unknown species name raises `KeyError`.
- [ ] `concentration_M` is exported from `g4utils.HDF5`.
- [ ] Tests check a hand-computed value, column selection, the empty case and the error.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `src/g4utils/HDF5/__init__.py`
- `tests/test_species_meso_spatial.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_species_meso_spatial.py -q -k concentration
```

Expected:
The concentration tests pass.

**Notes:**

The snapshot has no species list of its own, so `species=None` means all columns and a species list needs the file's names: give `MesoSpatialSnapshot` an optional `species: list[str]` field (set by `SpeciesMesoSpatialFile.read_snapshot`) and use it to resolve names to columns. Formula source: the "Concentration" section of the format document.

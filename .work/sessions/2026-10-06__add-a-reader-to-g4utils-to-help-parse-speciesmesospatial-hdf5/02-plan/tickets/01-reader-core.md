# Ticket 01: reader-core

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `src/g4utils/HDF5/species_meso_spatial.py` defines `MesoSpatialSnapshot` (run, event, index, time_ns, cell_size_nm, position_nm, counts) and `SpeciesMesoSpatialFile(path)`.
- [ ] The constructor raises `FileNotFoundError` for a missing path and `ValueError` when root attributes `species` or `formatVersion` are missing.
- [ ] `species`, `format_version`, `units`, `runs`, `events(run)`, `snapshot_indices(run, event)`, `snapshot_time_ns(run, event, index)` and `snapshot_cell_size_nm(run, event, index)` work from the cached index, with runs, events and snapshot indices sorted numerically.
- [ ] `read_snapshot(run, event, index)` returns `position_nm` (N, 3) float64 and `counts` (N, S) uint32, including N = 0 and hard-linked datasets; an unknown run, event or index raises `KeyError`.
- [ ] `iter_snapshots(run=None, event=None)` lazily yields snapshots in (run, event, index) order, optionally filtered.
- [ ] The two classes are exported from `g4utils.HDF5`.
- [ ] `tests/conftest.py` gains `write_species_meso_spatial` and a fixture; `tests/test_species_meso_spatial.py` covers the above.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `src/g4utils/HDF5/__init__.py`
- `tests/conftest.py`
- `tests/test_species_meso_spatial.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_species_meso_spatial.py -q
```

Expected:
All tests in the file pass, none fail.

**Notes:**

Format source: `C:/Users/rtonneau/DEV/GEANT4/SIM/dnachem-min/docs/output/SpeciesMesoSpatial-h5.md`. Layout is `run<R>/event<E>/snapshot<k>` (no zero padding, no underscores); `species` is a root attribute (h5py returns it as `str` items), `time_ns` and `cellSize_nm` are snapshot-group attributes. Follow the style of `vox_file_base.py`: `from __future__ import annotations`, `with h5py.File(path, "r")` per access, `isinstance` checks raising `TypeError` for unexpected object types. The constructor reads the index once (group names and snapshot attrs only, no array reads); `read_snapshot` reopens the file. Do not subclass `G4VoxFileBase`. Do not add concentration here (ticket 2). In the fixture, make two consecutive snapshots share one dataset via a hard link (`group["position_nm"] = other["position_nm"]`) and include an N = 0 snapshot of shapes (0, 3) and (0, S).

# Ticket 05: open-vox-file

**Model:** sonnet

**Acceptance Criteria:**
- [ ] New module `src/g4utils/HDF5/detect.py` with `detect_layout(path: str | Path) -> Literal["Snapshot3D", "Extendable4D"]` and `open_vox_file(path: str | Path) -> G4VoxFileBase`.
- [ ] `detect_layout` reads `/metadata` attr `mode` (str or bytes) first. An unknown value raises `ValueError` naming the value.
- [ ] Without `mode`: any root group named `subrun_*` means `"Snapshot3D"`; any root 4D dataset other than `metadata`/`run_log` means `"Extendable4D"`; neither raises `ValueError("Cannot determine layout of '<path>'")`.
- [ ] A missing file raises `FileNotFoundError`.
- [ ] `open_vox_file` returns `G4VoxFile3D` or `G4VoxFile4D` accordingly. Both functions are exported from `g4utils.HDF5`.
- [ ] Tests: each layout with metadata; each layout with `with_metadata=False` (warning expected from geometry inference); an unknown `mode`; an empty HDF5 file; a missing path.

**Files to Touch:**
- `src/g4utils/HDF5/detect.py` (create)
- `src/g4utils/HDF5/__init__.py`
- `tests/test_detect.py` (create)

**Verification Step:**

Run:
```bash
pytest tests/test_detect.py -v && pytest
```

Expected:
All pass.

**Notes:**

Use `isinstance(open_vox_file(p), G4VoxFile4D)` in the tests. Detection opens the file once read-only and closes it before constructing the class.

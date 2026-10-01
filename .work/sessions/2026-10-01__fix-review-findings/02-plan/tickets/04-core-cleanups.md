# Ticket 04: core-cleanups

**Model:** sonnet

**Acceptance Criteria:**
- [ ] `read_g4vox_hdf5_3d` and `_qty_whitelist` are deleted, and `read_g4vox_hdf5_3d` is removed from `g4utils.HDF5.__all__`.
- [ ] `select_quantities(available, quantities) -> list[str]` lives in `shared.py` (moved unchanged from `vti_export.py`), and `vox_file_base.py` imports it from there.
- [ ] `HDF5LayoutBackend.sum_subruns` accumulates floating input in float64, integer or bool input in int64, and any other dtype in its own dtype.
- [ ] The `_read_geometry` fallback calls `warnings.warn("/metadata absent – spacing=1 mm, origin=0 mm", UserWarning)` instead of `print`.
- [ ] Tests: `sum` on a float32 file returns float64 and equals `np.sum` of the float64 expected arrays; `sum` on an int32 file returns int64; a file with `with_metadata=False` triggers `pytest.warns(UserWarning, match="metadata absent")` and the inferred dims are `(4, 3, 2)`; `from g4utils.HDF5 import read_g4vox_hdf5_3d` raises `ImportError`.

**Files to Touch:**
- `src/g4utils/HDF5/shared.py`
- `src/g4utils/HDF5/vti_export.py`
- `src/g4utils/HDF5/vox_file_base.py`
- `src/g4utils/HDF5/__init__.py`
- `tests/test_core_cleanups.py` (create)

**Verification Step:**

Run:
```bash
pytest tests/test_core_cleanups.py -v && pytest
```

Expected:
All pass. `grep -rn "print(\|_qty_whitelist\|read_g4vox_hdf5_3d" src` finds nothing.

**Notes:**

`shared.py` no longer needs the `TYPE_CHECKING` import of `G4VoxFile3D` once the wrapper is gone, so remove it.

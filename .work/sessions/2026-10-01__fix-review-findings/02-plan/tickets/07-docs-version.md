# Ticket 07: docs-version

**Model:** haiku

**Acceptance Criteria:**
- [ ] `README.md` replaces the template text with:
  - what the package reads (G4Vox HDF5 output) and the two layouts;
  - install instructions;
  - a quick start using `open_vox_file`, `select_quantity`, `select_subrun`, iteration with `sim.data`, `get`, `sum` and `total_primaries`;
  - run_log columns;
  - VTI/PVD export with the `encoding`/`compress` options;
  - development setup (`pip install -e ".[dev]"`, `pytest`, and the CI coverage command).
- [ ] The `pyproject.toml` description is "Readers and VTK export for G4Vox voxel HDF5 output from Geant4 simulations", and its keywords include geant4, geant4-dna, hdf5, voxel and vtk.
- [ ] Version is `0.2.0` in both `pyproject.toml` and `src/g4utils/__init__.py`.
- [ ] Every README code example uses only names that exist in the final API.

**Files to Touch:**
- `README.md`
- `pyproject.toml`
- `src/g4utils/__init__.py`

**Verification Step:**

Run:
```bash
pip install -e ".[dev]" && python -c "import g4utils; print(g4utils.__version__)" && pytest
```

Expected:
`0.2.0`, then all tests pass.

**Notes:**

Reference `CONTEXT.md` for terminology (subrun ID vs slice index). Do not mention matplotlib or plotting: that is a future feature.

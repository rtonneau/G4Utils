# Ticket 03: meso-vti-timeseries

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `SpeciesMesoSpatialFile.to_vti_timeseries(run, event, path, species=None, dtype=np.float32, extent=None, *, encoding="binary", compress=False)` writes `<stem>_0000.vti`, ... and a `.pvd` at `path`, with timestep = `time_ns`.
- [ ] Default extent is the common box over all snapshots of the event; an explicit extent overrides it.
- [ ] Each frame uses its own cell size as spacing; snapshots are read one at a time.
- [ ] An event with no occupied cells in any snapshot raises `ValueError`; unknown run/event raises `KeyError`.
- [ ] The README documents `to_vti`, `to_vti_timeseries`, array names, nm units and the errors.

**Files to Touch:**
- `src/g4utils/HDF5/species_meso_spatial.py`
- `tests/test_species_meso_spatial.py`
- `README.md`

**Verification Step:**

Run:
```bash
python -m pytest -q
```

Expected:
Full suite passes, including a time series test checking file names, `.pvd` timesteps and a shared physical box across frames.

**Notes:**

The common box needs a first pass over the snapshots for their bounds (read each once, keep only min/max), then a second pass to write frames. Follow the file naming and `.pvd` relative paths of the Vox `dump_selection_to_vti_timeseries`.

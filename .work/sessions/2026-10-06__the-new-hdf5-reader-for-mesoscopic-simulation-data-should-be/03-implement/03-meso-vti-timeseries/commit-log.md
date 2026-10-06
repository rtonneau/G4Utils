# Ticket 03: meso-vti-timeseries

**Status:** ✅ Done

## Local Test Result

`C:/Users/rtonneau/miniconda3/envs/GEANT4_py311/python.exe -m pytest -q`: 141 passed, including the new time series tests (file names, `.pvd` timesteps, shared box, explicit extent, errors).

## Review Notes

Implemented by a sonnet subagent (effort medium); reviewed here against the diff and the full suite re-run. `SpeciesMesoSpatialFile.to_vti_timeseries` makes a first pass that keeps only per-snapshot min/max (empty snapshots skipped) to build the common box, then a second pass that writes each frame through `MesoSpatialSnapshot.to_vti` with that extent and a `.pvd` with timestep = `time_ns`. Frame names use the snapshot index (`<stem>_0000.vti`). An event with no occupied cell and no extent raises `ValueError`; unknown run or event raises `KeyError`. README section added. No changes made after review.

## Blockers / Challenges

None. Tests use a lattice-aligned fixture defined in the test file, because the shared conftest fixture has random off-lattice positions that `to_vti` rejects. Lattice regularity of real dnachem-min files is still unverified.

## Commits

- 112c780 feat(meso): add to_vti_timeseries and document the VTI export (ticket 03)

## Time Spent

2m (ticket-start.js to ticket-complete.js)

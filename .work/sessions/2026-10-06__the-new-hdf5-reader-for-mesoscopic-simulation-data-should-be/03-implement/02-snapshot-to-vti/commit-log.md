# Ticket 02: snapshot-to-vti

**Status:** ✅ Done

## Local Test Result

`C:/Users/rtonneau/miniconda3/envs/GEANT4_py311/python.exe -m pytest tests/test_vti_export.py tests/test_species_meso_spatial.py -q`: 63 passed. Full suite: 138 passed.

## Review Notes

Implemented by a sonnet subagent (effort medium); reviewed here against the diff and the full suite re-run. `MesoSpatialSnapshot.to_vti` densifies via `_densify_snapshot`, builds a `VoxGeometry` from nm values and writes `<raw species>_count` and `<raw species>_M` through the unchanged `write_vti`. Unknown species raises `KeyError`; a snapshot without species names also raises `KeyError`. Imports are local to avoid a cycle. Tests: round trip against `concentration_M`, species subset plus unknown species, extent with compressed binary. No changes made after review; one redundant `species is None` check was left as is.

## Blockers / Challenges

None.

## Commits

- 95d5698 feat(meso): add MesoSpatialSnapshot.to_vti (ticket 02)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

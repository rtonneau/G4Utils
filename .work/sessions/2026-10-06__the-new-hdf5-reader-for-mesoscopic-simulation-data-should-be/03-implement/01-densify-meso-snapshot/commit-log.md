# Ticket 01: densify-meso-snapshot

**Status:** ✅ Done

## Local Test Result

`C:/Users/rtonneau/miniconda3/envs/GEANT4_py311/python.exe -m pytest -q` (the default `python` has no pytest): 135 passed, 24 of them in `tests/test_species_meso_spatial.py`.

## Review Notes

Implemented by a sonnet subagent (effort high); reviewed here against the diff and the full suite re-run. Checked every Acceptance Criterion: dense (nZ, nY, nX) arrays with 0 in unoccupied cells, origin = min centre - cell_size/2, extent growth by whole cells on the snapshot's anchor, `ValueError` for off-lattice centres (1e-3 cell tolerance), colliding cells and empty snapshot without extent. No changes made after review.

Deviation accepted: the index is round((pos - origin)/cell_size - 0.5), since the literal ticket formula gives x.5 for cell centres. Collision check sorts flat indices instead of `np.add.at`. The helper is private (`_densify_snapshot`, `_DenseGrid`) and returns `counts` as (S, nZ, nY, nX) uint32 plus a 1-based `index` array, which ticket 02 can use.

## Blockers / Challenges

None. Note for ticket 3 and real data: the conftest fixture has random off-lattice positions, so valid placement is tested with hand-built snapshots only; real-file lattice regularity is still unverified.

## Commits

- 2e7efcb feat(meso): densify sparse meso snapshots onto a lattice (ticket 01)

## Time Spent

4m (ticket-start.js to ticket-complete.js)

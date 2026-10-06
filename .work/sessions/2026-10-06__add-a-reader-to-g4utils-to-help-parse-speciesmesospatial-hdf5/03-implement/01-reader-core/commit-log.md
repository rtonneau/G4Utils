# Ticket 01: reader-core

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_species_meso_spatial.py -q` (run in the subagent's venv under `.scratch`): 8 passed. The subagent also reported the full suite at 112 passed, 6 skipped.

## Review Notes

Implemented by a sonnet subagent (medium effort); I reviewed the diff and re-ran the verification myself. Checked against the Acceptance Criteria: both classes and the index accessors exist, indices are sorted numerically (fixture uses events 0, 2, 10 and snapshots 0, 1, 2, 10), `read_snapshot` handles N = 0 and the hard-linked snapshot, missing run/event/snapshot raise `KeyError`, constructor raises `FileNotFoundError` / `ValueError`, `iter_snapshots` is lazy and filterable, exports added to `g4utils.HDF5`. No changes made after review.

## Blockers / Challenges

None

## Commits

- 2222371 feat: add SpeciesMesoSpatialFile reader for dnachem-min spatial HDF5 (ticket 01)

## Time Spent

3m (ticket-start.js to ticket-complete.js)

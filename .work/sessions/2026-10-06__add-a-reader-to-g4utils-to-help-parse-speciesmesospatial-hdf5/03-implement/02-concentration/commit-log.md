# Ticket 02: concentration

**Status:** ✅ Done

## Local Test Result

`python -m pytest -q` (subagent venv under `.scratch`): 117 passed, 6 skipped. The concentration tests (`-k concentration`) all pass.

## Review Notes

Implemented by a haiku subagent (low effort); I reviewed the diff and re-ran the full suite. Checked against the Acceptance Criteria: module-level `concentration_M` uses count / (N_A * (cell_size_nm * 1e-8)^3) in float64; `MesoSpatialSnapshot.concentration_M(species=None)` selects columns in the requested order via the new `species` field set by `read_snapshot`; empty snapshot gives (0, S); unknown species raises `KeyError`; exported from `g4utils.HDF5`. Changed after review: wrapped the over-long import line in `__init__.py` (ruff line length 100).

## Blockers / Challenges

None

## Commits

- 2e1cd94 feat: add concentration_M for SpeciesMesoSpatial snapshots (ticket 02)

## Time Spent

2m (ticket-start.js to ticket-complete.js)

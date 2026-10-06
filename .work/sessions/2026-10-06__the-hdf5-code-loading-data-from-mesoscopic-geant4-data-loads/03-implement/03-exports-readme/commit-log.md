# Ticket 03: exports-readme

**Status:** ✅ Done

## Local Test Result

`PYTHONPATH=src python -m pytest -q` (conda env GEANT4_py311): 144 passed; `from g4utils.HDF5 import MesoGrid, DenseMesoPeriod` works.

## Review Notes

Subagent (haiku/low) did exports and README. Review fixes: reverted its CHANGELOG.md edit (it landed in the released 0.4.0 section; the fragment is written by /gps finish) and replaced an invented species name in the read_dense example with the one used elsewhere.

## Blockers / Challenges

None

## Commits

- 11a4397 docs: export MesoGrid/DenseMesoPeriod and document dense readers (ticket 03)

## Time Spent

3m (ticket-start.js to ticket-complete.js)

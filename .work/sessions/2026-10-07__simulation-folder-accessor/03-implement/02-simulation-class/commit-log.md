# Ticket 02: simulation-class

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_simulation.py -q` (GEANT4_py311 env): 7 passed; re-run in this session with the same result.

## Review Notes

Implemented by a sonnet/medium subagent. Reviewed `simulation.py` against the Acceptance Criteria: results/ preferred else the folder, find_dumps discovery without reading data, ordered `subruns` mapping, `subrun()` KeyError listing names, iteration, len, `table()` from `Dump._columns()`, flat and results folders handled, FileNotFoundError from find_dumps. No changes after review.

## Blockers / Challenges

None.

## Commits

- c451e10 feat: add Simulation folder accessor over lazy Dumps (ticket 02)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

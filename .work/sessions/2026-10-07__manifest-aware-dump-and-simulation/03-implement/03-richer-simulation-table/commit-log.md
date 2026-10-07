# Ticket 03: richer-simulation-table

**Status:** ✅ Done

## Local Test Result

pytest tests/test_dnachem_simulation.py tests/test_dnachem_loaders.py -q: 14 passed; full suite re-run here: 199 passed (GEANT4_py311).

## Review Notes

Sonnet/medium subagent. Reviewed the diff: table() rows keep dump_columns (chemistry, pH, scavengers, events, energy, particle, beam energy) plus labels, and add chemistryModel, hand-over/end times, voxel and meso settings, threads and summed wallTime_s; missing fields are None. dump_columns and load_* untouched. No changes after review.

## Blockers / Challenges

None.

## Commits

- 6dffd49 feat: richer Simulation.table() with model, meso and timing columns (ticket 03)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

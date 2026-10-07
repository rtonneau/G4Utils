# Ticket 03: loaders-exports-docs

**Status:** ✅ Done

## Local Test Result

`python -m pytest -q` (GEANT4_py311 env): 186 passed; re-run in this session with the same result.

## Review Notes

Implemented by a sonnet/high subagent. Reviewed the loaders diff: load_species/load_reactions concatenate each Dump's cached frame with ignore_index (same processing as before, moved into Dump); load_reaction_table keeps its signature and errors; duplicated constants removed; Dump and Simulation exported; README section added. dump.py was also touched (outside the ticket list) to hold the reaction-table helpers and break the import cycle. One behavior change: load_* now use a results/ subfolder when path has one.

## Blockers / Challenges

None. No direct old-vs-new DataFrame comparison beyond the unchanged existing tests.

## Commits

- a2d83b7 refactor: rebuild DnaChem loaders on Simulation/Dump, export and document (ticket 03)

## Time Spent

2m (ticket-start.js to ticket-complete.js)

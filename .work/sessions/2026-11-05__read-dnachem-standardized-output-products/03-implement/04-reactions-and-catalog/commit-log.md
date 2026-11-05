# Ticket 04: reactions-and-catalog

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_reaction_table.py tests/test_dnachem_loaders.py tests/test_dnachem_dump.py -q` passes; full suite 423 passed, 8 skipped, only the pre-existing test_version failure.

## Review Notes

Inline. Criteria met. New reactions() adds count (=countSum) and time_s (=binEnd_ns in s). Legacy mapped: binEnd_ns=time, binStart_ns NaN, nEvents=totalEvents, countSum=count. Catalogue read via tables with container Reactions (h5/root). loaders.py unchanged: manifest folders already route through Dump.reaction_table.

## Blockers / Challenges

None

## Commits

- 492aa8d feat: read standardized Reactions and ReactionsCatalog (ticket 04)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

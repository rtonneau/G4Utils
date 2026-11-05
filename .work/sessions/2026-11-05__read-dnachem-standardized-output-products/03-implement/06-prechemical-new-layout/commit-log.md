# Ticket 06: prechemical-new-layout

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_prechemical.py -q`: 41 passed (+1 new alias test). Full suite: 441 passed, 8 skipped, 1 pre-existing test_version failure (fixed by ticket 7).

## Review Notes

Base HEAD a681f26; pre-existing changes: session config only. Checked all four criteria: new csv/txt per-event and h5/root single-file reads with preference, legacy keeps old columns plus standardized aliases (level NaN), inventory-first selection, allow_missing empty frame. Inline, no independent review (localized change).

## Blockers / Challenges

None

## Commits

- f5f587f feat: read standardized PreChemical (ticket 06)

## Time Spent

2m (ticket-start.js to ticket-complete.js)

# Ticket 01: reaction-table

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests -q` → 104 passed, 6 skipped (pre-existing skips). Also ran it with `-W error` on the real `C:/DEV/GEANT4/DATA/260924/run_100keV` Dump: shape (42, 31), 6 reactions produce H2O2, no warnings.

## Review Notes

Implemented by a sonnet subagent and reviewed inline against every Acceptance Criterion (column order, tuples with repeats, `(no products)`, int wide columns, error cases). No changes after review. The subagent also updated `tests/test_dnachem_ntuple.py::test_short_name_table`, which pins the exact `SHORT_NAMES` dict; that was needed for the two new entries. The commit also includes the CONTEXT.md glossary updates from the grill (Short name, Reaction table).

## Blockers / Challenges

None.

## Commits

- 2b63827 feat(DnaChem): add load_reaction_table with reactant/product columns (ticket 01)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

# Ticket 02: table-reader

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_tables.py -q` -> 8 passed. Also checked on the real smoke12 dump: csv/h5/root equal (rtol 1e-12), txt equal (rtol 1e-4) for Species, SpeciesMeso, Reactions, ReactionsCatalog, PhysicsInteractions, TrackLengths.

## Review Notes

Inline. Criteria met. Signature is `read_table(directory, stem, table, formats, files=None, *, container=None)` (stem = file stem, container = h5/root file stem, needed for the catalogue which lives in Reactions.h5/root); also `pick_format`, `read_table_file`. A fixed column order/dtype schema (TABLE_COLUMNS) makes h5 (alphabetical) match csv.

## Blockers / Challenges

None

## Commits

- 2c7086b feat: format-agnostic table reader for csv/h5/root/txt (ticket 02)

## Time Spent

2m (ticket-start.js to ticket-complete.js)

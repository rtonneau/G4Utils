# Ticket 02: export-and-docs

**Status:** ✅ Done

## Local Test Result

Import check printed `ok`; `python -m pytest tests -q` → 104 passed, 6 skipped.

## Review Notes

Implemented by a haiku subagent. Reviewed inline: export and `__all__` order are correct. Fixed the README after review: it omitted the direct CSV-file input and the scan-root `ValueError`, and "0 or count" was reworded to int counts per species in the file.

## Blockers / Challenges

None.

## Commits

- 5f354c4 docs(DnaChem): export load_reaction_table and document it in README (ticket 02)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

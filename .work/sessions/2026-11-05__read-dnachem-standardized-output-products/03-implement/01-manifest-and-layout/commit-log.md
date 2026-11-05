# Ticket 01: manifest-and-layout

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_manifest.py tests/test_dnachem_simulation.py tests/test_dnachem_dump.py -q` -> 54 passed.

## Review Notes

Base 0747fcf. Inline. All 5 criteria met and covered by tests. New fixture tests/data/Manifest_standardized.json (copy of smoke12). Empty-product check is `Dump._product_formats`, used by later tickets.

## Blockers / Challenges

None

## Commits

- f683fb9 feat: read outputFormats/hdf5 and detect standardized layout (ticket 01)

## Time Spent

1m (ticket-start.js to ticket-complete.js)

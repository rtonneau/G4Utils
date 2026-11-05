# Ticket 05: physics-and-tracks

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_readers.py tests/test_dnachem_dump.py -q` passes; full suite 431 passed, 8 skipped, only the pre-existing test_version failure. Real smoke12: physical_interactions, track_lengths (summary validated against manifest), reactions load.

## Review Notes

Inline. Criteria met. TRACK_COLUMNS now the new names, LEGACY_TRACK_COLUMNS for legacy CSV header (renamed on read). Real manifests still carry legacy names in the trackLengths summary (primaryEkin0_keV etc.), so validate_track_summary accepts either key. Legacy physical_interactions keeps label/count and gains process/nEvents/countSum/countMean. Existing readers tests only needed the header constant swapped.

## Blockers / Challenges

None

## Commits

- bbce765 feat: read standardized PhysicsInteractions and TrackLengths (ticket 05)

## Time Spent

2m (ticket-start.js to ticket-complete.js)

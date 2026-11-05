# Ticket 03: species-and-g

**Status:** ✅ Done

## Local Test Result

`python -m pytest tests/test_dnachem_dump.py tests/test_dnachem_estimates.py tests/test_dnachem_meso_species.py tests/test_dnachem_loaders.py -q` -> 108 passed. Full suite: 400 passed; only test_version fails, and it fails identically before my changes (installed package metadata is 0.3.0). Real smoke12: species(), meso_species(), particle_g_estimates() load.

## Review Notes

Inline. Criteria met. Legacy frames keep their old columns and gain the standardized ones (nEvents, countSum, countMean, gSum, gSumSq, gMean; NaN for countSumSq/countStdErr/gStdErr) so existing tests/users stay green. Meso keeps raw species + count/mean_count_per_event in both layouts. New fixture make_standardized_dump in conftest (extended by later tickets). HO3 left unmapped: its raw Geant4 name could not be verified from source, so it passes through unchanged. loaders.py/readers.py needed no change.

## Blockers / Challenges

None

## Commits

- eca6f5b feat: read standardized Species/SpeciesMeso and stored G estimators (ticket 03)

## Time Spent

3m (ticket-start.js to ticket-complete.js)

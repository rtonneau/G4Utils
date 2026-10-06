# Ticket 04 Implementation

**Status:** ✅ Done

## Commits

- 87c5763 docs(DnaChem): export public API and document loaders in README (ticket 04)

## Local Test Result

```
PYTHONPATH=src python -m pytest -o addopts="" tests -q
19 passed in 0.71s

Smoke check against C:\Users\rtonneau\DATA\Geant4\260930\results:
o2_percent: [0.0, 0.3, 2.0, 21.0]
OH G at 1e-12 s: 4.84218 (0%), 4.84347 (0.3%), 4.84342 (2%), 4.84440 (21%)
load_reactions: 526 rows; load_manifests: 4 rows
21% O2: O2- G rises to 3.13 at 1 us, e_aq falls to 0.017 (scavenging visible)
```

## Review Notes

Implemented by a subagent on haiku. Inline follow-up on the README: replaced "chemistry model" with "Chemistry name" (the term the dnachem glossary says to avoid), said that time_s replaces the ns column, documented the `path` semantics, the name-pattern decimal/float rule and the ValueError on a mismatch, and noted that species are keyed by name.

## Time Spent

~15 minutes

## Blockers / Challenges

None

## Token Usage

- **Input:** 196
- **Output:** 5983
- **Cache read:** 1300258
- **Cache creation:** 47079
- **Total:** 1353516

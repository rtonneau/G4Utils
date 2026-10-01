# Ticket 07 Implementation

**Status:** ✅ Done

## Commits

- `4d2fcc3` docs: write real README, describe package, bump to 0.2.0 (ticket 07)

## Local Test Result

```
pip install -e ".[dev]" && python -c "import g4utils; print(g4utils.__version__)" -> 0.2.0
pytest -> 75 passed, 6 skipped in 4.01s
README examples executed against Snapshot3D and Extendable4D fixture files -> OK (both layouts)
```

## Review Notes

Implemented by a haiku subagent. Inline follow-up by the session model fixed README inaccuracies: VTI described as "rectilinear grid" (it is ImageData); unverified G4Vox GitHub link removed; PyPI install claim replaced with install-from-source; wrong comment on `sum("Dose")`; export options now listed for `to_vti` too; added the 4D subrun-ID rule, a per-subrun `to_vti` loop example, and a CONTEXT.md pointer. All README snippets were then executed against fixture files for both layouts.

## Time Spent

~0.3 hours

## Blockers / Challenges

None

## Token Usage

- **Input:** 176
- **Output:** 8468
- **Cache read:** 1734084
- **Cache creation:** 53544
- **Total:** 1796272

# Ticket 07: packaging-docs-regression

**Status:** ✅ Done

## Local Test Result

`python -m pytest -q`: 449 passed, 8 skipped (pre-existing symlink/spatial skips). Needed `pip install -e . --no-deps` to refresh installed metadata to 0.5.0 for test_version.

## Review Notes

Base HEAD 9f42106. All six criteria checked: extra root+dev uproot, 0.5.0, regression on a 563K real smoke12 sample (csv/h5/root equal), validator new-layout branch run on it, README/CONTEXT/CHANGELOG updated. CI installs .[dev] so uproot is picked up. Not done: has_meso consulting outputFormats (optional). Inline, no independent review.

## Blockers / Challenges

None

## Commits

- d54bc75 feat: packaging, docs and real-dump regression (ticket 07)

## Time Spent

4m (ticket-start.js to ticket-complete.js)

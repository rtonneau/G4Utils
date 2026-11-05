# Ticket 07: packaging-docs-regression

**Model:** fast
**Effort:** medium

**Acceptance Criteria:**
- [ ] `pyproject.toml` has an optional extra `root` (`uproot`) included in `dev`; version is 0.5.0; `tests/test_version.py` passes.
- [ ] A regression test loads a real smoke12 sample (small files copied into `tests/data/`) in csv, h5 and root and checks all accessors give equal frames across formats.
- [ ] `tests/validate_dnachem_output.py` accepts a new-layout dump.
- [ ] README describes the new layout, the format preference, `outputFormats`, the changed `G`, and corrects the `/chem/meso/spatialOutput` line (now `/output/speciesMesoSpatial/format h5`).
- [ ] CONTEXT.md renames `ReactionsMetadata` to `ReactionsCatalog` (noting legacy) and defines Output product, Output format and per-event statistics.
- [ ] CHANGELOG has a 0.5.0 breaking-change entry with the old to new file and column map.

**Files to Touch:**
- `pyproject.toml`
- `tests/test_dnachem_standardized_regression.py`
- `tests/data/`
- `tests/validate_dnachem_output.py`
- `README.md`
- `CONTEXT.md`
- `CHANGELOG.md`
- `.github/workflows/ci.yml`

**Verification Step:**

Run:
```bash
python -m pytest -q
```

Expected:
The full suite passes, including the cross-format regression; no tests are skipped except `uproot` ones where it is not installed.

**Notes:**

Keep copied test data small (2 events, strip PreChemical to a few rows). Check that the CI install of `.[dev]` picks up `uproot` on all four Python versions.

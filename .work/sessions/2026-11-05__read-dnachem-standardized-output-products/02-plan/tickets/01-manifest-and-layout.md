# Ticket 01: manifest-and-layout

**Model:** medium
**Effort:** medium

**Acceptance Criteria:**
- [ ] `Manifest` has `outputFormats` (dict of product to tuple of format names, or None) and `hdf5` (bool or None), validated in `from_dict`.
- [ ] `_summary()` text and HTML show both; `Simulation.table()` includes `outputFormats`.
- [ ] `Dump.output_formats` returns the dict (empty for legacy) and `Dump._layout` is `"standardized"` iff the manifest has `outputFormats`, else `"legacy"`.
- [ ] A product whose list is empty raises a `FileNotFoundError` naming the product and `outputFormats`.
- [ ] Existing manifest tests still pass.

**Files to Touch:**
- `src/g4utils/DnaChem/manifest.py`
- `src/g4utils/DnaChem/simulation.py`
- `src/g4utils/DnaChem/dump.py`
- `tests/test_dnachem_manifest.py`
- `tests/test_dnachem_simulation.py`
- `tests/data/Manifest_real.json`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_manifest.py tests/test_dnachem_simulation.py tests/test_dnachem_dump.py -q
```

Expected:
All tests pass, including new ones for `outputFormats`, `hdf5`, layout detection and the `none` product error.

**Notes:**

Add a new-layout manifest (copy of the smoke12 `Manifest.json`) next to the legacy `Manifest_real.json`. Unknown keys still go to `raw`.

# Ticket 02: dump-manifest-files

**Model:** sonnet
**Effort:** high

**Acceptance Criteria:**
- [ ] `Dump.files` returns the Manifest `files` as a tuple (empty if absent); `Dump.prefix` returns the Manifest `prefix` (empty string if absent or null).
- [ ] `species()`, `reactions()`, `reaction_table()` and `.meso` look up data files as prefix + filename; with an empty prefix the behavior is unchanged.
- [ ] `Dump.has_meso` is true only if `mesoSpatialOutput` is not false and the meso h5 exists.
- [ ] `Dump.meso` raises `FileNotFoundError` with a message saying whether meso output was disabled in the Manifest or the file is missing.
- [ ] `load_species` / `load_reactions` output is unchanged for existing tests.

**Files to Touch:**
- `src/g4utils/DnaChem/dump.py`
- `tests/test_dnachem_dump.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_dump.py tests/test_dnachem_loaders.py tests/test_dnachem_simulation.py -q
```

Expected:
All tests pass.

**Notes:**

The `make_dump` fixture in `tests/conftest.py` may need a `prefix` option (add it there if so). `loaders.py` imports file-name constants and helpers from `dump.py`; keep the public loader signatures.

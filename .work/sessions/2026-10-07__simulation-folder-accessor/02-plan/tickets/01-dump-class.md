# Ticket 01: dump-class

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `Dump(path, name_pattern=None)` reads nothing at construction except that the folder exists and has `Manifest.json`.
- [ ] `.manifest` (a `Manifest`), `.species()`, `.reactions()`, `.reaction_table()` and `.meso` load on first access and return the same object afterwards.
- [ ] `species()` / `reactions()` return the DataFrame that `load_species` / `load_reactions` give for this single Dump (same columns and values, including Dump columns and Name pattern labels).
- [ ] `.labels` returns the Name pattern values (empty dict without a pattern); `.name` is the folder name.
- [ ] A missing data file raises `FileNotFoundError` naming the file.

**Files to Touch:**
- `src/g4utils/DnaChem/dump.py`
- `tests/test_dnachem_dump.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_dump.py tests/test_dnachem_loaders.py -q
```

Expected:
All tests pass.

**Notes:**

Reuse the processing now inside `load_species` / `load_reactions` (`short_name`, `time_s`, `G`, `dump_columns`, `parse_dump_name`). Use the `make_dump` fixture from `tests/conftest.py`. Do not change `loaders.py` yet.

# Ticket 04: reactions-and-catalog

**Model:** medium
**Effort:** medium

**Acceptance Criteria:**
- [ ] `Dump.reactions()` returns, for new dumps, `binStart_ns`, `binEnd_ns`, `nEvents` and the count statistics joined with the catalogue labels; legacy dumps are mapped onto the same columns.
- [ ] `load_reaction_table` and `Dump.reaction_table()` read `ReactionsCatalog.<ext>` (any format) and still accept a legacy `ReactionsMetadata.csv`.
- [ ] A folder holding a single new Dump resolves without the legacy file check.
- [ ] `load_reactions` works on both layouts.

**Files to Touch:**
- `src/g4utils/DnaChem/dump.py`
- `src/g4utils/DnaChem/loaders.py`
- `tests/test_dnachem_reaction_table.py`
- `tests/test_dnachem_loaders.py`
- `tests/test_dnachem_dump.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_reaction_table.py tests/test_dnachem_loaders.py tests/test_dnachem_dump.py -q
```

Expected:
All tests pass for both layouts.

**Notes:**

Legacy `time_s` has one time per row; new frames have a bin. Provide `time_s` as the bin end for new dumps (document it) so existing plotting code keeps working.

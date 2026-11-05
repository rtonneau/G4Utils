# Ticket 05: physics-and-tracks

**Model:** medium
**Effort:** medium

**Acceptance Criteria:**
- [ ] `physical_interactions()` returns `process`, `nEvents`, `countSum`, `countSumSq`, `countMean`, `countStdErr` for new dumps; legacy `label`/`count` are mapped onto `process`/`countSum` with other columns NaN; `per_event=True` uses `countMean` for new dumps.
- [ ] `track_lengths()` uses `primaryEkinStart_keV`, `secondaryFirstGenLength_nm`, `secondaryAllLength_nm` for both layouts (legacy columns renamed).
- [ ] `validate_track_summary` and `warn_track_event_count` use the new column names and still agree with the manifest `trackLengths` object.
- [ ] Constants `TRACK_COLUMNS` and friends are updated.

**Files to Touch:**
- `src/g4utils/DnaChem/readers.py`
- `src/g4utils/DnaChem/dump.py`
- `tests/test_dnachem_readers.py`
- `tests/test_dnachem_dump.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_readers.py tests/test_dnachem_dump.py -q
```

Expected:
All tests pass for both layouts.

**Notes:**

Strict CSV validation of the legacy readers is kept for legacy files; new files go through `read_table` and a lighter schema check (required columns present, `primaryEnd` in `stopped`/`escaped`/`killed`, no duplicate `(run, event)`).

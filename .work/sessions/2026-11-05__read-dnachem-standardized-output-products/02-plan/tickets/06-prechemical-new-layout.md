# Ticket 06: prechemical-new-layout

**Model:** medium
**Effort:** medium

**Acceptance Criteria:**
- [ ] `Dump.prechemical_records()` returns `run`, `event`, `parentTrackId`, `molecule`, `modification`, `level`, `energy_eV`, `parentX_nm`, `parentY_nm`, `parentZ_nm`, `x_nm`, `y_nm`, `z_nm` for new dumps, from per-event `.csv`/`.txt` files or a single `PreChemical.h5`/`.root`, following the format preference.
- [ ] Legacy fixed-width `.txt` dumps still load and are mapped onto the same columns (`level` NaN).
- [ ] Selection stays inventory-first; a prefixed dump, a duplicate `(run, event)` and a listed missing file behave as before.
- [ ] `allow_missing=True` still returns an empty frame with the right columns.

**Files to Touch:**
- `src/g4utils/DnaChem/prechemical.py`
- `src/g4utils/DnaChem/dump.py`
- `tests/test_dnachem_prechemical.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_prechemical.py -q
```

Expected:
All tests pass for both layouts.

**Notes:**

In new dumps `PreChemical.h5` / `.root` sit beside the per-event files; read only the preferred format, never both. The old regex `…\.txt\Z` currently raises "ambiguous pre-chemical filename" on any other `PreChemical*` name, which must change.

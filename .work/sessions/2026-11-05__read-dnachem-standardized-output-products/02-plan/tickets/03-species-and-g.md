# Ticket 03: species-and-g

**Model:** medium
**Effort:** high

**Acceptance Criteria:**
- [ ] `Dump.species()` and `Dump.meso_species()` read `Species.<ext>` / `SpeciesMeso.<ext>` via `read_table` on new dumps and return the same column set for legacy dumps, mapped onto the new names (missing statistics NaN).
- [ ] For new dumps `species()["G"]` equals `gMean`; for legacy dumps it stays `number / (totalEnergyDeposit_eV / 100)`.
- [ ] `Dump.particle_g_estimates()` uses the stored `gMean`, `gStdErr` and `nEvents` for new dumps, with no moment-tolerance logic; the legacy path is unchanged.
- [ ] `time_s` and short species names are still added; `load_species` works on both layouts.
- [ ] The `has_meso_species` property works for both layouts.

**Files to Touch:**
- `src/g4utils/DnaChem/dump.py`
- `src/g4utils/DnaChem/estimates.py`
- `src/g4utils/DnaChem/readers.py`
- `src/g4utils/DnaChem/loaders.py`
- `tests/test_dnachem_dump.py`
- `tests/test_dnachem_estimates.py`
- `tests/test_dnachem_meso_species.py`
- `tests/test_dnachem_loaders.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_dump.py tests/test_dnachem_estimates.py tests/test_dnachem_meso_species.py tests/test_dnachem_loaders.py -q
```

Expected:
All tests pass for both layouts.

**Notes:**

The new `Species.csv` has raw species names (`H3O^1`), so `short_name` still applies. Check that `HO3` (Tonneau2025) is handled. `read_ntuple` stays for legacy dumps only.

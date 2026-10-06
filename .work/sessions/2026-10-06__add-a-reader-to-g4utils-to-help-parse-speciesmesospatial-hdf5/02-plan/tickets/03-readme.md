# Ticket 03: readme

**Model:** haiku
**Effort:** low

**Acceptance Criteria:**
- [ ] README has a section on reading `SpeciesMesoSpatial.h5` with a short example: open the file, list runs and events, read the last snapshot, compute concentrations.
- [ ] The example matches the implemented API names exactly.
- [ ] A one-line mention of the reader sits near the existing HDF5 and DnaChem overview.

**Files to Touch:**
- `README.md`

**Verification Step:**

Run:
```bash
python -m pytest -q
```

Expected:
The full test suite passes (README change must not break anything).

**Notes:**

Place the section after the DnaChem section. State that the file is written by dnachem-min when `/chem/meso/spatialOutput true` is set, and that the reader is single-file (no Dump discovery).

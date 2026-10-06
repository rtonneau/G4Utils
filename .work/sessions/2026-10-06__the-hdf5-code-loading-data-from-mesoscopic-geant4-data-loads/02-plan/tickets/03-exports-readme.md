# Ticket 03: exports-readme

**Model:** haiku
**Effort:** low

**Acceptance Criteria:**
- [ ] `MesoGrid` and `DenseMesoPeriod` are exported from `g4utils.HDF5`.
- [ ] README documents `to_dense` and `read_dense` with an example and the memory caveat.
- [ ] A changelog fragment is added following the repo's convention.

**Files to Touch:**
- `src/g4utils/HDF5/__init__.py`
- `README.md`
- `CHANGELOG.md`

**Verification Step:**

Run:
```bash
python -c "from g4utils.HDF5 import MesoGrid, DenseMesoPeriod" && python -m pytest -q
```

Expected:
Import succeeds and the full test suite passes.

**Notes:**

Check how earlier sessions recorded changelog fragments (see .work/ and CHANGELOG.md) and follow that.

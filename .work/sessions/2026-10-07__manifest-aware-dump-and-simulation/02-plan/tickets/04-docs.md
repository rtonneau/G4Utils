# Ticket 04: docs

**Model:** haiku
**Effort:** low

**Acceptance Criteria:**
- [ ] README "Opening a simulation folder" mentions `files`, `prefix`, `has_meso` and the richer `table()`.
- [ ] `CONTEXT.md` states that a Dump is one flush, may cover several `/run/beamOn`, and that each `/run/beamOn` is a Manifest run; it notes that `Manifest.json` is never prefixed and the Manifest `prefix` field prefixes the data files.

**Files to Touch:**
- `README.md`
- `CONTEXT.md`

**Verification Step:**

Run:
```bash
python -m pytest -q
```

Expected:
The whole suite passes.

**Notes:**

Keep the existing wording style; edit entries in place rather than adding new sections.

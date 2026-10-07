# Ticket 03: exports-docs

**Model:** haiku
**Effort:** low

**Acceptance Criteria:**
- [ ] `read_manifest` (and `Manifest`) are importable from `g4utils.DnaChem` and listed in `__all__`.
- [ ] README DnaChem section documents `read_manifest`, attribute access, `runs_table()`, `scavenger_molarity()`, and the notebook display, with a short example.
- [ ] A CHANGELOG fragment is added in the style of the existing `.work/changelog` files.

**Files to Touch:**
- `src/g4utils/DnaChem/__init__.py`
- `README.md`
- `.work/changelog/` (new fragment, following the existing ones)

**Verification Step:**

Run:
```bash
python -c "from g4utils.DnaChem import read_manifest, Manifest; print(read_manifest.__name__)" && python -m pytest -q
```

Expected:
Prints `read_manifest`, then the full test suite passes.

**Notes:**

Look at the existing fragments in `.work/changelog/` for the format before writing the new one.

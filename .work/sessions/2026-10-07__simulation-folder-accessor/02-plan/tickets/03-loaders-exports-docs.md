# Ticket 03: loaders-exports-docs

**Model:** sonnet
**Effort:** high

**Acceptance Criteria:**
- [ ] `load_species` and `load_reactions` are implemented with `Simulation`/`Dump` and return identical DataFrames; all existing tests pass unchanged.
- [ ] `Simulation` and `Dump` are exported from `g4utils.DnaChem` (`__all__`).
- [ ] The README documents `Simulation` with a short example (open the folder, pick a subrun, load species, access the meso file).

**Files to Touch:**
- `src/g4utils/DnaChem/loaders.py`
- `src/g4utils/DnaChem/__init__.py`
- `README.md`

**Verification Step:**

Run:
```bash
python -m pytest -q
```

Expected:
The whole suite passes.

**Notes:**

`find_dumps` stays. Keep the public signatures of `load_species`, `load_reactions`, `load_reaction_table`. Avoid circular imports: `loaders.py` imports from `simulation.py`, not the reverse.

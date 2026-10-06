# Ticket 04: exports-docs

**Model:** haiku

**Acceptance Criteria:**
- [ ] `g4utils.DnaChem.__init__` exports `load_species`, `load_reactions`, `load_manifests`, `find_dumps`, `read_ntuple`, `short_name` and `SHORT_NAMES` in `__all__`.
- [ ] The README gets a "DnaChem: loading dnachem-min output" section with the O2-scan example: `load_species(results, name_pattern=r"run_(?P<o2_percent>[\dp]+)pO2")`, filtered with `species.isin([...])` and pivoted on `time_s` × `species` for `G`. It also lists the added columns.
- [ ] A smoke check against the real data loads the 4 Dumps in `C:\Users\rtonneau\DATA\Geant4\260930\results` and prints `o2_percent` {0, 0.3, 2, 21} and the G of OH at the first time point for each.

**Files to Touch:**
- `src/g4utils/DnaChem/__init__.py`
- `README.md`

**Verification Step:**

Run:
```bash
PYTHONPATH=src python -m pytest -o addopts="" tests -q && PYTHONPATH=src python -c "from g4utils.DnaChem import load_species; df = load_species(r'C:\Users\rtonneau\DATA\Geant4\260930\results', name_pattern=r'run_(?P<o2_percent>[\dp]+)pO2'); print(sorted(df.o2_percent.unique())); print(df[(df.species=='OH') & (df.time_s==df.time_s.min())][['o2_percent','G']])"
```

Expected:
18 passed, then `[0.0, 0.3, 2.0, 21.0]` and four OH G rows of about 4.8.

**Notes:**

The smoke check only reads the DATA folder (it isn't a test), so CI stays independent of it.

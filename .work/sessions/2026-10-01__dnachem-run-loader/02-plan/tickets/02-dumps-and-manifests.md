# Ticket 02: dumps-and-manifests

**Model:** sonnet

**Acceptance Criteria:**
- [ ] `find_dumps(path: str | Path) -> list[Path]` in `src/g4utils/DnaChem/dumps.py`. If `path/Manifest.json` exists, it returns `[path]`. Otherwise it returns the direct subdirectories of `path` that contain a `Manifest.json`, sorted by name. If `path` has no manifest and no Dump subfolders, it raises `FileNotFoundError` naming `path`.
- [ ] `parse_dump_name(name: str, pattern: str | re.Pattern | None) -> dict[str, float | str]` in `dumps.py`. With `None` it returns `{}`. Otherwise it uses `re.search`. With no match it raises `ValueError` naming the folder and the pattern. Each named group's value has `p` replaced by `.` and is cast to `float` when that parses; otherwise the original string is kept.
- [ ] `dump_columns(manifest: dict, dump_name: str) -> dict` in `src/g4utils/DnaChem/manifest.py` returns `dump`, `chemistry`, `pH`, `totalEvents`, `totalEnergyDeposit_eV`, one `<species>_molarity_M` per scavenger, and `particle` / `beamEnergy_keV`. Those last two are set when all runs agree; otherwise they are NaN with a `UserWarning` that names the Dump.
- [ ] `load_manifests(path, name_pattern=None) -> pd.DataFrame` in `manifest.py` returns one row per run: the `dump_columns` + `parse_dump_name` columns, plus each run field as `run` (the index) and `run_<key>` for the other keys. A Dump with no runs gives one row with no run columns.
- [ ] Tests in `tests/test_dnachem_manifest.py`: `test_find_dumps_parent_skips_non_dumps` (parent also has `macro/` and `run_0pO2.log`), `test_find_dumps_single_dump`, `test_find_dumps_none_raises`, `test_parse_dump_name_values` (`run_0p3pO2` → 0.3, `run_21pO2` → 21.0, and a pattern `run_(?P<tag>[a-z]+)` on `run_abc` → `"abc"`), `test_parse_dump_name_mismatch_raises`, `test_dump_columns_o2_molarity`, `test_dump_columns_mixed_beams_warns`, `test_load_manifests_rows_and_pattern`.

**Files to Touch:**
- `src/g4utils/DnaChem/dumps.py`
- `src/g4utils/DnaChem/manifest.py`
- `tests/test_dnachem_manifest.py`

**Verification Step:**

Run:
```bash
PYTHONPATH=src python -m pytest -o addopts="" tests/test_dnachem_manifest.py -q
```

Expected:
8 passed

**Notes:**

Read manifests with `json.loads(path.read_text(encoding="utf-8"))`. Missing `scavengers` / `runs` keys are treated as empty lists. For the mixed-beams test, call `make_dump(..., runs=[{run 0, particle "e-", beamEnergy_keV 100}, {run 1, particle "e-", beamEnergy_keV 200}])` and check with `pytest.warns(UserWarning)` that `beamEnergy_keV` is NaN and `particle == "e-"`. The two columns are checked independently.

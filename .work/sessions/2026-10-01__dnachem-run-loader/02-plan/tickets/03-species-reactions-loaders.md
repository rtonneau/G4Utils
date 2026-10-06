# Ticket 03: species-reactions-loaders

**Model:** sonnet

**Acceptance Criteria:**
- [ ] `load_species(path, name_pattern=None) -> pd.DataFrame` in `src/g4utils/DnaChem/loaders.py`. For each Dump from `find_dumps`, it reads `Species_nt_species.csv` with `read_ntuple`, adds `species = short_name(speciesName)`, replaces `time` with `time_s = time * 1e-9`, adds `G = number / (totalEnergyDeposit_eV / 100)`, and adds the `dump_columns` and `parse_dump_name` columns. All Dumps are concatenated with `ignore_index=True`.
- [ ] `load_reactions(path, name_pattern=None) -> pd.DataFrame` in `loaders.py`. It reads `Reactions_nt_reactions.csv`, merges the `reaction` label from `ReactionsMetadata.csv` on `reactionId` (left join), replaces `time` with `time_s`, keeps the raw `count` and adds the same Dump columns. There is no G column.
- [ ] A missing data file raises `FileNotFoundError` naming the file.
- [ ] Tests in `tests/test_dnachem_loaders.py`: `test_load_species_g_value` (number 480000 with 1e7 eV gives G 4.8 for `OH` at `time_s` 1e-12), `test_load_species_time_s_no_rounding` (the max `time_s` equals `999.999e-9` via `pytest.approx(rel=1e-12)`), `test_load_species_scan_with_pattern` (three Dumps `run_0pO2`, `run_0p3pO2`, `run_21pO2` under one parent; `sorted(df.o2_percent.unique()) == [0.0, 0.3, 21.0]`, and `O2_molarity_M` is present), `test_load_species_single_dump_path`, `test_load_reactions_labels_and_counts`, `test_load_species_missing_file_raises`.

**Files to Touch:**
- `src/g4utils/DnaChem/loaders.py`
- `tests/test_dnachem_loaders.py`

**Verification Step:**

Run:
```bash
PYTHONPATH=src python -m pytest -o addopts="" tests -q
```

Expected:
18 passed

**Notes:**

The Dump-level columns are the same on every row of a Dump. Assign them with `df.assign(**cols)`, where `cols` is `dump_columns(...) | parse_dump_name(...)`. Use one private helper, `_load_table(path, filename, name_pattern)`, for the shared loop so that the two loaders differ only in their post-processing.

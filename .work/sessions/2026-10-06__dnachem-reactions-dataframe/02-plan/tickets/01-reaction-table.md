# Ticket 01: reaction-table

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `SHORT_NAMES` maps `"O^-1" -> "O-"` and `"O_3^-1" -> "O3-"`.
- [ ] `load_reaction_table(path: str | Path) -> pd.DataFrame` in `src/g4utils/DnaChem/loaders.py` accepts a Dump folder or the `ReactionsMetadata.csv` file directly, and both give equal frames.
- [ ] Columns, in this order: `reactionId`, `reaction` (raw), `equation`, `reactants`, `products`, then `reactant_<X>` for each species in sorted order, then `product_<X>` for each species in sorted order. One row per reaction, in file order.
- [ ] `reactants` / `products` are tuples of Short names that keep stoichiometric repeats in file order. `(no products)` gives `()`.
- [ ] `equation` joins the Short names with ` + ` and ` -> ` (e.g. `OH + OH -> H2O2`). An empty product side shows as `(no products)`.
- [ ] Wide columns cover every species that appears anywhere in the file, with both prefixes, as int counts (0 where absent). Names use the Short name verbatim (`product_OH-`, `reactant_H3O+`).
- [ ] Unknown species keep their raw name (via `short_name`).
- [ ] A path containing several Dumps raises `ValueError` naming the count. A folder that holds no metadata file and no Dumps raises `FileNotFoundError`, and so does a missing file path. A `reaction` without `->` raises `ValueError` that quotes the line.

**Files to Touch:**
- `src/g4utils/DnaChem/species_names.py`
- `src/g4utils/DnaChem/loaders.py`
- `tests/test_dnachem_reaction_table.py` (new)

**Verification Step:**

Run:
```bash
python -m pytest tests -q
```

Expected:
All tests pass, including the new `tests/test_dnachem_reaction_table.py`.

**Notes:**

Use TDD. Write tests first in `tests/test_dnachem_reaction_table.py` that create their own `ReactionsMetadata.csv` in `tmp_path` (and use `make_dump` from `tests/conftest.py` for the Dump-folder case; its metadata holds `1,H3O^1 + OH^-1 -> (no products)` and `25,°OH^0 + °OH^0 -> H2O2^0`). Required tests:
- `test_dump_folder_and_file_equal`: `assert_frame_equal(load_reaction_table(d), load_reaction_table(d / "ReactionsMetadata.csv"))`.
- `test_no_products`: row for id 1 has `products == ()`, `equation == "H3O+ + OH- -> (no products)"`, `reactant_H3O+ == 1`, `product_H3O+ == 0`.
- `test_stoichiometry`: line `19,O^-1 + O^-1 -> H2O2^0 + OH^-1 + OH^-1` gives `reactants == ("O-", "O-")`, `reactant_O- == 2`, `product_OH- == 2`, `product_H2O2 == 1`, and the wide columns have int dtype.
- `test_filter_by_product_and_reactant`: with ids 25 and 19, `df[df["product_H2O2"] > 0].reactionId` gives both, and `df[df["reactant_OH"] > 0].reactionId` gives `[25]`.
- `test_unknown_species_kept_raw`: `X^0 + °OH^0 -> Y^0` gives the column `reactant_X^0`.
- `test_scan_root_raises`: two `make_dump` folders under `tmp_path`, so `load_reaction_table(tmp_path)` raises `ValueError`.
- `test_missing_file_raises` and `test_malformed_line_raises` (`ValueError`, message contains the line).
- `test_short_names_o_minus`: `short_name("O^-1") == "O-"`, `short_name("O_3^-1") == "O3-"`.

Implementation: resolve the file. If `path` is a file, use it. If `path / REACTIONS_METADATA_FILE` exists, use that. Otherwise call `find_dumps(path)`: one Dump → its file, several → `ValueError`. Check what `find_dumps` raises or returns when there are none, and give `FileNotFoundError` there. Read with the existing `_read_metadata`. Split each side with `re.split(r"\s+\+\s+", side.strip())`. Build the wide columns in one `pd.DataFrame` and concat them, rather than inserting column by column (that triggers pandas fragmentation warnings). Follow the existing module style: `from __future__ import annotations`, short docstrings.

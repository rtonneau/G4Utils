# Ticket 02: table-reader

**Model:** heavy
**Effort:** high

**Acceptance Criteria:**
- [ ] `read_table(directory, product, table, formats, files)` returns a DataFrame from the first available format in the order csv, h5, root, txt, restricted to the manifest's list and the files present.
- [ ] csv, h5 and root of the same table give frames with equal values and identical dtypes (int64 integers, float64 doubles, `string` strings).
- [ ] h5 strings are decoded; root integer columns are cast back from double.
- [ ] Missing `uproot` gives an `ImportError` explaining `pip install g4utils[root]`.
- [ ] txt is used only when no other format exists; the reaction catalogue txt (labels containing spaces) parses correctly.

**Files to Touch:**
- `src/g4utils/DnaChem/tables.py`
- `tests/conftest.py`
- `tests/test_dnachem_tables.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_tables.py -q
```

Expected:
All tests pass, including the cross-format equality test and the txt catalogue test.

**Notes:**

Layouts: h5 has one group per table and one dataset per column (root attribute `formatVersion` = 1); root has one TTree per table with the column as branch name. Fixtures in `conftest.py` should build the same small table in all four formats. `uproot` tests use `pytest.importorskip`.

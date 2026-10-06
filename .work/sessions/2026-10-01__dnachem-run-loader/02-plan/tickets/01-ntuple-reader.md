# Ticket 01: ntuple-reader

**Model:** sonnet

**Acceptance Criteria:**
- [ ] `src/g4utils/DnaChem/` exists with `__init__.py` (empty `__all__` for now) and `py.typed`.
- [ ] `read_ntuple(path: Path) -> pd.DataFrame` in `src/g4utils/DnaChem/ntuple.py` takes the column names from the `#column <type> <name>` lines, skips every other `#` line, reads as UTF-8, and casts `int` columns to int64, `double` columns to float64 and `string` columns to str.
- [ ] `SHORT_NAMES: dict[str, str]` and `short_name(raw: str) -> str` in `src/g4utils/DnaChem/species_names.py`. Table: H3O^1→H3O+, °OH^0→OH, OH^-1→OH-, e_aq^-1→e_aq, H^0→H, H_2^0→H2, H2O2^0→H2O2, HO_2^-1→HO2-, O_2^0→O2, °O^0→O, O_2^-1→O2-, HO_2°^0→HO2. An unknown name is returned unchanged.
- [ ] `tests/conftest.py` provides a `make_dump` fixture (a factory; interface in Notes).
- [ ] Tests: `test_read_ntuple_species_columns_and_dtypes`, `test_read_ntuple_keeps_degree_names_crlf` (`"°OH^0"` and `"HO_2°^0"` come back exactly), `test_short_name_table`, `test_short_name_unknown_passthrough`.

**Files to Touch:**
- `src/g4utils/DnaChem/__init__.py`
- `src/g4utils/DnaChem/py.typed`
- `src/g4utils/DnaChem/ntuple.py`
- `src/g4utils/DnaChem/species_names.py`
- `tests/conftest.py`
- `tests/test_dnachem_ntuple.py`

**Verification Step:**

Run:
```bash
PYTHONPATH=src python -m pytest -o addopts="" tests/test_dnachem_ntuple.py -q
```

Expected:
4 passed

**Notes:**

`make_dump` fixture: it returns a callable
`make_dump(parent: Path, name: str, *, molarity_M: float = 0.0, energy_deposit_eV: float = 1e7, runs: list[dict] | None = None, with_manifest: bool = True) -> Path`. It creates `parent/name/` and writes:
- `Manifest.json` (unless `with_manifest=False`): `schemaVersion` 1, `chemistry` "BoscoloChem", `pH` 7, `scavengers` `[{"species": "O2", "molarity_M": molarity_M}]`, `totalEvents` 100, `totalEnergyDeposit_eV` energy_deposit_eV, `runs` defaulting to `[{"run": 0, "events": 100, "particle": "e-", "beamEnergy_keV": 100, "energyDeposit_eV": energy_deposit_eV, "seed": 12345}]`.
- `Species_nt_species.csv`, with the real header (`#class tools::wcsv::ntuple`, `#title species`, `#separator 44`, `#vector_separator 59`, then `#column int speciesID`, `#column int number`, `#column int nEvent`, `#column string speciesName`, `#column double time`, `#column double sumG`, `#column double sumG2`). Rows for times 0.001 and 999.999 and species `0,H3O^1`, `1,°OH^0`, `3,e_aq^-1`, `7,HO_2°^0`, with fixed `number` values (e.g. 400000 / 480000 / 400000 / 3 at 0.001, and 270000 / 250000 / 260000 / 50 at 999.999) and nEvent 100.
- `Reactions_nt_reactions.csv` (`#column int reactionId`, `#column double time`, `#column int count`; rows `1,0.001,646` / `25,0.001,4436` / `25,999.999,9000`).
- `ReactionsMetadata.csv` (`reactionId,reaction` header; `1,H3O^1 + OH^-1 -> (no products)` and `25,°OH^0 + °OH^0 -> H2O2^0`).

Write every file with `encoding="utf-8", newline="\r\n"`. Read the ntuple data with `pd.read_csv(path, comment=None, skiprows=<number of header lines>, header=None, names=..., encoding="utf-8")`. Don't use `comment="#"`, because a species name could contain `#`.

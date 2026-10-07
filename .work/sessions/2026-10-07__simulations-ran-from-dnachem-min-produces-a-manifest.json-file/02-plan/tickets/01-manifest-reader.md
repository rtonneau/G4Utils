# Ticket 01: manifest-reader

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `read_manifest(path)` accepts a Dump folder, a `*Manifest.json` file (including `EndOfRun_Manifest.json`), or a folder holding exactly one Dump; several Dumps raise `ValueError` mentioning `load_manifests`; no manifest raises `FileNotFoundError`.
- [ ] `Manifest`, `ManifestRun` and `Scavenger` are frozen dataclasses whose fields are the JSON keys; absent fields are `None`, `scavengers` and `runs` are tuples; unknown keys are kept in `.raw`.
- [ ] `schemaVersion` above 1 emits a `UserWarning` and still parses; non-object JSON or a missing `schemaVersion` raises `ValueError` naming the file.
- [ ] `Manifest.runs_table()` returns a DataFrame with one row per Manifest run; `Manifest.scavenger_molarity("O2")` returns the molarity or `None`.
- [ ] `dump_columns` is built on `Manifest` and returns exactly the same dict as before; all existing tests pass unchanged.
- [ ] A real dnachem-min manifest is stored as a test fixture and parsed in a test.

**Files to Touch:**
- `src/g4utils/DnaChem/manifest.py`
- `tests/test_dnachem_manifest.py`
- `tests/data/Manifest_real.json`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_manifest.py tests/test_dnachem_loaders.py -q
```

Expected:
All tests pass, no failures.

**Notes:**

Real manifest source: `C:/Users/rtonneau/DEV/GEANT4/SIM/dnachem-min/.scratch/tests/2026-10-02__irt-syn-mesoscopic/02b-mt/Manifest.json`. Use `Path.resolve`/`is_file` for the path rules; reuse `find_dumps` for folders but also accept a file path directly and a folder containing a single `*Manifest.json`. Keep `load_manifests` unchanged. Glossary terms: Manifest, Manifest run.

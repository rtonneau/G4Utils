# Ticket 01: manifest-model-fields

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `Manifest` has typed `chemistryModel`, `handOverTime_ns`, `voxelSize_nm`, `mesoPixels`, `mesoTimesPerDecade`, `mesoSpatialOutput`; absent keys are `None`.
- [ ] They no longer appear in `raw` or under "Other" in the text and HTML display; a "Chemistry model" section lists those present.
- [ ] `load_manifests` and `dump_columns` output is unchanged.
- [ ] Tests parse a Manifest like the `run_21pO2` example (all new fields plus scavenger and a run).

**Files to Touch:**
- `src/g4utils/DnaChem/manifest.py`
- `tests/test_dnachem_manifest.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_manifest.py tests/test_dnachem_loaders.py -q
```

Expected:
All tests pass.

**Notes:**

Use the GEANT4_py311 interpreter (`C:/Users/rtonneau/miniconda3/envs/GEANT4_py311/python.exe`). Follow how existing fields and `_summary` sections are written.

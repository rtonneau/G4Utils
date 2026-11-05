# Plan: read dnachem-min standardized output products (ADR 0009)

Status: DRAFT, awaiting approval. Nothing implemented.

## Decisions (user, 2026-11-05)
1. Support legacy dumps AND the new layout, behind the **same entry points** (`Dump.species()`, `load_species()`, ...). Layout is detected per Dump; callers never choose.
2. Format preference when several exist: **csv > h5 > root > txt**.
3. `species()["G"]` becomes the per-event mean `gMean` for new dumps (legacy dumps keep pooled `number/(E/100)`).

## Detection rule
New layout iff the manifest has `outputFormats`. Otherwise legacy. A product with an empty list (`none`) raises a clear `FileNotFoundError` naming the product and `outputFormats`.

## Steps

### 1. Manifest (`manifest.py`, `simulation.py`)
- Add `outputFormats: dict[str, tuple[str,...]] | None` and `hdf5: bool | None` to `Manifest`; validate types in `from_dict`.
- Show both in `_summary()` (text + HTML); add `outputFormats` to the `Simulation.table()` field list (simulation.py:44).
- `mesoSpatialOutput` stays (now = spatial format list non-empty).

### 2. Table backend (new `tables.py`)
- `read_table(dump_dir, product, table, formats, files)` -> DataFrame, tries formats in order csv, h5, root, txt (restricted to those listed in `outputFormats` and present in `files`).
- csv: `pd.read_csv`. h5: h5py, one dataset per column, decode bytes strings, keep int64. root: `uproot` (optional import, clear error if missing), cast integer columns back from double using the schema. txt: skip `#` lines, whitespace table; reaction-catalogue labels contain spaces so catalogue uses csv/h5/root first, txt via fixed fallback only if nothing else.
- Return unit/description from h5 attrs where available (not required).
- Unified column dtypes regardless of format (tested for equality).

### 3. Readers (`readers.py`, new-layout versions next to legacy ones)
- Species / SpeciesMeso: same 11 columns; add `time_s`, short `species`, provenance columns as today.
- Reactions + `ReactionsCatalog` (replaces `ReactionsMetadata`); `load_reaction_table` accepts a catalogue file in any format or a Dump folder; keep legacy `ReactionsMetadata.csv`.
- PhysicsInteractions: `process` + stats; `per_event=True` uses `countMean` (legacy: divide by totalEvents). Keep legacy `label`/`count` output via rename so the frame has one schema (decide column names in step 6).
- TrackLengths: renamed columns (`primaryEkinStart_keV`, `secondaryFirstGenLength_nm`, `secondaryAllLength_nm`); update `validate_track_summary` keys; legacy names mapped to the new ones.
- PreChemical: new columns (`parentTrackId`, `modification`, `level`, ...); per-event `.csv`/`.txt` files or single `PreChemical.h5/.root`; update the filename regex; legacy fixed-width parser kept.
- `particle_g_estimates`: new path uses stored `gMean/gStdErr/nEvents` (no tolerance check); legacy path unchanged.

### 4. Dump (`dump.py`)
- `_layout` property; each accessor dispatches new/legacy.
- `species()["G"] = gMean` for new dumps; legacy unchanged.
- Add `Dump.output_formats`; `has_meso`/`meso` consider `outputFormats["speciesMesoSpatial"]` (h5 only for `SpeciesMesoSpatialFile`).
- Optional later: reader for `SpeciesMesoSpatial.root` long table (out of scope unless requested).

### 5. Packaging
- `pyproject.toml`: `uproot` as optional extra (`[root]`) and in `[dev]`; CI installs dev.
- Check `HO3` (Tonneau2025) short name.
- Version bump to 0.5.0 (behavior change of `G` and column names for new dumps); `CHANGELOG.md`.

### 6. Tests
- Fixture builders in `conftest.py` for new layout in csv/h5/root/txt; keep legacy builders.
- Update/extend: dump, readers, estimates, meso_species, prechemical, loaders, reaction_table, manifest, simulation.
- Cross-format equality test; `none` product error test; legacy-still-works test.
- Real-output regression using dnachem-min `.scratch/tests/2026-11-04__data-output-should-be-standardized-each-type-of-data-should-have/smoke12/all`; update `tests/validate_dnachem_output.py`.

### 7. Docs
- README: sections on PhysicsInteractions, TrackLengths, PreChemical, SpeciesMeso, G estimators, `load_species/reactions/reaction_table`, and the stale `/chem/meso/spatialOutput` (line ~473 -> `/output/speciesMesoSpatial/format h5`).
- CONTEXT.md: rename ReactionsMetadata -> ReactionsCatalog (note legacy); add Output product, Output format, per-event statistics terms.

## Open points to confirm while implementing
- Unified column names for PhysicsInteractions (`process` vs legacy `label`) and meso/species legacy frames: proposal is new names everywhere, legacy mapped onto them, with legacy-only columns left as NaN.
- txt reader scope (minimal fallback only).

## Order / commits
One commit per step 1-7; run `pytest` after each.

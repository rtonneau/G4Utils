# Session: dnachem-run-loader

**Date:** 2026-10-01T10:33:35.163Z
**Status:** Grill phase complete

## Problem Statement

Data from dnachem-min (e.g. `C:\Users\rtonneau\DATA\Geant4\260930\results\run_*`) is loaded in JupyterLab with ad-hoc code: hand-written species-ID dicts, energy parsed from folder names, G-values computed by hand. The new scan varies O2 content (`run_0pO2`, `run_0p3pO2`, `run_2pO2`, `run_21pO2`) instead of beam energy, and the hardcoded `sp_label` dict no longer matches the species IDs in the files (O2=11, O2-=12, HO2°=7, HO2-=8; the dict also has the key "HO2" twice). G4Utils needs a generic, tested loader that is independent of the scan parameter.

## Context & Constraints

- Each `run_*` folder is a **Dump** with a `Manifest.json` (schemaVersion 1): `scavengers[{species, molarity_M}]`, `chemistry`, `pH`, `totalEvents`, `totalEnergyDeposit_eV`, `runs[{run, events, particle, beamEnergy_keV, position_um, direction, energyDeposit_eV, seed}]`.
- `Species_nt_species.csv` and `Reactions_nt_reactions.csv` are Geant4 wcsv ntuples: `#class`, `#title`, `#separator`, `#vector_separator` and `#column <type> <name>` header lines, then plain CSV rows. They are UTF-8 (species names like `°OH^0`) with CRLF line endings. Time is in ns, and the last point is 999.999.
- Species columns: speciesID, number, nEvent, speciesName, time, sumG, sumG2. Reaction columns: reactionId, time, count. `ReactionsMetadata.csv` (header `reactionId,reaction`) maps IDs to labels.
- Package: `src/g4utils/` (hatchling, py>=3.10, pandas>=2, numpy). Existing subpackages `HDF5/` and `Vox/` use PascalCase directory names and contain `py.typed`.
- Glossary in `CONTEXT.md` (Dump, Manifest, Scan, Name pattern, G-value, Short name).

## Success Metrics

- `load_species(results_dir, name_pattern=r"run_(?P<o2_percent>[\dp]+)pO2")` returns one long DataFrame for all 4 Dumps, with `o2_percent` in {0, 0.3, 2, 21}, `O2_molarity_M`, `species`, `time_s` and `G`.
- The notebook's G-value-vs-time plots can be reproduced with a pivot on `time_s` × `species`, without any ID dict.
- The pytest suite passes against synthetic Dumps (it doesn't need the DATA folder), and CI stays green.

## Architecture & Approach

New subpackage `g4utils.DnaChem` with three public functions:

- **Dump discovery** (shared by all three functions): `path` is either one Dump folder or a parent folder. In a parent folder, every direct subfolder that has a `Manifest.json` is a Dump. A single folder without a manifest raises an error that names it. If no Dumps are found, it raises.
- **`load_manifests(path)`**: one row per run, with the Dump's flattened columns: `dump` (folder name), `<species>_molarity_M` per scavenger, `chemistry`, `pH`, `totalEvents`, `totalEnergyDeposit_eV`, plus Dump-level `particle` and `beamEnergy_keV` when all runs agree. When they differ, those columns are NaN and a warning is issued.
- **`load_species(path, name_pattern=None)`**: the raw columns (speciesID, number, nEvent, speciesName, sumG, sumG2), plus `species` (short name from a fixed table, raw-name fallback), `time_s` = time × 1e-9 (replacing the ns column), `G = number / (totalEnergyDeposit_eV / 100)`, the Dump-level manifest columns and the name-pattern columns.
- **`load_reactions(path, name_pattern=None)`**: reactionId, `reaction` label (from ReactionsMetadata.csv), `time_s`, raw `count`, the Dump-level manifest columns and the name-pattern columns. No G for reactions.
- **Name pattern**: an optional regex. Each named group becomes a column. In a value, `p` is replaced by `.` and the value is cast to float when it parses as a number; otherwise it stays a string. A Dump folder that doesn't match raises an error that names the folder.
- **ntuple reader**: column names come from the `#column` lines, and files are read as UTF-8.
- **Short-name table**: H3O^1→H3O+, °OH^0→OH, OH^-1→OH-, e_aq^-1→e_aq, H^0→H, H_2^0→H2, H2O2^0→H2O2, HO_2^-1→HO2-, O_2^0→O2, °O^0→O, O_2^-1→O2-, HO_2°^0→HO2.

## Assumptions & Trade-offs

- G is computed from counts and the manifest total energy deposit (the user's choice), not from sumG/sumG2. There are no uncertainty columns; the raw sumG/sumG2 columns are kept so this can be added later.
- Functions rather than a class, which keeps notebook use simple. Each call re-reads the files.
- The fixed short-name table is safer than rule-based conversion. New species fall back to their raw name.
- Folder-name labels are only a convenience; the manifest stays the source of truth.

## Open Questions

None.

## Notes

- Tests: pytest fixtures write tiny synthetic Dumps (Manifest.json + ntuple CSVs in the real format, with `°` names and a 999.999 time) into `tmp_path`.
- README gets a short usage section.

## Token Usage

- **Input:** 52
- **Output:** 14002
- **Cache read:** 1720646
- **Cache creation:** 31977
- **Total:** 1766677

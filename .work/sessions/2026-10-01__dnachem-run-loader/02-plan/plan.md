# Implementation Plan

**Session:** dnachem-run-loader
**Date:** 2026-10-01T10:43:18.268Z
**Estimated effort:** half a day

## Strategy

Add a `g4utils.DnaChem` subpackage, built bottom-up with TDD:
1. A reader for Geant4 wcsv ntuples, plus the short-name table for species.
2. Dump discovery, the name-pattern parser and the Manifest flattening, which together give `load_manifests`.
3. `load_species` and `load_reactions` on top of 1 and 2.
4. Public exports and README usage, followed by a smoke run on the real 260930 data.

Everything is tested against synthetic Dumps written by one shared pytest fixture (`make_dump` in `tests/conftest.py`), using the real file format: `#` header lines, UTF-8 `°` names, CRLF line endings and a 999.999 time point.

**Spec:** `.work/sessions/2026-10-01__dnachem-run-loader/01-grill/resume.md`

**Global constraints:**
- Python >= 3.10, pandas >= 2.0, numpy. No new runtime dependencies.
- Code style matches `src/g4utils/HDF5/*`: `from __future__ import annotations`, type hints, private helpers prefixed with `_`, and the `# ═══` / `# ──` section rules where a file has several sections.
- The subpackage directory is PascalCase (`DnaChem/`) and contains a `py.typed` file, like `HDF5/` and `Vox/`.
- Species are always identified by name, never by numeric ID.
- G-value formula: `G = number / (totalEnergyDeposit_eV / 100)`. Time: `time_s = time_ns * 1e-9`, with no rounding.
- Local test command (base anaconda has no pytest-cov and g4utils isn't installed): `PYTHONPATH=src python -m pytest -o addopts="" tests -q`

**Review focus** (each one is tested in the ticket that owns the code):
1. CRLF + UTF-8 `°OH^0` / `HO_2°^0` names must round-trip exactly (ticket 01).
2. A parent folder that also holds `run_0pO2.log` files and a non-Dump folder (`macro/`): only subfolders with a manifest are taken (ticket 02).
3. `path` pointing directly at one Dump folder works the same as pointing at its parent (tickets 02 and 03).
4. Name-pattern values `"0p3"` → 0.3, `"21"` → 21.0, `"abc"` → `"abc"`, and a folder that doesn't match raises with its name in the message (ticket 02).
5. A Dump with two runs whose beams differ gets NaN `particle`/`beamEnergy_keV` and a `UserWarning` (ticket 02). The `time_s` value for 999.999 ns stays 9.99999e-07 (ticket 03).

## Tickets Overview

| # | Ticket | Delivers |
|---|---|---|
| 01 | ntuple-reader | `DnaChem/` skeleton, `read_ntuple`, `SHORT_NAMES` / `short_name`, `make_dump` fixture |
| 02 | dumps-and-manifests | `find_dumps`, `parse_dump_name`, `dump_columns`, `load_manifests` |
| 03 | species-reactions-loaders | `load_species`, `load_reactions` |
| 04 | exports-docs | `__init__` exports, README section, real-data smoke check |

## Sequencing Rationale

Each ticket uses only what the tickets before it produce: 03 joins the ntuple tables (01) with the Dump-level columns (02), and 04 exports and documents the finished API. The shared fixture lands in 01 so that every later test reuses it.

## Risks & Mitigation

- **The wcsv header format drifts** (new `#` lines). Mitigation: names come only from `#column <type> <name>` lines; any other `#` line is ignored.
- **The `°` encoding on Windows.** Mitigation: always open files with `encoding="utf-8"`, and the fixture writes UTF-8 with CRLF.
- **The real data differs from the fixture.** Mitigation: ticket 04 runs a smoke check against `C:\Users\rtonneau\DATA\Geant4\260930\results`.

## Assumptions

- A Manifest with `schemaVersion` 1 is the only format to support. Missing optional keys (`scavengers`, `runs`) become empty lists.
- Only the direct subfolders of a parent folder are scanned (no recursion).
- Name-pattern matching uses `re.search`, not `fullmatch`.

## Token Usage

- **Input:** 14
- **Output:** 9415
- **Cache read:** 653935
- **Cache creation:** 17881
- **Total:** 681245

# Session: simulations ran from 'dnachem-min' produces a 'manifest.json' file with relevant information about a simulation run. Add a reader and a way to easily access these pieces of information. It should also produce a pretty printer kind function to display it within a jupyter notebook.

**Date:** 2026-10-07T14:40:38.169Z
**Status:** Grill phase complete

## Problem Statement

dnachem-min writes a `Manifest.json` in every Dump describing the simulation (chemistry, scavengers, pH, beam per run, totals, files, timing, run mode). G4Utils has no reader for a single Manifest: `load_manifests()` flattens all Dumps into a DataFrame and `dump_columns()` keeps only a few fields. Users need an easy way to access every piece of information in one Manifest, and a pretty printer to display it in a Jupyter notebook.

## Context & Constraints

- **Current behavior:** `g4utils.DnaChem.manifest` has `dump_columns(manifest_dict, name)` and `load_manifests(path, name_pattern)` (one row per run, all Dumps). `find_dumps` only recognizes the exact name `Manifest.json`. Many fields (timestamp, geant4Version, macro, runMode, threads, halfBox_um, chemistryEndTime_ns, outputDir*, files, wallTime_s, meso settings) are dropped.
- **Pain point:** no object-style access to one Manifest, no readable display, prefixed names such as `EndOfRun_Manifest.json` are not reachable.
- **Dependencies:** keep `load_manifests()`, `dump_columns()` and loader output unchanged (existing tests act as regression check). CONTEXT.md terms Manifest, Dump, and the new Manifest run.
- **Tech stack:** Python, pandas, dataclasses, pytest; IPython optional, imported lazily.

## Success Metrics

- `read_manifest(path)` returns a `Manifest` for a Dump folder, a `*Manifest.json` file (incl. `EndOfRun_Manifest.json`), or a folder holding exactly one Dump; several Dumps raise `ValueError` pointing to `load_manifests()`.
- `Manifest`, `ManifestRun`, `Scavenger` are frozen dataclasses with JSON key names as fields; absent fields are `None`, empty scavengers/runs are tuples; unknown keys kept in `.raw`; `runs_table()` and `scavenger_molarity()` work.
- `_repr_html_`, `__str__` and `show()` render Overview, Chemistry, Totals, Runs, Files and Other sections, skipping absent fields; tested on a real dnachem-min manifest and on synthetic ones.

## Architecture & Approach

`read_manifest(path) -> Manifest` in `g4utils/DnaChem/manifest.py`, exported from `g4utils.DnaChem`. `Manifest` exposes `runs_table()` (DataFrame), `scavenger_molarity(species)` (float or None), `show()` (IPython `display`, lazy import), `__str__` (aligned text) and `_repr_html_`. Parsing is lenient: `schemaVersion` above 1 warns (UserWarning) but parses; non-object JSON or missing `schemaVersion` raises `ValueError` naming the file. `dump_columns` is rewritten on top of `Manifest`; `load_manifests` and loaders keep their output. README and CHANGELOG fragment updated.

## Assumptions & Trade-offs

Field names follow the JSON (camelCase with units) for consistency with existing DataFrame columns. No schema-strict mode. Not changing `find_dumps` discovery of prefixed manifests in parent folders. No ADR (reversible).

## Open Questions

None.

## Notes

Glossary term **Manifest run** added to CONTEXT.md during the grill. A real manifest example lives in the dnachem-min repo at `.scratch/tests/2026-10-02__irt-syn-mesoscopic/02b-mt/Manifest.json`.

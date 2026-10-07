# Implementation Plan

**Session:** simulations ran from 'dnachem-min' produces a 'manifest.json' file with relevant information about a simulation run. Add a reader and a way to easily access these pieces of information. It should also produce a pretty printer kind function to display it within a jupyter notebook.
**Date:** 2026-10-07T14:49:31.354Z
**Estimated effort:** 3 hours

## Strategy

Build the typed reader first (parsing, path resolution, access helpers, `dump_columns` rebuilt on it), then the Jupyter/text rendering on top of the object, then exports and docs. Each step is committable and tested on its own.

## Tickets Overview

- **Ticket 1:** `read_manifest`, the `Manifest` / `ManifestRun` / `Scavenger` dataclasses, `runs_table()`, `scavenger_molarity()`, and `dump_columns` rewritten on top of them, tested with a real manifest fixture.
- **Ticket 2:** `__str__`, `_repr_html_` and `show()` rendering of the Manifest.
- **Ticket 3:** Export from `g4utils.DnaChem`, README section, CHANGELOG fragment.

## Sequencing Rationale

The renderer needs the object; the exports and docs describe the finished API.

## Risks & Mitigation

- **Risk:** rewriting `dump_columns` changes DataFrame output. → **Mitigation:** existing manifest and loader tests must pass untouched.
- **Risk:** the real manifest has fields the synthetic fixture lacks. → **Mitigation:** copy a real dnachem-min manifest into the test fixtures.

## Assumptions

- IPython is not a dependency; `show()` imports it lazily.
- Field names equal the JSON keys.

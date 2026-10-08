# Implementation Plan

**Session:** manifest-aware-dump-and-simulation
**Date:** 2026-10-07T21:08:48.239Z
**Estimated effort:** 3 hours

## Strategy

Type the new Manifest fields first, then make `Dump` use the Manifest (files, prefix, meso), then enrich `sim.table()`, then document. Each ticket builds on committed work and keeps existing tests green.

## Tickets Overview

- **Ticket 1:** Add the six typed Manifest fields and a "Chemistry model" display section.
- **Ticket 2:** `Dump.files`, `Dump.prefix` (prefixed data filenames), `Dump.has_meso`, clear `.meso` error.
- **Ticket 3:** Richer `Simulation.table()` columns.
- **Ticket 4:** README and glossary updates.

## Sequencing Rationale

Tickets 2 and 3 read the new Manifest fields, so ticket 1 comes first. Docs come last.

## Risks & Mitigation

- **Risk:** prefix handling changes how `load_*` find files. → **Mitigation:** an empty prefix gives the old names; existing loader tests guard it.
- **Risk:** new typed fields change the "Other" display and existing display tests. → **Mitigation:** update only tests that listed those keys as unknown.

## Assumptions

- Prefix comes from the Manifest `prefix` field; `Manifest.json` itself is never prefixed.
- One Manifest per Dump; Manifest runs are `/run/beamOn` calls inside it.

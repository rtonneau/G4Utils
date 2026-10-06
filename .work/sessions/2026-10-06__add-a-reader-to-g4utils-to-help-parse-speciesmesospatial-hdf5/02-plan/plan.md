# Implementation Plan

**Session:** Add a Reader to G4Utils to help parse SpeciesMesoSpatial HDF5 files
**Date:** 2026-10-06T15:56:34.376Z
**Estimated effort:** 3 hours

## Strategy

Build the reader bottom-up in three committable steps: first the file index and snapshot reading with a synthetic-file test fixture, then the concentration helper on top of the snapshot, then the README. Every ticket leaves the test suite green.

## Tickets Overview

- **Ticket 1:** `SpeciesMesoSpatialFile` and `MesoSpatialSnapshot` in `g4utils.HDF5`: indexing, `read_snapshot`, `iter_snapshots`, error handling, exports, test fixture and tests.
- **Ticket 2:** `concentration_M` on the snapshot and as a module-level function, with tests.
- **Ticket 3:** README section documenting the reader.

## Sequencing Rationale

Ticket 2 needs the snapshot dataclass from ticket 1 and the fixture it adds. Ticket 3 documents the finished API, so it comes last.

## Risks & Mitigation

- **Risk:** The synthetic fixture drifts from the real dnachem-min output → **Mitigation:** Build it strictly from `dnachem-min/docs/output/SpeciesMesoSpatial-h5.md` (group names `run<R>`/`event<E>`/`snapshot<k>`, dtypes, attributes) and cover hard-linked and empty snapshots.
- **Risk:** Numeric sorting of non-zero-padded names is done lexically by mistake → **Mitigation:** Sort on the parsed integer and test with indices above 9.

## Assumptions

- h5py, numpy and pandas are already dependencies; no new dependency is needed.
- Single-file reader only: no Dump or `Manifest.json` discovery.
- Species names stay as raw Geant4 display names.

# Implementation Plan

**Session:** the HDF5 code loading data from mesoscopic geant4 data loads sparsed data. It should provides a way to reconstruct full dataframe (fileld with 0) to have correct sizes and compare easily one to another
**Date:** 2026-10-06T17:22:04.525Z
**Estimated effort:** 0.5 day

## Strategy

Build the grid geometry and single-snapshot densification first, then the event-level reader that groups snapshots into cell-size periods, then exports and docs.

## Tickets Overview

- **Ticket 1:** Grid geometry helper and `MesoSpatialSnapshot.to_dense`.
- **Ticket 2:** `DenseMesoPeriod` and `SpeciesMesoSpatialFile.read_dense`.
- **Ticket 3:** Exports, README section and changelog fragment.

## Sequencing Rationale

`read_dense` reuses the grid helper and the scatter logic from ticket 1; docs and exports come last, once the API is stable.

## Risks & Mitigation

- **Risk:** Cell-centre to index rounding errors when cell sizes grow. → **Mitigation:** Use the grid origin snapped to the cell size and round (pos - origin) / cell - 0.5; test with synthetic power-of-two sizes.
- **Risk:** Huge extents exhaust memory (no guard by design). → **Mitigation:** Document it in the README.

## Assumptions

- One species and one event per call.
- No resampling across cell sizes; no new dependencies.

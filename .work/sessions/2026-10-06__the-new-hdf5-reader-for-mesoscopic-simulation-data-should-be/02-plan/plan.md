# Implementation Plan

**Session:** the new hdf5 reader for mesoscopic simulation data should be able to export 3D data to vti format
**Date:** 2026-10-06T20:11:47.864Z
**Estimated effort:** 3 hours

## Strategy

Build bottom-up: first a pure function that turns one sparse snapshot into dense arrays on a lattice (with validation), then the per-snapshot `to_vti`, then the file-level time series and the docs. Each step builds on committed, tested work. `write_vti` and `write_pvd_collection` stay unchanged.

## Tickets Overview

- **Ticket 1:** Densify a sparse meso snapshot onto a lattice (`ValueError` on off-lattice or colliding cells), with an optional physical extent.
- **Ticket 2:** `MesoSpatialSnapshot.to_vti` writing `<species>_count` and `<species>_M` arrays in nm.
- **Ticket 3:** `SpeciesMesoSpatialFile.to_vti_timeseries` (common box, per-frame cell size, `.pvd` with `time_ns`) plus README section.

## Sequencing Rationale

Ticket 2 needs the dense grid from ticket 1. Ticket 3 needs `to_vti` with its `extent` parameter, and the common box needs per-snapshot bounds from ticket 1.

## Risks & Mitigation

- **Risk:** Cell centres are not on a regular lattice in real files. → **Mitigation:** validate with a tolerance and raise `ValueError` with the offending cell, as agreed.
- **Risk:** A common box does not align with a frame's lattice when cell sizes differ. → **Mitigation:** anchor each frame's lattice on its own cells and extend it by whole cells to cover the box.
- **Risk:** Float32 loses exact counts above 2^24. → **Mitigation:** `dtype` is caller-settable; documented.

## Assumptions

- `VoxGeometry` can carry nm values in its `*_mm` fields for the writer; the unit is documented as nm for meso exports.
- Test fixtures come from `tests/conftest.py::write_species_meso_spatial`.

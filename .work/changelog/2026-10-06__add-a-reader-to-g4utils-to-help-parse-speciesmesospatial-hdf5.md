---
bump: minor
floor: minor
---
- Add `SpeciesMesoSpatialFile` to `g4utils.HDF5`, a lazy reader for the `SpeciesMesoSpatial.h5` files dnachem-min writes: list runs, events and snapshots, and read a snapshot's cell positions and per-species counts on demand.
- Add `concentration_M` (function and `MesoSpatialSnapshot` method) to convert counts to mol/L, optionally for selected species.

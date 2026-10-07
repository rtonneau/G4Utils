---
bump: minor
floor: minor
---
- Add `MesoSpatialSnapshot.to_dense()` to rebuild one species of a sparse mesoscopic snapshot as a zero-filled 3D array (counts or mol/L), with an optional fixed `bounds_nm` extent.
- Add `SpeciesMesoSpatialFile.read_dense()` to read one event as zero-filled (time, x, y, z) arrays, one `DenseMesoPeriod` per cell size, each with a `MesoGrid` (origin, cell size, shape) so snapshots, events and files can be compared.
- Export `MesoGrid` and `DenseMesoPeriod` from `g4utils.HDF5`.

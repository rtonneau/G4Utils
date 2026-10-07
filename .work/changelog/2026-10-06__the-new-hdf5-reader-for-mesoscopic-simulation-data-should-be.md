---
bump: minor
floor: minor
---
- Export mesoscopic spatial snapshots to VTK ImageData with `MesoSpatialSnapshot.to_vti`: per-species `_count` and `_M` (mol/L) arrays on a dense lattice, with origin and spacing in nm.
- Export a whole event as a ParaView time series with `SpeciesMesoSpatialFile.to_vti_timeseries`: one `.vti` per snapshot plus a `.pvd` timed in ns, all frames sharing one physical box.

---
bump: minor
floor: minor
---
- `Manifest` now exposes the chemistry model and mesoscopic settings (`chemistryModel`, `handOverTime_ns`, `voxelSize_nm`, `mesoPixels`, `mesoTimesPerDecade`, `mesoSpatialOutput`) and shows them in a "Chemistry model" section.
- `Dump` gains `files`, `prefix` and `has_meso`; data files of a Dump whose Manifest has a `prefix` are found, and `Dump.meso` explains whether meso output was disabled or the file is missing.
- `Simulation.table()` now includes chemistry model, hand-over and end times, voxel and meso settings, threads and total wall time.

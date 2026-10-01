# G4Utils

Reading, analysing and exporting the voxelised scoring output that the G4Vox C++ library writes to HDF5 during Geant4 simulations.

## Language

### Files and layouts

**Vox file**:
One HDF5 file written by the G4Vox `HDF5Writer`, holding the voxel data of one or more subruns plus metadata and a run log.
_Avoid_: dump, output file

**Layout**:
How a vox file stores its voxel data: either **Snapshot3D** (one group per subrun) or **Extendable4D** (one 4D dataset per quantity, subruns stacked along the first axis). Recorded by the writer as the `mode` metadata attribute.
_Avoid_: format, mode (when talking about the Python side), 3D/4D file

**Quantity**:
A named scored field over the voxel grid, such as `Dose` or `Edep`.
_Avoid_: dataset, field, observable

**Geometry**:
The voxel grid description: voxel counts along x, y, z, voxel spacing and grid origin in mm.
_Avoid_: grid, metadata

### Subruns

**Subrun**:
One exported batch of primaries; the unit at which the writer appends voxel data and one run log row.
_Avoid_: batch, event, run

**Subrun ID**:
The identifier the simulation assigned to a subrun, as recorded in the run log (and in the group name for Snapshot3D). It is the ID users select and iterate over.
_Avoid_: index, subrun number

**Slice index**:
The position of a subrun along the first axis of an Extendable4D dataset. Equal to the row position in the run log, but not necessarily to the subrun ID.
_Avoid_: subrun ID

**Run log**:
The per-subrun table written by the writer: unix timestamp, number of primaries, runtime in seconds and subrun ID.
_Avoid_: seeds table

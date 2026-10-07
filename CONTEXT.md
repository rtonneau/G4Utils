# G4Utils

Python utilities for Geant4 simulation output. Two areas: the voxelised scoring output that the G4Vox C++ library writes to HDF5 (`g4utils.HDF5`, `g4utils.Vox`), and the chemistry output of the Geant4-DNA dnachem-min application (`g4utils.DnaChem`). This glossary fixes the vocabulary; it is not a spec.

## Language

### Vox files (HDF5)

#### Files and layouts

**Vox file**:
One HDF5 file written by the G4Vox `HDF5Writer`, holding the voxel data of one or more subruns plus metadata and a run log.
_Avoid_: dump (that is a DnaChem **Dump**), output file

**Layout**:
How a vox file stores its voxel data: either **Snapshot3D** (one group per subrun) or **Extendable4D** (one 4D dataset per quantity, subruns stacked along the first axis). Recorded by the writer as the `mode` metadata attribute.
_Avoid_: format, mode (when talking about the Python side), 3D/4D file

**Quantity**:
A named scored field over the voxel grid, such as `Dose` or `Edep`.
_Avoid_: dataset, field, observable

**Geometry**:
The voxel grid description: voxel counts along x, y, z, voxel spacing and grid origin in mm.
_Avoid_: grid, metadata

#### Subruns

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

### Mesoscopic spatial output (HDF5)

**Meso file**:
One `SpeciesMesoSpatial.h5` file written by dnachem-min, holding the mesoscopic spatial state of one or more runs.
_Avoid_: Vox file (different writer and layout), Dump

**Meso snapshot**:
The occupied cells of the mesoscopic mesh, with molecule counts per species, at one record time of one event of one run.
_Avoid_: frame, subrun

**Meso cell**:
One cubic cell of the mesoscopic mesh, identified by its centre. Only occupied cells are stored; its side (cell size) can differ from one snapshot to the next.
_Avoid_: voxel (a voxel belongs to a Vox file's Geometry)

**Event series**:
The ordered meso snapshots of one (run, event), exported as one time series. Runs and events are never merged.
_Avoid_: timeseries of a run

**Common box**:
The physical extent covering the occupied cells of every snapshot of an event series, so exported frames line up. May be overridden by the caller.
_Avoid_: bounding box (per-snapshot extents are not used by default)

### DnaChem output

**Dump**:
One folder of output written by a Geant4-DNA chemistry simulation (dnachem-min) in a
single flush: species and reaction tallies, pre-chemical files, and exactly one
**Manifest**. The unit that is loaded and compared (e.g. one O2 level of a scan).
_Avoid_: "run" (a Geant4 run is one `/run/beamOn`; a Dump may cover several),
"case", "simulation".

**Simulation folder**:
The top folder of a whole dnachem-min simulation (e.g. one scan script's output): the
executable, macros, console logs and a `results/` folder holding the **Dumps**. It
holds no Manifest itself; each Dump has its own.
_Avoid_: "Dump" (that is one subfolder of `results/`), "run".

**Dump subrun**:
In the Simulation-folder accessor's API, a Dump seen as one part of its Simulation
folder (e.g. one O2 level of the scan). Not a Vox **Subrun**, which is a batch inside
an HDF5 file.
_Avoid_: bare "subrun" in prose (Vox term).

**Manifest**:
The `Manifest.json` inside a Dump: what was simulated (beam per run, Chemistry,
scavengers, pH), the totals (events, energy deposit) and the files produced. Source
of truth for a Dump's physical parameters.
_Avoid_: "metadata" (collides with `ReactionsMetadata.csv`).

**Manifest run**:
One entry of a Manifest's `runs` list: the beam, energy deposit, seed and wall time of
one `/run/beamOn` folded into the Dump.
_Avoid_: "run" alone (a Geant4 run in general).

**Scan**:
A set of Dumps side by side that differ in one or more parameters (O2 content, beam
energy, ...), usually sibling folders under one results directory.

**Name pattern**:
A user-supplied regular expression whose named groups extract labels from a Dump's
folder name (e.g. `run_0p3pO2` → `o2_percent = 0.3`, with `p` read as a decimal
point). A convenience label only; the Manifest stays authoritative.

**G-value**:
Number of molecules of a species per 100 eV of energy deposited in the whole Dump:
count / (total energy deposit / 100 eV).
_Avoid_: "yield" alone (ambiguous between count and G-value).

**Short name**:
The plain species label (`OH`, `e_aq`, `HO2`, `HO2-`, `O2-`, `H3O+`, ...) mapped
from the Geant4 molecule name (`°OH^0`, `e_aq^-1`, `HO_2°^0`, ...). Species are
identified by name, never by numeric species ID. The one species vocabulary for both
species and reactions: radicals carry no `*` mark.
_Avoid_: radical notation (`OH*`, `H*`).

**Reaction table**:
The list of reactions declared in a Dump's `ReactionsMetadata.csv`, one row per
reaction, with its reactants and products as Short names (with stoichiometric
repeats). A reaction may have no products. Describes the chemistry, not what happened:
counts live in the reaction tallies.
_Avoid_: "reaction metadata" (collides with Manifest wording), "reaction list".

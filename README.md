# G4Utils

Readers and VTK export for G4Vox voxel HDF5 output from Geant4 simulations.

## What It Does

G4Utils reads the HDF5 voxel output written by the G4Vox library (`HDF5Writer`) during Geant4 simulations. It supports two storage layouts:

- **Snapshot3D**: One group per subrun, each holding 3D scored quantities (e.g., dose or energy deposit).
- **Extendable4D**: One 4D dataset per quantity, with subruns stacked along the first axis (one slice per subrun). Subrun IDs come from the run log's `subrun_id` column; if the run log is missing, has a different row count than the slices, or contains duplicate IDs, a `UserWarning` is emitted and slice indices `0..N-1` are used instead.

The package provides lazy per-subrun loading, quantity/subrun selection, sums over subruns, and export to VTI/PVD for visualization in ParaView. It also loads the chemistry output of the Geant4-DNA dnachem-min application (species and reaction tallies with G-values; see [DnaChem](#dnachem-loading-dnachem-min-output)), and reads its mesoscopic spatial output (see [SpeciesMesoSpatial](#speciesmesospatial-reading-the-mesoscopic-spatial-state)). Terminology (subrun ID, slice index, layout, quantity, Dump, Manifest, G-value) is defined in [CONTEXT.md](CONTEXT.md).

## Installation

Install from a clone of the repository:

```bash
pip install .
```

Or, for development, as an editable install with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Open a Vox File

```python
from g4utils.HDF5 import open_vox_file

sim = open_vox_file("path/to/voxel_output.h5")
```

The `open_vox_file()` function automatically detects the layout (Snapshot3D or Extendable4D) and returns the appropriate reader.

### Select Quantities and Subruns

```python
# Select specific quantities
sim.select_quantity(["Dose", "Edep"])

# Select subruns by ID
sim.select_subrun(subrun_ids=[0, 1, 2])

# Or select by ID range
sim.select_subrun(start=0, stop=5)
```

Both methods return `self` for chaining.

### Iterate Over Subruns

```python
for subrun_id in sim:
    dose = sim.data["Dose"]  # 3D numpy array
    print(f"Subrun {subrun_id}: shape={dose.shape}, sum={dose.sum():.2e}")
```

During iteration, `sim.data` holds the currently loaded subrun's selected quantities as numpy arrays of shape `(nZ, nY, nX)`. Only one subrun is in memory at a time.

To write one VTI per subrun while iterating:

```python
for subrun_id in sim:
    sim.to_vti(f"subrun_{subrun_id:04d}.vti")
```

### Direct Access

```python
# Fetch a single quantity and subrun
dose = sim.get("Dose", subrun_id=0)

# Sum one quantity over given subrun IDs (floats accumulate in float64)
total_dose = sim.sum("Dose", subrun_ids=[0, 1, 2])

# Sum over the current subrun selection (all subruns if none is selected)
selected_dose = sim.sum("Dose")
```

### Run Log

The run log is a pandas DataFrame with per-subrun metadata:

```python
print(sim.run_log)
```

Columns:

- **unix_timestamp**: Epoch time when the subrun was exported.
- **primaries**: Number of primary particles simulated in the subrun.
- **runtime_s**: Runtime in seconds.
- **subrun_id**: The logical subrun identifier.

Total primaries across all subruns:

```python
total = sim.total_primaries()
```

## VTI/PVD Export

Export selected data as VTI files (VTK XML ImageData, values stored as cell data) for use in ParaView.

### Single File (Summed Data)

```python
sim.select_quantity("Dose").select_subrun(subrun_ids=[0, 1, 2])
sim.dump_selection_to_vti("output.vti")
```

### Per-Subrun Timeseries

Export one VTI per subrun and a PVD collection file for animation:

```python
sim.select_quantity("Dose")
sim.dump_selection_to_vti_timeseries("output.pvd")
```

This creates `output.pvd` and `output_0000.vti`, `output_0001.vti`, etc.

### Export Options

`to_vti()`, `dump_selection_to_vti()` and `dump_selection_to_vti_timeseries()` accept these keyword-only options:

- **encoding** (default: `"binary"`): Data format in the XML.
  - `"binary"` (default): Inline base64-encoded binary data. Fast and compact.
  - `"ascii"`: Plain text values. Useful for debugging or text-based workflows.
- **compress** (default: `False`): Enable zlib compression of binary data.
  - `False` (default): No compression.
  - `True`: Compress with vtkZLibDataCompressor. Useful for large datasets.

Example:

```python
sim.dump_selection_to_vti("output.vti", encoding="binary", compress=True)
```

## DnaChem: loading dnachem-min output

The `g4utils.DnaChem` module provides loaders for Geant4-DNA chemical output files. The main entry point is `load_species()`, which reads species counts and computes G values (molecules per 100 eV of deposited energy).

### Example: O2 concentration scan

```python
from g4utils.DnaChem import load_species
import pandas as pd

# Load all Dumps matching the O2 concentration pattern
df = load_species(
    "results",
    name_pattern=r"run_(?P<o2_percent>[\dp]+)pO2"
)

# Filter to species of interest
species_list = ["OH", "e_aq", "H"]
df = df[df.species.isin(species_list)]

# Pivot to G values by time and species
g_by_time = df.pivot_table(
    index="time_s",
    columns=["o2_percent", "species"],
    values="G"
)
```

`path` is either one Dump folder (it contains `Manifest.json`) or a parent folder: every direct subfolder with a `Manifest.json` is loaded, other files and folders are ignored. Species are identified by name, never by numeric ID.

### Opening a simulation folder

`Simulation` opens a dnachem-min folder without reading any data file. `path` is a flat Dump folder, a `results` folder holding Dumps, or a simulation folder with a `results/` subfolder. Each Dump is a lazy `Dump`: its tables and its meso file load on first access and are cached.

```python
from g4utils.DnaChem import Simulation

sim = Simulation("path/to/simulation", name_pattern=r"run_(?P<o2_percent>[\dp]+)pO2")
print(sim)            # Simulation('.../results', 3 subruns)
sim.table()           # one row per subrun: name, labels, Manifest summary columns

dump = sim.subrun("run_21pO2")   # pick a subrun by folder name (KeyError lists the names)
dump.files            # tuple of file names listed in Manifest
dump.prefix           # data file prefix from Manifest (empty if absent)
dump.has_meso         # True if meso output enabled and SpeciesMesoSpatial.h5 exists

species = dump.species()          # same frame as load_species() gives for this Dump
dump.reactions()                  # reaction counts
dump.reaction_table()             # one row per reaction
dump.manifest                     # parsed Manifest

meso = dump.meso                  # SpeciesMesoSpatialFile (SpeciesMesoSpatial.h5)
```

`Simulation.table()` provides one row per Dump with columns including `dump` (the folder name), one column per Name pattern group, and Manifest summary fields such as `chemistryModel`, `handOverTime_ns`, `chemistryEndTime_ns`, `voxelSize_nm`, `mesoPixels`, `mesoTimesPerDecade`, `mesoSpatialOutput`, `threads`, and `wallTime_s` (total wall time across all runs in the Dump).

`load_species()`, `load_reactions()` and `load_reaction_table()` below are built on these classes. A missing data file raises `FileNotFoundError` naming the file.

### Columns added by loaders

**`load_species()`** adds these columns to the ntuple data:
- `species`: short chemical formula (e.g., "OH", "e_aq"; mapped via `SHORT_NAMES`)
- `time_s`: time in seconds (replaces the ns `time` column)
- `G`: G value = number / (totalEnergyDeposit_eV / 100)
- `dump`: Dump folder name
- `chemistry`: Chemistry name from the Manifest (e.g. `BoscoloChem`)
- `pH`: pH from Manifest
- `totalEvents`: total events in the Dump
- `totalEnergyDeposit_eV`: total energy deposited (eV)
- `{species}_molarity_M`: molarity of each scavenger (e.g., `O2_molarity_M`)
- `particle`: particle type (from run metadata, if consistent)
- `beamEnergy_keV`: beam energy in keV (from run metadata, if consistent)
- Named groups from `name_pattern` (e.g., `o2_percent` from the regex above); `p` is read as a decimal point and numbers become floats. A Dump folder that does not match raises `ValueError`.

**`load_reactions()`** similarly loads reaction data with reaction labels and timing.

**`load_reaction_table()`** loads one row per reaction (from a single Dump) with reaction equations and stoichiometry columns. It accepts a Dump folder, a folder holding exactly one Dump, or the `ReactionsMetadata.csv` file itself; a folder with several Dumps raises `ValueError` (the reactions are the same across the Dumps of one setup). Returns a DataFrame with:
- `reactionId`: numeric reaction identifier
- `reaction`: raw reaction string from metadata
- `equation`: reaction equation using Short names (e.g., `"OH + OH -> H2O2"`)
- `reactants`, `products`: tuple of species involved on each side (Short names, no radical marks like `*`)
- `reactant_<species>`, `product_<species>`: int stoichiometric counts, one pair for every species that appears in the file (0 where absent)

Example usage:
```python
from g4utils.DnaChem import load_reaction_table

df = load_reaction_table("path/to/Dump")

# Filter reactions producing H2O2
products_h2o2 = df[df["product_H2O2"] > 0]

# Filter reactions consuming OH (reactant)
consumes_oh = df[df["reactant_OH"] > 0]

# Access charged species using quoted column names
consumes_oh_minus = df[df["reactant_OH-"] > 0]

# Join with reaction counts from load_reactions() on reactionId
from g4utils.DnaChem import load_reactions
reactions_data = load_reactions("path/to/parent")
merged = reactions_data.merge(df, on="reactionId")
```

**`load_manifests()`** loads one row per run with Dump and run metadata.

### Reading Manifests

`read_manifest()` reads one `Manifest.json` file from a Dump folder, a single Manifest file, or a folder holding exactly one Dump:

```python
from g4utils.DnaChem import read_manifest

manifest = read_manifest("path/to/Dump")
# or
manifest = read_manifest("path/to/Manifest.json")
```

The returned `Manifest` object provides attribute access to simulation metadata:

```python
# Simulation info
print(manifest.timestamp)
print(manifest.geant4Version)
print(manifest.macro)

# Chemistry settings
print(manifest.chemistry)
print(manifest.pH)
print(manifest.halfBox_um)
print(manifest.chemistryEndTime_ns)

# Scavengers
for scavenger in manifest.scavengers:
    print(f"{scavenger.species}: {scavenger.molarity_M} M")

# Or query one scavenger directly
o2_molarity = manifest.scavenger_molarity("O2")

# Run metadata
print(manifest.totalEvents)
print(manifest.totalEnergyDeposit_eV)
```

All fields from `Manifest.json` are accessible as attributes. Unknown keys are stored in the `raw` attribute for forward compatibility.

#### Per-run data: `runs_table()`

Access per-run metadata (beamEnergy, position, seed, etc.) as a DataFrame:

```python
runs_df = manifest.runs_table()
# One row per run, with columns like: run, events, particle, beamEnergy_keV, position_um, direction, energyDeposit_eV, seed, wallTime_s
```

#### Notebook display

`Manifest` has a text representation (`str()`) and notebook HTML rendering (`_repr_html_()`):

```python
# In a Jupyter notebook
manifest  # displays as formatted HTML table

# Or explicitly
print(manifest)  # text format
manifest.show()  # display() using IPython
```

### Other utilities

- `find_dumps()`: locate Dump folders by Manifest.json
- `read_ntuple()`: parse Geant4 wcsv CSV ntuples
- `short_name()` / `SHORT_NAMES`: map raw species names to short labels

## SpeciesMesoSpatial: reading the mesoscopic spatial state

dnachem-min writes `SpeciesMesoSpatial.h5` when a macro sets `/chem/meso/spatialOutput true`: for each run, event and record time, the occupied cells of the mesoscopic mesh with their molecule counts per species. `g4utils.HDF5.SpeciesMesoSpatialFile` reads one such file lazily. It indexes runs, events and snapshots when opened and reads the arrays of a snapshot only on request. It reads a single file: it does not look for it inside a Dump.

```python
from g4utils.HDF5 import SpeciesMesoSpatialFile

f = SpeciesMesoSpatialFile("SpeciesMesoSpatial.h5")
print(f.species, f.format_version, f.runs)

run = f.runs[0]
event = f.events(run)[0]
last = f.snapshot_indices(run, event)[-1]

snap = f.read_snapshot(run, event, last)
print(snap.time_ns, snap.cell_size_nm)   # cell size grows with time
print(snap.position_nm.shape)            # (N, 3), nm
print(snap.counts.shape)                 # (N, S), molecules

conc = snap.concentration_M()            # mol/L, (N, S)
oh = snap.concentration_M(["°OH^0"])     # selected species only

for s in f.iter_snapshots(run=run, event=event):
    ...
```

Species names are the raw Geant4 display names stored in the file. `concentration_M(counts, cell_size_nm)` is also available as a function. The file format is described in the dnachem-min documentation (`docs/output/SpeciesMesoSpatial-h5.md`).

### Export to VTI / PVD

```python
snap.to_vti("snap.vti", species=["°OH^0"])
f.to_vti_timeseries(run, event, "oh/series.pvd", species=["°OH^0"])
```

The sparse cells are placed on a dense lattice and written as a VTK ImageData; unoccupied cells are 0. Origin and spacing are in nm. Each species gives two CellData arrays, `<species>_count` (molecules) and `<species>_M` (mol/L), named after the raw species names. `species=None` exports all species. `extent=(min_xyz_nm, max_xyz_nm)` makes the lattice cover that box (grown by whole cells).

`to_vti_timeseries(run, event, path, ...)` writes `<stem>_<index>.vti` (snapshot index, 4 digits) and a `.pvd` whose timesteps are `time_ns`. By default all frames share the common physical box of every cell of the event; each frame uses its own cell size as spacing, so the grid gets coarser with time. Snapshots are read one at a time.

Errors: `KeyError` for an unknown run, event or species; `ValueError` if the snapshot (or, for the time series, every snapshot of the event) has no occupied cell and no `extent` is given, if cells are off a common lattice or collide, or if `extent` is malformed.

### Dense Time Series: `to_dense` and `read_dense`

For analysis over time or to feed grids to other tools, convert sparse snapshots to dense 3D arrays. Two approaches:

**Per-snapshot:** `MesoSpatialSnapshot.to_dense()` rebuilds one species as a zero-filled 3D array with a single snapshot:

```python
from g4utils.HDF5 import SpeciesMesoSpatialFile

f = SpeciesMesoSpatialFile("SpeciesMesoSpatial.h5")
snap = f.read_snapshot(run=0, event=0, index=0)

# Rebuild OH as uint32 counts
dense, grid = snap.to_dense("°OH^0")  # shape (nx, ny, nz)
print(grid.origin_nm, grid.cell_size_nm, grid.shape)

# Or as molar concentration
dense_conc, grid = snap.to_dense("°OH^0", concentration=True)  # float64, mol/L
```

Optionally pass `bounds_nm` to define a fixed extent (snapped outwards to the cell lattice) for consistent array shapes across snapshots.

**Time series:** `SpeciesMesoSpatialFile.read_dense()` reads all snapshots of one event as a 4D array grouped by cell-size period. Cell sizes often change between snapshots; this method groups consecutive equal sizes into one `DenseMesoPeriod`:

```python
periods = f.read_dense(run=0, event=0, species="°OH^0")

for period in periods:
    print(f"Cell size: {period.grid.cell_size_nm} nm, times: {period.times_ns}")
    print(f"Data shape: {period.data.shape}")  # (T, nx, ny, nz)
    print(f"Grid: {period.grid}")
    # Process time series at constant cell size
```

Same options as `to_dense`: pass `bounds_nm` to lock the extent and `concentration=True` to get mol/L instead of counts.

**Memory caveat:** Dense arrays can grow quickly. For a 100×100×100 grid with 10 snapshots, uint32 counts need ~40 MB; float64 concentrations need ~80 MB. Multi-period events duplicate data, so a 10-step event with cell-size changes can use several hundred MB. Reading a large file without specifying `bounds_nm` scans all snapshots once to find the union of occupied cells. Use `bounds_nm` when you know the region of interest.

## Development

### Setup

Clone and install in editable mode with dev dependencies:

```bash
git clone https://github.com/rtonneau/G4Utils.git
cd G4Utils
pip install -e ".[dev]"
```

### Testing

Run the test suite:

```bash
pytest
```

With coverage report:

```bash
pytest --cov=g4utils --cov-report=term-missing
```

## License

MIT License — see [LICENSE](LICENSE) for details.

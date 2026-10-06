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

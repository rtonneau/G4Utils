# G4Utils

Readers and VTK export for G4Vox voxel HDF5 output from Geant4 simulations.

## What It Does

G4Utils reads the HDF5 voxel output written by the G4Vox library (`HDF5Writer`) during Geant4 simulations. It supports two storage layouts:

- **Snapshot3D**: One group per subrun, each holding 3D scored quantities (e.g., dose or energy deposit).
- **Extendable4D**: One 4D dataset per quantity, with subruns stacked along the first axis (one slice per subrun). Subrun IDs come from the run log's `subrun_id` column; if the run log is missing, has a different row count than the slices, or contains duplicate IDs, a `UserWarning` is emitted and slice indices `0..N-1` are used instead.

The package provides lazy per-subrun loading, quantity/subrun selection, sums over subruns, and export to VTI/PVD for visualization in ParaView. It also loads the chemistry output of the Geant4-DNA dnachem-min application (species and reaction tallies with G-values; see [DnaChem](#dnachem-loading-dnachem-min-output)), and reads its mesoscopic spatial output (see [SpeciesMesoSpatial](#speciesmesospatial-reading-the-mesoscopic-spatial-state)). Terminology (subrun ID, slice index, layout, quantity, Dump, Manifest, G-value) is defined in [CONTEXT.md](CONTEXT.md).

## Installation

Use Python 3.10 or newer. Installation includes numpy >= 1.24, pandas >= 2.0,
and h5py >= 3.8; reading existing output does not require a Geant4 installation.
Install from a clone of the repository using the intended Python environment:

```bash
python -m pip install .
```

Or, for development, as an editable install with dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

For a notebook, install G4Utils in the environment selected by its kernel.
For example, after `conda activate GEANT4`, run the editable installation above
from this checkout. Plotting and Jupyter are optional, separate dependencies:
`python -m pip install matplotlib ipykernel`. In a notebook cell, verify
`sys.executable` and `g4utils.__file__` before reading data. An existing kernel
must be restarted after library changes to load the updated modules.
An editable install follows the checkout's current branch; select the branch
or merged revision containing the APIs you intend to use before restarting.

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

`path` may be a Dump folder, a selected manifest file, a results directory, or a simulation folder containing `results/`. Discovery follows a results index when present, including prefixed and nested dumps. Species are identified by name, never by numeric ID.

### Discovering and selecting dumps

```python
from g4utils.DnaChem import Dump, discover_dumps

locations = discover_dumps("results")
for location in locations:
    print(location.key, location.directory, location.manifest_path, location.prefix)

dump = Dump.from_manifest("results/EndOfRun_Manifest.json")
```

Each frozen `DumpLocation` identifies one manifest. Its key is the resolved manifest path relative to the discovery root, using `/` separators: for example `Manifest.json`, `pfx_Manifest.json`, or `nested/run02/Manifest.json`. Adding siblings does not change existing keys. Paths are resolved on the current machine; stale `outputDirAbsolute` and `outputDirAsConfigured` metadata are not used to locate files.

When root `Manifest.json` has `kind: "resultsIndex"` or a `dumps` array, discovery follows its entries in order. A plain dump's manifest may also hold that array and reference itself; this dump is included once at its indexed position. Each entry must provide string `folder`, `prefix`, and `manifest` fields. Missing manifests, duplicate keys, malformed entries, folder/prefix mismatches and paths escaping the index directory (including resolved symlinks) raise contextual errors. Both `/` and `\\` work in relative paths. An index cannot itself be selected as a dump unless it also represents a plain dump (without `kind: "resultsIndex"`). Index event totals are discovery hints; each selected dump manifest supplies the calculation metadata.

Without an index, an ordinary root `Manifest.json` keeps the legacy single-dump behavior. Otherwise discovery scans `*Manifest.json` at the root and in direct child directories, sorting root files first and then children by name. Within each direct child, a plain `Manifest.json` takes precedence: its index is followed if present, otherwise it selects that one dump. This also supports historical prefixed output without an index. It does not recursively scan deeper directories; deeper dumps need explicit index references. An empty discovery raises `FileNotFoundError`. `find_dumps()` retains its original `list[Path]` folder result and plain-`Manifest.json` single/direct-child search.

`Dump(path)` continues to select only `path/Manifest.json`. `Dump.from_manifest(path, name_pattern=None)` explicitly selects a manifest, including a prefixed one or a plain manifest that also contains an index. Both defer data loading until needed; an index-only manifest is rejected on first manifest or data access. Data paths listed in the selected manifest's `files` are literal, relative to that manifest's directory; they may already contain the prefix. For conventional files not listed there, readers prepend the selected manifest's literal `prefix` (empty if absent or null), even when it overlaps the start of a conventional filename. A listed file that is missing raises with its selected path; another file on disk is never substituted. Data paths must stay within the manifest directory after resolving symlinks.

### Opening a simulation folder

`Simulation` opens a dnachem-min folder by reading its manifests without reading data files. `path` is a flat Dump folder, a selected manifest file, a `results` folder holding Dumps, or a simulation folder with a `results/` subfolder. Each Dump is a lazy `Dump`: its tables and its meso file load on first access and are cached.

```python
from g4utils.DnaChem import Simulation

sim = Simulation("path/to/simulation", name_pattern=r"run_(?P<o2_percent>[\dp]+)pO2")
print(sim)            # Simulation('.../results', 3 subruns)
sim.table()           # one row per subrun: name, labels, Manifest summary columns

dump = sim.subrun("run_21pO2/Manifest.json")  # canonical manifest key
# sim.subrun("run_21pO2") and sim.subruns["run_21pO2"] also work if unambiguous
dump.files            # tuple of file names listed in Manifest
dump.prefix           # data file prefix from Manifest (empty if absent)
dump.has_meso         # True if meso output enabled and SpeciesMesoSpatial.h5 exists
dump.has_meso_species # True if the mesoscopic stage is enabled and SpeciesMeso.csv exists

species = dump.species()          # same frame as load_species() gives for this Dump
dump.reactions()                  # reaction counts
dump.reaction_table()             # one row per reaction
dump.physical_interactions()      # raw process firing totals
dump.physical_interactions(per_event=True)  # also mean firings per event
dump.track_lengths()              # one row per non-aborted event
dump.prechemical_records()        # pre-chemical records with nullable final positions
dump.meso_species()               # summed mesoscopic molecules and mean counts per event
dump.particle_g_estimates()       # aggregate-energy and mean-event particle G
dump.manifest                     # parsed Manifest

meso = dump.meso                  # SpeciesMesoSpatialFile (SpeciesMesoSpatial.h5)
```

`Simulation.subruns` is a mapping whose iteration, keys, values and length include each canonical key once. Folder-name lookup remains available when exactly one selected dump has that folder name; ambiguous aliases raise `KeyError` listing the manifest keys. `dump.name` and the `dump` column keep the folder label. `dump.key` is the discovery-relative key in a Simulation and the manifest filename for a standalone Dump. `dump.manifest_path` is the resolved selected manifest.

`Simulation.table()` provides one row per Dump with columns including `dump` (the folder name), `dump_key`, `manifest_path`, `prefix`, one column per Name pattern group, and Manifest summary fields such as `chemistryModel`, `handOverTime_ns`, `chemistryEndTime_ns`, `voxelSize_nm`, `mesoPixels`, `mesoTimesPerDecade`, `mesoSpatialOutput`, `threads`, and `wallTime_s` (total wall time across all runs in the Dump).

`load_species()`, `load_reactions()` and `load_reaction_table()` below are built on these classes. A missing data file raises `FileNotFoundError` naming the file.

### Physical interactions and track lengths

`Dump.physical_interactions(*, per_event=False)` reads `PhysicsInteractions.csv` with the writer's raw `label` strings and integer `count` values. Counts are dimensionless process firing totals for the dump. With `per_event=True`, it adds `count_per_event = count / manifest.totalEvents`, a mean number of firings per event. This requires a finite positive event total. It is not a G value.

`Dump.track_lengths()` reads `TrackLengths.csv` with the writer's ten fields: `run`, `event`, `primaryLength_nm`, `primaryEkin0_keV`, `primaryEkinEnd_keV`, `primaryEnd`, `secondaryFirstGen_nm`, `nSecondaryFirstGen`, `secondaryAll_nm`, and `nSecondaryAll`. The `(run, event)` pair identifies an event. Lengths are nm, primary kinetic energies are keV, and secondary counts are integers. `primaryEnd` is `stopped`, `escaped`, or `killed`. Rows have no event normalization. Both readers add the Dump's provenance and summary columns, including `dump`, `dump_key`, `manifest_path`, and `prefix`.

Readers select the CSV through the chosen manifest's file inventory and literal prefix. Missing CSVs raise `FileNotFoundError` with the resolved path; TXT files are not substituted. A header-only CSV returns typed empty columns. The track reader checks the available `trackLengths` manifest events, end counts, means, and SEMs against the rows within writer rounding precision. If its row count differs from `totalEvents`, it warns because events may have been aborted or omitted.

### Pre-chemical event records

`Dump.prechemical_records(*, allow_missing=False)` returns `run`, `event`,
`parent_id`, `molecule`, `electron_modification`, `energy_eV`, `parent_x_nm`,
`parent_y_nm`, `parent_z_nm`, `x_nm`, `y_nm`, and `z_nm`, plus the Dump's
provenance and summary columns. IDs are integers, molecule and modification
codes are raw strings, and numeric fields are floating point. Final coordinates
use pandas nullable `Float64`: missing coordinates remain null for water and
electron records alike and are never filled from parent positions. Rows are
records, without event normalization; positions are nm and energy is eV.

Files must be named `<prefix>PreChemical_run<R>_event<E>.txt`. Selection uses
the manifest's `files` inventory and exact literal prefix, then sorts numerically
by `(run, event)`. Listed paths are relative to the selected manifest and may
be nested. An inventory, including an empty list, is authoritative: unlisted
files and other prefixes are not consumed. Only if `files` is absent or null
does the reader scan the manifest's immediate directory for its prefix; it
does not recurse into other dumps or staging folders. Missing selected files
always raise. If no files are selected, the default error explains regeneration;
`allow_missing=True` instead returns typed empty columns. A valid empty or
comment-only event file also returns a typed empty frame without this option.

The observed text layout uses widths `[11,10,14,13,22,22,22,14,14,14]` for
the ten fields starting at `parent_id`. Blank lines and `#` comments are ignored.
Required fields, finite numeric values, unambiguous filenames, and unique
run/event files are checked; final coordinate fields may be blank. This layout
was observed in Geant4 11.4.3 output while the configured source was 11.4.1.
It is not matched-source semantic verification. Modification codes receive no
scientific labels in G4Utils; interpretation needs source verification against
the version that generated the data.

### Aggregate mesoscopic species

`Dump.meso_species()` reads `SpeciesMeso.csv`, returning `time_ns` (ns),
`species` (the raw string), `count` (summed integer molecules), and
`mean_count_per_event = count / manifest.totalEvents` (molecules per event).
For example, 97 `H2O2^0` molecules over two events gives 48.5 molecules per
event. These are count curves, distinct from particle G values in molecules
per 100 eV. Species spelling is preserved without mapping to particle-stage
names. The frame includes Dump provenance and summary columns, plus
`handOverTime_ns` and `chemistryEndTime_ns` copied from the manifest.

`Dump.has_meso_species` requires both enabled mesoscopic output and an existing
resolved CSV. It is independent of `Dump.has_meso`, which refers to spatial
HDF5. Modern manifests must provide a boolean `mesoEnabled`; that flag takes
precedence over `chemistryModel`. For historical manifests **without**
`mesoEnabled`, only the exact recorded `chemistryModel: "IRT_syn+mesoscopic"`
enables this reader. Missing or unknown models do not establish enablement;
neither spatial HDF5 presence nor `mesoSpatialOutput` supplies it. An explicit
null or nonboolean `mesoEnabled` is a malformed manifest.

Disabled or unconfirmed output raises `FileNotFoundError` explaining that
status. Enabled output with a missing CSV instead raises a missing-expected-file
error naming the selected path. Inventory and literal-prefix resolution follow
the other readers; TXT output is not substituted. Header-only CSVs return typed
empty frames, including when the end time is at or before hand-over. The
reader validates the exact `time_ns,species,count` header, finite nonnegative
times, nonempty species names, nonnegative integer counts, and a finite positive
`totalEvents` denominator even for empty output.

### Particle-stage G estimates

`Dump.particle_g_estimates(*, assume_valid_event_g=False)` returns the species
rows with their raw `number`, `nEvent`, `sumG`, `sumG2`, species/time columns,
and Dump provenance. It adds two estimators in molecules per 100 eV:

- `aggregate_energy_G = 100 * number / manifest.totalEnergyDeposit_eV`, using
  the Dump's total count and deposited energy. The legacy `species().G` has
  this normalization and remains available.
- `mean_event_G = sumG / nEvent`, where the scorer forms each event value as
  `G_i = 100 * count_i / deposited_energy_i_eV`, then stores
  `sumG = Σ G_i` and `sumG2 = Σ G_i²`. Unequal event deposits can make the
  two estimators differ.

`mean_event_G_sem` is pandas nullable `Float64` and is null by default. Pass
`assume_valid_event_g=True` only when you can assert that the same `nEvent`
valid event G values, **including zeros**, contributed to both moments and
every contributing event had positive deposited energy. For `nEvent >= 2`,
it then computes
`sqrt(max(0, (sumG2 - sumG²/nEvent) / ((nEvent - 1) * nEvent)))`;
for `nEvent = 1` it remains null. The CSV itself cannot verify the assertion.
Interpreting this SEM as uncertainty also requires independent events, which
the API does not establish. It is not an uncertainty for the aggregate-energy
ratio; the stored totals cannot recover event count/energy correlations.

The method rejects nonpositive or nonfinite total energy, nonpositive or
noninteger `nEvent`, nonfinite moments, and moments violating
`sumG2 >= sumG²/nEvent` beyond the writer's decimal rounding. The check allows
a deficit of at most `2e-5 * max(sumG2, sumG²/nEvent)` plus 64 floating-point
ulps of that scale; tolerated negative variance is clamped to zero for SEM.
Errors identify the manifest or species CSV row. The returned frame is separate
from the cached `species()` frame.

### Columns added by loaders

Name-pattern labels must use names distinct from reader data, calculated
estimates and manifest metadata. Conflicts raise a contextual `ValueError`
instead of overwriting fields such as `run`, `count` or `totalEvents`.
Declared integer ntuple columns are validated before conversion; fractional
event counts are rejected rather than truncated.

**`load_species()`** adds these columns to the ntuple data:
- `species`: short chemical formula (e.g., "OH", "e_aq"; mapped via `SHORT_NAMES`)
- `time_s`: time in seconds (replaces the ns `time` column)
- `G`: aggregate-energy G value = number / (totalEnergyDeposit_eV / 100)
- `dump`: Dump folder name
- `dump_key`: canonical manifest key within the selected discovery root
- `manifest_path`: resolved path of the selected manifest (distinct provenance even for shared folders)
- `prefix`: literal prefix from that manifest
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

`read_manifest()` reads a selected manifest file, or uses the same discovery rules as `Simulation` for a folder that selects exactly one Dump. A folder selecting several dumps raises `ValueError`; use `load_manifests()` for one row per Manifest run across all selected dumps, including provenance columns. Explicit file selection stays on that manifest even if it also holds a `dumps` array:

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

- `discover_dumps()`: locate selected manifests with stable keys, paths and prefixes
- `find_dumps()`: legacy folder search by plain Manifest.json
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
python -m pip install -e ".[dev]"
```

### Testing

The regular suite uses small generated fixtures and requires no simulation
output. Run it in the environment where G4Utils is installed:

```bash
python -m pytest tests -q
```

For opt-in comparisons against your own dnachem-min output, use
[`tests/validate_dnachem_output.py`](tests/validate_dnachem_output.py):

```bash
python tests/validate_dnachem_output.py --help
python tests/validate_dnachem_output.py --irt-dump /data/irt --meso-dump /data/meso-spatial --discovery-root /data/results --explicit-manifest /data/historical/EndOfRun_Manifest.json --output .scratch/validation.json
```

The two demo arguments accept dump directories or selected manifest paths.
Both require physical, track, pre-chemical and particle-species output; the
mesoscopic demo also requires aggregate CSV/TXT and the spatial HDF5 file.
Repeat `--discovery-root` for indexed, shared-folder prefixed, and nested
results, and `--explicit-manifest` for dumps outside automatic selection.
The script compares all raw CSV fields, fixed-width pre-chemical records and
null positions, particle G calculations, and mesoscopic CSV/TXT normalization.
Discovery checks compare manifest order, stable keys, each selected species
table and provenance; recursive inventory reports dumps outside the documented
selection rules without silently extending them.

Optional `--expected-irt-records 2152 --expected-irt-final-electrons 999`
compares the original two-event report baseline. A mismatch is recorded as
`changed_data_discrepancy`, while raw-value mismatches fail the script. Those
counts are observations, not assumptions about other or regenerated runs.
JSON evidence records paths, input SHA-256 hashes, package versions and
editable-install metadata. Keep generated data and evidence outside Git (for
example under the ignored `.scratch/`). The script does not run a simulation.
It checks spatial-file availability, not HDF5 contents or scientific validity;
it leaves the particle SEM assertion disabled.

With coverage report:

```bash
python -m pytest --cov=g4utils --cov-report=term-missing
```

## License

MIT License — see [LICENSE](LICENSE) for details.

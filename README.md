# G4Utils

Readers and VTK export for G4Vox voxel HDF5 output from Geant4 simulations.

## What It Does

G4Utils reads the HDF5 voxel output written by the G4Vox library (`HDF5Writer`) during Geant4 simulations. It supports two storage layouts:

- **Snapshot3D**: One group per subrun, each holding 3D scored quantities (e.g., dose or energy deposit).
- **Extendable4D**: One 4D dataset per quantity, with subruns stacked along the first axis (one slice per subrun). Subrun IDs come from the run log's `subrun_id` column; if the run log is missing, has a different row count than the slices, or contains duplicate IDs, a `UserWarning` is emitted and slice indices `0..N-1` are used instead.

The package provides lazy per-subrun loading, quantity/subrun selection, sums over subruns, and export to VTI/PVD for visualization in ParaView. Terminology (subrun ID, slice index, layout, quantity) is defined in [CONTEXT.md](CONTEXT.md).

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

# G4Utils

A Python utility library.

## Installation

```bash
pip install g4utils
```

Or install from source with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import g4utils
```

### DnaChem: loading dnachem-min output

The `g4utils.DnaChem` module provides loaders for Geant4-DNA chemical output files. The main entry point is `load_species()`, which reads species counts and computes G values (molecules per 100 eV of deposited energy).

#### Example: O2 concentration scan

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

#### Columns added by loaders

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

**`load_manifests()`** loads one row per run with Dump and run metadata.

#### Other utilities

- `find_dumps()`: locate Dump folders by Manifest.json
- `read_ntuple()`: parse Geant4 wcsv CSV ntuples
- `short_name()` / `SHORT_NAMES`: map raw species names to short labels

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/G4Utils.git
cd G4Utils

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

## License

MIT License — see [LICENSE](LICENSE) for details.

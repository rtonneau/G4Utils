# G4Utils

Python utilities for reading and post-processing Geant4 / Geant4-DNA simulation
output. This glossary fixes the vocabulary; it is not a spec.

## Language

**Dump**:
One folder of output written by a Geant4-DNA chemistry simulation (dnachem-min) in a
single flush: species and reaction tallies, pre-chemical files, and exactly one
**Manifest**. The unit that is loaded and compared (e.g. one O2 level of a scan).
_Avoid_: "run" (a Geant4 run is one `/run/beamOn`; a Dump may cover several),
"case", "simulation".

**Manifest**:
The `Manifest.json` inside a Dump: what was simulated (beam per run, Chemistry,
scavengers, pH), the totals (events, energy deposit) and the files produced. Source
of truth for a Dump's physical parameters.
_Avoid_: "metadata" (collides with `ReactionsMetadata.csv`).

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
identified by name, never by numeric species ID.

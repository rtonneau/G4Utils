---
bump: minor
floor: minor
---
- Added `Simulation` and `Dump` to `g4utils.DnaChem`: open a dnachem-min simulation folder (with a `results/` subfolder, a results folder or a single Dump) and reach each Dump's manifest, species, reactions, reaction table and mesoscopic file on demand, loaded on first access and cached.
- `load_species` and `load_reactions` now also accept a simulation folder that has a `results/` subfolder.

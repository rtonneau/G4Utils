---
bump: minor
floor: minor
---
- Add `read_manifest()` and the `Manifest` object to `g4utils.DnaChem`: read one dnachem-min `Manifest.json` (Dump folder, manifest file, or a folder with a single Dump) and access every field as an attribute, including per-run data and scavenger molarities.
- Add `Manifest.runs_table()` (per-run DataFrame) and `Manifest.scavenger_molarity(species)`.
- Display a Manifest in a Jupyter notebook as formatted HTML tables (or as text with `print`), and explicitly with `Manifest.show()`.

---
bump: minor
floor: minor
---
- Export `read_manifest()` and `Manifest` from `g4utils.DnaChem`: read one Manifest.json file with full attribute access.
- Add `Manifest.runs_table()` to get per-run metadata as a DataFrame; supports unknown run keys as extra columns.
- Add `Manifest.scavenger_molarity(species)` to query the molarity of a scavenger species.
- Manifest objects display as formatted text and as HTML tables in notebooks via `_repr_html_()` and `show()`.

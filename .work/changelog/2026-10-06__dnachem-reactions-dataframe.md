---
bump: minor
floor: minor
---
- `g4utils.DnaChem.load_reaction_table(path)`: one row per reaction from `ReactionsMetadata.csv`, with a Short-name `equation`, `reactants`/`products` tuples and `reactant_<X>`/`product_<X>` count columns for filtering (e.g. `df[df["product_H2O2"] > 0]`).
- Short names now cover `O^-1` → `O-` and `O_3^-1` → `O3-` (also in `load_species`).

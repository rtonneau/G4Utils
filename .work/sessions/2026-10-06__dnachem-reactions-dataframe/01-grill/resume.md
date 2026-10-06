# Session: dnachem-reactions-dataframe

**Date:** 2026-10-06T10:02:12.114Z
**Status:** Grill phase complete

## Problem Statement

`g4utils.DnaChem` has no way to list the reactions a Dump declares in `ReactionsMetadata.csv` (`REACTIONS_METADATA_FILE`) in a form that can be filtered by species: "all reactions producing H2O2", "all reactions with OH as reactant". The raw `reaction` strings use Geant4 molecule names (`°OH^0`, `H3O^1`, `OH^-1`), which are hard to read and to match.

## Context & Constraints

- **Current behavior:** `load_reactions` merges the raw `reaction` string (Geant4 names) onto reaction counts; nothing parses reactants and products. `SHORT_NAMES` maps Geant4 names to Short names but lacks `O^-1` and `O_3^-1`, which occur in real Dumps.
- **Pain point:** filtering reactions by reactant or product needs ad hoc string parsing on Geant4 names.
- **Dependencies:** keep `load_reactions` unchanged (users join on `reactionId`). Reuse `short_name` / `SHORT_NAMES` as the single species vocabulary (Short name, no `*` radical marks). Adding entries to `SHORT_NAMES` also changes `load_species` output for those species.
- **Tech stack:** Python, pandas, pytest; test Dumps from the `make_dump` fixture in `tests/conftest.py`.

## Success Metrics

- `load_reaction_table(dump_folder)` and `load_reaction_table(dump / "ReactionsMetadata.csv")` return the same DataFrame, one row per reaction, with columns `reactionId`, `reaction` (raw), `equation` (Short names, e.g. `OH + OH -> H2O2`), `reactants`, `products` (tuples of Short names with stoichiometric repeats; `()` for `(no products)`), and int `reactant_X` / `product_X` for every species appearing anywhere in the file (0 where absent).
- `df[df["product_H2O2"] > 0]` lists all reactions producing H2O2; `df[df["reactant_OH"] > 0]` all with OH as reactant; `O^-1 + O^-1 -> ...` gives `reactant_O- == 2`.
- A scan root with several Dumps raises a clear error; a missing file raises `FileNotFoundError`; a line without `->` raises `ValueError`.
- `SHORT_NAMES` maps `O^-1 → O-` and `O_3^-1 → O3-`; unknown species keep their raw name.
- Exported in `g4utils.DnaChem.__all__`, documented in README, tests pass.

## Architecture & Approach

New function `load_reaction_table(path)` in `g4utils/DnaChem/loaders.py`, next to `_read_metadata` (reuse it). Resolve `path`: a file is read directly; a folder that is a Dump (contains the metadata file) is read; anything with several Dumps (via `find_dumps`) raises `ValueError`. Parse each `reaction` string: split on `->`, split each side on ` + `, map each token with `short_name`, `(no products)` → empty tuple. Build `equation` from the Short names (`(no products)` kept as text on the product side). Wide columns from the union of species seen, named `reactant_<Short name>` / `product_<Short name>` verbatim (e.g. `product_OH-`, `reactant_H3O+`), int counts.

## Assumptions & Trade-offs

- One vocabulary: Short name without `*` (user rejected the `H*` / `OH*` notation).
- Column names keep `+`/`-`, so charged species need `df["product_OH-"]` rather than attribute access; accepted to avoid a second spelling.
- No Dump columns and no scan support: the reaction table describes chemistry, identical across Dumps of one setup.
- `load_reactions` is not enriched; join on `reactionId`.

## Open Questions

None.

## Notes

Glossary updated during the grill: Short name is the single species vocabulary (no radical marks); new term Reaction table. Real file sample: `C:/DEV/GEANT4/DATA/260924/run_100keV/ReactionsMetadata.csv` (42 reactions, includes `(no products)`, `O^-1`, `O_3^-1`).

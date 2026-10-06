# Implementation Plan

**Session:** dnachem-reactions-dataframe
**Date:** 2026-10-06T10:07:49.949Z
**Estimated effort:** 2 hours

## Strategy

Ticket 1 implements the behaviour: the two new `SHORT_NAMES` entries and `load_reaction_table(path)` in `loaders.py`, with tests. Ticket 2 exposes the function in the package API and documents it in the README.

## Tickets Overview

- **Ticket 1:** `load_reaction_table(path)` builds the Reaction table (equation, reactant/product tuples, wide stoichiometric count columns), and `SHORT_NAMES` gains `O-` and `O3-`.
- **Ticket 2:** Export `load_reaction_table` from `g4utils.DnaChem` and document it in the README.

## Sequencing Rationale

The docs and export need the function to exist, so ticket 2 comes after ticket 1.

## Risks & Mitigation

- **Risk:** Reaction strings in other Geant4 versions use different spacing around `+` / `->` → **Mitigation:** split on `->` and on `+` surrounded by whitespace, then strip the tokens. Never split on a bare `+`, because `H3O+`-style raw names could appear.
- **Risk:** Adding `SHORT_NAMES` entries changes the `species` column of `load_species` for `O^-1` / `O_3^-1` → **Mitigation:** the change is intended (agreed in the grill). Existing tests don't use those species.

## Assumptions

- `ReactionsMetadata.csv` always has the `reactionId,reaction` header (as read by `_read_metadata`).
- Products equal to the literal `(no products)` mean an empty product side.

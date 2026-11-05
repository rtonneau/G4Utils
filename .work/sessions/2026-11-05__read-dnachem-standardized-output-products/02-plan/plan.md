# Implementation Plan

**Session:** read-dnachem-standardized-output-products
**Date:** 2026-11-05T20:30:34.972Z
**Estimated effort:** 2-3 days

## Strategy

Add the new-layout support behind the existing `Dump` accessors in small, independently testable layers: manifest fields first, then a format-agnostic table reader, then one ticket per group of products, and finally packaging and docs. Legacy readers stay in place and are mapped onto the new column names, so every ticket leaves the legacy test suite green. The real dnachem-min smoke12 dump (all four formats) is the shared fixture for verification.

## Tickets Overview

- **Ticket 1:** `Manifest` gains `outputFormats` and `hdf5`; `Dump` gets layout detection and `output_formats`.
- **Ticket 2:** New `tables.py` reading csv/h5/root/txt with a csv > h5 > root > txt preference and identical dtypes.
- **Ticket 3:** Species, SpeciesMeso and G estimators on the new layout (`G` = `gMean`), legacy mapped to the same columns.
- **Ticket 4:** Reactions (time bins), `ReactionsCatalog` and `load_reaction_table` on both layouts.
- **Ticket 5:** PhysicsInteractions and TrackLengths on the new layout with new column names.
- **Ticket 6:** PreChemical records on the new layout (per-event csv/txt, single h5/root).
- **Ticket 7:** Packaging (uproot extra, version 0.5.0), README, CONTEXT.md, CHANGELOG, and a real-dump regression test.

## Sequencing Rationale

Tickets 1 and 2 are the foundation: layout detection and format-agnostic table reading. Tickets 3-6 each use them for one product group and are independent of each other, ordered by how central the product is. Ticket 7 comes last because the docs and the cross-format regression describe the finished behavior.

## Risks & Mitigation

- **Risk:** `species()["G"]` changes meaning for new dumps. → **Mitigation:** documented as a breaking change in CHANGELOG and README; version bump to 0.5.0; legacy dumps keep their old value.
- **Risk:** root integers are stored as doubles and h5 strings come back as bytes/objects, so frames differ between formats. → **Mitigation:** the table reader normalizes dtypes and a cross-format equality test on the real smoke12 dump guards it.
- **Risk:** `uproot` missing in a user environment. → **Mitigation:** optional import with a clear error; csv is the default and preferred format, so most users never need it.
- **Risk:** txt labels with spaces (reaction catalogue) parse wrongly. → **Mitigation:** txt is only a last-resort fallback, with a dedicated catalogue parse and a test.

## Assumptions

- Layout is new iff the manifest has `outputFormats`.
- The smoke12 dump at `C:\DEV\GEANT4\SIM\dnachem-min\.scratch\tests\2026-11-04__data-output-should-be-standardized-each-type-of-data-should-have\smoke12\all` is available; tests copy the small files they need into `tests/data/`.
- The `SpeciesMesoSpatial` h5 layout is unchanged, so `g4utils.HDF5` is untouched.
- Legacy frames use the new column names, with columns they lack left as NaN.

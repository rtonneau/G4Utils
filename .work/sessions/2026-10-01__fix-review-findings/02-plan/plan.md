# Implementation Plan

**Session:** fix-review-findings
**Date:** 2026-10-01T08:41:28.055Z
**Estimated effort:** 1 day

## Strategy

Test infrastructure goes first. Every later ticket is then written test-first, against fixture files that copy what the C++ writer (`G4Vox/src/HDF5Writer.cc`) actually produces. Fixes then go in order of the data they depend on. The run log must parse correctly before 4D subrun IDs can come from it. Layout detection and binary VTI export are independent additions after that, and docs and the version bump come last, describing the final API.

Global constraints (they apply to every ticket):
- Python >= 3.10. No new runtime dependencies: use the standard library only (`base64`, `zlib`, `warnings`, `xml`).
- Ground truth for the file format: `/metadata` attrs `dims_xyz`, `spacing_mm`, `origin_mm` (float64 arrays), `mode` (`"Snapshot3D"` or `"Extendable4D"`), `quantities` (comma-separated string). Voxel data is float64. `/run_log` is float64 `(N, 4)` with attr `columns = "unix_timestamp, primaries, runtime_s, subrun_id"`.
- Breaking API changes are allowed, with no deprecation shims.
- Warnings use `warnings.warn(..., UserWarning)`, never `print`.
- Match the existing style: `from __future__ import annotations`, numpy-style docstrings, ruff line length 100.
- Glossary terms come from `CONTEXT.md`: subrun ID vs slice index, layout, quantity, run log.

## Tickets Overview

| # | Ticket | Model | Delivers |
|---|---|---|---|
| 01 | test-infra-ci | sonnet | Local editable install, `--cov` moved to CI, CI on Python 3.10–3.13, writer-faithful fixtures, baseline tests |
| 02 | run-log-columns | sonnet | `run_log` columns named from the `columns` attr; `total_primaries()` reads `primaries` |
| 03 | 4d-subrun-ids | opus | 4D subrun IDs from `run_log.subrun_id`, with a warn-and-fallback path; `G4VoxFile4D` reduced to a thin subclass |
| 04 | core-cleanups | sonnet | Duplicate helper and old wrapper removed; float64/int64 accumulation; `warnings.warn` in `_read_geometry` |
| 05 | open-vox-file | sonnet | `detect_layout()` and `open_vox_file()` |
| 06 | binary-vti | opus | Base64 binary VTI (default), zlib compression, ASCII kept, options passed through the façade |
| 07 | docs-version | haiku | Real README, pyproject description, version 0.2.0 |

## Sequencing Rationale

- 01 first: it gives every other ticket a test harness and a passing CI baseline.
- 02 before 03: the 4D subrun-ID mapping reads `run_log["subrun_id"]`, which only exists after 02.
- 03 before 04: 03 deletes the `G4VoxFile4D` overrides, so 04 only has to clean the base and shared modules.
- 05 after 03: detection returns the final thin `G4VoxFile3D`/`G4VoxFile4D` classes.
- 06 after 04: 04 moves `select_quantities` out of `vti_export.py`, so 06 edits a file that only holds export code.
- 07 last: the README documents the final API (`open_vox_file`, run_log columns, VTI options).

## Risks & Mitigation

- **Fixtures drift from the real writer.** Mitigation: the conftest writers copy `HDF5Writer.cc` attribute names, dtypes and the run_log layout exactly, and the source file is cited in a conftest comment.
- **Compressed VTI header gets the wrong layout, so ParaView can't open the file.** Mitigation: follow VTK's `vtkXMLWriter` layout (UInt64 header `[nblocks, blocksize, partial_last_size (0 if last block full), compressed sizes...]`, header and data base64-encoded separately). Tests decode it with an independent decoder, plus a `vtkXMLImageDataReader` round-trip test that runs when `vtk` is installed (`pytest.importorskip`).
- **Subrun-ID fallback hides real problems.** Mitigation: each fallback reason has its own warning message, and each one is tested.
- **Breaking changes affect the user's own scripts.** Accepted: Alpha package, version bump to 0.2.0, and changes listed in commit messages.

## Assumptions

- The Python on PATH (3.13.9, conda) is the environment to install into, as the user approved.
- `Snapshot3D` group names (`subrun_XXXX`) are unique per file: the writer raises on a duplicate dataset, so no fallback is needed for 3D.
- One `Export` call always appends exactly one 4D slice and one run_log row, so row *i* describes slice *i*.
- VTI arrays use C order of `(nZ, nY, nX)`, which matches VTK's x-fastest cell ordering. No transpose is needed.

## Token Usage

- **Input:** 12
- **Output:** 13755
- **Cache read:** 640666
- **Cache creation:** 22195
- **Total:** 676628

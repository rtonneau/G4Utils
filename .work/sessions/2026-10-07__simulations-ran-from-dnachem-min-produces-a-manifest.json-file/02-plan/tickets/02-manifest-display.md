# Ticket 02: manifest-display

**Model:** sonnet
**Effort:** medium

**Acceptance Criteria:**
- [ ] `str(manifest)` returns an aligned text summary with sections Overview, Chemistry, Totals, Runs, Files and a trailing Other section for unknown keys.
- [ ] `manifest._repr_html_()` returns HTML tables for the same sections, with Files in a collapsed `<details>`; values are HTML-escaped.
- [ ] Absent (`None`) fields are skipped in both renderings; the output location (`outputDirAsConfigured`, `outputDirAbsolute`, `prefix`, `subdir`) appears in Overview; meso settings appear in Chemistry only when present.
- [ ] `manifest.show()` calls `IPython.display.display` with the Manifest; IPython is imported lazily and `show()` raises a clear `ImportError` message if it is missing.
- [ ] Tests cover the real fixture and a minimal manifest.

**Files to Touch:**
- `src/g4utils/DnaChem/manifest.py`
- `tests/test_dnachem_manifest.py`

**Verification Step:**

Run:
```bash
python -m pytest tests/test_dnachem_manifest.py -q
```

Expected:
All tests pass, no failures.

**Notes:**

Do not add IPython to `pyproject.toml` dependencies. In tests, monkeypatch `IPython.display` via a fake module in `sys.modules`. Keep rendering helpers private (`_`-prefixed).

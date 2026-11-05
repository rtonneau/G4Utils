"""Real smoke12 sample: every accessor gives equal frames in csv, h5 and root."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from g4utils.DnaChem import Dump

SAMPLE = Path(__file__).parent / "data" / "smoke12_dump"
FORMATS = ("csv", "h5", "root")
ACCESSORS = (
    "species", "meso_species", "reactions", "reaction_table",
    "physical_interactions", "track_lengths",
)


def _dump_for(tmp_path: Path, fmt: str) -> Dump:
    if fmt == "root":
        pytest.importorskip("uproot")
    directory = tmp_path / fmt
    directory.mkdir()
    manifest = json.loads((SAMPLE / "Manifest.json").read_text(encoding="utf-8"))
    manifest["outputFormats"] = {k: [fmt] for k in manifest["outputFormats"]}
    names = [n for n in manifest["files"] if n.endswith(f".{fmt}")]
    manifest["files"] = names
    for name in names:
        shutil.copy(SAMPLE / name, directory / name)
    (directory / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return Dump(directory)


def _call(dump: Dump, name: str) -> pd.DataFrame:
    frame = getattr(dump, name)()
    return frame.drop(columns=[c for c in ("dump", "dump_key", "manifest_path") if c in frame])


@pytest.mark.parametrize("accessor", ACCESSORS)
def test_accessors_equal_across_formats(tmp_path, accessor):
    frames = {fmt: _call(_dump_for(tmp_path, fmt), accessor) for fmt in FORMATS
              if fmt != "root" or pytest.importorskip("uproot")}
    reference = frames["csv"]
    assert len(reference) > 0
    for fmt, frame in frames.items():
        pd.testing.assert_frame_equal(frame, reference, check_exact=False, rtol=1e-9)


def test_particle_g_estimates_use_stored_g(tmp_path):
    frame = _dump_for(tmp_path, "csv").particle_g_estimates()
    assert (frame["mean_event_G"] == frame["gMean"]).all()

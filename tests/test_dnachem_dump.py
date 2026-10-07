from __future__ import annotations

import pandas as pd
import pytest

from g4utils.DnaChem.dump import Dump
from g4utils.DnaChem.loaders import load_reaction_table, load_reactions, load_species
from g4utils.DnaChem.manifest import Manifest
from tests.conftest import write_species_meso_spatial

PATTERN = r"run_(?P<o2_percent>[0-9p]+)pO2"


def test_construction_reads_nothing(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_0pO2")
    (d / "Species_nt_species.csv").unlink()
    dump = Dump(d)
    assert dump.name == "run_0pO2"
    assert dump.labels == {}


def test_construction_requires_manifest(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_a", with_manifest=False)
    with pytest.raises(FileNotFoundError, match="Manifest.json"):
        Dump(d)


def test_lazy_cached(tmp_path, make_dump):
    d = Dump(make_dump(tmp_path, "run_a"))
    write_species_meso_spatial(d.path / "SpeciesMesoSpatial.h5")
    assert isinstance(d.manifest, Manifest)
    assert d.manifest is d.manifest
    assert d.species() is d.species()
    assert d.reactions() is d.reactions()
    assert d.reaction_table() is d.reaction_table()
    assert d.meso is d.meso


def test_matches_loaders(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_0p3pO2")
    dump = Dump(path, PATTERN)
    pd.testing.assert_frame_equal(
        dump.species(), load_species(path, PATTERN)
    )
    pd.testing.assert_frame_equal(
        dump.reactions(), load_reactions(path, PATTERN)
    )
    pd.testing.assert_frame_equal(dump.reaction_table(), load_reaction_table(path))
    assert dump.labels == {"o2_percent": 0.3}


@pytest.mark.parametrize(
    "fname, call",
    [
        ("Species_nt_species.csv", lambda d: d.species()),
        ("Reactions_nt_reactions.csv", lambda d: d.reactions()),
        ("ReactionsMetadata.csv", lambda d: d.reaction_table()),
        ("SpeciesMesoSpatial.h5", lambda d: d.meso),
    ],
)
def test_missing_file_raises(tmp_path, make_dump, fname, call):
    path = make_dump(tmp_path, "run_a")
    (path / fname).unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match=fname):
        call(Dump(path))

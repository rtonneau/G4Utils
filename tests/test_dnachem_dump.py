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


def test_files_and_prefix_default(tmp_path, make_dump):
    dump = Dump(make_dump(tmp_path, "run_a"))
    assert dump.files == ()
    assert dump.prefix == ""


def test_files_from_manifest(tmp_path, make_dump):
    names = ["Species_nt_species.csv", "Manifest.json"]
    dump = Dump(make_dump(tmp_path, "run_a", manifest_extra={"files": names}))
    assert dump.files == tuple(names)


def test_null_prefix_is_empty(tmp_path, make_dump):
    dump = Dump(make_dump(tmp_path, "run_a", manifest_extra={"prefix": None}))
    assert dump.prefix == ""
    assert len(dump.species()) > 0


def test_prefixed_files(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", prefix="pre_")
    write_species_meso_spatial(path / "pre_SpeciesMesoSpatial.h5")
    dump = Dump(path)
    assert dump.prefix == "pre_"
    assert not (path / "Species_nt_species.csv").exists()
    assert len(dump.species()) > 0
    assert len(dump.reactions()) > 0
    assert len(dump.reaction_table()) == 2
    assert dump.has_meso
    assert dump.meso is not None


def test_prefixed_loaders(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", prefix="pre_")
    assert len(load_species(path)) == 8
    assert len(load_reactions(path)) == 3
    assert len(load_reaction_table(path)) == 2


def test_has_meso(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    assert not Dump(path).has_meso  # file missing
    write_species_meso_spatial(path / "SpeciesMesoSpatial.h5")
    assert Dump(path).has_meso
    path2 = make_dump(
        tmp_path, "run_b", manifest_extra={"mesoSpatialOutput": False}
    )
    write_species_meso_spatial(path2 / "SpeciesMesoSpatial.h5")
    assert not Dump(path2).has_meso
    path3 = make_dump(tmp_path, "run_c", manifest_extra={"mesoSpatialOutput": True})
    assert not Dump(path3).has_meso  # enabled but file missing


def test_meso_disabled_message(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", manifest_extra={"mesoSpatialOutput": False})
    with pytest.raises(FileNotFoundError, match="disabled in the Manifest"):
        Dump(path).meso


def test_meso_missing_message(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", manifest_extra={"mesoSpatialOutput": True})
    with pytest.raises(FileNotFoundError, match="missing"):
        Dump(path).meso

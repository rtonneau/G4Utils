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


def test_standardized_species_columns_and_g(tmp_path, make_standardized_dump):
    dump = Dump(make_standardized_dump(tmp_path, "new", energy_deposit_eV=400))
    frame = dump.species()
    assert len(frame) == 3
    assert frame["speciesName"].tolist() == ["°OH^0", "e_aq^-1", "°OH^0"]
    assert frame["species"].tolist() == ["OH", "e_aq", "OH"]
    assert frame["time_s"].tolist() == pytest.approx([1e-12, 1e-12, 5e-9])
    assert frame["G"].tolist() == [0.5, 0.4, 0.3]  # gMean, not countSum / (E/100)
    assert {"nEvents", "countSum", "gStdErr", "dump", "chemistry"} <= set(frame.columns)


def test_legacy_species_maps_onto_standardized_columns(tmp_path, make_dump):
    frame = Dump(make_dump(tmp_path, "old", energy_deposit_eV=400)).species()
    first = frame.iloc[0]
    assert first["nEvents"] == 100
    assert first["countSum"] == 400000
    assert first["countMean"] == 4000
    assert first["gSum"] == 0.5 and first["gMean"] == pytest.approx(0.005)
    assert first["time_ns"] == pytest.approx(0.001)
    assert first["G"] == pytest.approx(400000 / (400 / 100))  # legacy pooled ratio
    assert frame[["countSumSq", "countStdErr", "gStdErr"]].isna().all().all()


@pytest.mark.parametrize("fmt", ["csv", "h5", "root", "txt"])
def test_standardized_species_every_format(tmp_path, make_standardized_dump, fmt):
    frame = Dump(make_standardized_dump(tmp_path, "new", formats=(fmt,))).species()
    assert frame["G"].tolist() == pytest.approx([0.5, 0.4, 0.3])
    assert frame["speciesName"].tolist() == ["°OH^0", "e_aq^-1", "°OH^0"]


@pytest.mark.parametrize("fmt", ["csv", "h5", "root", "txt"])
def test_standardized_reactions_every_format(tmp_path, make_standardized_dump, fmt):
    dump = Dump(make_standardized_dump(tmp_path, "new", formats=(fmt,)))
    frame = dump.reactions()
    assert frame["reactionId"].tolist() == [1, 25, 25]
    assert frame["binStart_ns"].tolist() == [0.0, 0.0, 0.001]
    assert frame["binEnd_ns"].tolist() == [0.001, 0.001, 0.01]
    assert frame["count"].tolist() == [646, 4436, 9000]
    assert frame["time_s"].tolist() == pytest.approx([1e-12, 1e-12, 1e-11])
    assert frame["reaction"].tolist()[1] == "°OH^0 + °OH^0 -> H2O2^0"
    assert frame["countStdErr"].tolist() == [1.0, 2.0, 3.0]
    table = dump.reaction_table()
    assert table["equation"].tolist() == ["H3O+ + OH- -> (no products)", "OH + OH -> H2O2"]


def test_legacy_reactions_mapped_onto_standardized_columns(tmp_path, make_dump):
    frame = Dump(make_dump(tmp_path, "old")).reactions()
    assert frame["binEnd_ns"].tolist() == pytest.approx([0.001, 0.001, 999.999])
    assert frame["binStart_ns"].isna().all()
    assert frame["nEvents"].tolist() == [100, 100, 100]
    assert frame["countSum"].tolist() == [646.0, 4436.0, 9000.0]
    assert frame["countMean"].tolist() == pytest.approx([6.46, 44.36, 90.0])
    assert frame["count"].tolist() == [646, 4436, 9000]
    assert frame["countStdErr"].isna().all()

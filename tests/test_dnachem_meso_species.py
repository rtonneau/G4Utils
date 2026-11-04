from __future__ import annotations

import pandas as pd
import pytest

from g4utils.DnaChem import Dump


def _csv(path, rows=("5,H2O2^0,97", "10,°OH^0,0"), name="SpeciesMeso.csv"):
    file = path / name
    file.write_text("time_ns,species,count\n" + "".join(f"{row}\n" for row in rows), encoding="utf-8")
    return file


def test_meso_species_normalization_and_metadata(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", manifest_extra={
        "mesoEnabled": True, "mesoSpatialOutput": False, "totalEvents": 2,
        "handOverTime_ns": 5, "chemistryEndTime_ns": 1000,
    })
    _csv(path)
    dump = Dump(path)
    assert dump.has_meso_species
    assert not dump.has_meso
    frame = dump.meso_species()
    assert frame["time_ns"].tolist() == [5.0, 10.0]
    assert frame["species"].tolist() == ["H2O2^0", "°OH^0"]
    assert frame["count"].tolist() == [97, 0]
    assert frame["mean_count_per_event"].tolist() == [48.5, 0.0]
    assert frame["handOverTime_ns"].tolist() == [5, 5]
    assert frame["chemistryEndTime_ns"].tolist() == [1000, 1000]
    assert frame["dump_key"].tolist() == ["Manifest.json"] * 2
    assert frame["manifest_path"].tolist() == [str(dump.manifest_path)] * 2
    assert frame["prefix"].tolist() == [""] * 2
    assert pd.api.types.is_integer_dtype(frame["count"])
    assert "G" not in frame


@pytest.mark.parametrize("prefix", ["EndOfRun_", "S"])
def test_meso_species_prefix_inventory_and_missing(tmp_path, make_dump, prefix):
    name = f"{prefix}SpeciesMeso.csv"
    path = make_dump(tmp_path, "run_a", prefix=prefix, manifest_extra={
        "mesoEnabled": True, "files": [f"nested/{name}"],
    })
    (path / "nested").mkdir()
    selected = _csv(path / "nested", name=name)
    _csv(path, rows=("5,wrong,999",))
    _csv(path, rows=("5,also wrong,888",), name=name)
    dump = Dump(path)
    assert dump.meso_species()["count"].tolist() == [97, 0]
    selected.unlink()
    assert not dump.has_meso_species
    with pytest.raises(FileNotFoundError, match="expected.*nested"):
        dump.meso_species()


def test_meso_species_prefix_fallback(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", prefix="S", manifest_extra={"mesoEnabled": True})
    _csv(path, name="SSpeciesMeso.csv")
    assert Dump(path).meso_species()["count"].tolist() == [97, 0]


@pytest.mark.parametrize("metadata,enabled", [
    ({"mesoEnabled": False, "chemistryModel": "IRT_syn+mesoscopic"}, False),
    ({"mesoEnabled": True, "chemistryModel": "IRT_syn"}, True),
    ({"chemistryModel": "IRT_syn+mesoscopic"}, True),
    ({"chemistryModel": "IRT_syn"}, False),
    ({"chemistryModel": "SBS"}, False),
    ({"chemistryModel": "unknown+mesoscopic"}, False),
    ({}, False),
])
def test_meso_species_enablement(tmp_path, make_dump, metadata, enabled):
    path = make_dump(tmp_path, "run_a", manifest_extra=metadata)
    _csv(path)
    (path / "SpeciesMesoSpatial.h5").touch()
    dump = Dump(path)
    assert dump.has_meso  # Aggregate availability is independent of spatial files.
    assert dump.has_meso_species is enabled
    if enabled:
        assert len(dump.meso_species()) == 2
    else:
        with pytest.raises(FileNotFoundError, match="disabled|not enabled"):
            dump.meso_species()


def test_meso_species_missing_expected_csv(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", manifest_extra={"mesoEnabled": True})
    (path / "SpeciesMeso.Txt").write_text("mean counts", encoding="utf-8")
    dump = Dump(path)
    assert not dump.has_meso_species
    with pytest.raises(FileNotFoundError, match="expected.*SpeciesMeso.csv"):
        dump.meso_species()


@pytest.mark.parametrize("end_time", [1, 5, 1000])
def test_meso_species_header_only(tmp_path, make_dump, end_time):
    path = make_dump(tmp_path, "run_a", manifest_extra={
        "mesoEnabled": True, "handOverTime_ns": 5, "chemistryEndTime_ns": end_time,
    })
    _csv(path, rows=())
    dump = Dump(path)
    assert dump.has_meso_species
    frame = dump.meso_species()
    assert frame.empty
    for column in ("time_ns", "mean_count_per_event"):
        assert pd.api.types.is_float_dtype(frame[column])
    assert pd.api.types.is_string_dtype(frame["species"])
    assert pd.api.types.is_integer_dtype(frame["count"])
    assert {"handOverTime_ns", "chemistryEndTime_ns", "manifest_path"} <= set(frame)
    assert dump.manifest.handOverTime_ns == 5
    assert dump.manifest.chemistryEndTime_ns == end_time


@pytest.mark.parametrize("row", [
    "-1,x,1", "nan,x,1", "inf,x,1", "bad,x,1", ",x,1",
    "5,x,-1", "5,x,1.5", "5,x,nan", "5,x,inf", "5,x,",
    "5,x,9223372036854775808", "5,,1", "5,x", "5,x,1,extra",
])
def test_meso_species_bad_fields(tmp_path, make_dump, row):
    path = make_dump(tmp_path, "run_a", manifest_extra={"mesoEnabled": True})
    _csv(path, rows=(row,))
    with pytest.raises(ValueError, match="SpeciesMeso.csv"):
        Dump(path).meso_species()


@pytest.mark.parametrize("header", ["", "time,species,count\n", "time_ns,species\n"])
def test_meso_species_bad_header(tmp_path, make_dump, header):
    path = make_dump(tmp_path, "run_a", manifest_extra={"mesoEnabled": True})
    (path / "SpeciesMeso.csv").write_text(header, encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        Dump(path).meso_species()


@pytest.mark.parametrize("total", [0, -1, None, True, "2", float("nan"), float("inf")])
@pytest.mark.parametrize("rows", [(), ("5,x,97",)])
def test_meso_species_invalid_denominator(tmp_path, make_dump, total, rows):
    path = make_dump(tmp_path, "run_a", manifest_extra={"mesoEnabled": True, "totalEvents": total})
    _csv(path, rows=rows)
    with pytest.raises(ValueError, match="totalEvents.*finite and positive"):
        Dump(path).meso_species()


@pytest.mark.parametrize("flag", [None, "false", 0, 1])
def test_meso_species_invalid_enablement(tmp_path, make_dump, flag):
    path = make_dump(tmp_path, "run_a", manifest_extra={
        "mesoEnabled": flag, "chemistryModel": "IRT_syn+mesoscopic",
    })
    _csv(path)
    with pytest.raises(ValueError, match="mesoEnabled.*boolean"):
        Dump(path).meso_species()

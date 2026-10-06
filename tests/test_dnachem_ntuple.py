from __future__ import annotations

import pandas as pd

from g4utils.DnaChem.ntuple import read_ntuple
from g4utils.DnaChem.species_names import SHORT_NAMES, short_name


def test_read_ntuple_species_columns_and_dtypes(make_dump, tmp_path):
    d = make_dump(tmp_path, "dump")
    df = read_ntuple(d / "Species_nt_species.csv")
    assert list(df.columns) == [
        "speciesID", "number", "nEvent", "speciesName", "time", "sumG", "sumG2",
    ]
    assert len(df) == 8
    assert df["speciesID"].dtype == "int64"
    assert df["number"].dtype == "int64"
    assert df["nEvent"].dtype == "int64"
    assert df["time"].dtype == "float64"
    assert df["sumG"].dtype == "float64"
    assert pd.api.types.is_string_dtype(df["speciesName"])
    assert df["speciesName"].map(type).eq(str).all()


def test_read_ntuple_keeps_degree_names_crlf(make_dump, tmp_path):
    d = make_dump(tmp_path, "dump")
    df = read_ntuple(d / "Species_nt_species.csv")
    names = set(df["speciesName"])
    assert "°OH^0" in names
    assert "HO_2°^0" in names
    assert not any("\r" in n for n in names)


def test_short_name_table():
    expected = {
        "H3O^1": "H3O+", "°OH^0": "OH", "OH^-1": "OH-", "e_aq^-1": "e_aq",
        "H^0": "H", "H_2^0": "H2", "H2O2^0": "H2O2", "HO_2^-1": "HO2-",
        "O_2^0": "O2", "°O^0": "O", "O_2^-1": "O2-", "HO_2°^0": "HO2",
        "O^-1": "O-", "O_3^-1": "O3-",
    }
    assert SHORT_NAMES == expected
    for raw, short in expected.items():
        assert short_name(raw) == short


def test_short_name_unknown_passthrough():
    assert short_name("Foo^3") == "Foo^3"

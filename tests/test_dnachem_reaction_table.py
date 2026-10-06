from __future__ import annotations

import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from g4utils.DnaChem.loaders import load_reaction_table
from g4utils.DnaChem.species_names import short_name


def _write_meta(tmp_path, lines):
    f = tmp_path / "ReactionsMetadata.csv"
    f.write_text("\n".join(["reactionId,reaction", *lines]) + "\n", encoding="utf-8")
    return f


def test_dump_folder_and_file_equal(tmp_path, make_dump):
    d = make_dump(tmp_path, "d")
    assert_frame_equal(
        load_reaction_table(d), load_reaction_table(d / "ReactionsMetadata.csv")
    )


def test_no_products(tmp_path, make_dump):
    df = load_reaction_table(make_dump(tmp_path, "d"))
    row = df[df["reactionId"] == 1].iloc[0]
    assert row["products"] == ()
    assert row["equation"] == "H3O+ + OH- -> (no products)"
    assert row["reactant_H3O+"] == 1
    assert row["product_H3O+"] == 0
    assert list(df.columns[:5]) == [
        "reactionId", "reaction", "equation", "reactants", "products"
    ]


def test_stoichiometry(tmp_path):
    f = _write_meta(tmp_path, ["19,O^-1 + O^-1 -> H2O2^0 + OH^-1 + OH^-1"])
    df = load_reaction_table(f)
    row = df.iloc[0]
    assert row["reactants"] == ("O-", "O-")
    assert row["reactant_O-"] == 2
    assert row["product_OH-"] == 2
    assert row["product_H2O2"] == 1
    assert all(pd.api.types.is_integer_dtype(df[c]) for c in df.columns[5:])


def test_filter_by_product_and_reactant(tmp_path):
    f = _write_meta(
        tmp_path,
        [
            "25,°OH^0 + °OH^0 -> H2O2^0",
            "19,O^-1 + O^-1 -> H2O2^0 + OH^-1 + OH^-1",
        ],
    )
    df = load_reaction_table(f)
    assert list(df[df["product_H2O2"] > 0].reactionId) == [25, 19]
    assert list(df[df["reactant_OH"] > 0].reactionId) == [25]


def test_unknown_species_kept_raw(tmp_path):
    df = load_reaction_table(_write_meta(tmp_path, ["1,X^0 + °OH^0 -> Y^0"]))
    assert "reactant_X^0" in df.columns
    assert "product_Y^0" in df.columns


def test_scan_root_raises(tmp_path, make_dump):
    make_dump(tmp_path, "a")
    make_dump(tmp_path, "b")
    with pytest.raises(ValueError, match="2"):
        load_reaction_table(tmp_path)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reaction_table(tmp_path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reaction_table(tmp_path / "nope.csv")


def test_malformed_line_raises(tmp_path):
    f = _write_meta(tmp_path, ["1,H^0 + H^0 H2^0"])
    with pytest.raises(ValueError, match=re.escape("H^0 + H^0 H2^0")):
        load_reaction_table(f)


def test_short_names_o_minus():
    assert short_name("O^-1") == "O-"
    assert short_name("O_3^-1") == "O3-"

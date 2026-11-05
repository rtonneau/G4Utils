from __future__ import annotations

import builtins

import pandas as pd
import pytest

from g4utils.DnaChem.tables import pick_format, read_table
from tests.conftest import write_standardized_table

ALL = ["txt", "csv", "h5", "root"]

SPECIES = {
    "time_ns": [0.001, 0.001, 5.0],
    "species": ["H2O2^0", "°OH^0", "e_aq^-1"],
    "nEvents": [2, 2, 2],
    "countSum": [15, 971, 819],
    "countSumSq": [117, 471505, 335405],
    "countMean": [7.5, 485.5, 409.5],
    "countStdErr": [1.5, 6.5, 3.5],
    "gSum": [0.15, 9.71, 8.19],
    "gSumSq": [0.0117, 47.15, 33.54],
    "gMean": [0.075, 4.855, 4.095],
    "gStdErr": [0.015, 0.065, 0.035],
}
CATALOG = {
    "reactionId": [0, 1, 2],
    "reaction": ["H2O2^0 + OH^-1 -> HO_2^-1", "H^0 + H^0 -> H_2^0", "A + B -> (no products)"],
}


def test_cross_format_equality_and_dtypes(tmp_path):
    write_standardized_table(tmp_path, "Species", "species", SPECIES, ALL)
    frames = {f: read_table(tmp_path, "Species", "species", [f]) for f in ALL}
    ref = frames["csv"]
    assert str(ref["species"].dtype) == "string"
    assert str(ref["nEvents"].dtype) == "int64"
    assert str(ref["countSum"].dtype) == "float64"
    assert list(ref.columns) == list(SPECIES)
    for fmt in ("h5", "root", "txt"):
        pd.testing.assert_frame_equal(frames[fmt], ref, check_exact=False)
    assert list(ref["species"]) == SPECIES["species"]


def test_preference_order_and_manifest_restriction(tmp_path):
    write_standardized_table(tmp_path, "Species", "species", SPECIES, ALL)
    assert pick_format(tmp_path, "Species", ALL)[0] == "csv"
    assert pick_format(tmp_path, "Species", ["txt", "h5", "root"])[0] == "h5"
    assert pick_format(tmp_path, "Species", ["txt", "root"])[0] == "root"
    assert pick_format(tmp_path, "Species", ["txt"])[0] == "txt"


def test_files_inventory_and_missing(tmp_path):
    write_standardized_table(tmp_path, "Species", "species", SPECIES, ["csv", "h5"])
    assert pick_format(tmp_path, "Species", ALL, files=["Species.h5"])[0] == "h5"
    with pytest.raises(FileNotFoundError, match="Species"):
        pick_format(tmp_path, "Species", ["root"])
    with pytest.raises(FileNotFoundError):
        pick_format(tmp_path, "Species", ["csv"], files=["Other.csv"])


def test_txt_only_used_when_nothing_else(tmp_path):
    write_standardized_table(tmp_path, "Species", "species", SPECIES, ["txt", "csv"])
    (tmp_path / "Species.csv").write_text("garbage", encoding="utf-8")
    # csv present and preferred: txt is never consulted.
    with pytest.raises(ValueError):
        read_table(tmp_path, "Species", "species", ["csv", "txt"])
    (tmp_path / "Species.csv").unlink()
    assert len(read_table(tmp_path, "Species", "species", ["csv", "txt"])) == 3


def test_txt_catalogue_labels_with_spaces(tmp_path):
    write_standardized_table(tmp_path, "ReactionsCatalog", "reactionCatalog", CATALOG, ["txt", "csv"])
    csv_frame = read_table(tmp_path, "ReactionsCatalog", "reactionCatalog", ["csv"])
    txt_frame = read_table(tmp_path, "ReactionsCatalog", "reactionCatalog", ["txt"])
    pd.testing.assert_frame_equal(txt_frame, csv_frame)
    assert txt_frame["reaction"][0] == "H2O2^0 + OH^-1 -> HO_2^-1"


def test_catalogue_container_for_h5_root(tmp_path):
    write_standardized_table(
        tmp_path, "ReactionsCatalog", "reactionCatalog", CATALOG, ["h5", "root", "csv"],
        container="Reactions",
    )
    for fmt in ("h5", "root"):
        frame = read_table(
            tmp_path, "ReactionsCatalog", "reactionCatalog", [fmt], container="Reactions"
        )
        assert list(frame["reactionId"]) == [0, 1, 2]
        assert str(frame["reactionId"].dtype) == "int64"
        assert frame["reaction"][2] == "A + B -> (no products)"


def test_csv_keeps_na_like_strings(tmp_path):
    cols = {"process": ["NA", "None"], "nEvents": [1, 1], "countSum": [1, 2],
            "countSumSq": [1, 4], "countMean": [1.0, 2.0], "countStdErr": [0.0, 0.0]}
    write_standardized_table(tmp_path, "PhysicsInteractions", "physicsInteractions", cols, ["csv"])
    frame = read_table(tmp_path, "PhysicsInteractions", "physicsInteractions", ["csv"])
    assert list(frame["process"]) == ["NA", "None"]


def test_missing_uproot_message(tmp_path, monkeypatch):
    write_standardized_table(tmp_path, "Species", "species", SPECIES, ["h5", "root"]) \
        if False else write_standardized_table(tmp_path, "Species", "species", SPECIES, ["csv"])
    (tmp_path / "Species.root").write_bytes(b"x")
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "uproot":
            raise ImportError("no uproot")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"pip install g4utils\[root\]"):
        read_table(tmp_path, "Species", "species", ["root"])

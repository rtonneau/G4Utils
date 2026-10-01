from __future__ import annotations

import json
import math
import re

import pytest

from g4utils.DnaChem.dumps import find_dumps, parse_dump_name
from g4utils.DnaChem.manifest import dump_columns, load_manifests



def _manifest(d):
    return json.loads((d / "Manifest.json").read_text(encoding="utf-8"))


def test_find_dumps_parent_skips_non_dumps(tmp_path, make_dump):
    make_dump(tmp_path, "run_1pO2")
    make_dump(tmp_path, "run_0pO2")
    (tmp_path / "macro").mkdir()
    (tmp_path / "run_0pO2.log").write_text("x")
    found = find_dumps(tmp_path)
    assert [p.name for p in found] == ["run_0pO2", "run_1pO2"]


def test_find_dumps_single_dump(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_0pO2")
    assert find_dumps(d) == [d]


def test_find_dumps_none_raises(tmp_path):
    (tmp_path / "macro").mkdir()
    with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path))):
        find_dumps(tmp_path)


def test_parse_dump_name_values():
    pat = r"run_(?P<O2>[0-9p]+)pO2"
    assert parse_dump_name("run_0p3pO2", pat) == {"O2": 0.3}
    assert parse_dump_name("run_21pO2", pat) == {"O2": 21.0}
    assert parse_dump_name("run_abc", r"run_(?P<tag>[a-z]+)") == {"tag": "abc"}
    assert parse_dump_name("anything", None) == {}


def test_parse_dump_name_mismatch_raises():
    with pytest.raises(ValueError, match="foo") as exc:
        parse_dump_name("foo", r"run_(?P<x>\d+)")
    assert "run_" in str(exc.value)


def test_dump_columns_o2_molarity(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_0", molarity_M=0.00026)
    cols = dump_columns(_manifest(d), "run_0")
    assert cols["dump"] == "run_0"
    assert cols["chemistry"] == "BoscoloChem"
    assert cols["pH"] == 7
    assert cols["totalEvents"] == 100
    assert cols["totalEnergyDeposit_eV"] == 1e7
    assert cols["O2_molarity_M"] == 0.00026
    assert cols["particle"] == "e-"
    assert cols["beamEnergy_keV"] == 100


def test_dump_columns_mixed_beams_warns(tmp_path, make_dump):
    runs = [
        {"run": 0, "particle": "e-", "beamEnergy_keV": 100},
        {"run": 1, "particle": "e-", "beamEnergy_keV": 200},
    ]
    d = make_dump(tmp_path, "run_mix", runs=runs)
    with pytest.warns(UserWarning, match="run_mix"):
        cols = dump_columns(_manifest(d), "run_mix")
    assert math.isnan(cols["beamEnergy_keV"])
    assert cols["particle"] == "e-"


def test_load_manifests_rows_and_pattern(tmp_path, make_dump):
    runs = [
        {"run": 0, "events": 50, "seed": 1},
        {"run": 1, "events": 50, "seed": 2},
    ]
    make_dump(tmp_path, "run_0p3pO2", runs=runs)
    make_dump(tmp_path, "run_21pO2", runs=[])
    df = load_manifests(tmp_path, name_pattern=r"run_(?P<O2>[0-9p]+)pO2")
    assert len(df) == 3
    a = df[df["dump"] == "run_0p3pO2"]
    assert list(a["run"]) == [0, 1]
    assert list(a["run_seed"]) == [1, 2]
    assert set(a["O2"]) == {0.3}
    b = df[df["dump"] == "run_21pO2"]
    assert len(b) == 1
    assert b["O2"].iloc[0] == 21.0
    assert b["run"].isna().all()


def test_parse_dump_name_non_numeric_words_stay_strings():
    out = parse_dump_name("run_nan_inf", r"run_(?P<a>[a-z]+)_(?P<b>[a-z]+)")
    assert out == {"a": "nan", "b": "inf"}

from __future__ import annotations

import pytest

from g4utils.DnaChem.loaders import load_reactions, load_species

PATTERN = r"run_(?P<o2_percent>[0-9p]+)pO2"


def test_load_species_g_value(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_a")
    df = load_species(d)
    row = df[(df.species == "OH") & (df.time_s < 1e-9)].iloc[0]
    assert row.G == pytest.approx(4.8)
    assert row.time_s == pytest.approx(1e-12)
    assert "time" not in df.columns


def test_load_species_time_s_no_rounding(tmp_path, make_dump):
    df = load_species(make_dump(tmp_path, "run_a"))
    assert df.time_s.max() == pytest.approx(999.999e-9, rel=1e-12)


def test_load_species_scan_with_pattern(tmp_path, make_dump):
    for name in ("run_0pO2", "run_0p3pO2", "run_21pO2"):
        make_dump(tmp_path, name)
    df = load_species(tmp_path, name_pattern=PATTERN)
    assert sorted(df.o2_percent.unique()) == [0.0, 0.3, 21.0]
    assert "O2_molarity_M" in df.columns
    assert df.index.is_unique


def test_load_species_single_dump_path(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_0pO2")
    df = load_species(d)
    assert set(df.dump) == {"run_0pO2"}
    assert len(df) == 8


def test_load_reactions_labels_and_counts(tmp_path, make_dump):
    df = load_reactions(make_dump(tmp_path, "run_a"))
    assert "G" not in df.columns
    assert "time" not in df.columns
    assert list(df["count"]) == [646, 4436, 9000]
    assert df.reaction.iloc[1] == "°OH^0 + °OH^0 -> H2O2^0"
    assert df.time_s.iloc[0] == pytest.approx(1e-12)
    assert set(df.dump) == {"run_a"}


def test_load_species_missing_file_raises(tmp_path, make_dump):
    d = make_dump(tmp_path, "run_a")
    (d / "Species_nt_species.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Species_nt_species.csv"):
        load_species(d)

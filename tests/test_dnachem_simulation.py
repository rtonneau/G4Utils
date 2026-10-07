from __future__ import annotations

import pytest

from g4utils.DnaChem.dump import Dump
from g4utils.DnaChem.simulation import Simulation

PATTERN = r"run_(?P<o2_percent>[0-9p]+)pO2"


def _results(tmp_path, make_dump, names=("run_0pO2", "run_2pO2")):
    res = tmp_path / "results"
    for n in names:
        make_dump(res, n)
    return res


def test_reads_no_data_file(tmp_path, make_dump):
    res = _results(tmp_path, make_dump)
    for f in res.glob("*/*.csv"):
        f.unlink()
    assert len(Simulation(res)) == 2


def test_simulation_folder_uses_results(tmp_path, make_dump):
    _results(tmp_path, make_dump)
    sim = Simulation(tmp_path)
    assert sim.path == tmp_path / "results"
    assert list(sim.subruns) == ["run_0pO2", "run_2pO2"]


def test_results_folder_and_flat_dump(tmp_path, make_dump):
    res = _results(tmp_path, make_dump)
    assert len(Simulation(res)) == 2
    flat = Simulation(res / "run_0pO2")
    assert list(flat.subruns) == ["run_0pO2"]


def test_no_dump_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Simulation(tmp_path)


def test_subrun_lookup(tmp_path, make_dump):
    sim = Simulation(_results(tmp_path, make_dump))
    assert isinstance(sim.subrun("run_0pO2"), Dump)
    with pytest.raises(KeyError, match="run_2pO2"):
        sim.subrun("nope")


def test_iter_len(tmp_path, make_dump):
    sim = Simulation(_results(tmp_path, make_dump))
    assert len(sim) == 2
    assert [d.name for d in sim] == ["run_0pO2", "run_2pO2"]


def test_table(tmp_path, make_dump):
    sim = Simulation(_results(tmp_path, make_dump), PATTERN)
    df = sim.table()
    assert len(df) == 2
    assert list(df["dump"]) == ["run_0pO2", "run_2pO2"]
    assert list(df["o2_percent"]) == [0.0, 2.0]
    assert {"chemistry", "pH", "totalEvents"} <= set(df.columns)

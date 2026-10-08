from __future__ import annotations

import pandas as pd
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


def test_table_richer_columns(tmp_path, make_dump):
    res = tmp_path / "results"
    make_dump(
        res,
        "run_0pO2",
        manifest_extra={
            "chemistryModel": "IRT",
            "handOverTime_ns": 1.0,
            "chemistryEndTime_ns": 1000.0,
            "voxelSize_nm": 5.0,
            "mesoPixels": 64,
            "mesoTimesPerDecade": 10,
            "mesoSpatialOutput": True,
            "threads": 4,
            "runs": [
                {"run": 0, "particle": "e-", "beamEnergy_keV": 100, "wallTime_s": 1.5},
                {"run": 1, "particle": "e-", "beamEnergy_keV": 100, "wallTime_s": 2.0},
            ],
        },
    )
    make_dump(res, "run_2pO2")
    df = Simulation(res, PATTERN).table()
    a, b = df.iloc[0], df.iloc[1]
    assert a["chemistryModel"] == "IRT"
    assert a["mesoPixels"] == 64
    assert a["threads"] == 4
    assert a["wallTime_s"] == 3.5
    assert a["O2_molarity_M"] == a["O2_molarity_M"]
    assert pd.isna(b["chemistryModel"])
    assert pd.isna(b["wallTime_s"])

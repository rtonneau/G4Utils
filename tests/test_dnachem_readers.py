from __future__ import annotations

import pandas as pd
import pytest

from g4utils.DnaChem import Dump
from g4utils.DnaChem.readers import TRACK_COLUMNS


TRACK_HEADER = ",".join(TRACK_COLUMNS)
TRACK_ROWS = [
    "0,0,3940.635938,10.000000,0.000025,stopped,4192.985256,226,6309.350231,511",
    "0,1,2534.414142,10.000000,0.000025,stopped,3655.026025,203,6768.150826,503",
]
TRACK_SUMMARY = {
    "events": 2, "stopped": 2, "escaped": 0, "killed": 0,
    "primaryLength_nm": {"mean": 3237.52504026, "sem": 703.110898045},
    "primaryEkin0_keV": {"mean": 10, "sem": 0},
    "primaryEkinEnd_keV": {"mean": 0.000025, "sem": 0},
    "secondaryFirstGen_nm": {"mean": 3924.00564062, "sem": 268.979615334},
    "nSecondaryFirstGen": {"mean": 214.5, "sem": 11.5},
    "secondaryAll_nm": {"mean": 6538.7505284, "sem": 229.400297302},
    "nSecondaryAll": {"mean": 507, "sem": 4},
}


def _csv(path, name, header, rows):
    file = path / name
    file.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8")
    return file


def test_physical_raw_equality_and_normalization(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", manifest_extra={"totalEvents": 2})
    _csv(path, "PhysicsInteractions.csv", "label,count", ["e-_G4DNAAttachment,17", "raw label,0"])
    dump = Dump(path)
    raw = dump.physical_interactions()
    assert raw["label"].tolist() == ["e-_G4DNAAttachment", "raw label"]
    assert raw["count"].tolist() == [17, 0]
    assert pd.api.types.is_integer_dtype(raw["count"])
    assert "count_per_event" not in raw
    assert raw["dump_key"].tolist() == ["Manifest.json"] * 2
    assert raw["manifest_path"].tolist() == [str(dump.manifest_path)] * 2
    assert dump.physical_interactions(per_event=True)["count_per_event"].tolist() == [8.5, 0.0]


def test_physical_prefix_inventory_and_missing_csv(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", prefix="pfx_", manifest_extra={"files": ["nested/pfx_PhysicsInteractions.csv"]})
    (path / "nested").mkdir()
    _csv(path / "nested", "pfx_PhysicsInteractions.csv", "label,count", ["process,3"])
    _csv(path, "PhysicsInteractions.csv", "label,count", ["wrong,99"])
    dump = Dump(path)
    assert dump.physical_interactions()["label"].tolist() == ["process"]
    (path / "nested" / "pfx_PhysicsInteractions.csv").unlink()
    (path / "PhysicsInteractions.Txt").write_text("process count = 3", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nested.*pfx_PhysicsInteractions.csv"):
        dump.physical_interactions()


@pytest.mark.parametrize("row", ["x,-1", "x,1.5", "x,nan", "x,inf", "x,9223372036854775808", ",1"])
def test_physical_bad_rows(tmp_path, make_dump, row):
    path = make_dump(tmp_path, "run_a")
    _csv(path, "PhysicsInteractions.csv", "label,count", [row])
    with pytest.raises(ValueError, match="PhysicsInteractions.csv"):
        Dump(path).physical_interactions()


def test_physical_header_and_empty(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    _csv(path, "PhysicsInteractions.csv", "label,counts", [])
    with pytest.raises(ValueError, match="header"):
        Dump(path).physical_interactions()
    _csv(path, "PhysicsInteractions.csv", "label,count", [])
    empty = Dump(path).physical_interactions(per_event=True)
    assert empty.empty
    assert list(empty.columns[:2]) == ["label", "count"]
    assert pd.api.types.is_string_dtype(empty["label"])
    assert pd.api.types.is_integer_dtype(empty["count"])
    assert pd.api.types.is_float_dtype(empty["count_per_event"])


@pytest.mark.parametrize("total", [0, -1, None, float("inf"), float("nan"), "2"])
def test_physical_invalid_denominator(tmp_path, make_dump, total):
    path = make_dump(tmp_path, "run_a", manifest_extra={"totalEvents": total})
    _csv(path, "PhysicsInteractions.csv", "label,count", ["x,0"])
    with pytest.raises(ValueError, match="totalEvents"):
        Dump(path).physical_interactions(per_event=True)


def test_track_raw_equality_summary_and_prefix(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", prefix="EndOfRun_", manifest_extra={"totalEvents": 2, "trackLengths": TRACK_SUMMARY})
    _csv(path, "EndOfRun_TrackLengths.csv", TRACK_HEADER, TRACK_ROWS)
    _csv(path, "TrackLengths.csv", TRACK_HEADER, [TRACK_ROWS[1]])
    frame = Dump(path).track_lengths()
    assert frame["run"].tolist() == [0, 0]
    assert frame["event"].tolist() == [0, 1]
    assert frame["primaryLength_nm"].tolist() == [3940.635938, 2534.414142]
    assert frame["primaryEnd"].tolist() == ["stopped", "stopped"]
    assert frame["nSecondaryAll"].tolist() == [511, 503]
    assert frame["prefix"].tolist() == ["EndOfRun_"] * 2


def test_track_header_only_and_missing(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", prefix="pfx_", manifest_extra={"totalEvents": 0, "trackLengths": {"events": 0, "primaryLength_nm": {"mean": None, "sem": None}}})
    _csv(path, "pfx_TrackLengths.csv", TRACK_HEADER, [])
    frame = Dump(path).track_lengths()
    assert frame.empty
    assert list(frame.columns[:10]) == list(TRACK_COLUMNS)
    assert pd.api.types.is_integer_dtype(frame["run"])
    assert pd.api.types.is_float_dtype(frame["primaryLength_nm"])
    assert pd.api.types.is_string_dtype(frame["primaryEnd"])
    (path / "pfx_TrackLengths.csv").unlink()
    (path / "pfx_TrackLengths.Txt").write_text("# track lengths", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="pfx_TrackLengths.csv"):
        Dump(path).track_lengths()


@pytest.mark.parametrize("row", [
    "0,0,-1,10,0,stopped,2,1,3,1",
    "0,0,nan,10,0,stopped,2,1,3,1",
    "0,0,1,inf,0,stopped,2,1,3,1",
    "0,0,1,10,0,other,2,1,3,1",
    "0,0,1,10,0,stopped,2,1.2,3,1",
    "-1,0,1,10,0,stopped,2,1,3,1",
    "0,0,1,10,0,stopped,2,1,3,9223372036854775808",
])
def test_track_bad_fields(tmp_path, make_dump, row):
    path = make_dump(tmp_path, "run_a", manifest_extra={"totalEvents": 1})
    _csv(path, "TrackLengths.csv", TRACK_HEADER, [row])
    with pytest.raises(ValueError, match="TrackLengths.csv"):
        Dump(path).track_lengths()


def test_track_header_duplicate_and_omitted_warning(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a", manifest_extra={"totalEvents": 2})
    _csv(path, "TrackLengths.csv", "run,event,bad", [])
    with pytest.raises(ValueError, match="header"):
        Dump(path).track_lengths()
    _csv(path, "TrackLengths.csv", TRACK_HEADER, [TRACK_ROWS[0], TRACK_ROWS[0]])
    with pytest.raises(ValueError, match="duplicate"):
        Dump(path).track_lengths()
    _csv(path, "TrackLengths.csv", TRACK_HEADER, [TRACK_ROWS[0]])
    with pytest.warns(UserWarning, match="aborted or otherwise omitted"):
        assert len(Dump(path).track_lengths()) == 1


@pytest.mark.parametrize("change", [
    {"events": 1}, {"stopped": 1},
    {"primaryLength_nm": {"mean": 3238, "sem": 703.110898045}},
    {"primaryLength_nm": {"mean": 3237.52504026, "sem": 700}},
])
def test_track_summary_inconsistency(tmp_path, make_dump, change):
    summary = TRACK_SUMMARY | change
    path = make_dump(tmp_path, "run_a", manifest_extra={"totalEvents": 2, "trackLengths": summary})
    _csv(path, "TrackLengths.csv", TRACK_HEADER, TRACK_ROWS)
    with pytest.raises(ValueError, match="trackLengths"):
        Dump(path).track_lengths()

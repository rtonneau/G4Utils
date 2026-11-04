from __future__ import annotations

import pandas as pd
import pytest

from g4utils.DnaChem import Dump


WIDTHS = (11, 10, 14, 13, 22, 22, 22, 14, 14, 14)
COLUMNS = (
    "run", "event", "parent_id", "molecule", "electron_modification", "energy_eV",
    "parent_x_nm", "parent_y_nm", "parent_z_nm", "x_nm", "y_nm", "z_nm",
)
VALUES = ("123", "H2O", "0:4", "12.61", "-1.23456789e+03", "2.5", "-3", "", "", "")


def _line(values=VALUES):
    return "".join(f"{value:>{width}}" for value, width in zip(values, WIDTHS)).rstrip()


def _event(path, name="PreChemical_run0_event0.txt", rows=None):
    file = path / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(rows if rows is not None else [_line()]), encoding="utf-8")
    return file


def _assert_typed_empty(frame):
    assert frame.empty
    assert tuple(frame.columns[:12]) == COLUMNS
    for name in ("run", "event", "parent_id"):
        assert str(frame[name].dtype) == "int64"
    for name in ("molecule", "electron_modification"):
        assert isinstance(frame[name].dtype, pd.StringDtype)
    for name in COLUMNS[5:9]:
        assert str(frame[name].dtype) == "float64"
    for name in COLUMNS[9:]:
        assert str(frame[name].dtype) == "Float64"
    assert {"dump", "dump_key", "manifest_path", "prefix"} <= set(frame)


def test_raw_substrings_and_optional_positions(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    electron = ("124", "e_aq", "-1", "0.025", "-5", "6", "7", "8", "-9", "10")
    lines = [_line(), _line(electron), _line((*electron[:7], "", "", ""))]
    _event(path, rows=["# header", "  # comment", "", *lines])
    dump = Dump(path)
    frame = dump.prechemical_records()
    for i, line in enumerate(lines):
        start = 0
        for column, width in zip(COLUMNS[2:], WIDTHS):
            raw = line[start:start + width].strip()
            actual = frame.iloc[i][column]
            if column in ("molecule", "electron_modification"):
                assert actual == raw
            elif raw:
                assert actual == float(raw)
            else:
                assert pd.isna(actual)
            start += width
    assert frame["electron_modification"].tolist() == ["0:4", "-1", "-1"]
    assert frame.loc[[0, 2], ["x_nm", "y_nm", "z_nm"]].isna().all().all()
    assert frame["manifest_path"].tolist() == [str(dump.manifest_path)] * 3


@pytest.mark.parametrize("inventory", [False, True])
def test_prefix_numeric_sort_and_reused_events(tmp_path, make_dump, inventory):
    prefix = "p[1].+_"
    names = [f"{prefix}PreChemical_run{run}_event{event}.txt"
             for run, event in ((10, 0), (2, 10), (2, 2), (0, 0), (2, 0))]
    extra = {"files": [f"nested/{name}" for name in names]} if inventory else {}
    path = make_dump(tmp_path, "run_a", prefix=prefix, manifest_extra=extra)
    for name in names:
        _event(path / "nested" if inventory else path, name)
    _event(path, "PreChemical_run99_event99.txt")
    _event(path, "p1x_PreChemical_run99_event99.txt")
    _event(path / "other_dump", f"{prefix}PreChemical_run99_event99.txt")
    frame = Dump(path).prechemical_records()
    assert list(zip(frame.run, frame.event)) == [(0, 0), (2, 0), (2, 2), (2, 10), (10, 0)]
    assert frame.prefix.tolist() == [prefix] * 5


@pytest.mark.parametrize("files", [[], ["Species_nt_species.csv"], ["other_PreChemical_run0_event0.txt"]])
def test_inventory_does_not_fall_back(tmp_path, make_dump, files):
    path = make_dump(tmp_path, "run_a", manifest_extra={"files": files})
    _event(path)
    _event(path, "other_PreChemical_run0_event0.txt")
    with pytest.raises(FileNotFoundError, match="[Rr]egenerate"):
        Dump(path).prechemical_records()
    _assert_typed_empty(Dump(path).prechemical_records(allow_missing=True))


@pytest.mark.parametrize("allow_missing", [False, True])
def test_listed_missing_always_raises(tmp_path, make_dump, allow_missing):
    name = "pfx_PreChemical_run0_event0.txt"
    path = make_dump(tmp_path, "run_a", prefix="pfx_", manifest_extra={"files": [f"nested/{name}"]})
    _event(path, name)
    with pytest.raises(FileNotFoundError, match="nested.*pfx_PreChemical"):
        Dump(path).prechemical_records(allow_missing=allow_missing)


@pytest.mark.parametrize("rows", [[], ["# header", "   ", "# end"]])
def test_empty_events_are_valid(tmp_path, make_dump, rows):
    path = make_dump(tmp_path, "run_a")
    _event(path, rows=rows)
    _assert_typed_empty(Dump(path).prechemical_records())


def test_absent_output(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    with pytest.raises(FileNotFoundError, match="[Rr]egenerate"):
        Dump(path).prechemical_records()
    _assert_typed_empty(Dump(path).prechemical_records(allow_missing=True))


@pytest.mark.parametrize("index,value", [
    *((i, "") for i in range(7)), (0, "1.5"), (0, "nan"),
    (3, "bad"), (3, "NaN"), (4, "inf"), (5, "-inf"), (6, "1e999"),
    (7, "NaN"), (8, "inf"), (9, "bad"),
])
def test_invalid_fields(tmp_path, make_dump, index, value):
    path = make_dump(tmp_path, "run_a")
    values = list(VALUES)
    values[index] = value
    _event(path, rows=["# header", _line(values)])
    with pytest.raises(ValueError, match=r"PreChemical_run0_event0.txt:2"):
        Dump(path).prechemical_records()


@pytest.mark.parametrize("name", ["PreChemical_run0_event1_extra.txt", "PreChemical_runx_event1.txt", "PreChemical_run0_event.txt"])
def test_ambiguous_names(tmp_path, make_dump, name):
    path = make_dump(tmp_path, "run_a")
    _event(path, name)
    with pytest.raises(ValueError, match="filename"):
        Dump(path).prechemical_records(allow_missing=True)


@pytest.mark.parametrize("inventory", [False, True])
def test_duplicate_event_files(tmp_path, make_dump, inventory):
    names = ["PreChemical_run0_event0.txt", "PreChemical_run00_event00.txt"]
    extra = {"files": names} if inventory else {}
    path = make_dump(tmp_path, "run_a", manifest_extra=extra)
    for name in names:
        _event(path, name, rows=[])
    with pytest.raises(ValueError, match="duplicate"):
        Dump(path).prechemical_records()


def test_duplicate_nested_event_files(tmp_path, make_dump):
    names = ["a/PreChemical_run0_event0.txt", "b/PreChemical_run0_event0.txt"]
    path = make_dump(tmp_path, "run_a", manifest_extra={"files": names})
    for name in names:
        _event(path, name)
    with pytest.raises(ValueError, match="duplicate"):
        Dump(path).prechemical_records()


def test_unexpected_trailing_data(tmp_path, make_dump):
    path = make_dump(tmp_path, "run_a")
    _event(path, rows=[_line().ljust(sum(WIDTHS)) + "extra"])
    with pytest.raises(ValueError, match="trailing"):
        Dump(path).prechemical_records()

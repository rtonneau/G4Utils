from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from g4utils.DnaChem import (
    Dump,
    DumpLocation,
    Simulation,
    discover_dumps,
    find_dumps,
    load_manifests,
    load_reaction_table,
    load_species,
    read_manifest,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _dump(make_dump, root, folder="", prefix="", number=11, **extra):
    directory = make_dump(root, folder, prefix=prefix, manifest_extra=extra)
    manifest = directory / "Manifest.json"
    if prefix:
        manifest = manifest.rename(directory / f"{prefix}Manifest.json")
    species = directory / f"{prefix}Species_nt_species.csv"
    species.write_text(
        species.read_text(encoding="utf-8").replace("400000", str(number)),
        encoding="utf-8",
    )
    return {"folder": folder, "prefix": prefix,
            "manifest": manifest.relative_to(root).as_posix(), "events": 999999}


def _index(root, entries, base=None):
    data = {"schemaVersion": 1, "kind": "resultsIndex"} if base is None else base
    _write_json(root / "Manifest.json", data | {"dumps": entries})


@pytest.mark.parametrize("collided", [False, True])
def test_index_order_identity_and_selected_species(tmp_path, make_dump, collided):
    entries = [
        _dump(make_dump, tmp_path, prefix="EndOfRun_", number=22),
        _dump(make_dump, tmp_path, "nested/run_a", number=33),
        _dump(make_dump, tmp_path, prefix="pfx_", number=44),
    ]
    base = None
    if collided:
        entries.insert(1, _dump(make_dump, tmp_path, number=11))
        base = json.loads((tmp_path / "Manifest.json").read_text())
    _index(tmp_path, entries, base)
    locations = discover_dumps(tmp_path)
    assert all(isinstance(loc, DumpLocation) for loc in locations)
    assert [loc.key for loc in locations] == [e["manifest"] for e in entries]
    assert [loc.prefix for loc in locations] == [e["prefix"] for e in entries]
    with pytest.raises(dataclasses.FrozenInstanceError):
        locations[0].key = "changed"
    sim = Simulation(tmp_path)
    assert list(sim.subruns) == [loc.key for loc in locations]
    assert len(sim) == len(entries) == len(list(sim.subruns.values()))
    expected = [22, 11, 33, 44] if collided else [22, 33, 44]
    assert [dump.species().number.iloc[0] for dump in sim] == expected
    frame = load_species(tmp_path)
    assert list(frame.dump_key.drop_duplicates()) == list(sim.subruns)
    assert frame.manifest_path.nunique() == len(entries)
    assert set(frame.totalEvents) == {100}  # Index events are only hints.
    assert list(load_manifests(tmp_path).dump_key) == list(sim.subruns)
    assert list(sim.table().dump_key) == list(sim.subruns)
    with pytest.raises(KeyError, match="[Aa]mbiguous"):
        sim.subrun(tmp_path.name)
    assert sim.subrun("run_a").key == "nested/run_a/Manifest.json"
    assert sim.subruns["run_a"] is sim.subrun("run_a")
    with pytest.raises(ValueError, match="load_manifests"):
        read_manifest(tmp_path)
    if collided:
        assert Dump(tmp_path).species().number.iloc[0] == 11
        assert read_manifest(tmp_path / "Manifest.json").totalEvents == 100


def test_no_index_fallback_keys_stable_when_siblings_added(tmp_path, make_dump):
    _dump(make_dump, tmp_path, prefix="pfx_", number=21)
    first = discover_dumps(tmp_path)[0]
    direct = Dump.from_manifest(first.manifest_path)
    assert direct.key == first.key
    assert direct.species().number.iloc[0] == 21
    _dump(make_dump, tmp_path, "run02", number=32)
    assert [loc.key for loc in discover_dumps(tmp_path)] == [
        first.key, "run02/Manifest.json"
    ]
    assert [dump.species().number.iloc[0] for dump in Simulation(tmp_path)] == [21, 32]
    _dump(make_dump, tmp_path, "unindexed/deeper", number=50)
    assert len(discover_dumps(tmp_path)) == 2  # No recursive scan.
    assert find_dumps(tmp_path) == [tmp_path / "run02"]


def test_index_key_stable_after_sibling_added(tmp_path, make_dump):
    first = _dump(make_dump, tmp_path, "nested/first", number=12)
    _index(tmp_path, [first])
    original = discover_dumps(tmp_path)[0]
    sibling = _dump(make_dump, tmp_path, "second", number=13)
    _index(tmp_path, [sibling, first])
    assert discover_dumps(tmp_path)[1] == original
    assert list(load_species(tmp_path).dump_key.drop_duplicates()) == [
        sibling["manifest"], first["manifest"]
    ]


def test_duplicate_folder_aliases_keep_distinct_keys(tmp_path, make_dump):
    entries = [_dump(make_dump, tmp_path, "left/same", number=1),
               _dump(make_dump, tmp_path, "right/same", number=2)]
    _index(tmp_path, entries)
    sim = Simulation(tmp_path)
    with pytest.raises(KeyError, match="Ambiguous"):
        sim.subruns["same"]
    assert sim.subruns["left/same/Manifest.json"].species().number.iloc[0] == 1
    assert sim.subruns["right/same/Manifest.json"].species().number.iloc[0] == 2


def test_direct_child_index_is_followed_without_duplicate_fallback(tmp_path, make_dump):
    child = tmp_path / "child"
    entries = [_dump(make_dump, child, prefix="pre_", number=8),
               _dump(make_dump, child, "nested", number=9)]
    _index(child, entries)
    sim = Simulation(tmp_path)
    assert list(sim.subruns) == ["child/pre_Manifest.json", "child/nested/Manifest.json"]
    assert [dump.species().number.iloc[0] for dump in sim] == [8, 9]


def test_legacy_plain_dump_precedence_and_lookup(tmp_path, make_dump):
    _dump(make_dump, tmp_path, number=7)
    _dump(make_dump, tmp_path, "child", number=8)
    assert [loc.key for loc in discover_dumps(tmp_path)] == ["Manifest.json"]
    assert find_dumps(tmp_path) == [tmp_path]
    sim = Simulation(tmp_path)
    assert sim.subrun(tmp_path.name) is sim.subruns["Manifest.json"]
    assert sim.subruns.get(tmp_path.name) is sim.subrun(tmp_path.name)
    assert len(sim) == 1


def test_selected_manifest_and_literal_file_inventory(tmp_path, make_dump):
    _dump(make_dump, tmp_path, number=4)
    entry = _dump(
        make_dump, tmp_path, prefix="pfx_", number=9,
        files=["tables\\pfx_Species_nt_species.csv", "pfx_ReactionsMetadata.csv"],
        outputDirAbsolute="Z:/old/machine/results",
    )
    (tmp_path / "tables").mkdir()
    species = tmp_path / "pfx_Species_nt_species.csv"
    species.rename(tmp_path / "tables" / species.name)
    dump = Dump.from_manifest(tmp_path / entry["manifest"])
    assert dump.manifest_path == (tmp_path / entry["manifest"]).resolve()
    assert dump.species().number.iloc[0] == 9
    assert load_species(dump.manifest_path).number.iloc[0] == 9
    assert load_reaction_table(dump.manifest_path).equals(dump.reaction_table())
    assert dump._data_file("pfx_ReactionsMetadata.csv").name == "pfx_ReactionsMetadata.csv"
    assert dump._data_file("tables/pfx_Species_nt_species.csv").parent.name == "tables"
    (tmp_path / "tables" / species.name).unlink()
    # A same-name fallback beside the manifest must not replace a listed file.
    species.write_text("wrong data", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="tables"):
        Dump.from_manifest(dump.manifest_path).species()


def test_from_manifest_name_pattern_and_missing_literal_file(tmp_path, make_dump):
    entry = _dump(make_dump, tmp_path, "run_0p3", prefix="EndOfRun_", number=9,
                  files=["EndOfRun_Species_nt_species.csv"])
    path = tmp_path / entry["manifest"]
    dump = Dump.from_manifest(path, r"run_(?P<value>[0-9p]+)")
    assert dump.labels == {"value": 0.3}
    assert dump.species().number.iloc[0] == 9
    file = path.parent / "EndOfRun_Species_nt_species.csv"
    file.rename(path.parent / "Species_nt_species.csv")
    with pytest.raises(FileNotFoundError, match="EndOfRun_Species_nt_species.csv"):
        Dump.from_manifest(path).species()


@pytest.mark.parametrize("listed", [False, True])
def test_literal_prefix_overlapping_canonical_filename(tmp_path, make_dump, listed):
    extra = {"files": ["SSpecies_nt_species.csv"]} if listed else {}
    path = make_dump(tmp_path, "dump", prefix="S", manifest_extra=extra)
    dump = Dump(path)
    assert dump._data_file("Species_nt_species.csv").name == "SSpecies_nt_species.csv"
    assert len(dump.species()) == 8
    assert dump._listed_file("SSpecies_nt_species.csv").name == "SSpecies_nt_species.csv"
    (path / "SSpecies_nt_species.csv").rename(path / "Species_nt_species.csv")
    with pytest.raises(FileNotFoundError, match="SSpecies_nt_species.csv"):
        Dump(path).species()


def test_reaction_loader_respects_selected_manifest(tmp_path, make_dump):
    _dump(make_dump, tmp_path, prefix="pfx_", number=9)
    _index(tmp_path, [{"folder": "", "prefix": "pfx_", "manifest": "pfx_Manifest.json"}])
    (tmp_path / "ReactionsMetadata.csv").write_text("unrelated", encoding="utf-8")
    assert len(load_reaction_table(tmp_path)) == 2


def test_reaction_loader_prefers_discovered_child_over_loose_metadata(tmp_path, make_dump):
    _dump(make_dump, tmp_path, "child")
    (tmp_path / "ReactionsMetadata.csv").write_text("unrelated", encoding="utf-8")
    assert len(load_reaction_table(tmp_path)) == 2


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_index_cross_platform_relative_paths(tmp_path, make_dump, separator):
    entry = _dump(make_dump, tmp_path, "nested/run", prefix="pre_", number=71)
    entry = {k: v.replace("/", separator) if isinstance(v, str) else v
             for k, v in entry.items()}
    _index(tmp_path, [entry])
    assert discover_dumps(tmp_path)[0].key == "nested/run/pre_Manifest.json"
    assert next(iter(Simulation(tmp_path))).species().number.iloc[0] == 71


@pytest.mark.parametrize("entries", [None, {}, "bad", [None], [{}],
    [{"folder": 1, "prefix": "", "manifest": "Manifest.json"}],
    [{"folder": "", "prefix": None, "manifest": "Manifest.json"}],
    [{"folder": "", "prefix": "", "manifest": ""}],
])
def test_malformed_index_entries(tmp_path, entries):
    _index(tmp_path, entries)
    with pytest.raises(ValueError, match="Manifest.json"):
        discover_dumps(tmp_path)


@pytest.mark.parametrize("value", ["{bad", "[]", "{}", '{"schemaVersion": "1"}'])
def test_json_and_schema_errors(tmp_path, value):
    (tmp_path / "Manifest.json").write_text(value, encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest.json"):
        discover_dumps(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("prefix", 4), ("prefix", "../"), ("files", "species.csv"),
    ("files", [3]), ("runs", {}), ("runs", [None]), ("scavengers", "O2"),
])
def test_manifest_type_errors(tmp_path, field, value):
    path = tmp_path / "Manifest.json"
    _write_json(path, {"schemaVersion": 1, field: value})
    for read in (read_manifest, lambda p: Dump.from_manifest(p).manifest):
        with pytest.raises(ValueError, match="Manifest.json"):
            read(path)


def test_missing_duplicate_and_prefix_mismatch(tmp_path, make_dump):
    entry = _dump(make_dump, tmp_path, prefix="pre_")
    _index(tmp_path, [entry, entry | {"manifest": "./pre_Manifest.json"}])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        discover_dumps(tmp_path)
    _index(tmp_path, [entry | {"manifest": "missing/Manifest.json", "folder": "missing", "prefix": ""}])
    with pytest.raises(FileNotFoundError, match="missing"):
        discover_dumps(tmp_path)
    _index(tmp_path, [entry])
    manifest = tmp_path / "pre_Manifest.json"
    data = json.loads(manifest.read_text())
    _write_json(manifest, data | {"prefix": "wrong_"})
    with pytest.raises(ValueError, match="prefix"):
        discover_dumps(tmp_path)


@pytest.mark.parametrize("change", [
    {"folder": "elsewhere"}, {"prefix": "wrong_"}, {"prefix": "../"},
])
def test_index_folder_and_prefix_must_match_path(tmp_path, make_dump, change):
    entry = _dump(make_dump, tmp_path, prefix="pre_")
    _index(tmp_path, [entry | change])
    with pytest.raises(ValueError, match="Manifest.json.*prefix"):
        discover_dumps(tmp_path)


def test_index_cannot_reference_itself_as_a_dump(tmp_path):
    _index(tmp_path, [{"folder": "", "prefix": "", "manifest": "Manifest.json"}])
    with pytest.raises(ValueError, match="resultsIndex"):
        discover_dumps(tmp_path)
    with pytest.raises(ValueError, match="resultsIndex"):
        Dump(tmp_path).manifest


@pytest.mark.parametrize("reference", ["../outside/Manifest.json", "..\\outside\\Manifest.json",
    "/outside/Manifest.json", "C:/outside/Manifest.json", "C:Manifest.json",
    "\\\\server\\share\\Manifest.json"])
def test_index_root_escape(tmp_path, reference):
    _index(tmp_path, [{"folder": "", "prefix": "", "manifest": reference}])
    with pytest.raises(ValueError, match="relative|escape"):
        discover_dumps(tmp_path)


def test_resolved_symlinks_cannot_escape_root(tmp_path, make_dump):
    outside = make_dump(tmp_path, "outside")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    _index(root, [{"folder": "linked", "prefix": "", "manifest": "linked/Manifest.json"}])
    with pytest.raises(ValueError, match="escape"):
        discover_dumps(root)
    (root / "Manifest.json").unlink()
    with pytest.raises(ValueError, match="escape"):
        discover_dumps(root)


@pytest.mark.parametrize("reference", ["../outside.csv", "..\\outside.csv", "C:/data.csv"])
def test_manifest_data_paths_cannot_escape(tmp_path, make_dump, reference):
    _dump(make_dump, tmp_path, files=[reference])
    with pytest.raises(ValueError, match="relative|escape"):
        Dump(tmp_path)._data_file(reference)


def test_data_symlink_cannot_escape_manifest_directory(tmp_path, make_dump):
    path = make_dump(tmp_path, "dump")
    species = path / "Species_nt_species.csv"
    outside = species.rename(tmp_path / species.name)
    try:
        species.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="escape"):
        Dump(path).species()


def test_ambiguous_listed_data_files_are_rejected(tmp_path, make_dump):
    path = make_dump(tmp_path, "dump", manifest_extra={
        "files": ["Species_nt_species.csv", "nested/Species_nt_species.csv"]
    })
    with pytest.raises(ValueError, match="ambiguous files"):
        Dump(path).species()

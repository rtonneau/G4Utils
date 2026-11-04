"""Opt-in, read-only comparisons with user-supplied dnachem-min output.

This is deliberately not a pytest test: real datasets are not distributed with
G4Utils. Only --output writes a file. No simulation is run or data regenerated.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import g4utils
import g4utils.DnaChem as dnachem
from g4utils.DnaChem import Dump, Simulation, discover_dumps
from g4utils.HDF5 import SpeciesMesoSpatialFile


PRECHEM_COLUMNS = [
    "parent_id", "molecule", "electron_modification", "energy_eV",
    "parent_x_nm", "parent_y_nm", "parent_z_nm", "x_nm", "y_nm", "z_nm",
]
INPUTS: dict[str, str] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_text(path: Path) -> str:
    contents = path.read_bytes()
    INPUTS[str(path.resolve())] = hashlib.sha256(contents).hexdigest()
    return contents.decode("utf-8-sig")


def raw_manifest(path: Path) -> dict:
    return json.loads(read_text(path))


def selected_file(dump: Dump, filename: str) -> Path:
    """Resolve the raw inventory independently of Dump's private helpers."""
    manifest = raw_manifest(dump.manifest_path)
    prefix = manifest.get("prefix") or ""
    matches = [Path(name.replace("\\", "/")) for name in manifest.get("files") or []
               if Path(name.replace("\\", "/")).name in (filename, prefix + filename)]
    require(len(matches) <= 1, f"Ambiguous raw inventory: {filename}")
    selected = matches[0] if matches else Path(prefix + filename)
    result = (dump.manifest_path.parent / selected).resolve()
    require(result.is_relative_to(dump.manifest_path.parent), f"Escaping raw path: {result}")
    return result


def raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(read_text(path)), keep_default_na=False)


def compare_columns(actual: pd.DataFrame, raw: pd.DataFrame, context: str) -> None:
    pd.testing.assert_frame_equal(
        actual.loc[:, raw.columns].reset_index(drop=True), raw.reset_index(drop=True),
        check_dtype=False, check_exact=True, obj=context,
    )


def provenance(dump: Dump, frame: pd.DataFrame) -> None:
    for column, expected in {
        "dump": dump.name, "dump_key": dump.key,
        "manifest_path": str(dump.manifest_path), "prefix": dump.prefix,
    }.items():
        require(frame[column].eq(expected).all(), f"{dump.key}: wrong {column}")


def raw_species(dump: Dump) -> pd.DataFrame:
    text = read_text(selected_file(dump, "Species_nt_species.csv"))
    headers = [line.split()[1:] for line in text.splitlines() if line.startswith("#column ")]
    rows = list(csv.reader(line for line in text.splitlines() if not line.startswith("#")))
    raw = pd.DataFrame(rows, columns=[name for _, name in headers])
    for kind, name in headers:
        if kind in ("int", "double"):
            raw[name] = raw[name].astype("int64" if kind == "int" else "float64")
    return raw


def check_species(dump: Dump, *, estimates: bool = False) -> dict:
    raw = raw_species(dump)
    normalized = raw.rename(columns={"time": "time_s"}).copy()
    normalized["time_s"] *= 1e-9
    frame = dump.particle_g_estimates() if estimates else dump.species()
    compare_columns(frame, normalized, f"{dump.key}: species CSV")
    provenance(dump, frame)
    energy = raw_manifest(dump.manifest_path)["totalEnergyDeposit_eV"]
    np.testing.assert_allclose(frame.G, 100 * raw.number / energy, rtol=1e-14)
    if estimates:
        np.testing.assert_allclose(frame.aggregate_energy_G, 100 * raw.number / energy)
        np.testing.assert_allclose(frame.mean_event_G, raw.sumG / raw.nEvent)
        require(frame.mean_event_G_sem.isna().all(), "SEM must remain null without assertion")
    return {"rows": len(frame), "raw_first_row": raw.iloc[0].to_dict() if len(raw) else None,
            "estimators_checked": estimates, "sem_opt_in": False}


def check_prechemical(dump: Dump) -> dict:
    manifest = raw_manifest(dump.manifest_path)
    names = manifest.get("files")
    if names is None:
        names = [p.name for p in dump.manifest_path.parent.iterdir()]
    pattern = re.compile(re.escape(dump.prefix) + r"PreChemical_run(\d+)_event(\d+)\.txt\Z")
    files = []
    for name in names:
        path = Path(name.replace("\\", "/"))
        match = pattern.fullmatch(path.name)
        if match:
            files.append((*map(int, match.groups()), dump.manifest_path.parent / path))
    require(bool(files), f"No raw pre-chemical files in {dump.manifest_path}")
    tables = []
    for run, event, path in sorted(files):
        lines = [line.partition("#")[0] for line in read_text(path).splitlines()]
        text = "\n".join(line for line in lines if line.strip())
        if not text:
            continue
        frame = pd.read_fwf(
            io.StringIO(text), widths=[11, 10, 14, 13, 22, 22, 22, 14, 14, 14],
            names=PRECHEM_COLUMNS, dtype={"molecule": str, "electron_modification": str},
        )
        frame.insert(0, "event", event)
        frame.insert(0, "run", run)
        tables.append(frame)
    raw = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["run", "event", *PRECHEM_COLUMNS])
    actual = dump.prechemical_records()
    # Compare null masks explicitly; convert nullable extension types for raw comparison.
    columns = ["x_nm", "y_nm", "z_nm"]
    np.testing.assert_array_equal(actual[columns].isna(), raw[columns].isna())
    comparable = actual.copy()
    for column in columns:
        comparable[column] = comparable[column].astype("float64")
    compare_columns(comparable, raw, f"{dump.key}: fixed-width raw fields")
    provenance(dump, actual)
    return {"files": len(files), "rows": len(actual),
            "event_identities": actual[["run", "event"]].drop_duplicates().values.tolist(),
            "e_aq_final_positions": int((actual.molecule.eq("e_aq")
                                         & actual[columns].notna().all(axis=1)).sum()),
            "rows_with_missing_final_positions": int(actual[columns].isna().any(axis=1).sum())}


def check_dump(path: Path, *, meso: bool) -> dict:
    dump = Dump.from_manifest(path if path.is_file() else path / "Manifest.json")
    manifest = raw_manifest(dump.manifest_path)
    physical_raw = raw_csv(selected_file(dump, "PhysicsInteractions.csv"))
    physical = dump.physical_interactions(per_event=True)
    compare_columns(physical, physical_raw, "physical CSV")
    provenance(dump, physical)
    np.testing.assert_allclose(physical.count_per_event,
                               physical_raw["count"] / manifest["totalEvents"])
    track_raw = raw_csv(selected_file(dump, "TrackLengths.csv"))
    tracks = dump.track_lengths()
    compare_columns(tracks, track_raw, "track CSV")
    provenance(dump, tracks)
    result = {
        "manifest": str(dump.manifest_path), "events": manifest["totalEvents"],
        "physical_rows": len(physical),
        "physical_first_row": physical_raw.iloc[0].to_dict() if len(physical_raw) else None,
        "track_rows": len(tracks),
        "track_first_row": track_raw.iloc[0].to_dict() if len(track_raw) else None,
        "prechemical": check_prechemical(dump), "particle": check_species(dump, estimates=True),
    }
    if meso:
        require(dump.has_meso_species, "Mesoscopic aggregate CSV unavailable")
        require(dump.has_meso, "Mesoscopic spatial demonstration has no spatial HDF5")
        raw = raw_csv(selected_file(dump, "SpeciesMeso.csv"))
        frame = dump.meso_species()
        compare_columns(frame, raw, "mesoscopic CSV")
        provenance(dump, frame)
        np.testing.assert_allclose(frame.mean_count_per_event, raw["count"] / manifest["totalEvents"])
        for column in ("handOverTime_ns", "chemistryEndTime_ns"):
            require(frame[column].eq(manifest[column]).all(), f"Wrong {column}")
        # The human-readable companion independently reports event-normalized counts.
        txt = read_text(selected_file(dump, "SpeciesMeso.Txt")).splitlines()
        txt_rows = []
        time = None
        for line in txt[1:]:
            fields = line.split()
            if len(fields) == 1:
                time = float(fields[0])
            elif len(fields) == 2:
                txt_rows.append((time, fields[0], float(fields[1])))
        require(len(txt_rows) == len(frame), "Meso TXT/CSV row count differs")
        np.testing.assert_allclose(frame.time_ns, [r[0] for r in txt_rows], rtol=1e-9)
        require(frame.species.tolist() == [r[1] for r in txt_rows], "Meso TXT species differ")
        np.testing.assert_allclose(frame.mean_count_per_event, [r[2] for r in txt_rows], rtol=1e-9)
        result["meso"] = {"rows": len(frame), "first_row": frame.loc[
            0, ["time_ns", "species", "count", "mean_count_per_event"]].to_dict()
            if len(frame) else None,
            "txt_normalization_checked": True, "spatial_file_present": True}
    return result


def expected_manifests(root: Path) -> list[Path]:
    """Read index order or the documented single/root/direct-child selection."""
    plain = root / "Manifest.json"
    if plain.is_file():
        data = raw_manifest(plain)
        if "dumps" in data:
            return [(root / entry["manifest"].replace("\\", "/")).resolve()
                    for entry in data["dumps"]]
        return [plain.resolve()]
    selected = sorted(root.glob("*Manifest.json"))
    for child in sorted(root.iterdir()):
        if child.is_dir():
            if (child / "Manifest.json").is_file():
                selected.extend(expected_manifests(child))
            else:
                selected.extend(sorted(child.glob("*Manifest.json")))
    return [p.resolve() for p in selected]


def check_discovery(root: Path) -> dict:
    root = root.resolve()
    locations = discover_dumps(root)
    require([loc.manifest_path for loc in locations] == expected_manifests(root),
            f"Discovery differs from raw manifest selection: {root}")
    sim = Simulation(root)
    require(list(sim.subruns) == [loc.key for loc in locations], "Simulation keys/order differ")
    require(len({loc.key for loc in locations}) == len(locations), "Duplicate dump keys")
    selected = []
    for location, dump in zip(locations, sim):
        require(location.key == location.manifest_path.relative_to(root).as_posix(), "Unstable key")
        require(location.prefix == (raw_manifest(location.manifest_path).get("prefix") or ""),
                "Wrong discovery prefix")
        selected.append({"key": dump.key, "prefix": dump.prefix,
                         "manifest": str(dump.manifest_path), "species": check_species(dump)})
    selected_paths = {loc.manifest_path for loc in locations}
    # This inventory is diagnostic, not a promise of recursive library discovery.
    omitted = [str(p.relative_to(root)) for p in sorted(root.rglob("*Manifest.json"))
               if p.resolve() not in selected_paths
               and raw_manifest(p).get("kind") != "resultsIndex"]
    return {"root": str(root), "selected": selected, "outside_selection": omitted,
            "limitation": ("Index entries are authoritative; a plain root manifest selects one "
                           "dump; fallback scans only root/direct children. Select omitted dumps "
                           "explicitly with Dump.from_manifest().") if omitted else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irt-dump", required=True, type=Path,
                        help="IRT demo directory or explicit dump manifest; all direct readers required")
    parser.add_argument("--meso-dump", required=True, type=Path,
                        help="Meso-spatial demo directory or manifest, including SpeciesMeso.Txt")
    parser.add_argument("--discovery-root", action="append", default=[], type=Path,
                        help="Repeat for prefixed/nested/indexed/historical results directories")
    parser.add_argument("--explicit-manifest", action="append", default=[], type=Path,
                        help="Repeat to validate species/provenance of dumps outside automatic selection")
    parser.add_argument("--expected-irt-records", type=int,
                        help="Optional historical baseline; discrepancies are reported, not parser failures")
    parser.add_argument("--expected-irt-final-electrons", type=int,
                        help="Optional baseline for e_aq rows with all final coordinates")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path; use an ignored scratch directory")
    args = parser.parse_args()
    public_names = dnachem.__all__
    require(all(hasattr(dnachem, name) for name in public_names), "A public DnaChem import is missing")
    require(SpeciesMesoSpatialFile is not None, "Spatial public import is missing")
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "g4utils_source": g4utils.__file__,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("g4utils", "pandas", "numpy", "h5py")},
        "installation": json.loads(importlib.metadata.distribution("g4utils").read_text(
            "direct_url.json") or "null"), "public_imports": public_names,
        "irt": check_dump(args.irt_dump, meso=False),
        "meso": check_dump(args.meso_dump, meso=True),
        "discovery": [check_discovery(root) for root in args.discovery_root],
        "explicit": [], "baseline_comparisons": [],
    }
    for path in args.explicit_manifest:
        dump = Dump.from_manifest(path)
        report["explicit"].append({"manifest": str(dump.manifest_path),
                                   "key": dump.key, "species": check_species(dump)})
    for field, expected in [("rows", args.expected_irt_records),
                            ("e_aq_final_positions", args.expected_irt_final_electrons)]:
        if expected is not None:
            actual = report["irt"]["prechemical"][field]
            report["baseline_comparisons"].append({
                "field": field, "expected": expected, "actual": actual,
                "status": "matches" if actual == expected else "changed_data_discrepancy",
            })
    report["input_sha256"] = dict(sorted(INPUTS.items()))
    report["status"] = "passed_raw_value_checks"
    result = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared test helpers: synthetic G4Vox HDF5 files.

The file layout mirrors what the C++ writer produces; the format source is
``G4Vox/src/HDF5Writer.cc``:

- ``/metadata`` group with attrs ``dims_xyz``, ``spacing_mm``, ``origin_mm``
  (float64 arrays), ``mode`` and ``quantities`` (variable-length strings).
- Snapshot3D: ``/subrun_NNNN/<qty>`` datasets of shape ``(nZ, nY, nX)``.
- Extendable4D: ``/<qty>`` datasets of shape ``(N, nZ, nY, nX)``.
- ``/run_log``: float64 ``(N, 4)``, rows ``[unix_timestamp, primaries, runtime_s, subrun_id]``,
  with a ``columns`` string attribute.

Also: ``write_species_meso_spatial`` (SpeciesMesoSpatial.h5 files), ``make_dump``, a factory for synthetic dnachem-min Dumps (Manifest.json +
Geant4 wcsv ntuple CSVs) used by the DnaChem tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pytest

DIMS_XYZ = (4, 3, 2)
SPACING_MM = (0.5, 1.0, 2.0)
ORIGIN_MM = (-1.0, -1.5, -2.0)
QUANTITIES = ("Dose", "Edep")

_RUN_LOG_COLUMNS = "unix_timestamp, primaries, runtime_s, subrun_id"


def expected_array(
    qty_index: int,
    subrun_id: int,
    dims_xyz: Sequence[int] = DIMS_XYZ,
    dtype=np.float64,
) -> np.ndarray:
    """Return the deterministic ``(nz, ny, nx)`` array for a quantity and subrun."""
    nx, ny, nz = dims_xyz
    base = np.arange(nz * ny * nx).reshape(nz, ny, nx)
    return (base + 1000 * qty_index + 100 * subrun_id).astype(dtype)


def _write_str_attr(obj, name: str, value: str) -> None:
    obj.attrs.create(name, value, dtype=h5py.string_dtype())


def _write_metadata(f, mode, dims_xyz, quantities) -> None:
    meta = f.create_group("metadata")
    meta.attrs["dims_xyz"] = np.asarray(dims_xyz, dtype=np.float64)
    meta.attrs["spacing_mm"] = np.asarray(SPACING_MM, dtype=np.float64)
    meta.attrs["origin_mm"] = np.asarray(ORIGIN_MM, dtype=np.float64)
    _write_str_attr(meta, "mode", mode)
    _write_str_attr(meta, "quantities", ",".join(quantities))


def _write_run_log(f, ids: Sequence[int], primaries: int, columns_attr: bool) -> None:
    rows = np.array(
        [[1.7e9 + i, primaries, 1.5, sid] for i, sid in enumerate(ids)],
        dtype=np.float64,
    ).reshape(len(ids), 4)
    ds = f.create_dataset("run_log", data=rows, maxshape=(None, 4), chunks=(1, 4))
    if columns_attr:
        _write_str_attr(ds, "columns", _RUN_LOG_COLUMNS)


def write_snapshot3d(
    path,
    *,
    subrun_ids: Sequence[int] = (0, 1, 2),
    dims_xyz: Sequence[int] = DIMS_XYZ,
    quantities: Sequence[str] = QUANTITIES,
    dtype=np.float64,
    primaries: int = 100,
    with_metadata: bool = True,
    with_run_log: bool = True,
    run_log_columns_attr: bool = True,
) -> Path:
    """Write a Snapshot3D file matching ``HDF5Writer.cc``."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        if with_metadata:
            _write_metadata(f, "Snapshot3D", dims_xyz, quantities)
        for sid in subrun_ids:
            group = f.create_group(f"subrun_{sid:04d}")
            for qi, qty in enumerate(quantities):
                group.create_dataset(qty, data=expected_array(qi, sid, dims_xyz, dtype))
        if with_run_log:
            _write_run_log(f, list(subrun_ids), primaries, run_log_columns_attr)
    return path


def write_extendable4d(
    path,
    *,
    subrun_ids: Sequence[int] = (0, 1, 2),
    n_slices: int | None = None,
    run_log_ids: Sequence[int] | None = None,
    dims_xyz: Sequence[int] = DIMS_XYZ,
    quantities: Sequence[str] = QUANTITIES,
    dtype=np.float64,
    primaries: int = 100,
    with_metadata: bool = True,
    with_run_log: bool = True,
    run_log_columns_attr: bool = True,
) -> Path:
    """Write an Extendable4D file matching ``HDF5Writer.cc``.

    Slice ``i`` holds ``expected_array(qi, subrun_ids[i])``. ``n_slices`` limits the
    number of slices written; ``run_log_ids`` overrides the run_log subrun_id column
    and may differ in length from the slices.
    """
    path = Path(path)
    nx, ny, nz = dims_xyz
    slice_ids = list(subrun_ids)[: n_slices if n_slices is not None else len(subrun_ids)]
    with h5py.File(path, "w") as f:
        if with_metadata:
            _write_metadata(f, "Extendable4D", dims_xyz, quantities)
        for qi, qty in enumerate(quantities):
            data = np.zeros((len(slice_ids), nz, ny, nx), dtype=dtype)
            for i, sid in enumerate(slice_ids):
                data[i] = expected_array(qi, sid, dims_xyz, dtype)
            f.create_dataset(
                qty,
                data=data,
                maxshape=(None, nz, ny, nx),
                chunks=(1, nz, ny, nx),
            )
        if with_run_log:
            ids = list(run_log_ids) if run_log_ids is not None else slice_ids
            _write_run_log(f, ids, primaries, run_log_columns_attr)
    return path


@pytest.fixture
def snapshot3d_file(tmp_path) -> Path:
    return write_snapshot3d(tmp_path / "snapshot3d.h5")


@pytest.fixture
def extendable4d_file(tmp_path) -> Path:
    return write_extendable4d(tmp_path / "extendable4d.h5")


# ═════════════════════════════════════════════════════════════════════════════
#  DnaChem: synthetic dnachem-min Dumps
# ═════════════════════════════════════════════════════════════════════════════

_SPECIES_HEADER = [
    "#class tools::wcsv::ntuple",
    "#title species",
    "#separator 44",
    "#vector_separator 59",
    "#column int speciesID",
    "#column int number",
    "#column int nEvent",
    "#column string speciesName",
    "#column double time",
    "#column double sumG",
    "#column double sumG2",
]

_REACTIONS_HEADER = [
    "#class tools::wcsv::ntuple",
    "#title reactions",
    "#separator 44",
    "#vector_separator 59",
    "#column int reactionId",
    "#column double time",
    "#column int count",
]


def _write(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


@pytest.fixture
def make_dump():
    def _make_dump(
        parent: Path,
        name: str,
        *,
        molarity_M: float = 0.0,
        energy_deposit_eV: float = 1e7,
        runs: list[dict] | None = None,
        with_manifest: bool = True,
    ) -> Path:
        d = Path(parent) / name
        d.mkdir(parents=True, exist_ok=True)
        if with_manifest:
            if runs is None:
                runs = [
                    {
                        "run": 0,
                        "events": 100,
                        "particle": "e-",
                        "beamEnergy_keV": 100,
                        "energyDeposit_eV": energy_deposit_eV,
                        "seed": 12345,
                    }
                ]
            manifest = {
                "schemaVersion": 1,
                "chemistry": "BoscoloChem",
                "pH": 7,
                "scavengers": [{"species": "O2", "molarity_M": molarity_M}],
                "totalEvents": 100,
                "totalEnergyDeposit_eV": energy_deposit_eV,
                "runs": runs,
            }
            with open(d / "Manifest.json", "w", encoding="utf-8", newline="\r\n") as f:
                json.dump(manifest, f, indent=2)
        sp = [
            (0, 400000, "H3O^1"),
            (1, 480000, "°OH^0"),
            (3, 400000, "e_aq^-1"),
            (7, 3, "HO_2°^0"),
        ]
        sp_late = [270000, 250000, 260000, 50]
        rows = [f"{i},{n},100,{s},0.001,0.5,0.25" for i, n, s in sp]
        rows += [
            f"{i},{n},100,{s},999.999,0.1,0.01"
            for (i, _, s), n in zip(sp, sp_late)
        ]
        _write(d / "Species_nt_species.csv", _SPECIES_HEADER + rows)
        _write(
            d / "Reactions_nt_reactions.csv",
            _REACTIONS_HEADER + ["1,0.001,646", "25,0.001,4436", "25,999.999,9000"],
        )
        _write(
            d / "ReactionsMetadata.csv",
            [
                "reactionId,reaction",
                "1,H3O^1 + OH^-1 -> (no products)",
                "25,°OH^0 + °OH^0 -> H2O2^0",
            ],
        )
        return d

    return _make_dump


# ═════════════════════════════════════════════════════════════════════════════
#  SpeciesMesoSpatial.h5
# ═════════════════════════════════════════════════════════════════════════════

MESO_SPECIES = ("H3O^1", "°OH^0", "e_aq^-1")


def write_species_meso_spatial(
    path,
    *,
    species: Sequence[str] = MESO_SPECIES,
    format_version: int | None = 2,
    with_species: bool = True,
) -> Path:
    """Write a SpeciesMesoSpatial.h5 file.

    Runs 0 and 2, each with events 0, 2 and 10 (to check numeric sorting).
    run0/event0 has snapshots 0, 1, 2 and 10: snapshot1 hard-links the datasets
    of snapshot0 and snapshot2 has N = 0. Other events hold a single snapshot.
    """
    path = Path(path)
    s = len(species)
    with h5py.File(path, "w") as f:
        if with_species:
            f.attrs.create("species", list(species), dtype=h5py.string_dtype())
        if format_version is not None:
            f.attrs["formatVersion"] = np.int32(format_version)
        _write_str_attr(f, "units", "positions nm, times ns, counts molecules")

        def make(group, k, n, time_ns, cell, seed):
            g = group.create_group(f"snapshot{k}")
            g.attrs["time_ns"] = np.float64(time_ns)
            g.attrs["cellSize_nm"] = np.float64(cell)
            rng = np.random.default_rng(seed)
            g.create_dataset("position_nm", data=rng.random((n, 3)) * 100.0)
            g.create_dataset(
                "counts", data=rng.integers(1, 9, size=(n, s)).astype(np.uint32)
            )
            return g

        for run in (0, 2):
            for ev in (0, 2, 10):
                eg = f.create_group(f"run{run}/event{ev}")
                if run == 0 and ev == 0:
                    s0 = make(eg, 0, 4, 5.0, 6.25, 1)
                    s1 = eg.create_group("snapshot1")
                    s1.attrs["time_ns"] = np.float64(6.3)
                    s1.attrs["cellSize_nm"] = np.float64(6.25)
                    s1["position_nm"] = s0["position_nm"]
                    s1["counts"] = s0["counts"]
                    make(eg, 2, 0, 8.0, 12.5, 2)
                    make(eg, 10, 3, 100.0, 25.0, 3)
                else:
                    make(eg, 0, 2, 5.0, 6.25, 10 * run + ev)
    return path


@pytest.fixture
def species_meso_spatial_file(tmp_path) -> Path:
    return write_species_meso_spatial(tmp_path / "SpeciesMesoSpatial.h5")

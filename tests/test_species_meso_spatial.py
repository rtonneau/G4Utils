import h5py
import numpy as np
import pytest

from g4utils.HDF5 import MesoSpatialSnapshot, SpeciesMesoSpatialFile, concentration_M
from g4utils.HDF5.species_meso_spatial import (
    DenseMesoPeriod,
    MesoGrid,
    _densify_snapshot,
)

from .conftest import MESO_SPECIES, write_species_meso_spatial


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SpeciesMesoSpatialFile(tmp_path / "nope.h5")


@pytest.mark.parametrize("kw", [{"with_species": False}, {"format_version": None}])
def test_missing_root_attrs(tmp_path, kw):
    p = write_species_meso_spatial(tmp_path / "x.h5", **kw)
    with pytest.raises(ValueError):
        SpeciesMesoSpatialFile(p)


def test_index(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    assert f.species == list(MESO_SPECIES)
    assert f.format_version == 2
    assert "nm" in f.units
    assert f.runs == [0, 2]
    assert f.events(0) == [0, 2, 10]
    assert f.snapshot_indices(0, 0) == [0, 1, 2, 10]
    assert f.snapshot_time_ns(0, 0, 1) == pytest.approx(6.3)
    assert f.snapshot_cell_size_nm(0, 0, 2) == pytest.approx(12.5)


def test_unknown_keys_raise(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    with pytest.raises(KeyError):
        f.events(1)
    with pytest.raises(KeyError):
        f.snapshot_indices(0, 1)
    with pytest.raises(KeyError):
        f.snapshot_time_ns(0, 0, 3)
    for args in [(1, 0, 0), (0, 1, 0), (0, 0, 3)]:
        with pytest.raises(KeyError):
            f.read_snapshot(*args)


def test_read_snapshot(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 0)
    assert isinstance(s, MesoSpatialSnapshot)
    assert (s.run, s.event, s.index) == (0, 0, 0)
    assert s.time_ns == 5.0 and s.cell_size_nm == 6.25
    assert s.position_nm.shape == (4, 3) and s.position_nm.dtype == np.float64
    assert s.counts.shape == (4, len(MESO_SPECIES)) and s.counts.dtype == np.uint32


def test_hard_linked_and_empty(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    a, b = f.read_snapshot(0, 0, 0), f.read_snapshot(0, 0, 1)
    np.testing.assert_array_equal(a.position_nm, b.position_nm)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert b.time_ns != a.time_ns
    e = f.read_snapshot(0, 0, 2)
    assert e.position_nm.shape == (0, 3) and e.position_nm.dtype == np.float64
    assert e.counts.shape == (0, len(MESO_SPECIES)) and e.counts.dtype == np.uint32
    with h5py.File(species_meso_spatial_file, "r") as h:
        assert h["run0/event0/snapshot0/counts"] == h["run0/event0/snapshot1/counts"]


def test_iter_snapshots(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    it = f.iter_snapshots()
    assert iter(it) is it
    keys = [(s.run, s.event, s.index) for s in it]
    assert keys == sorted(keys)
    assert len(keys) == 4 + 5
    assert keys[:5] == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 10), (0, 2, 0)]
    assert {s.run for s in f.iter_snapshots(run=2)} == {2}
    assert [s.index for s in f.iter_snapshots(run=0, event=0)] == [0, 1, 2, 10]
    assert {s.event for s in f.iter_snapshots(event=10)} == {10}


def test_concentration_M_hand_computed(species_meso_spatial_file):
    """Test concentration_M with hand-computed values."""
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 0)
    # Cell size is 6.25 nm, volume = (6.25 * 1e-8)^3 L
    cell_size_nm = 6.25
    volume_L = (cell_size_nm * 1e-8) ** 3
    N_A = 6.02214076e23

    # Test with all columns (species=None)
    conc = s.concentration_M()
    assert conc.dtype == np.float64
    assert conc.shape == s.counts.shape

    # Manual calculation for first cell, first species
    expected_first = s.counts[0, 0] / (N_A * volume_L)
    np.testing.assert_allclose(conc[0, 0], expected_first)


def test_concentration_M_column_selection(species_meso_spatial_file):
    """Test concentration_M with species column selection."""
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 0)

    # Select specific species
    conc = s.concentration_M(species=["°OH^0", "H3O^1"])
    assert conc.shape == (4, 2)
    assert conc.dtype == np.float64

    # Compare with all columns version
    conc_all = s.concentration_M()
    # Species order in file: H3O^1 (0), °OH^0 (1), e_aq^-1 (2)
    # Requested order: °OH^0 (1), H3O^1 (0)
    np.testing.assert_allclose(conc[:, 0], conc_all[:, 1])  # °OH^0
    np.testing.assert_allclose(conc[:, 1], conc_all[:, 0])  # H3O^1


def test_concentration_M_empty_snapshot(species_meso_spatial_file):
    """Test concentration_M with an empty snapshot (N=0)."""
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 2)  # Empty snapshot

    conc = s.concentration_M()
    assert conc.shape == (0, len(MESO_SPECIES))
    assert conc.dtype == np.float64


def test_concentration_M_unknown_species(species_meso_spatial_file):
    """Test concentration_M raises KeyError for unknown species."""
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 0)

    with pytest.raises(KeyError):
        s.concentration_M(species=["UnknownSpecies"])


# ----------------------------------------------------------------------
# _densify_snapshot
# ----------------------------------------------------------------------
def _snap(pos, cs=10.0, counts=None):
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    if counts is None:
        counts = np.arange(1, 2 * len(pos) + 1, dtype=np.uint32).reshape(len(pos), 2)
    return MesoSpatialSnapshot(
        run=0, event=0, index=0, time_ns=1.0, cell_size_nm=cs,
        position_nm=pos, counts=np.asarray(counts, dtype=np.uint32).reshape(len(pos), 2),
        species=["a", "b"],
    )


def test_densify_placement():
    # Centres at x=5,15,35 (gap at 25), y=5,15, z=5 with cell size 10.
    snap = _snap([[5, 5, 5], [15, 5, 5], [35, 15, 5]])
    g = _densify_snapshot(snap)
    np.testing.assert_allclose(g.origin_nm, [0, 0, 0])
    assert g.cell_size_nm == 10.0
    assert g.dims == (4, 2, 1)
    assert g.index.shape == (1, 2, 4) and g.counts.shape == (2, 1, 2, 4)
    assert g.counts.dtype == np.uint32
    np.testing.assert_array_equal(g.index[0], [[1, 2, 0, 0], [0, 0, 0, 3]])
    np.testing.assert_array_equal(g.counts[0, 0], [[1, 3, 0, 0], [0, 0, 0, 5]])
    np.testing.assert_array_equal(g.counts[1, 0], [[2, 4, 0, 0], [0, 0, 0, 6]])


def test_densify_origin_from_min_centre():
    g = _densify_snapshot(_snap([[-100.0, 20.0, 7.5], [-90.0, 20.0, 7.5]], cs=10.0))
    np.testing.assert_allclose(g.origin_nm, [-105.0, 15.0, 2.5])
    assert g.dims == (2, 1, 1)


def test_densify_axis_order():
    g = _densify_snapshot(_snap([[5, 5, 5], [15, 25, 35]]))
    assert g.dims == (2, 3, 4)
    assert g.index.shape == (4, 3, 2)
    assert g.index[0, 0, 0] == 1 and g.index[3, 2, 1] == 2


def test_densify_extent_grows_by_whole_cells():
    snap = _snap([[5, 5, 5], [15, 5, 5]])
    g = _densify_snapshot(snap, extent=([-12, 0, 0], [33, 10, 10]))
    # anchor (0,0,0): x grows 2 cells below (-12 -> -20) and to 40 above.
    np.testing.assert_allclose(g.origin_nm, [-20, 0, 0])
    assert g.dims == (6, 1, 1)
    np.testing.assert_array_equal(g.index[0, 0], [0, 0, 1, 2, 0, 0])
    # Result stays on the snapshot's own lattice.
    assert ((g.origin_nm - 0.0) / 10.0 == np.round((g.origin_nm - 0.0) / 10.0)).all()


def test_densify_extent_inside_data_does_not_shrink():
    snap = _snap([[5, 5, 5], [35, 5, 5]])
    g = _densify_snapshot(snap, extent=([10, 0, 0], [20, 10, 10]))
    np.testing.assert_allclose(g.origin_nm, [0, 0, 0])
    assert g.dims == (4, 1, 1)


def test_densify_extent_exact_edges_no_extra_cell():
    snap = _snap([[5, 5, 5]])
    g = _densify_snapshot(snap, extent=([0, 0, 0], [10, 10, 10]))
    assert g.dims == (1, 1, 1)
    np.testing.assert_allclose(g.origin_nm, [0, 0, 0])


def test_densify_empty_snapshot_with_extent():
    snap = _snap(np.zeros((0, 3)), cs=5.0)
    g = _densify_snapshot(snap, extent=([0, 0, 0], [12, 5, 5]))
    np.testing.assert_allclose(g.origin_nm, [0, 0, 0])
    assert g.dims == (3, 1, 1)
    assert g.counts.shape == (2, 1, 1, 3)
    assert not g.index.any() and not g.counts.any()


def test_densify_off_lattice_raises():
    snap = _snap([[5, 5, 5], [15, 5, 5], [25.0, 5, 5], [34.0, 5, 5]])
    with pytest.raises(ValueError, match="off the lattice"):
        _densify_snapshot(snap)


def test_densify_collision_raises():
    snap = _snap([[5, 5, 5], [15, 5, 5], [5.0001, 5, 5]])
    with pytest.raises(ValueError, match="share lattice index"):
        _densify_snapshot(snap)


def test_densify_empty_without_extent_raises():
    with pytest.raises(ValueError, match="no cells"):
        _densify_snapshot(_snap(np.zeros((0, 3))))


def test_densify_bad_extent_raises():
    with pytest.raises(ValueError):
        _densify_snapshot(_snap([[5, 5, 5]]), extent=([10, 0, 0], [0, 10, 10]))


def test_densify_real_file_empty_snapshot(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    e = f.read_snapshot(0, 0, 2)
    with pytest.raises(ValueError, match="no cells"):
        _densify_snapshot(e)
    g = _densify_snapshot(e, extent=([0, 0, 0], [25, 25, 25]))
    assert g.dims == (2, 2, 2) and g.cell_size_nm == e.cell_size_nm
    assert g.counts.shape == (len(MESO_SPECIES), 2, 2, 2)


# ---------------------------------------------------------------------------
# to_vti_timeseries
# ---------------------------------------------------------------------------
def _vti_geometry(path):
    import xml.etree.ElementTree as ET

    img = ET.parse(path).getroot().find("ImageData")
    return (
        np.array(img.get("Origin").split(), dtype=float),
        np.array(img.get("Spacing").split(), dtype=float),
        np.array(img.get("WholeExtent").split(), dtype=int)[1::2],
    )


def _lattice_file(tmp_path):
    """Event with lattice-aligned cells; cell size doubles between frames."""
    path = tmp_path / "lattice.h5"
    with h5py.File(path, "w") as f:
        f.attrs.create("species", ["A", "B", "C"], dtype=h5py.string_dtype())
        f.attrs["formatVersion"] = np.int32(2)
        eg = f.create_group("run0/event0")
        frames = [
            (0, 1.0, 10.0, [[5, 5, 5], [15, 5, 5]]),
            (1, 2.0, 20.0, [[10, 10, 10], [30, 10, 10], [10, 30, 30]]),
            (5, 3.0, 40.0, [[20, 20, 20]]),
        ]
        for k, t, cs, pos in frames:
            g = eg.create_group(f"snapshot{k}")
            g.attrs["time_ns"] = t
            g.attrs["cellSize_nm"] = cs
            g.create_dataset("position_nm", data=np.array(pos, dtype=float))
            g.create_dataset("counts", data=np.ones((len(pos), 3), dtype=np.uint32))
    return path


def test_timeseries_files_pvd_and_common_box(tmp_path):
    import xml.etree.ElementTree as ET

    f = SpeciesMesoSpatialFile(_lattice_file(tmp_path))
    pvd = f.to_vti_timeseries(0, 0, tmp_path / "out" / "ts.pvd")
    assert pvd.name == "ts.pvd"
    entries = [d.attrib for d in ET.parse(pvd).getroot().iter("DataSet")]
    assert [e["file"] for e in entries] == [
        "ts_0000.vti", "ts_0001.vti", "ts_0005.vti"
    ]
    assert [float(e["timestep"]) for e in entries] == [1.0, 2.0, 3.0]
    lo = hi = None
    for s in f.iter_snapshots(0, 0):
        if len(s.position_nm):
            a = s.position_nm.min(0) - s.cell_size_nm / 2
            b = s.position_nm.max(0) + s.cell_size_nm / 2
            lo = a if lo is None else np.minimum(lo, a)
            hi = b if hi is None else np.maximum(hi, b)
    for e, k in zip(entries, (0, 1, 5)):
        origin, spacing, dims = _vti_geometry(pvd.parent / e["file"])
        assert np.allclose(spacing, f.snapshot_cell_size_nm(0, 0, k))
        end = origin + dims * spacing
        assert np.all(origin <= lo + 1e-6) and np.all(end >= hi - 1e-6)
        assert np.all(origin > lo - spacing - 1e-6) and np.all(end < hi + spacing + 1e-6)


def test_timeseries_explicit_extent(tmp_path):
    f = SpeciesMesoSpatialFile(_lattice_file(tmp_path))
    pvd = f.to_vti_timeseries(0, 0, tmp_path / "ts", extent=([-50] * 3, [250] * 3))
    assert pvd.suffix == ".pvd"
    origin, spacing, dims = _vti_geometry(pvd.parent / "ts_0005.vti")
    assert np.all(origin <= -50 + 1e-6)
    assert np.all(origin + dims * spacing >= 250 - 1e-6)


def test_timeseries_errors(species_meso_spatial_file, tmp_path):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    with pytest.raises(KeyError):
        f.to_vti_timeseries(1, 0, tmp_path / "a.pvd")
    with pytest.raises(KeyError):
        f.to_vti_timeseries(0, 1, tmp_path / "a.pvd")
    empty = tmp_path / "empty.h5"
    with h5py.File(species_meso_spatial_file) as src, h5py.File(empty, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        g = dst.create_group("run0/event0/snapshot0")
        g.attrs["time_ns"] = 1.0
        g.attrs["cellSize_nm"] = 5.0
        g.create_dataset("position_nm", data=np.zeros((0, 3)))
        g.create_dataset("counts", data=np.zeros((0, 3), dtype=np.uint32))
    with pytest.raises(ValueError):
        SpeciesMesoSpatialFile(empty).to_vti_timeseries(0, 0, tmp_path / "e.pvd")

# ---------------------------------------------------------------------------
# MesoGrid and to_dense
# ---------------------------------------------------------------------------
def _dense_snapshot(cell=6.25, ijk=((2, 3, 4), (3, 3, 4), (5, 7, 4)), species=MESO_SPECIES):
    """Snapshot whose cell centres sit on the lattice (k + 0.5) * cell."""
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    pos = (ijk + 0.5) * cell
    counts = np.arange(1, ijk.shape[0] * len(species) + 1, dtype=np.uint32).reshape(
        -1, len(species)
    )
    return MesoSpatialSnapshot(
        run=0, event=0, index=0, time_ns=1.0, cell_size_nm=cell,
        position_nm=pos, counts=counts, species=list(species),
    )


def test_mesogrid_geometry():
    g = MesoGrid(origin_nm=(10.0, 0.0, -5.0), cell_size_nm=2.0, shape=(3, 2, 4))
    np.testing.assert_allclose(g.centers(0), [11.0, 13.0, 15.0])
    np.testing.assert_allclose(g.centers(1), [1.0, 3.0])
    assert g.centers(2).shape == (4,)
    idx = g.index_of([[11.0, 1.0, -4.0], [15.0, 3.0, 2.0]])
    assert idx.dtype == np.int64
    np.testing.assert_array_equal(idx, [[0, 0, 0], [2, 1, 3]])
    np.testing.assert_array_equal(g.index_of([9.9, 0.0, -5.0]), [-1, 0, 0])
    with pytest.raises(ValueError):
        g.centers(3)


@pytest.mark.parametrize("cell", [6.25, 12.5, 25.0, 50.0])
def test_to_dense_round_trip_default_extent(cell):
    s = _dense_snapshot(cell=cell)
    dense, grid = s.to_dense("°OH^0")
    assert dense.dtype == np.uint32
    assert grid.cell_size_nm == cell
    assert grid.origin_nm == (2 * cell, 3 * cell, 4 * cell)
    assert grid.shape == (4, 5, 1) == dense.shape
    # every row lands on the cell whose centre is its position
    idx = grid.index_of(s.position_nm)
    np.testing.assert_array_equal(idx, [[0, 0, 0], [1, 0, 0], [3, 4, 0]])
    np.testing.assert_array_equal(dense[tuple(idx.T)], s.counts[:, 1])
    assert dense.sum() == s.counts[:, 1].sum()
    # cell centres of the grid reproduce the input positions
    for ax in range(3):
        c = grid.centers(ax)
        assert set(np.round(s.position_nm[:, ax], 9)) <= set(np.round(c, 9))


def test_to_dense_concentration():
    s = _dense_snapshot()
    n, _ = s.to_dense("H3O^1")
    c, grid = s.to_dense("H3O^1", concentration=True)
    assert c.dtype == np.float64 and c.shape == n.shape
    np.testing.assert_allclose(c, concentration_M(n, 6.25))
    assert c.sum() > 0


def test_to_dense_explicit_bounds():
    s = _dense_snapshot()
    cell = 6.25
    bounds = [[0.0, 10 * cell], [0.0, 10 * cell], [0.0, 10 * cell]]
    dense, grid = s.to_dense("e_aq^-1", bounds_nm=bounds)
    assert grid.origin_nm == (0.0, 0.0, 0.0)
    assert grid.shape == (10, 10, 10)
    assert dense.sum() == s.counts[:, 2].sum()
    np.testing.assert_array_equal(dense[2, 3, 4], s.counts[0, 2])
    # same bounds -> same shape for different snapshots
    other = _dense_snapshot(ijk=((0, 0, 0),))
    d2, g2 = other.to_dense("e_aq^-1", bounds_nm=bounds)
    assert d2.shape == dense.shape and g2 == grid


def test_to_dense_bounds_drop_outside_cells():
    s = _dense_snapshot()
    cell = 6.25
    bounds = [[2 * cell, 4 * cell], [3 * cell, 4 * cell], [4 * cell, 5 * cell]]
    dense, grid = s.to_dense("H3O^1", bounds_nm=bounds)
    assert grid.shape == (2, 1, 1)
    # only the first two rows are inside
    np.testing.assert_array_equal(dense[:, 0, 0], s.counts[:2, 0])


def test_to_dense_bounds_snapped_outwards():
    s = _dense_snapshot()
    cell = 6.25
    bounds = [[0.3 * cell, 1.2 * cell], [0.0, cell], [0.0, cell]]
    _, grid = s.to_dense("H3O^1", bounds_nm=bounds)
    assert grid.origin_nm == (0.0, 0.0, 0.0)
    assert grid.shape == (2, 1, 1)


def test_to_dense_empty_snapshot(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 2)
    dense, grid = s.to_dense("H3O^1")
    assert dense.size == 0 and grid.shape == (0, 0, 0)
    bounds = [[0.0, 50.0]] * 3
    dense, grid = s.to_dense("H3O^1", bounds_nm=bounds)
    assert dense.shape == grid.shape == (4, 4, 4)
    assert dense.dtype == np.uint32 and not dense.any()
    conc, _ = s.to_dense("H3O^1", bounds_nm=bounds, concentration=True)
    assert conc.dtype == np.float64 and not conc.any()


def test_to_dense_real_fixture_snapshot(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    s = f.read_snapshot(0, 0, 0)
    dense, grid = s.to_dense("°OH^0")
    assert dense.ndim == 3 and dense.shape == grid.shape
    assert dense.sum() == s.counts[:, 1].sum()


def test_to_dense_sums_duplicate_cells():
    s = _dense_snapshot(ijk=((1, 1, 1), (1, 1, 1)))
    dense, grid = s.to_dense("H3O^1")
    assert dense.shape == (1, 1, 1)
    assert dense[0, 0, 0] == s.counts[:, 0].sum()


def test_to_dense_unknown_species():
    s = _dense_snapshot()
    with pytest.raises(KeyError):
        s.to_dense("Nope")
    s.species = None
    with pytest.raises(KeyError):
        s.to_dense("H3O^1")


# ---------------------------------------------------------------------------
# read_dense
# ---------------------------------------------------------------------------
def test_read_dense_period_grouping(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    periods = f.read_dense(0, 0, "°OH^0")
    assert all(isinstance(p, DenseMesoPeriod) for p in periods)
    assert [p.snapshot_indices for p in periods] == [[0, 1], [2], [10]]
    assert [p.grid.cell_size_nm for p in periods] == [6.25, 12.5, 25.0]
    first = periods[0]
    assert (first.run, first.event, first.species) == (0, 0, "°OH^0")
    np.testing.assert_allclose(first.times_ns, [5.0, 6.3])
    assert first.data.shape == (2, *first.grid.shape)
    assert first.data.dtype == np.uint32
    # Hard-linked snapshots read the same data
    np.testing.assert_array_equal(first.data[0], first.data[1])
    s0 = f.read_snapshot(0, 0, 0)
    assert first.data[0].sum() == s0.counts[:, 1].sum()


def test_read_dense_empty_snapshot_period(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    periods = f.read_dense(0, 0, "H3O^1")
    empty = periods[1]
    assert empty.data.shape == (1, *empty.grid.shape)
    assert not empty.data.any()
    assert empty.data.size > 0  # shared bounding box, not (0, 0, 0)


def test_read_dense_shared_bounding_box(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    # Independent bounding box of the occupied cells, in nm
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for s in f.iter_snapshots(0, 0):
        if len(s.position_nm) == 0:
            continue
        lat = np.floor(s.position_nm / s.cell_size_nm)
        lo = np.minimum(lo, lat.min(axis=0) * s.cell_size_nm)
        hi = np.maximum(hi, (lat.max(axis=0) + 1) * s.cell_size_nm)
    for p in f.read_dense(0, 0, "H3O^1"):
        origin = np.asarray(p.grid.origin_nm)
        upper = origin + np.asarray(p.grid.shape) * p.grid.cell_size_nm
        assert np.all(origin <= lo + 1e-9) and np.all(upper >= hi - 1e-9)
        # Snapped outwards by less than one cell
        assert np.all(lo - origin < p.grid.cell_size_nm + 1e-9)
        assert np.all(upper - hi < p.grid.cell_size_nm + 1e-9)
        assert p.data.sum() == sum(
            f.read_snapshot(0, 0, k).counts[:, 0].sum() for k in p.snapshot_indices
        )


def test_read_dense_concentration(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    counts = f.read_dense(0, 0, "H3O^1")[0]
    conc = f.read_dense(0, 0, "H3O^1", concentration=True)[0]
    assert conc.data.dtype == np.float64
    np.testing.assert_allclose(
        conc.data, concentration_M(counts.data, counts.grid.cell_size_nm)
    )


def test_read_dense_same_bounds_same_shape_across_files(tmp_path, species_meso_spatial_file):
    other = write_species_meso_spatial(tmp_path / "other.h5")
    # Rewrite one event with different positions but same cell sizes
    with h5py.File(other, "r+") as h:
        pos = h["run0/event0/snapshot10/position_nm"]
        pos[...] = pos[...] * 0.5
    a = SpeciesMesoSpatialFile(species_meso_spatial_file)
    b = SpeciesMesoSpatialFile(other)
    bounds = [[0.0, 120.0]] * 3
    pa = a.read_dense(0, 0, "H3O^1", bounds_nm=bounds)
    pb = b.read_dense(0, 0, "H3O^1", bounds_nm=bounds)
    assert len(pa) == len(pb)
    for x, y in zip(pa, pb):
        assert x.data.shape == y.data.shape
        assert x.grid == y.grid
    # Default bounds differ between the two files
    assert [p.data.shape for p in a.read_dense(0, 0, "H3O^1")] != [
        p.data.shape for p in b.read_dense(0, 0, "H3O^1")
    ]


def test_read_dense_single_period_event(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    periods = f.read_dense(2, 10, "e_aq^-1")
    assert len(periods) == 1
    assert periods[0].snapshot_indices == [0]
    assert periods[0].data.shape == (1, *periods[0].grid.shape)


def test_read_dense_all_empty_event(tmp_path):
    p = tmp_path / "empty.h5"
    with h5py.File(p, "w") as h:
        h.attrs.create("species", ["A", "B"], dtype=h5py.string_dtype())
        h.attrs["formatVersion"] = np.int32(2)
        g = h.create_group("run0/event0/snapshot0")
        g.attrs["time_ns"] = 1.0
        g.attrs["cellSize_nm"] = 5.0
        g.create_dataset("position_nm", data=np.zeros((0, 3)))
        g.create_dataset("counts", data=np.zeros((0, 2), dtype=np.uint32))
    f = SpeciesMesoSpatialFile(p)
    (period,) = f.read_dense(0, 0, "A")
    assert period.data.shape == (1, 0, 0, 0)
    (period,) = f.read_dense(0, 0, "A", bounds_nm=[[0.0, 10.0]] * 3)
    assert period.data.shape == (1, 2, 2, 2)


def test_read_dense_unknown_keys(species_meso_spatial_file):
    f = SpeciesMesoSpatialFile(species_meso_spatial_file)
    for args in [(1, 0, "H3O^1"), (0, 1, "H3O^1"), (0, 0, "Nope")]:
        with pytest.raises(KeyError):
            f.read_dense(*args)

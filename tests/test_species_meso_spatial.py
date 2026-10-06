import h5py
import numpy as np
import pytest

from g4utils.HDF5 import MesoSpatialSnapshot, SpeciesMesoSpatialFile, concentration_M
from g4utils.HDF5.species_meso_spatial import DenseMesoPeriod, MesoGrid

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

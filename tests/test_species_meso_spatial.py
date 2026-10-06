import h5py
import numpy as np
import pytest

from g4utils.HDF5 import MesoSpatialSnapshot, SpeciesMesoSpatialFile, concentration_M

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

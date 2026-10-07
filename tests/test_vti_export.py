"""VTI export: ascii, inline binary and zlib-compressed binary encodings."""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pytest

from g4utils.HDF5 import G4VoxFile3D, write_vti
from g4utils.Vox.vox_geometry import VoxGeometry

from .conftest import QUANTITIES, expected_array, write_snapshot3d

BLOCK_SIZE = 32768
BIG_DIMS_XYZ = (41, 30, 20)  # 24,600 cells -> 196,800 bytes as float64

_VTK_TO_NUMPY = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
}

ENCODINGS = [
    pytest.param("ascii", False, id="ascii"),
    pytest.param("binary", False, id="binary"),
    pytest.param("binary", True, id="binary-compressed"),
]


# ── Decoder helper ───────────────────────────────────────────────────────────


def _b64_len(nbytes: int) -> int:
    return 4 * ((nbytes + 2) // 3)


def _decode_compressed(text: str) -> tuple[bytes, list[int]]:
    """Decode a vtkZLibDataCompressor payload with a UInt64 header.

    Returns the raw bytes and the header ``[nblocks, blocksize, last_partial, sizes...]``.
    """
    first = base64.b64decode(text[: _b64_len(24)])
    nblocks = int(np.frombuffer(first[:8], dtype="<u8")[0])
    header_bytes = (3 + nblocks) * 8
    header_text_len = _b64_len(header_bytes)
    header = np.frombuffer(
        base64.b64decode(text[:header_text_len])[:header_bytes], dtype="<u8"
    ).tolist()
    blob = base64.b64decode(text[header_text_len:])
    sizes = header[3:]
    assert sum(sizes) == len(blob)
    out = bytearray()
    offset = 0
    for size in sizes:
        out += zlib.decompress(blob[offset : offset + size])
        offset += size
    return bytes(out), header


def decode_vti(path: Path) -> tuple[ET.Element, dict[str, np.ndarray]]:
    """Parse a .vti file and return its root element and ``(nZ, nY, nX)`` cell arrays."""
    root = ET.parse(path).getroot()
    assert root.get("byte_order") == "LittleEndian"
    image = root.find("ImageData")
    x0, x1, y0, y1, z0, z1 = map(int, image.get("WholeExtent").split())
    shape = (z1 - z0, y1 - y0, x1 - x0)

    arrays: dict[str, np.ndarray] = {}
    for da in root.iter("DataArray"):
        dt = np.dtype("<" + _VTK_TO_NUMPY[da.get("type")])
        text = "".join((da.text or "").split())
        fmt = da.get("format")
        if fmt == "ascii":
            values = np.array((da.text or "").split(), dtype=np.float64 if dt.kind == "f" else dt)
            arr = values.astype(dt)
        elif fmt == "binary":
            assert root.get("header_type") == "UInt64"
            if root.get("compressor") == "vtkZLibDataCompressor":
                raw, _ = _decode_compressed(text)
            else:
                assert root.get("compressor") is None
                decoded = base64.b64decode(text)
                nbytes = int(np.frombuffer(decoded[:8], dtype="<u8")[0])
                raw = decoded[8:]
                assert nbytes == len(raw)
            arr = np.frombuffer(raw, dtype=dt)
        else:
            raise AssertionError(f"unexpected format {fmt!r}")
        arrays[da.get("Name")] = arr.reshape(shape)
    return root, arrays


def _first_payload(path: Path) -> str:
    root = ET.parse(path).getroot()
    da = next(root.iter("DataArray"))
    return "".join((da.text or "").split())


# ── Round trips ──────────────────────────────────────────────────────────────


def _geometry(dims_xyz) -> VoxGeometry:
    return VoxGeometry(
        dims_xyz=np.array(dims_xyz),
        spacing_mm=np.array([0.5, 1.0, 2.0]),
        origin_mm=np.array([-1.0, -1.5, -2.0]),
    )


def _big_arrays(dtype) -> dict[str, np.ndarray]:
    nx, ny, nz = BIG_DIMS_XYZ
    rng = np.random.default_rng(1234)
    if np.issubdtype(dtype, np.floating):
        a = rng.standard_normal((nz, ny, nx)) * 1e3
        b = rng.random((nz, ny, nx))
    else:
        a = rng.integers(-(2**31), 2**31 - 1, size=(nz, ny, nx))
        b = np.arange(nz * ny * nx).reshape(nz, ny, nx)
    return {"Dose": a.astype(dtype), "Edep": b.astype(dtype)}


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32])
@pytest.mark.parametrize("encoding, compress", ENCODINGS)
def test_round_trip(tmp_path, dtype, encoding, compress):
    arrays = _big_arrays(dtype)
    out = write_vti(
        tmp_path / "rt.vti",
        _geometry(BIG_DIMS_XYZ),
        arrays,
        dtype=dtype,
        encoding=encoding,
        compress=compress,
    )
    root, decoded = decode_vti(out)
    assert list(decoded) == ["Dose", "Edep"]
    for name, arr in arrays.items():
        assert decoded[name].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(decoded[name], arr)
    for da in root.iter("DataArray"):
        assert da.get("format") == encoding


def test_binary_vtkfile_attributes(tmp_path):
    out = write_vti(tmp_path / "b.vti", _geometry(BIG_DIMS_XYZ), _big_arrays(np.float64))
    root = ET.parse(out).getroot()
    assert root.get("header_type") == "UInt64"
    assert root.get("compressor") is None

    out_c = write_vti(
        tmp_path / "c.vti", _geometry(BIG_DIMS_XYZ), _big_arrays(np.float64), compress=True
    )
    root_c = ET.parse(out_c).getroot()
    assert root_c.get("header_type") == "UInt64"
    assert root_c.get("compressor") == "vtkZLibDataCompressor"


def test_default_encoding_is_binary(tmp_path):
    out = write_vti(tmp_path / "d.vti", _geometry(BIG_DIMS_XYZ), _big_arrays(np.float32))
    root = ET.parse(out).getroot()
    assert {da.get("format") for da in root.iter("DataArray")} == {"binary"}


def test_uncompressed_payload_layout(tmp_path):
    arrays = _big_arrays(np.float64)
    out = write_vti(
        tmp_path / "u.vti", _geometry(BIG_DIMS_XYZ), arrays, dtype=np.float64, encoding="binary"
    )
    raw = base64.b64decode(_first_payload(out))
    data = arrays["Dose"].astype("<f8").tobytes(order="C")
    assert raw == np.array([len(data)], dtype="<u8").tobytes() + data


def test_compressed_header_with_partial_last_block(tmp_path):
    arrays = _big_arrays(np.float64)
    out = write_vti(
        tmp_path / "p.vti", _geometry(BIG_DIMS_XYZ), arrays, dtype=np.float64, compress=True
    )
    raw, header = _decode_compressed(_first_payload(out))
    nbytes = 196_800
    assert len(raw) == nbytes
    assert header[:3] == [7, BLOCK_SIZE, nbytes % BLOCK_SIZE]
    assert nbytes % BLOCK_SIZE == 192
    assert len(header) == 3 + 7

    data = arrays["Dose"].astype("<f8").tobytes(order="C")
    blocks = [data[i : i + BLOCK_SIZE] for i in range(0, nbytes, BLOCK_SIZE)]
    assert header[3:] == [len(zlib.compress(b)) for b in blocks]


def test_compressed_header_with_exactly_one_full_block(tmp_path):
    dims_xyz = (16, 16, 16)  # 4096 float64 values = 32768 bytes
    arr = np.arange(4096, dtype=np.float64).reshape(16, 16, 16) * 0.25
    out = write_vti(
        tmp_path / "f.vti", _geometry(dims_xyz), {"Dose": arr}, dtype=np.float64, compress=True
    )
    raw, header = _decode_compressed(_first_payload(out))
    assert len(raw) == BLOCK_SIZE
    assert header[:3] == [1, BLOCK_SIZE, 0]
    _, decoded = decode_vti(out)
    np.testing.assert_array_equal(decoded["Dose"], arr)


def test_big_endian_input_written_little_endian(tmp_path):
    arr = _big_arrays(np.float64)["Dose"].astype(">f8")
    out = write_vti(tmp_path / "be.vti", _geometry(BIG_DIMS_XYZ), {"Dose": arr}, dtype=np.float64)
    _, decoded = decode_vti(out)
    np.testing.assert_array_equal(decoded["Dose"], arr)


def test_compress_with_ascii_raises(tmp_path):
    with pytest.raises(ValueError):
        write_vti(
            tmp_path / "x.vti",
            _geometry(BIG_DIMS_XYZ),
            _big_arrays(np.float32),
            encoding="ascii",
            compress=True,
        )


def test_unknown_encoding_raises(tmp_path):
    with pytest.raises(ValueError):
        write_vti(
            tmp_path / "x.vti", _geometry(BIG_DIMS_XYZ), _big_arrays(np.float32), encoding="raw"
        )


@pytest.mark.parametrize("encoding, compress", ENCODINGS)
def test_overflow_checked_for_every_encoding(tmp_path, encoding, compress):
    arr = np.full((2, 3, 4), 1e300)
    with pytest.raises(OverflowError):
        write_vti(
            tmp_path / "o.vti",
            _geometry((4, 3, 2)),
            {"Dose": arr},
            dtype=np.float32,
            encoding=encoding,
            compress=compress,
        )


def test_ascii_output_unchanged(tmp_path):
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    out = write_vti(
        tmp_path / "a.vti", _geometry((4, 3, 2)), {"Dose": arr}, dtype=np.int32, encoding="ascii"
    )
    values = " ".join(str(i) for i in range(24))
    assert out.read_text(encoding="utf-8") == (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 4 0 3 0 2"\n'
        '             Origin="-1.0 -1.5 -2.0" Spacing="0.5 1.0 2.0">\n'
        '    <Piece Extent="0 4 0 3 0 2">\n'
        '      <CellData Scalars="Dose">\n'
        '      <DataArray type="Int32" Name="Dose" format="ascii">\n'
        f"        {values}\n"
        "      </DataArray>\n"
        "      </CellData>\n"
        "      <PointData/>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )


# ── Forwarding from G4VoxFileBase ────────────────────────────────────────────


@pytest.fixture
def sim(tmp_path) -> G4VoxFile3D:
    return G4VoxFile3D(write_snapshot3d(tmp_path / "snap.h5"))


@pytest.mark.parametrize("encoding, compress", ENCODINGS)
def test_to_vti_forwards_encoding(sim, tmp_path, encoding, compress):
    next(iter(sim))
    out = sim.to_vti(tmp_path / "to.vti", dtype=np.float64, encoding=encoding, compress=compress)
    root, decoded = decode_vti(out)
    assert {da.get("format") for da in root.iter("DataArray")} == {encoding}
    assert (root.get("compressor") is not None) == compress
    for name, arr in sim.data.items():
        np.testing.assert_array_equal(decoded[name], arr)


@pytest.mark.parametrize("encoding, compress", ENCODINGS)
def test_dump_selection_forwards_encoding(sim, tmp_path, encoding, compress):
    out = sim.dump_selection_to_vti(
        tmp_path / "sel.vti", dtype=np.float64, encoding=encoding, compress=compress
    )
    root, decoded = decode_vti(out)
    assert {da.get("format") for da in root.iter("DataArray")} == {encoding}
    assert (root.get("compressor") is not None) == compress
    for qi, name in enumerate(QUANTITIES):
        expected = sum(expected_array(qi, sid) for sid in (0, 1, 2))
        np.testing.assert_array_equal(decoded[name], expected)


@pytest.mark.parametrize("encoding, compress", ENCODINGS)
def test_timeseries_forwards_encoding(sim, tmp_path, encoding, compress):
    pvd = sim.dump_selection_to_vti_timeseries(
        tmp_path / "ts" / "series.pvd", dtype=np.float64, encoding=encoding, compress=compress
    )
    files = [ds.get("file") for ds in ET.parse(pvd).getroot().iter("DataSet")]
    assert len(files) == 3
    for sid, fname in zip((0, 1, 2), files):
        root, decoded = decode_vti(pvd.parent / fname)
        assert {da.get("format") for da in root.iter("DataArray")} == {encoding}
        assert (root.get("compressor") is not None) == compress
        np.testing.assert_array_equal(decoded["Dose"], expected_array(0, sid))


# ── Cross-check with VTK's own reader ────────────────────────────────────────


@pytest.mark.parametrize("compress", [False, True], ids=["binary", "binary-compressed"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32])
def test_vtk_reader_reads_binary(tmp_path, compress, dtype):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    arrays = _big_arrays(dtype)
    out = write_vti(
        tmp_path / "vtk.vti",
        _geometry(BIG_DIMS_XYZ),
        arrays,
        dtype=dtype,
        encoding="binary",
        compress=compress,
    )
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(out))
    reader.Update()
    image = reader.GetOutput()
    nx, ny, nz = BIG_DIMS_XYZ
    assert image.GetNumberOfCells() == nx * ny * nz
    cell_data = image.GetCellData()
    for name, arr in arrays.items():
        got = vtk_to_numpy(cell_data.GetArray(name)).reshape(nz, ny, nx)
        np.testing.assert_array_equal(got, arr)


# ── MesoSpatialSnapshot.to_vti ───────────────────────────────────────────────


def _meso_snapshot():
    from g4utils.HDF5 import MesoSpatialSnapshot

    pos = np.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0], [5.0, 15.0, 25.0]])
    counts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint32)
    return MesoSpatialSnapshot(
        run=0, event=0, index=0, time_ns=1.0, cell_size_nm=10.0,
        position_nm=pos, counts=counts, species=["°OH^0", "H3O^1"],
    )


def test_meso_to_vti_round_trip(tmp_path):
    snap = _meso_snapshot()
    out = snap.to_vti(tmp_path / "meso.vti", dtype=np.float64)
    assert out == tmp_path / "meso.vti"
    root, arrays = decode_vti(out)
    assert set(arrays) == {"°OH^0_count", "°OH^0_M", "H3O^1_count", "H3O^1_M"}
    image = root.find("ImageData")
    assert [float(v) for v in image.get("Origin").split()] == [0.0, 0.0, 0.0]
    assert [float(v) for v in image.get("Spacing").split()] == [10.0, 10.0, 10.0]
    assert image.get("WholeExtent") == "0 2 0 2 0 3"
    conc = snap.concentration_M()
    cells = [(0, 0, 0), (0, 0, 1), (2, 1, 0)]  # (z, y, x)
    for row, (z, y, x) in enumerate(cells):
        for col, sp in enumerate(snap.species):
            assert arrays[f"{sp}_count"][z, y, x] == snap.counts[row, col]
            np.testing.assert_allclose(arrays[f"{sp}_M"][z, y, x], conc[row, col])
    assert arrays["H3O^1_count"].sum() == 12
    assert arrays["H3O^1_M"][1, 1, 1] == 0


def test_meso_to_vti_species_subset_and_unknown(tmp_path):
    snap = _meso_snapshot()
    _, arrays = decode_vti(snap.to_vti(tmp_path / "a.vti", species=["H3O^1"], encoding="ascii"))
    assert set(arrays) == {"H3O^1_count", "H3O^1_M"}
    with pytest.raises(KeyError):
        snap.to_vti(tmp_path / "b.vti", species=["nope"])


def test_meso_to_vti_extent(tmp_path):
    snap = _meso_snapshot()
    root, arrays = decode_vti(
        snap.to_vti(tmp_path / "e.vti", extent=([-10, 0, 0], [30, 20, 30]), compress=True)
    )
    image = root.find("ImageData")
    assert [float(v) for v in image.get("Origin").split()] == [-10.0, 0.0, 0.0]
    assert arrays["H3O^1_count"].shape == (3, 2, 4)

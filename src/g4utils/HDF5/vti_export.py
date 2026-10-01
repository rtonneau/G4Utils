from __future__ import annotations

import base64
import zlib
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from g4utils.Vox.vox_geometry import VoxGeometry


def _vtk_scalar_type(dtype: np.dtype) -> str:
    dt = np.dtype(dtype)
    mapping = {
        np.dtype(np.int8): "Int8",
        np.dtype(np.uint8): "UInt8",
        np.dtype(np.int16): "Int16",
        np.dtype(np.uint16): "UInt16",
        np.dtype(np.int32): "Int32",
        np.dtype(np.uint32): "UInt32",
        np.dtype(np.int64): "Int64",
        np.dtype(np.uint64): "UInt64",
        np.dtype(np.float32): "Float32",
        np.dtype(np.float64): "Float64",
    }
    if dt not in mapping:
        raise TypeError(f"Unsupported dtype for VTI export: {dt}")
    return mapping[dt]


_ZLIB_BLOCK_SIZE = 32768
_UINT64_LE = np.dtype("<u8")


def _format_data_array(name: str, arr: np.ndarray, vtk_type: str) -> str:
    flat = np.asarray(arr).ravel(order="C")
    values = " ".join(map(str, flat.tolist()))
    return (
        f'      <DataArray type="{vtk_type}" Name="{name}" format="ascii">\n'
        f"        {values}\n"
        f"      </DataArray>"
    )


def _encode_binary(data: bytes, compress: bool) -> str:
    """
    Encode raw bytes as a VTK XML inline-binary payload (``header_type="UInt64"``).

    Uncompressed: ``base64(uint64 nbytes + data)``.
    Compressed (``vtkZLibDataCompressor``): ``base64(header) + base64(blocks)``, where
    ``header = uint64 [nblocks, blocksize, last_partial_size, compressed sizes...]`` and
    ``last_partial_size`` is 0 when the last block is full.
    """
    nbytes = len(data)
    if not compress:
        header = np.array([nbytes], dtype=_UINT64_LE).tobytes()
        return base64.b64encode(header + data).decode("ascii")

    blocks = [
        zlib.compress(data[i : i + _ZLIB_BLOCK_SIZE])
        for i in range(0, nbytes, _ZLIB_BLOCK_SIZE)
    ]
    header = np.array(
        [len(blocks), _ZLIB_BLOCK_SIZE, nbytes % _ZLIB_BLOCK_SIZE]
        + [len(b) for b in blocks],
        dtype=_UINT64_LE,
    ).tobytes()
    return (
        base64.b64encode(header).decode("ascii")
        + base64.b64encode(b"".join(blocks)).decode("ascii")
    )


def _format_binary_data_array(
    name: str, arr: np.ndarray, vtk_type: str, compress: bool
) -> str:
    a = np.asarray(arr)
    data = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes(order="C")
    return (
        f'      <DataArray type="{vtk_type}" Name="{name}" format="binary">\n'
        f"        {_encode_binary(data, compress)}\n"
        f"      </DataArray>"
    )


def _cast_for_vti(
    arr: np.ndarray,
    target_dtype: np.dtype,
    name: str,
) -> np.ndarray:
    """Cast with explicit overflow checks to avoid RuntimeWarning spam."""
    a = np.asarray(arr)

    if np.issubdtype(target_dtype, np.floating):
        finfo = np.finfo(target_dtype)
        finite = (
            a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
        )
        if finite.size and (
            finite.min() < finfo.min or finite.max() > finfo.max
        ):
            raise OverflowError(
                f"Array '{name}' cannot be represented as {target_dtype}. "
                "Use a wider dtype such as np.float64."
            )

    elif np.issubdtype(target_dtype, np.integer):
        iinfo = np.iinfo(target_dtype)
        if np.issubdtype(a.dtype, np.floating):
            finite = a[np.isfinite(a)]
            if finite.size and (
                finite.min() < iinfo.min or finite.max() > iinfo.max
            ):
                raise OverflowError(
                    f"Array '{name}' cannot be represented as {target_dtype}."
                )
        else:
            if a.size and (a.min() < iinfo.min or a.max() > iinfo.max):
                raise OverflowError(
                    f"Array '{name}' cannot be represented as {target_dtype}."
                )

    return a.astype(target_dtype, copy=False)


def write_vti(
    filepath: str | Path,
    geometry: VoxGeometry,
    cell_arrays: dict[str, np.ndarray],
    dtype: npt.DTypeLike = np.float32,
    *,
    encoding: Literal["binary", "ascii"] = "binary",
    compress: bool = False,
) -> Path:
    """
    Write cell-centered 3D arrays (nZ,nY,nX) to a VTK ImageData (.vti) file.

    Parameters
    ----------
    filepath : str or Path
        Output ``.vti`` path. Parent directories are created.
    geometry : VoxGeometry
        Grid dimensions, spacing and origin.
    cell_arrays : dict of str to ndarray
        Arrays of shape ``(nZ, nY, nX)``, one per quantity.
    dtype : numpy dtype, default np.float32
        Output data type. Values that do not fit raise ``OverflowError``.
    encoding : {"binary", "ascii"}, default "binary"
        ``"binary"`` writes base64 inline data with a UInt64 header;
        ``"ascii"`` writes space-separated text.
    compress : bool, default False
        Compress binary data with zlib (``vtkZLibDataCompressor``, 32768-byte
        blocks). Only valid with ``encoding="binary"``.

    Notes
    -----
    - Arrays are exported as CellData.
        - VTK extent is encoded as cell extent, so WholeExtent is
            [0..nx, 0..ny, 0..nz].
    - Binary data is little-endian, in C order of ``(nZ, nY, nX)``.
    """
    if encoding not in ("binary", "ascii"):
        raise ValueError(f"encoding must be 'binary' or 'ascii', got {encoding!r}")
    if compress and encoding != "binary":
        raise ValueError("compress=True requires encoding='binary'")

    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    expected_shape = (nz, ny, nx)
    target_dtype = np.dtype(dtype)
    vtk_type = _vtk_scalar_type(target_dtype)

    if not cell_arrays:
        raise ValueError("cell_arrays must contain at least one quantity")

    arrays_xml: list[str] = []
    active_scalar: str | None = None
    for name, arr in cell_arrays.items():
        a = np.asarray(arr)
        if a.shape != expected_shape:
            raise ValueError(
                f"Array '{name}' has shape {a.shape}, "
                f"expected {expected_shape}"
            )
        if active_scalar is None:
            active_scalar = name
        cast = _cast_for_vti(a, target_dtype, name)
        if encoding == "binary":
            arrays_xml.append(_format_binary_data_array(name, cast, vtk_type, compress))
        else:
            arrays_xml.append(_format_data_array(name, cast, vtk_type))

    origin = " ".join(
        map(str, np.asarray(geometry.origin_mm, dtype=float).tolist())
    )
    spacing = " ".join(
        map(str, np.asarray(geometry.spacing_mm, dtype=float).tolist())
    )
    whole_extent = f"0 {nx} 0 {ny} 0 {nz}"

    vtkfile_attrs = 'type="ImageData" version="0.1" byte_order="LittleEndian"'
    if encoding == "binary":
        vtkfile_attrs += ' header_type="UInt64"'
        if compress:
            vtkfile_attrs += ' compressor="vtkZLibDataCompressor"'

    xml = (
        '<?xml version="1.0"?>\n'
        f"<VTKFile {vtkfile_attrs}>\n"
        f'  <ImageData WholeExtent="{whole_extent}"\n'
        f'             Origin="{origin}" Spacing="{spacing}">\n'
        f'    <Piece Extent="{whole_extent}">\n'
        f'      <CellData Scalars="{active_scalar}">\n'
        + "\n".join(arrays_xml)
        + "\n"
        + "      </CellData>\n"
        + "      <PointData/>\n"
        + "    </Piece>\n"
        + "  </ImageData>\n"
        + "</VTKFile>\n"
    )

    out.write_text(xml, encoding="utf-8")
    return out


def write_pvd_collection(
    filepath: str | Path,
    datasets: list[tuple[float, str]],
) -> Path:
    """Write a ParaView .pvd collection referencing timestep files."""
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        (
            f'    <DataSet timestep="{time}" group="" part="0" '
            f'file="{filename}"/>'
        )
        for time, filename in datasets
    ]

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n"
        + "\n".join(entries)
        + "\n"
        + "  </Collection>\n"
        + "</VTKFile>\n"
    )

    out.write_text(xml, encoding="utf-8")
    return out

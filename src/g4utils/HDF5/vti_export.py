from __future__ import annotations

import typing as tp
from pathlib import Path

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


def _format_data_array(name: str, arr: np.ndarray, vtk_type: str) -> str:
    flat = np.asarray(arr).ravel(order="C")
    values = " ".join(map(str, flat.tolist()))
    return (
        f'      <DataArray type="{vtk_type}" Name="{name}" format="ascii">\n'
        f"        {values}\n"
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
) -> Path:
    """
    Write cell-centered 3D arrays (nZ,nY,nX) to a VTK ImageData (.vti) file.

    Notes
    -----
    - Arrays are exported as CellData.
        - VTK extent is encoded as cell extent, so WholeExtent is
            [0..nx, 0..ny, 0..nz].
    """
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
        arrays_xml.append(
            _format_data_array(
                name, _cast_for_vti(a, target_dtype, name), vtk_type
            )
        )

    origin = " ".join(
        map(str, np.asarray(geometry.origin_mm, dtype=float).tolist())
    )
    spacing = " ".join(
        map(str, np.asarray(geometry.spacing_mm, dtype=float).tolist())
    )
    whole_extent = f"0 {nx} 0 {ny} 0 {nz}"

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
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


def select_quantities(
    available: tp.Iterable[str],
    quantities: tp.Iterable[str] | None,
) -> list[str]:
    avail = list(available)
    if quantities is None:
        return avail
    wanted = list(quantities)
    missing = set(wanted) - set(avail)
    if missing:
        raise KeyError(f"Quantities not found: {sorted(missing)}")
    return [q for q in avail if q in wanted]

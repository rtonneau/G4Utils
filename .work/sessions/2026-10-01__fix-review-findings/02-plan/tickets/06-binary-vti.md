# Ticket 06: binary-vti

**Model:** opus

**Acceptance Criteria:**
- [ ] `write_vti(filepath, geometry, cell_arrays, dtype=np.float32, *, encoding: Literal["binary", "ascii"] = "binary", compress: bool = False) -> Path`.
- [ ] `encoding="ascii"` output is unchanged from today. `compress=True` with `"ascii"` raises `ValueError`.
- [ ] Binary: the `VTKFile` element gets `header_type="UInt64"`, and each `DataArray` has `format="binary"`. Uncompressed payload is `base64(uint64_le(nbytes) + data_le_bytes)`.
- [ ] Compressed: the `VTKFile` element also gets `compressor="vtkZLibDataCompressor"`. Data is split into 32768-byte blocks, each compressed with `zlib.compress`. The header is uint64 `[nblocks, 32768, last_partial_size (nbytes % 32768, i.e. 0 when the last block is full), compressed_size_1, ...]`. The payload text is `base64(header) + base64(concatenated compressed blocks)`, encoded separately.
- [ ] Array bytes are little-endian, in C order of `(nZ, nY, nX)`.
- [ ] `G4VoxFileBase.to_vti`, `dump_selection_to_vti` and `dump_selection_to_vti_timeseries` accept and forward keyword-only `encoding` and `compress`.
- [ ] Tests include a decoder helper (ElementTree + base64 + zlib, following the layout above). Round trips must cover float64, float32 and int32 in `ascii`, `binary` and `binary+compress`, using dims `(41, 30, 20)` float64 (196,800 bytes: several blocks plus a partial last block). Another test uses an array of exactly 32768 bytes (partial size 0).
- [ ] A test using `pytest.importorskip("vtk")` reads binary and compressed files with `vtkXMLImageDataReader` and compares the cell arrays.

**Files to Touch:**
- `src/g4utils/HDF5/vti_export.py`
- `src/g4utils/HDF5/vox_file_base.py`
- `tests/test_vti_export.py` (create)

**Verification Step:**

Run:
```bash
pytest tests/test_vti_export.py -v && pytest
```

Expected:
All pass. The vtk test may show as skipped if `vtk` isn't installed.

**Notes:**

Keep `_cast_for_vti` overflow checks in front of every encoding. A float64 array with 4096 values is exactly 32768 bytes.

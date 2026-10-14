"""Independent inputs and bit comparisons for storage tests; no production helpers."""

import hashlib

import numpy as np


def assert_bits(actual, expected):
    """Compare shape, dtype, and every bit, including NaN payloads and signed zero."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    uint = np.uint64 if expected.dtype.itemsize == 8 else np.uint32
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(uint),
        np.ascontiguousarray(expected).view(uint),
    )


def matrix_case(kind, n=4, windows=7, dtype=np.float64):
    """Construct each relationship directly, preserving distinct channel/window values."""
    dtype = np.dtype(dtype)
    data = np.empty((n, n, windows), dtype=dtype)
    for i in range(n):
        data[i, i] = np.arange(windows) + 0.125 + i
        for j in range(i + 1, n):
            upper = np.asarray(100 * i + 10 * j + np.arange(windows) + 0.25, dtype=dtype)
            data[i, j] = upper
            if kind.startswith("anti"):
                data[j, i] = -upper
            elif kind == "full":
                data[j, i] = -upper * 1.375 - 2
            else:
                data[j, i] = upper
    if kind.endswith("_res") and n > 1:
        data[1, 0, -1] = np.nextafter(data[1, 0, -1], dtype.type(np.inf))
    return data


def special_values(dtype):
    """Explicit IEEE patterns, including quiet/signaling NaNs of both signs."""
    if np.dtype(dtype) == np.dtype(np.float64):
        patterns = [0, 1 << 63, 1, (1 << 63) | 1, 0x7FEFFFFFFFFFFFFF,
                    0x7FF0000000000000, 0xFFF0000000000000,
                    0x7FF8000000000000, 0x7FF8000000000123,
                    0xFFF8000000000456, 0x7FF0000000000001]
        return np.array(patterns, dtype=np.uint64).view(np.float64)
    patterns = [0, 1 << 31, 1, (1 << 31) | 1, 0x7F7FFFFF,
                0x7F800000, 0xFF800000, 0x7FC00000, 0x7FC00123,
                0xFFC00456, 0x7F800001]
    return np.array(patterns, dtype=np.uint32).view(np.float32)


def tree_fingerprint(path):
    """Detect file creation/removal, byte changes, or mtime changes, ignoring read atime."""
    return {
        str(p.relative_to(path)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in path.rglob("*") if p.is_file()
    }


def data_files(array_path):
    return sorted(p for p in (array_path / "c").rglob("*") if p.is_file())

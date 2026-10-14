"""Lossless matrix representation invariants, with independently constructed inputs."""

import numpy as np
import pytest

from storage_cases import assert_bits, matrix_case, special_values


@pytest.mark.parametrize("kind", ["full", "sym", "anti", "sym_res", "anti_res"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_every_exact_mode_preserves_entries(packing, kind, dtype):
    data = matrix_case(kind, dtype=dtype)
    packed = packing.pack(data, residual_ok=lambda residual, full: kind != "full")
    assert packed.mode == kind
    assert_bits(packing.unpack(packed), data)
    assert_bits(packing.unpack(packed, windows=[6, 0, 6, 2]), data[:, :, [6, 0, 6, 2]])
    assert_bits(packing.unpack(packed, windows=np.array([], dtype=int)), data[:, :, :0])


@pytest.mark.parametrize("n,windows", [(1, 1), (1, 9), (2, 1), (7, 3)])
def test_small_shapes_and_full_mode(packing, n, windows):
    data = matrix_case("full", n=n, windows=windows)
    assert_bits(packing.unpack(packing.pack(data, structure="full")), data)
    assert_bits(packing.unpack(packing.pack(data)), data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("relationship", ["sym", "anti", "sym_res", "anti_res"])
def test_special_bits_in_both_triangles_and_diagonal(packing, dtype, relationship):
    values = special_values(dtype)
    data = matrix_case(relationship, n=3, windows=len(values), dtype=dtype)
    data[0, 0] = values
    data[0, 1] = values
    lower = values.copy()
    if relationship.startswith("anti"):
        nonzero_not_nan = ~np.isnan(values) & (values != 0)
        lower[nonzero_not_nan] = -values[nonzero_not_nan]
    data[1, 0] = lower
    with np.errstate(invalid="ignore", over="ignore"):
        packed = packing.pack(data)
        assert_bits(packing.unpack(packed), data)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [1, 13, 42])
def test_arbitrary_float_bits_round_trip(packing, dtype, seed):
    uint = np.uint64 if dtype == np.float64 else np.uint32
    bits = np.random.default_rng(seed).integers(0, np.iinfo(uint).max, size=(5, 5, 9), dtype=uint)
    with np.errstate(invalid="ignore", over="ignore"):
        assert_bits(packing.unpack(packing.pack(bits.view(dtype))), bits.view(dtype))


@pytest.mark.parametrize("layout", ["fortran", "sliced", "reversed"])
def test_noncontiguous_inputs_preserve_orientation(packing, layout):
    data = matrix_case("full", n=5, windows=8)
    if layout == "fortran":
        data = np.asfortranarray(data)
    elif layout == "sliced":
        data = data[::2, ::2, ::2]
    else:
        data = data[::-1, ::-1, ::-1]
    assert not data.flags.c_contiguous
    assert_bits(packing.unpack(packing.pack(data)), data)


@pytest.mark.parametrize("shape", [(4,), (2, 2), (2, 3, 4), (2, 2, 1, 1)])
def test_rejects_nonmatrix_time_shapes(packing, shape):
    with pytest.raises(ValueError, match="expected"):
        packing.pack(np.ones(shape))


@pytest.mark.parametrize("dtype", [np.int64, np.complex128, np.float16])
def test_rejects_unsupported_payload_dtype(packing, dtype):
    with pytest.raises(ValueError, match="unsupported dtype"):
        packing.pack(np.ones((2, 2, 3), dtype=dtype))


@pytest.mark.parametrize("upper,lower", [(np.inf, 7.0), (-np.inf, 7.0)])
def test_exact_packing_preserves_incompatible_infinities(packing, upper, lower):
    data = np.zeros((2, 2, 1))
    data[0, 1, 0], data[1, 0, 0] = upper, lower
    assert_bits(packing.unpack(packing.pack(data)), data)


def test_lossy_structure_is_not_supported(packing):
    with pytest.raises(ValueError, match="structure must be exact or full"):
        packing.pack(matrix_case("sym"), structure="tol")

"""Unit tests for pyspi.base: matrix orientation, symmetrisation and input parsing."""
import numpy as np
import pytest

from pyspi.base import Directed, Undirected, parse_bivariate, parse_multivariate
from pyspi.data import Data


class ToyDirected(Directed):
    identifier = "toy_directed"
    labels = ["toy"]

    def bivariate(self, data, i=None, j=None):  # undecorated: multivariate passes i, j
        return 10 * i + j


class ToyUndirected(Undirected):
    identifier = "toy_undirected"
    labels = ["toy"]

    def bivariate(self, data, i=None, j=None):
        return 10 * i + j


class ParsedBivariate(Directed):
    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        return data, i, j


class CachingMultivariate(Directed):
    @parse_multivariate
    def multivariate(self, data):
        data.toy_cache = "written"
        return data


def _data(n_processes=4, n_observations=50, seed=0):
    x = np.random.default_rng(seed).standard_normal((n_processes, n_observations))
    return Data(x)


def test_directed_orientation_row_source_col_target():
    A = ToyDirected().multivariate(_data(4))
    assert A.shape == (4, 4)
    assert np.isnan(np.diag(A)).all()
    for i in range(4):
        for j in range(4):
            if i != j:
                assert A[i, j] == 10 * i + j


def test_undirected_copies_upper_triangle_to_lower():
    A = ToyUndirected().multivariate(_data(4))
    assert np.isnan(np.diag(A)).all()
    for i in range(4):
        for j in range(i + 1, 4):
            assert A[i, j] == 10 * i + j
            assert A[j, i] == 10 * i + j
    np.testing.assert_array_equal(A, A.T)


def test_parse_bivariate_accepts_two_1d_arrays():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(50), rng.standard_normal(50)
    data, i, j = ParsedBivariate().bivariate(x, y)
    assert isinstance(data, Data)
    assert data.n_processes == 2
    assert (i, j) == (0, 1)


def test_parse_bivariate_rejects_single_array():
    x = np.random.default_rng(2).standard_normal(50)
    with pytest.raises(TypeError, match="two 1D-array inputs"):
        ParsedBivariate().bivariate(x)


def test_parse_multivariate_accepts_list_of_arrays():
    rng = np.random.default_rng(3)
    procs = [rng.standard_normal(50) for _ in range(3)]
    A = ToyDirected().multivariate(procs)
    assert A.shape == (3, 3)
    assert np.isnan(np.diag(A)).all()
    assert A[2, 0] == 20 and A[0, 2] == 2


def test_inplace_false_deep_copies():
    original = _data(3, seed=4)
    before = original.to_numpy().copy()

    seen = CachingMultivariate().multivariate(original, inplace=False)
    assert seen is not original
    # Deep, not shallow: the underlying array buffer is not shared.
    assert not np.shares_memory(seen._data, original._data)
    assert seen.toy_cache == "written"
    assert not hasattr(original, "toy_cache")
    np.testing.assert_array_equal(original.to_numpy(), before)

    # Contrast: the default inplace=True works on the caller's object.
    seen = CachingMultivariate().multivariate(original)
    assert seen is original
    assert original.toy_cache == "written"


def test_get_group_asserts_on_subset_classes():
    with pytest.raises(AssertionError, match="is a subset of class"):
        ToyDirected().get_group([["toy"], ["toy", "other"]])


def test_two_array_preprocessing_matches_complete_data():
    raw = np.random.default_rng(11).normal(size=(2, 1024))
    raw = raw * np.array([[3.], [20.]]) + np.array([[12.], [-90.]])
    before = raw.copy()
    seen, i, j = ParsedBivariate().bivariate(raw[0], raw[1])
    np.testing.assert_array_equal(seen.to_numpy(), Data(raw).to_numpy())
    np.testing.assert_array_equal(raw, before)
    assert (i, j) == (0, 1)


@pytest.mark.parametrize("value", [np.nan, np.inf, 3.])
def test_invalid_second_array_is_rejected(value):
    raw = np.random.default_rng(12).normal(size=(2, 1024))
    if np.isfinite(value):
        raw[1] = value
    else:
        raw[1, 17] = value
    with pytest.raises(ValueError):
        ParsedBivariate().bivariate(raw[0], raw[1])


def test_raw_multivariate_matches_data_for_scale_sensitive_metric():
    from pyspi.statistics.basic import Covariance
    raw = np.random.default_rng(13).normal(size=(3, 1024))
    raw = raw * np.array([[1.], [20.], [.001]]) + np.array([[12.], [-90.], [3.]])
    np.testing.assert_array_equal(
        Covariance().multivariate(list(raw)), Covariance().multivariate(Data(raw))
    )

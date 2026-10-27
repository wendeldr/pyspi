"""Spearman's statistical definition, input contract, and matrix integration."""

import numpy as np
import pytest
from scipy import stats

from pyspi.data import Data
from pyspi.statistics.basic import SpearmanR


@pytest.mark.parametrize("shape", [(1, 8), (2, 3), (5, 37), (8, 129)])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("squared", [False, True])
def test_matrix_matches_scipy_pairwise(shape, tied, squared):
    raw = np.random.default_rng(72).normal(size=shape)
    if tied:
        raw = np.round(raw, 1)
    expected = np.full((shape[0], shape[0]), np.nan)
    for i in range(shape[0]):
        for j in range(i + 1, shape[0]):
            value = stats.spearmanr(raw[i], raw[j]).statistic
            expected[i, j] = expected[j, i] = value ** 2 if squared else value

    data = Data(raw, normalise=False)
    actual = SpearmanR(squared=squared).multivariate(data)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual, actual.T)
    assert np.isnan(np.diag(actual)).all()
    np.testing.assert_array_equal(data.to_numpy(realisation=0), raw)


@pytest.mark.parametrize("squared", [False, True])
def test_hand_calculated_ties_and_monotonic_transformations(squared):
    # Average ranks [1, 2.5, 2.5, 4] and [4, 1.5, 1.5, 3] give rho=-1/3.
    x = np.array([1., 2., 2., 4.])
    raw = np.stack([x, [4., 1., 1., 2.], x ** 3, -x])
    actual = SpearmanR(squared=squared).multivariate(Data(raw, normalise=False))
    expected = np.array([-1 / 3, 1., -1.])
    if squared:
        expected **= 2
    np.testing.assert_allclose(actual[0, 1:], expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("squared", [False, True])
def test_constant_channels_only_invalidate_their_own_pairs(squared):
    raw = np.array([[3., 3., 3., 3.], [1., 2., 2., 4.],
                    [4., 1., 1., 2.], [0., 0., 0., 0.]])
    data = Data(raw, normalise=False)
    spi = SpearmanR(squared=squared)
    actual = spi.multivariate(data)
    assert np.isnan(actual[[0, 3], :]).all()
    assert np.isnan(actual[:, [0, 3]]).all()
    np.testing.assert_allclose(actual[1, 2], 1 / 9 if squared else -1 / 3)
    assert np.isnan(spi.bivariate(data, i=0, j=1))


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("normalise,detrend", [(False, False), (True, False),
                                               (False, True), (True, True)])
def test_pair_and_matrix_apis_agree_after_preprocessing(squared, normalise, detrend):
    raw = np.random.default_rng(73).normal(size=(4, 91))
    raw *= np.array([1., 7., .02, 2.])[:, None]
    raw += np.arange(91)[None, :] * np.array([.1, -.2, .04, 0.])[:, None]
    data = Data(raw, normalise=normalise, detrend=detrend)
    spi = SpearmanR(squared=squared)
    matrix = spi.multivariate(data)
    for i, j in zip(*np.triu_indices(4, 1)):
        np.testing.assert_allclose(spi.bivariate(data, i=i, j=j), matrix[i, j],
                                   rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(spi.bivariate(data, i=j, j=i), matrix[i, j],
                                   rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("squared", [False, True])
def test_direct_arrays_match_data_and_do_not_modify_input(squared):
    raw = np.random.default_rng(74).normal(size=(2, 61))
    raw = raw * np.array([[3.], [.01]]) + np.array([[20.], [-9.]])
    before = raw.copy()
    spi = SpearmanR(squared=squared)
    expected = spi.multivariate(Data(raw))[0, 1]
    np.testing.assert_allclose(spi.bivariate(raw[0], raw[1]), expected, atol=1e-14)
    np.testing.assert_allclose(spi.multivariate(raw)[0, 1], expected, atol=1e-14)
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("squared", [False, True])
def test_one_observation_has_undefined_correlation(squared):
    spi = SpearmanR(squared=squared)
    data = Data(np.array([[1.], [2.], [3.]]), normalise=False)
    assert np.isnan(spi.multivariate(data)).all()
    assert np.isnan(spi.bivariate(data, i=0, j=1))


@pytest.mark.parametrize("method", ["bivariate", "multivariate"])
def test_multiple_realisations_are_explicitly_rejected(method):
    data = Data(np.random.default_rng(75).normal(size=(2, 32, 2)), dim_order="psr")
    with pytest.raises(ValueError, match="requires a single realisation"):
        getattr(SpearmanR(), method)(data)


def test_labels_are_independent_for_each_instance():
    original = SpearmanR.labels.copy()
    plain, squared, another = SpearmanR(), SpearmanR(squared=True), SpearmanR()
    assert plain.identifier == "spearmanr"
    assert squared.identifier == "spearmanr-sq"
    assert plain.issigned() and not squared.issigned()
    assert plain.labels == original + ["signed"]
    assert squared.labels == original + ["unsigned"]
    squared.labels.append("only-this-instance")
    assert another.labels == plain.labels == original + ["signed"]
    assert SpearmanR.labels == original


def test_repeated_calls_use_current_data_and_return_independent_results():
    raw = np.random.default_rng(76).normal(size=(3, 43))
    data = Data(raw, normalise=False)
    spi = SpearmanR()
    original = spi.multivariate(data)
    expected = original.copy()
    original[0, 1] = 999.
    np.testing.assert_array_equal(spi.multivariate(data), expected)
    raw[1] = -raw[0]
    data.set_data(raw, dim_order="ps")
    np.testing.assert_allclose(spi.multivariate(data)[0, 1], -1., atol=1e-14)

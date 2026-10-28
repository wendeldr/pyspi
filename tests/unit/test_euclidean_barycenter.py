"""Euclidean summaries, operation order and fixed-window redundancies."""

import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.distance import Barycenter


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("statistic", ["mean", "max", "max_time"])
def test_barycenter_summaries_against_explicit_time_series(squared, statistic):
    x = np.random.default_rng(1101).normal(size=(5, 87))
    x[0] -= 10
    x[2] = -x[1] + 1e-8*x[2]
    expected = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            barycenter = np.average(np.stack([x[i], x[j]]), axis=0)
            if squared:
                barycenter = barycenter**2
            expected[i, j] = {"mean": np.mean, "max": np.max, "max_time": np.argmax}[statistic](barycenter)
    spi = Barycenter(statistic=statistic, squared=squared)
    data = Data(x, normalise=False)
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_allclose(spi.bivariate(data, i=3, j=1), actual[3, 1])
    actual[:] = 10
    np.testing.assert_allclose(spi.multivariate(data), expected, atol=1e-14)
    data.to_numpy()[0, :, 0] += 20
    assert spi.multivariate(data)[0, 1] != expected[0, 1] or statistic == "max_time"
    data.to_numpy()[1, 3, 0] = np.nan
    invalid = spi.multivariate(data)
    assert np.isnan(invalid[1]).all() and np.isnan(invalid[:, 1]).all()
    np.testing.assert_allclose(invalid[2:, 2:], expected[2:, 2:], atol=1e-14)


def test_barycenter_square_before_max_is_not_square_after_max():
    x = np.array([[-1.8, .4], [0., 0.]])
    data = Data(x, normalise=False)
    assert Barycenter(statistic="max").bivariate(data) == .2
    assert Barycenter(statistic="max", squared=True).bivariate(data) == .81


def test_redundant_means_and_ridge_remain_redundant_after_window_averaging():
    from pyspi.statistics.misc import LinearModel
    rng = np.random.default_rng(1102)
    n = 1024
    covariance, plain, squared, ridge, weak_ridge = [], [], [], [], []
    for strength in (.1, .8, -.5):
        x = rng.normal(size=(3, n))
        x[1] += strength*x[0]
        data = Data(x)
        z = data.to_numpy(realisation=0)
        cov = z @ z.T/n
        covariance.append(cov)
        plain.append(Barycenter().multivariate(data))
        squared.append(Barycenter(squared=True).multivariate(data))
        ridge.append(LinearModel("Ridge").multivariate(data))
        weak_ridge.append(LinearModel("Ridge", alpha=.0001, penalty_scale="mean").multivariate(data))
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(np.mean(plain, axis=0)[off], 0, atol=1e-15)
    np.testing.assert_allclose(np.mean(squared, axis=0)[off],
                               ((n-1)/(2*n)+np.mean(covariance, axis=0)/2)[off], atol=1e-14)
    expected = (n-1)/n-(n+1)/n*np.mean(np.square(covariance), axis=0)
    np.testing.assert_allclose(np.mean(ridge, axis=0)[off], expected[off], atol=1e-14)
    v = (n-1)/n
    expected = v-(v+.0002)/(v+.0001)**2*np.mean(np.square(covariance), axis=0)
    np.testing.assert_allclose(np.mean(weak_ridge, axis=0)[off], expected[off], atol=1e-14)

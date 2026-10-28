"""Banded DTW against a full-grid dynamic-programming reference."""

import numpy as np
import pytest
from tslearn.metrics import dtw

from pyspi.data import Data
from pyspi.statistics.distance import DynamicTimeWarping


def full_grid(x, y, radius):
    grid = np.full((len(x) + 1, len(y) + 1), np.inf)
    grid[0, 0] = 0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if abs(i - j) <= radius:
                grid[i, j] = (x[i-1] - y[j-1])**2 + min(grid[i-1, j], grid[i, j-1], grid[i-1, j-1])
    return np.sqrt(grid[-1, -1])


@pytest.mark.parametrize("n,radius", [(1, 1), (2, 1), (19, 0), (31, 1), (17, 3), (7, 20)])
def test_exact_banded_paths(n, radius):
    x = np.random.default_rng(1920).normal(size=(4, n))
    x[2] = np.roll(x[0], 1, axis=0)
    x[3] = 2.
    spi = DynamicTimeWarping(global_constraint="sakoe_chiba", sakoe_chiba_radius=radius)
    data = Data(x, normalise=False)
    actual = spi.multivariate(data)
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            expected = full_grid(x[i], x[j], radius)
            assert actual[i, j] == expected
            assert actual[i, j] == pytest.approx(dtw(x[i], x[j], global_constraint="sakoe_chiba",
                                                    sakoe_chiba_radius=radius), abs=1e-13)
    np.testing.assert_array_equal(actual, actual.T)
    assert np.isnan(actual.diagonal()).all()
    permutation = [2, 0, 3, 1]
    np.testing.assert_array_equal(spi.multivariate(Data(x[permutation], normalise=False)),
                                  actual[np.ix_(permutation, permutation)])
    actual[0, 1] = 555
    assert spi.bivariate(data, i=0, j=1) != 555
    data.to_numpy()[1, :, 0] += 100
    assert spi.bivariate(data, i=0, j=1) != actual[1, 0]


def test_banded_extreme_units_and_invalid_channel_isolation():
    x = np.random.default_rng(1921).normal(size=(3, 31))
    spi = DynamicTimeWarping(global_constraint="sakoe_chiba")
    expected = spi.multivariate(Data(x, normalise=False))
    for scale in [1e200, 1e-200]:
        d = Data(x*scale, normalise=False)
        np.testing.assert_allclose(spi.multivariate(d)/scale, expected, rtol=1e-13)
    d.to_numpy()[1, 0, 0] = np.nan
    bad = spi.multivariate(d)
    assert np.isnan(bad[1]).all()
    assert np.isfinite(bad[0, 2])


def test_banded_radius_settings():
    default = DynamicTimeWarping(global_constraint="sakoe_chiba")
    assert default.identifier == "dtw_constraint-sakoe-chiba"
    assert DynamicTimeWarping(global_constraint="sakoe_chiba", sakoe_chiba_radius=4).identifier.endswith("radius-4")
    for radius in [True, -1, 1.5]:
        with pytest.raises(ValueError):
            DynamicTimeWarping(global_constraint="sakoe_chiba", sakoe_chiba_radius=radius)
    with pytest.raises(ValueError):
        DynamicTimeWarping(sakoe_chiba_radius=4)

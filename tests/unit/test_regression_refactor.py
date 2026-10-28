"""Independent fitted-model checks of batched single-predictor regressions."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from cdt.causality.pairwise import RECI

from pyspi._moments import moment_workspace
from pyspi.data import Data
from pyspi.statistics.misc import LinearModel
from pyspi.statistics.causal import RegressionErrorCausalInference


def make_data(x, **kwargs):
    return Data(np.array(x, copy=True), normalise=False, **kwargs)


def metric(model):
    return RegressionErrorCausalInference() if model == "RECI" else LinearModel(model)


def reference(x, model):
    out = np.full((len(x), len(x)), np.nan)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            if model == "Ridge":
                # Augmented least squares, independent of sklearn's Ridge solver.
                design = np.column_stack([np.ones(x.shape[1]), x[i]])
                augmented = np.vstack([design, [0., 1.]])
                beta = np.linalg.lstsq(augmented, np.append(x[j], 0.), rcond=None)[0]
                predicted = design @ beta
                target = x[j]
            else:
                a, b = x[[i, j]].copy()
                a = (a-a.min())/(np.ptp(a) or 1.)
                b = (b-b.min())/(np.ptp(b) or 1.)
                design = np.column_stack([np.ones(len(a)), a**3])
                predicted = design @ np.linalg.lstsq(design, b, rcond=None)[0]
                target = b
            out[i, j] = np.mean((target-predicted)**2)
    return out


@pytest.mark.parametrize("model", ["Ridge", "RECI"])
@pytest.mark.parametrize("kind", ["raw", "normalized", "constant", "cubic", "offset"])
def test_batched_regressions_against_independent_least_squares(model, kind):
    x = np.random.default_rng(991).normal(size=(5, 127))
    x[1] = .7*x[0] + .2*x[1]
    if kind == "raw":
        x *= np.arange(1, 6)[:, None]
    elif kind == "normalized":
        x = (x-x.mean(axis=1, keepdims=True))/x.std(axis=1, ddof=1, keepdims=True)
    elif kind == "constant":
        x[0], x[2] = 4., 0.
    elif kind == "cubic":
        x[0] = np.linspace(0., 1., x.shape[1])
        x[1] = x[0]**3
    elif kind == "offset":
        x += np.arange(5)[:, None]*10
    actual = metric(model).multivariate(make_data(x))
    expected = reference(x, model)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)
    assert np.nanmin(actual) >= 0
    # Two existing delegates are additional compatibility checks, not the
    # mathematical oracle above.
    if model == "Ridge":
        fit = Ridge().fit(x[0, :, None], x[1])
        delegated = np.mean((x[1]-fit.predict(x[0, :, None]))**2)
    else:
        delegated = RECI().b_fit_score(x[0], x[1])
    np.testing.assert_allclose(actual[0, 1], delegated, rtol=2e-11, atol=2e-13)


@pytest.mark.parametrize("model", ["Ridge", "RECI"])
def test_regression_permutation_orientation_and_cache_ownership(model):
    x = np.random.default_rng(992).normal(size=(4, 95))*[[1], [2], [5], [.1]]
    spi, data = metric(model), make_data(x)
    before = spi.multivariate(data)
    assert not np.allclose(before, before.T, equal_nan=True)
    np.testing.assert_allclose(spi.bivariate(data, i=2, j=0), before[2, 0])
    order = [3, 1, 0, 2]
    np.testing.assert_allclose(spi.multivariate(make_data(x[order])), before[np.ix_(order, order)])
    returned = spi.multivariate(data)
    returned[:] = 99
    np.testing.assert_array_equal(spi.multivariate(data), before)
    workspace = moment_workspace(data)
    data.to_numpy()[0, :, 0] += x[1]**2
    assert moment_workspace(data) is not workspace
    np.testing.assert_allclose(spi.multivariate(data), reference(data.to_numpy(realisation=0), model), atol=1e-12)
    data.set_data(x[::-1])
    np.testing.assert_allclose(spi.multivariate(data), before[::-1, ::-1], atol=1e-12)


@pytest.mark.parametrize("model", ["Ridge", "RECI"])
def test_regression_array_input_uses_consistent_preprocessing(model):
    x = np.random.default_rng(993).normal(size=(3, 80))*[[2], [7], [3]] + [[100], [-5], [9]]
    spi = metric(model)
    expected = spi.multivariate(Data(x, detrend=False, normalise=True))
    np.testing.assert_allclose(spi.multivariate(x), expected, atol=1e-13)
    np.testing.assert_allclose(spi.bivariate(x[0], x[1]), expected[0, 1], atol=1e-13)


@pytest.mark.parametrize("model", ["Ridge", "RECI"])
def test_regression_nonfinite_view_is_local_and_multiple_realisations_rejected(model):
    data = make_data(np.random.default_rng(994).normal(size=(4, 65)))
    spi = metric(model)
    expected = spi.multivariate(data)
    data.to_numpy()[0, 2, 0] = np.nan
    actual = spi.multivariate(data)
    assert np.isnan(actual[0]).all() and np.isnan(actual[:, 0]).all()
    np.testing.assert_array_equal(actual[1:, 1:], expected[1:, 1:])
    repeated = Data(np.ones((3, 10, 2)), dim_order="psr", normalise=False)
    with pytest.raises(ValueError, match="single realisation"):
        spi.multivariate(repeated)


def test_reci_scale_independence_does_not_flatten_tiny_channels():
    x = np.random.default_rng(995).uniform(size=(3, 200))
    x[0, :2] = [0., 1.]
    x[1] = x[0]**3 + 1e-8*x[1]
    x[1, :2] = [0., 1.]
    spi = metric("RECI")
    actual = spi.multivariate(make_data(x))
    scaled = spi.multivariate(make_data(x*[[1e-150], [1e150], [1e-100]]))
    np.testing.assert_allclose(scaled, actual, atol=1e-15)
    assert 0 < actual[0, 1] < 1e-15
    assert actual[1, 0] > .001


def test_ridge_near_perfect_large_scale_uses_residuals():
    x = 1e6*np.random.default_rng(996).normal(size=(2, 100))
    x[1] = 2*x[0]
    actual = metric("Ridge").multivariate(make_data(x))[0, 1]
    fit = Ridge().fit(x[0, :, None], x[1])
    expected = np.mean((x[1]-fit.predict(x[0, :, None]))**2)
    assert 0 < actual < 1e-14
    # Relative tolerance reflects subtraction at machine resolution here.
    np.testing.assert_allclose(actual, expected, rtol=.2, atol=1e-20)


def test_moment_intervals_and_cache_bound():
    data = make_data(np.random.default_rng(997).normal(size=(3, 40)))
    work = moment_workspace(data)
    full = work.centered()[0]
    partial = work.centered(start=1)[0]
    assert full.shape[1] == 40 and partial.shape[1] == 39
    np.testing.assert_allclose(partial.mean(axis=1), 0., atol=1e-15)
    assert work.centered()[0] is full
    work.arrays.max_bytes = 2*full.nbytes
    for start in range(1, 30):
        work.cross(start=start)
    assert work.arrays.nbytes <= work.arrays.max_bytes

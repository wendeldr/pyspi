"""Independent formulas and cache/input contracts for the candidate additions."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pyspi.data import Data
from pyspi._moments import moment_workspace
from pyspi.statistics.distance import DistanceCorrelation, HistogramRelativeEntropy
from pyspi.statistics.misc import InterDependenceScore


def distance_reference(x, biased):
    centered = []
    for row in x.astype(np.longdouble):
        a = abs(row[:, None] - row[None, :])
        n = len(row)
        if biased:
            a = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
        else:
            a = (a - a.sum(0)[None, :] / (n - 2) - a.sum(1)[:, None] / (n - 2)
                 + a.sum() / ((n - 1) * (n - 2)))
            np.fill_diagonal(a, 0.)
        centered.append(a)
    out = np.full((len(x), len(x)), np.nan)
    for i, a in enumerate(centered):
        for j, b in enumerate(centered):
            if i == j:
                continue
            denominator = np.sqrt(np.sum(a*a) * np.sum(b*b))
            value = np.sum(a*b) / denominator if denominator else 0.
            out[i, j] = np.sqrt(max(value, 0.)) if biased else value
    return out


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("kind", ["ordinary", "ties", "degenerate", "units"])
def test_distance_correlation_independent_centering(biased, kind):
    x = np.random.default_rng(1910).normal(size=(6, 63))
    if kind == "ties":
        x = np.round(x)
    if kind == "degenerate":
        x = np.array([[0., 0., 0., 1.], [1., 0., 1., 0.], [2., 2., 2., 2.]])
    if kind == "units":
        x *= np.array([1e200, 1e-200, 1e-100, 1., 1e100, 1.])[:, None]
    expected = distance_reference(x, biased)
    data = Data(x, normalise=False)
    # Exercise many block boundaries, rather than only the default large block.
    moment_workspace(data).block_bytes = len(x) * 8 * 19
    actual = DistanceCorrelation(biased=biased).multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=2e-14)


def test_distance_shares_both_conventions_and_preserves_negative_coefficients(monkeypatch):
    import pyspi._distance_correlation as module
    calls = []
    original = module._products
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(module, "_products", counted)
    x = np.random.default_rng(1911).normal(size=(8, 31))
    d = Data(x, normalise=False)
    unbiased = DistanceCorrelation().multivariate(d)
    biased = DistanceCorrelation(biased=True).multivariate(d)
    assert len(calls) == 1
    assert np.nanmin(unbiased) < 0
    assert np.nanmin(biased) >= 0
    assert DistanceCorrelation().issigned()
    assert not DistanceCorrelation(biased=True).issigned()


@pytest.mark.parametrize("spi", [DistanceCorrelation(), HistogramRelativeEntropy(), InterDependenceScore()])
def test_candidates_channel_order_mutation_and_owned_outputs(spi):
    x = np.random.default_rng(1912).normal(size=(5, 71))
    d = Data(x.copy(), normalise=False)
    expected = spi.multivariate(d)
    order = [3, 0, 4, 1, 2]
    permuted = spi.multivariate(Data(x[order], normalise=False))
    np.testing.assert_allclose(permuted, expected[np.ix_(order, order)], atol=2e-12)
    assert spi.bivariate(d, i=0, j=3) == pytest.approx(expected[0, 3])
    actual = spi.multivariate(d)
    actual[:] = 123
    np.testing.assert_allclose(spi.multivariate(d), expected, atol=2e-12)
    d.to_numpy()[1, :, 0] = x[1]**2
    changed = spi.multivariate(d)
    assert not np.allclose(changed[1], expected[1], equal_nan=True)
    d.to_numpy()[2, 3, 0] = np.nan
    invalid = spi.multivariate(d)
    assert np.isnan(invalid[2]).all() and np.isnan(invalid[:, 2]).all()
    keep = [0, 1, 3, 4]
    np.testing.assert_allclose(invalid[np.ix_(keep, keep)], changed[np.ix_(keep, keep)], atol=2e-12)


def test_histogram_shape_matches_explicit_kl_and_ignores_sample_order():
    x = np.random.default_rng(1913).normal(size=(4, 1024))
    x[1] = x[1]**3
    d = Data(x, normalise=False)
    actual = HistogramRelativeEntropy().multivariate(d)
    h = np.stack([np.histogram(row, bins=10)[0] for row in x]).astype(float) + 1e-10
    p = h / h.sum(1, keepdims=True)
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            expected = max(np.sum(p[i]*np.log(p[i]/p[j])), np.sum(p[j]*np.log(p[j]/p[i])))
            assert actual[i, j] == pytest.approx(expected, abs=1e-13)
    shuffled = x.copy()
    shuffled[0] = shuffled[0, ::-1]
    np.testing.assert_allclose(HistogramRelativeEntropy().multivariate(Data(shuffled, normalise=False)), actual)
    scaled = x * np.array([1e200, 1e-200, 3., 2.])[:, None]
    np.testing.assert_allclose(HistogramRelativeEntropy().multivariate(Data(scaled, normalise=False)), actual)


@pytest.mark.parametrize("pseudocount", [np.nextafter(0., 1.), 1e300])
def test_histogram_extreme_regularization_stays_defined(pseudocount):
    x = np.random.default_rng(1915).normal(size=(3, 67))
    x[0] = x[0]**5
    result = HistogramRelativeEntropy(pseudocount=pseudocount).multivariate(Data(x, normalise=False))
    assert np.isfinite(result[~np.eye(3, dtype=bool)]).all()
    assert np.nanmin(result) >= 0


@pytest.mark.parametrize("spi", [HistogramRelativeEntropy(), InterDependenceScore()])
def test_undefined_constant_channels_stay_missing(spi):
    x = np.random.default_rng(1914).normal(size=(3, 31))
    x[0] = 5.
    result = spi.multivariate(Data(x, normalise=False))
    assert np.isnan(result[0]).all() and np.isnan(result[:, 0]).all()
    assert np.isfinite(result[1, 2])


@pytest.mark.parametrize("kwargs", [{"bins": 1}, {"bins": True}, {"pseudocount": 0}, {"pseudocount": np.inf}])
def test_histogram_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        HistogramRelativeEntropy(**kwargs)


@pytest.mark.parametrize("kwargs", [{"terms": 0}, {"terms": True}, {"pnorm": "2"}, {"bandwidth": 0}])
def test_ids_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        InterDependenceScore(**kwargs)


def test_ids_parameter_identifiers_and_import_preserve_rng():
    assert InterDependenceScore().identifier == "ids_pnormmax_terms06_bandwidth0p5"
    assert InterDependenceScore(terms=2).identifier != InterDependenceScore(bandwidth=.25).identifier
    code = """
import json, numpy as np
np.random.seed(271)
expected = np.random.random(4)
np.random.seed(271)
import pyspi.statistics.misc
print(json.dumps(bool(np.array_equal(np.random.random(4), expected))))
"""
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    assert json.loads(child.stdout.strip().splitlines()[-1]) is True

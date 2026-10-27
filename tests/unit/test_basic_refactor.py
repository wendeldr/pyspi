"""Independent tau-b checks and cache correctness across the basic metrics."""

import itertools
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from scipy import signal, stats

from pyspi._cache import data_cache
from pyspi._rank import KendallWorkspace
from pyspi.data import Data
from pyspi.statistics import basic


def make_data(values):
    return Data(np.array(values, copy=True), normalise=False, detrend=False)


def brute_tau_b(x, y):
    """Count observation-pair agreements directly, without ranks or sorting."""
    numerator = x_untied = y_untied = 0
    for a, b in itertools.combinations(range(len(x)), 2):
        sx = int(x[a] > x[b]) - int(x[a] < x[b])
        sy = int(y[a] > y[b]) - int(y[a] < y[b])
        numerator += sx * sy
        x_untied += sx != 0
        y_untied += sy != 0
    if not x_untied or not y_untied:
        return np.nan
    return numerator / np.sqrt(x_untied * y_untied)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 31])
@pytest.mark.parametrize("kind", ["continuous", "ties", "constant", "joint_ties", "large_int"])
@pytest.mark.parametrize("squared", [False, True])
def test_kendall_against_independent_pair_counts(n, kind, squared):
    rng = np.random.default_rng(310 + n)
    x = rng.normal(size=(5, n))
    if kind in ("ties", "joint_ties"):
        x = rng.integers(-2, 3, size=x.shape)
    if kind == "joint_ties":
        x[1] = x[0]
        x[2] = -x[0]
    if kind == "constant":
        x[0] = 3
    if kind == "large_int":
        x = rng.integers(0, 20, size=x.shape).astype(np.uint64) + np.uint64(2**63)
    expected = np.full((5, 5), np.nan)
    for i, j in itertools.combinations(range(5), 2):
        tau = brute_tau_b(x[i], x[j])
        expected[i, j] = expected[j, i] = tau ** 2 if squared else tau
    data = make_data(x)
    spi = basic.KendallTau(squared=squared)
    # Populate one pair first, in the reverse direction, then the matrix.
    np.testing.assert_allclose(spi.bivariate(data, i=2, j=0), expected[0, 2], atol=1e-14)
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual, actual.T)
    for i, j in itertools.combinations(range(5), 2):
        np.testing.assert_allclose(spi.bivariate(data, i=i, j=j), actual[i, j], atol=1e-14)


@pytest.mark.parametrize("n", [128, 1024, 4096])
@pytest.mark.parametrize("decimals", [None, 0, 1])
def test_kendall_matches_scipy_at_larger_sizes_and_permutations(n, decimals):
    rng = np.random.default_rng(311)
    x = rng.normal(size=(6, n))
    if decimals is not None:
        x = np.round(x, decimals)
    x[4], x[5] = x[0] ** 3, -x[0]
    expected = np.full((6, 6), np.nan)
    for i, j in itertools.combinations(range(6), 2):
        expected[i, j] = expected[j, i] = stats.kendalltau(x[i], x[j]).statistic
    spi = basic.KendallTau()
    actual = spi.multivariate(make_data(x))
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    channels = rng.permutation(6)
    observations = rng.permutation(n)
    permuted = spi.multivariate(make_data(x[channels][:, observations]))
    np.testing.assert_allclose(permuted, actual[np.ix_(channels, channels)], atol=1e-14)


def test_kendall_exhaustive_short_tied_sequences():
    # Covers all possible tie arrangements for two ternary series of length 4.
    sequences = np.array(list(itertools.product(range(3), repeat=4)))
    work = KendallWorkspace(sequences)
    for i, j in itertools.combinations_with_replacement(range(len(sequences)), 2):
        np.testing.assert_allclose(work.pair(i, j), brute_tau_b(sequences[i], sequences[j]), atol=1e-14)


@pytest.mark.parametrize("cls", [basic.KendallTau, basic.SpearmanR])
def test_rank_labels_are_instance_owned(cls):
    original = cls.labels.copy()
    plain, squared = cls(), cls(squared=True)
    assert plain.issigned() and not squared.issigned()
    assert plain.labels == original + ["signed"]
    assert squared.labels == original + ["unsigned"]
    squared.labels.append("private")
    assert plain.labels == cls().labels == original + ["signed"]
    assert cls.labels == original


@pytest.mark.parametrize("cls", [basic.KendallTau, basic.SpearmanR])
def test_signed_and_squared_rank_variants_share_work(cls):
    data = make_data(np.random.default_rng(312).normal(size=(5, 128)))
    target = "pyspi._rank._kendall_dis" if cls is basic.KendallTau else "pyspi._rank.stats.rankdata"
    from pyspi._rank import _kendall_dis
    wrapped = _kendall_dis if cls is basic.KendallTau else stats.rankdata
    with patch(target, wraps=wrapped) as counter:
        squared = cls(squared=True).multivariate(data)
        plain = cls().multivariate(data)
        cls().bivariate(data, i=4, j=1)
        cls(squared=True).multivariate(data)
    assert counter.call_count == (10 if cls is basic.KendallTau else 1)
    np.testing.assert_array_equal(squared, plain ** 2)


@pytest.mark.parametrize("cls", [basic.KendallTau, basic.SpearmanR])
def test_rank_pair_calls_reuse_preprocessing_and_do_not_compute_other_pairs(cls):
    data = make_data(np.random.default_rng(313).normal(size=(5, 128)))
    from pyspi._rank import _kendall_dis
    target = "pyspi._rank._kendall_dis" if cls is basic.KendallTau else "pyspi._rank.np.corrcoef"
    wrapped = _kendall_dis if cls is basic.KendallTau else np.corrcoef
    with patch(target, wraps=wrapped) as counter:
        value = cls().bivariate(data, i=4, j=1)
        squared = cls(squared=True).bivariate(data, i=1, j=4)
    assert counter.call_count == 1
    assert squared == value ** 2


@pytest.mark.parametrize("method", ["multivariate", "bivariate"])
def test_kendall_rejects_multiple_realisations(method):
    data = Data(np.ones((2, 8, 2)), dim_order="psr", normalise=False)
    with pytest.raises(ValueError, match="requires a single realisation"):
        getattr(basic.KendallTau(), method)(data)


def test_kendall_direct_arrays_match_preprocessed_data():
    raw = np.random.default_rng(314).normal(size=(2, 91)) * [[3], [10]] + [[-4], [7]]
    expected = basic.KendallTau().multivariate(Data(raw))[0, 1]
    np.testing.assert_allclose(basic.KendallTau().bivariate(*raw), expected, atol=1e-14)


@pytest.mark.parametrize("cls", [basic.KendallTau, basic.SpearmanR])
def test_nan_introduced_through_view_only_invalidates_affected_rank_pairs(cls):
    data = make_data(np.random.default_rng(315).normal(size=(3, 31)))
    expected = cls().multivariate(data)
    data.to_numpy()[0, 7, 0] = np.nan
    actual = cls().multivariate(data)
    assert np.isnan(actual[0]).all() and np.isnan(actual[:, 0]).all()
    assert actual[1, 2] == expected[1, 2]


FACTORIES = [basic.KendallTau, basic.SpearmanR, basic.Covariance, basic.Precision,
             lambda: basic.CrossCorrelation(sigonly=False)]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("mutation", ["replace", "view", "owner", "add", "remove", "dtype"])
def test_basic_caches_follow_data_changes(factory, mutation):
    raw = np.random.default_rng(316).normal(size=(4, 128))
    data = Data(raw, normalise=False)
    spi = factory()
    spi.multivariate(data)
    old_cache = data_cache(data)
    if mutation == "replace":
        data.set_data(raw * [[1], [2], [-3], [4]])
    elif mutation == "view":
        data.to_numpy()[1, :, 0] = -data.to_numpy()[0, :, 0]
    elif mutation == "owner":
        raw[1] = -raw[0]
    elif mutation == "add":
        data.add_process(np.random.default_rng(317).normal(size=128))
    elif mutation == "remove":
        data.remove_process(1)
    else:
        data.set_data(raw.astype(np.float32))
    actual = spi.multivariate(data)
    expected = factory().multivariate(make_data(data.to_numpy(realisation=0)))
    assert data_cache(data) is not old_cache
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("factory", FACTORIES)
def test_results_do_not_expose_cache(factory):
    data = make_data(np.random.default_rng(318).normal(size=(4, 64)))
    spi = factory()
    matrix = spi.multivariate(data)
    expected = matrix.copy()
    matrix[:] = 123
    np.testing.assert_array_equal(spi.multivariate(data), expected)


@pytest.mark.parametrize("estimator", ["EmpiricalCovariance", "LedoitWolf", "OAS", "ShrunkCovariance", "GraphicalLasso"])
def test_covariance_and_precision_keep_fitted_models_intact_and_share_fit(estimator):
    data = make_data(np.random.default_rng(319).normal(size=(4, 128)))
    cls = getattr(basic.cov, estimator)
    original = cls.fit
    with patch.object(cls, "fit", autospec=True, side_effect=original) as fits:
        for kind in (basic.Covariance, basic.Precision):
            plain = kind(estimator=estimator).multivariate(data)
            squared = kind(estimator=estimator, squared=True).multivariate(data)
            np.testing.assert_array_equal(squared, plain ** 2)
    assert fits.call_count == 1
    fitted = basic.Covariance(estimator=estimator)._from_cache(data)
    assert np.isfinite(fitted.covariance_).all()
    assert np.isfinite(fitted.precision_).all()
    assert np.isfinite(fitted.score(data.to_numpy(realisation=0).T))


def test_fitting_warnings_are_visible():
    original = basic.cov.EmpiricalCovariance.fit
    def fit(model, values):
        warnings.warn("test fitting warning", UserWarning)
        return original(model, values)
    with patch.object(basic.cov.EmpiricalCovariance, "fit", fit):
        with pytest.warns(UserWarning, match="test fitting warning"):
            basic.Covariance().multivariate(make_data(np.arange(40.).reshape(2, 20)))


@pytest.mark.parametrize("cls", [basic.Covariance, basic.Precision, basic.KendallTau, basic.SpearmanR])
def test_single_channel_matrix_has_only_undefined_diagonal(cls):
    actual = cls().multivariate(make_data(np.arange(32.)[None, :]))
    assert actual.shape == (1, 1) and np.isnan(actual[0, 0])


@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("sigonly", [False, True])
def test_cross_correlation_preserves_existing_window_summary_and_pair_order(statistic, squared, sigonly):
    x = np.random.default_rng(320).normal(size=(4, 128))
    data = make_data(x)
    expected = np.full((4, 4), np.nan)
    for j in range(4):
        for i in range(j + 1, 4):
            # Freeze the current definition pending its mathematical review.
            r = signal.correlate(x[i, :, None], x[j, :, None], "full").squeeze()
            r = r / x[i].std() / x[j].std() / 127
            r = r[96:160]
            if sigonly:
                f = np.where(abs(r[32:]) <= 1.96 / np.sqrt(32))[0][0]
                b = np.where(abs(r[:32]) <= 1.96 / np.sqrt(32))[0][-1]
                r = r[32 - b:32 + f]
            if squared:
                r = r ** 2
            expected[i, j] = expected[j, i] = getattr(np, statistic)(r)
    actual = basic.CrossCorrelation(statistic=statistic, squared=squared, sigonly=sigonly).multivariate(data)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_cross_correlation_shares_sequences_and_keeps_first_direction_after_eviction():
    data = make_data(np.random.default_rng(321).normal(size=(3, 32)))
    spi = basic.CrossCorrelation(statistic="mean", sigonly=False)
    work = basic._basic_workspace(data)
    work.correlations.max_bytes = 16 * 8
    forward = spi.bivariate(data, i=0, j=1)
    spi.bivariate(data, i=0, j=2)  # evicts the first sequence
    assert list(work.correlations.entries) == [(0, 2)]
    assert spi.bivariate(data, i=1, j=0) == forward


def test_cross_correlation_computes_unique_pairs_once_across_variants():
    data = make_data(np.random.default_rng(322).normal(size=(4, 128)))
    with patch("pyspi.statistics.basic.signal.correlate", wraps=signal.correlate) as counter:
        for squared in (False, True):
            for statistic in ("max", "mean"):
                basic.CrossCorrelation(squared=squared, statistic=statistic, sigonly=False).multivariate(data)
    assert counter.call_count == 6


def test_basic_and_spectral_share_snapshot_and_invalidate_together():
    from pyspi._spectral import data_spectral_cache, spectral_workspace
    data = make_data(np.random.default_rng(323).normal(size=(3, 64)))
    basic.SpearmanR().multivariate(data)
    basic.Covariance().multivariate(data)
    old = spectral_workspace(data, 1)
    cache = data_cache(data)
    assert np.shares_memory(basic._basic_workspace(data).values, cache.source)
    assert data_spectral_cache(data).source is cache.source
    data.to_numpy()[1, :, 0] *= -1
    basic.SpearmanR().multivariate(data)
    assert data_cache(data) is not cache
    assert spectral_workspace(data, 1) is not old

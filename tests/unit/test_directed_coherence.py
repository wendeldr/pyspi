"""Squared DC checked independently of the unfixed factorization policy."""

from unittest.mock import patch

import numpy as np
import pytest

from pyspi._spectral import SpectralConnectivity
from pyspi.data import Data
from pyspi.statistics.spectral import DirectedCoherence


def model(transfer, covariance):
    transfer = np.asarray(transfer, complex)
    windows, frequencies, channels, _ = transfer.shape
    nfft = 2*(frequencies-1)
    conn = SpectralConnectivity(np.zeros((windows, 1, 5, nfft, channels), complex),
                                np.fft.fftfreq(nfft), np.arange(windows))
    conn.__dict__["_transfer_function"] = transfer
    conn.__dict__["_noise_covariance"] = np.asarray(covariance, float)
    return conn


def analytic_model():
    a = np.array([[.6, 0, 0], [.9, .3, 0], [.1, .4, .2]])
    f = np.fft.rfftfreq(128)
    transfer = np.linalg.inv(np.eye(3) - np.exp(-2j*np.pi*f)[:, None, None]*a)
    return model(transfer[None], np.diag([4., .25, 9.])[None])


def scalar_reference(transfer, covariance):
    result = np.empty(transfer.shape)
    for w in range(transfer.shape[0]):
        for f in range(transfer.shape[1]):
            for target in range(transfer.shape[2]):
                weights = [covariance[w, source, source]*abs(transfer[w, f, target, source])**2
                           for source in range(transfer.shape[3])]
                result[w, f, target] = np.array(weights)/sum(weights)
    return result


def test_exact_var_formula_bounds_and_direction():
    conn = analytic_model()
    actual = conn.directed_coherence()
    expected = scalar_reference(conn._transfer_function, conn._noise_covariance)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=1e-14)
    assert np.max(actual[0, :, 1, 0]) == pytest.approx(.9878048780487805)
    np.testing.assert_allclose(actual[0, :, 0, 1], 0, atol=1e-30)


def test_source_variances_and_target_normalization_include_self():
    h = np.broadcast_to([[1., 0, 0], [1, 1, 1], [0, 0, 1]], (2, 5, 3, 3)).copy()
    covariance = np.array([np.diag([1., 4, 9]), np.diag([9., 1, 0])])
    actual = model(h, covariance).directed_coherence()
    np.testing.assert_allclose(actual[0, :, 1], np.tile([1/14, 4/14, 9/14], (5, 1)))
    np.testing.assert_allclose(actual[1, :, 1], np.tile([.9, .1, 0], (5, 1)))
    assert np.isnan(actual[1, :, 2]).all()  # zero total innovation inflow


@pytest.mark.parametrize("scales", [[2, 7, .3], [-1, 1, 1], [1e120, 1e-120, .3],
                                    [1e-120]*3, [1e120]*3])
def test_changing_channel_units_preserves_dc_without_squaring_overflow(scales):
    conn = analytic_model()
    d = np.asarray(scales)
    transformed_h = conn._transfer_function*d[None, None, :, None]/d[None, None, None, :]
    transformed_cov = conn._noise_covariance*d[None, :, None]*d[None, None, :]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = model(transformed_h, transformed_cov).directed_coherence()
    np.testing.assert_allclose(actual, conn.directed_coherence(), atol=1e-14, rtol=1e-13)


def test_permutation_equivariance_of_formula_and_no_cached_model_mutation():
    conn = analytic_model()
    h, covariance = conn._transfer_function.copy(), conn._noise_covariance.copy()
    expected = conn.directed_coherence()
    order = [2, 0, 1]
    actual = model(h[..., order, :][..., order], covariance[..., order, :][..., order]).directed_coherence()
    np.testing.assert_allclose(actual, expected[..., order, :][..., order], atol=1e-14)
    expected[:] = 123
    np.testing.assert_array_equal(conn._transfer_function, h)
    np.testing.assert_array_equal(conn._noise_covariance, covariance)
    np.testing.assert_allclose(conn.directed_coherence(), scalar_reference(h, covariance), atol=1e-14)


def test_correlated_innovations_keep_documented_diagonal_weighting():
    conn = analytic_model()
    expected = conn.directed_coherence()
    covariance = conn._noise_covariance.copy()
    covariance[0, 0, 1] = covariance[0, 1, 0] = .3
    np.testing.assert_allclose(model(conn._transfer_function, covariance).directed_coherence(), expected)


@pytest.mark.parametrize("statistic,expected", [("mean", .34), ("max", .64)])
def test_square_at_each_frequency_before_band_summary(statistic, expected):
    strength = np.array([.99, .04, .64, .98, .97])
    h = np.broadcast_to(np.eye(2, dtype=complex), (1, 5, 2, 2)).copy()
    h[0, :, 0, 0], h[0, :, 0, 1] = np.sqrt(1-strength), np.sqrt(strength)
    conn = model(h, np.eye(2)[None])
    spi = DirectedCoherence(fmin=.125, fmax=.25, statistic=statistic)
    data = Data(np.random.default_rng(55).normal(size=(2, 32)), normalise=False)
    with patch("pyspi.statistics.spectral.directed_workspace", return_value=conn):
        with patch.object(conn, "_directed_blocks", return_value=iter([(np.array([[0, 1]]), conn)])):
            result = spi.multivariate(data)
    assert result[1, 0] == pytest.approx(expected)
    assert np.isnan(result.diagonal()).all()


def test_zero_inflow_is_undefined_and_does_not_poison_other_targets():
    h = np.broadcast_to(np.eye(3), (1, 5, 3, 3)).copy()
    h[..., 0, :] = 0
    with np.errstate(all="raise"):
        actual = model(h, np.eye(3)[None]).directed_coherence()
        absent = model(h, np.zeros((1, 3, 3))).directed_coherence()
    assert np.isnan(actual[..., 0, :]).all()
    np.testing.assert_allclose(actual[..., 1:, :], np.broadcast_to(np.eye(3)[1:], (1, 5, 2, 3)))
    assert np.isnan(absent).all()

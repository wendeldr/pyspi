"""Directed formulas checked without estimating or factoring a spectrum."""

import numpy as np
import pytest

from pyspi import _directed as directed
from pyspi._wilson import adjoint


def model(p=3, n=128):
    a = np.array([[.6, 0, 0], [.9, .3, 0], [.1, .4, .2]])[:p, :p]
    covariance = np.array([[4., .2, 0], [.2, .25, .03], [0, .03, 2.]])[:p, :p]
    ar = np.eye(p)-np.exp(-2j*np.pi*np.fft.rfftfreq(n))[:, None, None]*a
    return np.linalg.inv(ar)[None], covariance[None], ar[None]


@pytest.mark.parametrize("p", [2, 3])
def test_dtf_pdc_gpdc_against_direct_matrix_definitions(p):
    h, covariance, ar = model(p)
    power = abs(h)**2
    ar_power = abs(ar)**2
    weighted = ar_power/np.diag(covariance[0])[None, None, :, None]
    np.testing.assert_allclose(directed.directed_transfer_function(h), power/power.sum(axis=-1, keepdims=True), atol=1e-14)
    np.testing.assert_allclose(directed.partial_directed_coherence(h), ar_power/ar_power.sum(axis=-2, keepdims=True), atol=1e-14)
    np.testing.assert_allclose(directed.generalized_partial_directed_coherence(h, covariance), weighted/weighted.sum(axis=-2, keepdims=True), atol=1e-14)


@pytest.mark.parametrize("p", [2, 3])
def test_ddtf_is_full_frequency_squared_dtf_times_squared_partial_coherence(p):
    h, covariance, _ = model(p)
    spectrum = h @ covariance[:, None] @ adjoint(h)
    precision = np.linalg.inv(spectrum)
    expected = np.empty_like(h.real)
    power = abs(h)**2
    for target in range(p):
        denominator = power[0, :, target].sum()
        for source in range(p):
            for f in range(h.shape[1]):
                partial = abs(precision[0, f, target, source])**2 / (precision[0, f, target, target].real*precision[0, f, source, source].real)
                expected[0, f, target, source] = power[0, f, target, source]*partial/denominator
    actual = directed.direct_directed_transfer_function(h, covariance)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
    assert np.all((actual >= 0) & (actual <= 1+1e-14))
    if p == 2:
        coh_squared = abs(spectrum[0, :, 0, 1])**2 / (spectrum[0, :, 0, 0].real*spectrum[0, :, 1, 1].real)
        np.testing.assert_allclose(directed.squared_partial_coherence(h, covariance)[0, :, 0, 1], coh_squared, atol=1e-13)


def test_ddtf_native_grid_keeps_nyquist_once_and_band_product_order():
    h, covariance, _ = model(2)
    full = directed.direct_directed_transfer_function(h, covariance)
    band = slice(3, 20)
    partial = directed.squared_partial_coherence(h, covariance)
    ffdtf = abs(h)**2/np.sum(abs(h)**2, axis=(1, 3), keepdims=True)
    np.testing.assert_allclose(full[:, band].mean(axis=1), (ffdtf*partial)[:, band].mean(axis=1), atol=1e-14)
    assert not np.allclose(full[:, band].mean(axis=1), ffdtf[:, band].mean(axis=1)*partial[:, band].mean(axis=1), atol=1e-8)
    without_nyquist = directed.direct_directed_transfer_function(h[:, :-1], covariance)
    assert not np.allclose(full[:, :-1], without_nyquist, atol=1e-8)
    # Padding for a solver must not redefine the evaluation/normalization grid.
    dense, _, _ = model(2, n=512)
    np.testing.assert_allclose(directed.direct_directed_transfer_function(dense[:, ::4], covariance), full, atol=1e-14)


def test_pairwise_granger_against_original_geweke_power_decomposition():
    h, covariance, _ = model(2)
    spectrum = h @ covariance[:, None] @ adjoint(h)
    expected = np.full(h.shape, np.nan, dtype=float)
    for target, source in ((0, 1), (1, 0)):
        total = spectrum[..., target, target].real
        sigma = covariance[0]
        residual = sigma[source, source]-sigma[target, source]**2/sigma[target, target]
        intrinsic = total-residual*abs(h[..., target, source])**2
        expected[..., target, source] = np.log(total/intrinsic)
    actual = directed.pairwise_granger(h, covariance)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(actual[..., 0, 1], 0, atol=1e-30)  # inversion roundoff
    assert np.all(actual[..., 1, 0] > 0)


def test_granger_resolves_small_positive_effect_without_subtractive_cancellation():
    h = np.broadcast_to(np.eye(2, dtype=complex), (1, 3, 2, 2)).copy()
    h[..., 1, 0] = 1e-9
    actual = directed.pairwise_granger(h, np.eye(2)[None])
    np.testing.assert_allclose(actual[..., 1, 0], np.log1p(1e-18), rtol=1e-14, atol=0)
    assert np.all(actual[..., 0, 1] == 0)


def test_granger_zero_intrinsic_power_has_no_epsilon_floor():
    h = np.array([[[[1., 0], [1., 0]]]], dtype=complex)
    result = directed.pairwise_granger(h, np.eye(2)[None])
    assert np.isnan(result[..., 1, 0]).all()
    assert np.all(result[..., 0, 1] == 0)

"""Independent identities, known models, and failure isolation for Wilson fits."""

import numpy as np
import pytest

from pyspi._wilson import REASONS, adjoint, factorize, inverse, plus_operator


def exact_var(n=256):
    a = np.array([[.6, 0, 0], [.9, .3, 0], [.1, .4, .2]])
    covariance = np.array([[4., .2, 0], [.2, .25, .03], [0, .03, 2.]])
    transfer = np.linalg.inv(np.eye(3)-np.exp(-2j*np.pi*np.fft.fftfreq(n))[:, None, None]*a)
    return transfer, covariance, transfer @ covariance @ adjoint(transfer)


def transfer_covariance(factor):
    zero = np.fft.ifft(factor, axis=-3)[..., 0, :, :].real
    return factor @ np.linalg.inv(zero)[..., None, :, :], zero @ adjoint(zero)


@pytest.mark.parametrize("n", [15, 16, 127, 128])
def test_plus_split_reconstructs_every_hermitian_matrix_entry(n):
    a = np.random.default_rng(n).normal(size=(2, n, 3, 3))
    a = np.fft.fft(a, axis=1)
    g = a @ adjoint(a)
    positive = plus_operator(g)
    np.testing.assert_allclose(positive+adjoint(positive), g, rtol=1e-14, atol=1e-12)
    lag = np.fft.ifft(g, axis=1)
    result = np.fft.ifft(positive, axis=1)
    np.testing.assert_allclose(result[:, 0], lag[:, 0]/2, atol=1e-13)
    np.testing.assert_allclose(result[:, 1:(n+1)//2], lag[:, 1:(n+1)//2], atol=1e-13)
    if n % 2 == 0:
        np.testing.assert_allclose(result[:, n//2], lag[:, n//2]/2, atol=1e-13)
    np.testing.assert_allclose(result[:, n//2+1:], 0, atol=1e-13)


@pytest.mark.parametrize("n", [255, 256])
def test_exact_var_recovers_transfer_and_correlated_innovations(n):
    h, covariance, spectrum = exact_var(n)
    fit = factorize(spectrum)
    assert fit.reason == 0 and fit.residual < 1e-10
    actual_h, actual_covariance = transfer_covariance(fit.factor)
    np.testing.assert_allclose(actual_h, h, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(actual_covariance, covariance, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("n", [127, 128, 1024])
def test_estimated_spectra_permute_with_channels_and_reconstruct(n):
    raw = np.random.default_rng(n).normal(size=(5, n, 3))
    raw[:, 1:, 1] += .7*raw[:, :-1, 0]
    coefficients = np.fft.fft(raw, axis=1)
    s = np.einsum("kfi,kfj->fij", coefficients, coefficients.conj())/5
    order = [2, 0, 1]
    fit = factorize(s)
    permuted = factorize(s[:, order][:, :, order])
    assert fit.reason == permuted.reason == 0
    h, covariance = transfer_covariance(fit.factor)
    ph, pcovariance = transfer_covariance(permuted.factor)
    np.testing.assert_allclose(ph, h[:, order][:, :, order], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(pcovariance, covariance[np.ix_(order, order)], rtol=1e-10)
    np.testing.assert_allclose(fit.factor @ adjoint(fit.factor), s, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", [[1e120, 1e-120, 3.], [1e-140]*3, [1e140]*3, [-2., 3., .4]])
def test_channel_units_do_not_change_normalized_fit_quality(scale):
    _, _, s = exact_var()
    d = np.array(scale)
    fit = factorize(s*d[None, :, None]*d[None, None, :])
    assert fit.reason == 0
    actual = fit.factor/d[None, :, None]
    h, covariance = transfer_covariance(actual)
    expected_h, expected_covariance, _ = exact_var()
    np.testing.assert_allclose(h, expected_h, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(covariance, expected_covariance, atol=2e-10, rtol=2e-10)


def test_pair_local_failure_reasons_and_unaffected_good_fit():
    _, _, s = exact_var()
    batch = np.broadcast_to(s, (6, *s.shape)).copy()
    batch[1, 3, 0, 0] = np.nan
    batch[2, :, 2, :] = batch[2, :, 0, :]
    batch[2, :, :, 2] = batch[2, :, :, 0]
    batch[3, :, 0, 0] = -1
    batch[4, 4, 0, 1] += 1j
    batch[5] = 0
    fit = factorize(batch)
    assert REASONS[fit.reason].tolist() == ["estimated", "nonfinite_spectrum", "singular_spectrum",
                                          "invalid_spectrum", "invalid_spectrum", "singular_spectrum"]
    assert np.isnan(fit.factor[1:]).all()
    np.testing.assert_allclose(fit.factor[0], factorize(s).factor, atol=1e-12)


def test_unconverged_fit_is_not_returned_as_valid():
    fit = factorize(exact_var()[2], max_iterations=1)
    assert REASONS[fit.reason] == "nonconverged"
    assert fit.iterations == 1 and fit.residual > 1e-10
    assert np.isnan(fit.factor).all()


def test_uncoupled_and_low_power_channels_are_valid_not_weak_signal_failures():
    s = np.broadcast_to(np.diag([1., 1e-200]), (128, 2, 2)).astype(complex).copy()
    fit = factorize(s)
    assert fit.reason == 0
    h, covariance = transfer_covariance(fit.factor)
    np.testing.assert_allclose(h, np.broadcast_to(np.eye(2), h.shape), atol=1e-14)
    np.testing.assert_allclose(covariance, s[0].real, rtol=1e-14, atol=0)


def test_pair_inverse_matches_lapack_and_isolates_singular_matrices():
    a = np.random.default_rng(17).normal(size=(200, 2, 2, 2))
    a = a[:, 0]+1j*a[:, 1]
    for scale in (1e-150, 1., 1e150):
        np.testing.assert_allclose(inverse(a*scale)*scale, np.linalg.inv(a), rtol=1e-12, atol=1e-12)
    a[0] = 0
    actual = inverse(a)
    assert not np.isfinite(actual[0]).any()
    np.testing.assert_allclose(actual[1:], np.linalg.inv(a[1:]), rtol=1e-12, atol=1e-12)

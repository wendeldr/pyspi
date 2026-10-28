"""Nolte eq. 3: adjacent FFT bins, inclusive bands, and raw directionality."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.spectral import PhaseSlopeIndex


def analytic_connection(c, fs):
    """Inject an exact coherency spectrum to test the estimator independently."""
    nfft = 2 * (len(c) - 1)
    conn = SpectralConnectivity(
        np.zeros((1, 1, 1, nfft, 2), complex), np.fft.fftfreq(nfft, 1/fs), np.array([0.]),
    )
    spectrum = np.full((1, len(c), 2, 2), np.nan, complex)
    spectrum[0, :, 0, 1] = c
    spectrum[0, :, 1, 0] = np.conj(c)
    conn.__dict__["_coherency"] = spectrum
    return conn


def reference_matrix(raw, fs, low, high):
    """Scalar channel/frequency loops, independent of the production reduction.

    The transform is unchanged. Construct coherency directly from its Fourier
    coefficients, including Nyquist, then evaluate the paper's adjacent terms.
    """
    coefficients = sc.Multitaper(raw.T, sampling_frequency=fs).fft()[0]
    frequencies = np.fft.rfftfreq(coefficients.shape[-2], 1/fs)
    selected = np.flatnonzero((low <= frequencies) & (frequencies <= high))
    expected = np.full((raw.shape[0], raw.shape[0]), np.nan)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[0]):
            if i == j:
                continue
            coherencies = []
            for f in selected:
                x, y = coefficients[:, :, f, i], coefficients[:, :, f, j]
                coherencies.append(np.mean(x * y.conj()) /
                                   np.sqrt(np.mean(abs(x)**2) * np.mean(abs(y)**2)))
            expected[i, j] = sum((a.conjugate() * b).imag
                                 for a, b in zip(coherencies[:-1], coherencies[1:]))
    return expected


@pytest.mark.parametrize("n", [125, 128])
@pytest.mark.parametrize("band", [(0, .5), (0, .25), (.25, .5), (.08, .31)])
def test_pipeline_matches_independent_paper_reference(n, band):
    raw = np.random.default_rng(481).normal(size=(3, n))
    raw[1] += np.roll(raw[0], 3)
    data = Data(raw, normalise=False)
    spi = PhaseSlopeIndex(fmin=band[0], fmax=band[1])
    expected = reference_matrix(raw, 1, *band)
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual, -actual.T, rtol=1e-12, atol=1e-13)
    actual[:] = 123  # A returned matrix must not expose the cached PSI array.
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-13)


def test_weighted_adjacent_terms_not_all_frequency_combinations():
    c = np.ones(6, complex)
    c[1:4] = np.array([.9, .1, .9]) * np.exp(1j * np.deg2rad([0, 120, 240]))
    result = analytic_connection(c, 10).phase_slope_index((1, 3))[0]
    # Two adjacent weights of 0.9*0.1. The nonadjacent 1->3 term would reverse sign.
    expected = 2 * .09 * np.sin(2*np.pi/3)
    np.testing.assert_allclose(result[0, 1], expected, atol=1e-14)
    np.testing.assert_allclose(result[1, 0], -expected, atol=1e-14)


def test_inclusive_bands_share_boundary_but_not_frequency_pair():
    c = np.exp(1j * np.arange(9) * .2)
    conn = analytic_connection(c, 32)  # 0, 2, 4, ..., 16 Hz
    lower = conn.phase_slope_index((4, 8))
    upper = conn.phase_slope_index((8, 12))
    whole = conn.phase_slope_index((4, 12))
    np.testing.assert_allclose(lower[0, 0, 1], 2 * np.sin(.2), atol=1e-14)
    np.testing.assert_allclose(upper[0, 0, 1], 2 * np.sin(.2), atol=1e-14)
    np.testing.assert_allclose(whole, lower + upper, atol=1e-14)


def test_no_pair_uses_a_frequency_outside_the_band():
    c = np.exp(1j * np.array([2.8, -1.1, 0., .2, .4, -2.4, 1.8]))
    conn = analytic_connection(c, 12)
    np.testing.assert_allclose(conn.phase_slope_index((1.5, 4.5))[0, 0, 1],
                               2 * np.sin(.2), atol=1e-14)


def test_wide_band_uses_every_bin_without_delay_direction_aliasing():
    frequencies = np.arange(0, 1025, 2)
    c = np.exp(2j * np.pi * frequencies * .1)  # analytical 100 ms delay
    conn = analytic_connection(c, 2048)
    result = conn.phase_slope_index((1, 250))[0, 0, 1]
    # 125 bins from 2 through 250 Hz, and 124 adjacent comparisons at 2 Hz.
    np.testing.assert_allclose(result, 124 * np.sin(.4*np.pi), atol=1e-12)
    assert result > 0


@pytest.mark.parametrize("band, count", [((.1, .9), 0), ((1, 1), 1), ((1, 1.9), 1)])
def test_insufficient_frequency_bins_raise_identified_error(band, count):
    conn = analytic_connection(np.ones(6, complex), 10)
    with pytest.raises(ValueError, match=f"at least two FFT bins.*got {count}"):
        conn.phase_slope_index(band)


@pytest.mark.parametrize("n", [125, 128])
def test_dc_and_last_positive_bin_available_on_even_and_odd_fft(n):
    raw = np.random.default_rng(482).normal(size=(2, n))
    data = Data(raw, normalise=False)
    conn = spectral_workspace(data, 128)
    expected_frequencies = np.fft.rfftfreq(n, 1/128)
    np.testing.assert_array_equal(conn.frequencies, expected_frequencies)
    for low, high in ((0, expected_frequencies[1]), tuple(expected_frequencies[-2:])):
        expected = reference_matrix(raw, 128, low, high)
        actual = PhaseSlopeIndex(fs=128, fmin=low, fmax=high).multivariate(data)
        assert np.isfinite(actual[0, 1])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("band", [(1, 4), (4, 8), (8, 12), (1, 250)])
def test_eeg_bands_and_normalized_units_agree(band):
    raw = np.random.default_rng(483).normal(size=(3, 1024))
    data = Data(raw, normalise=False)
    hz = PhaseSlopeIndex(fs=2048, fmin=band[0], fmax=band[1]).multivariate(data)
    normalized = PhaseSlopeIndex(fmin=band[0]/2048, fmax=band[1]/2048).multivariate(data)
    assert np.isfinite(hz[~np.eye(3, dtype=bool)]).all()
    np.testing.assert_allclose(hz, normalized, rtol=1e-12, atol=1e-13)


def test_leading_channels_have_positive_psi_and_permutation_preserves_direction():
    source = np.random.default_rng(484).normal(size=2060)
    raw = np.stack([source[12:], source[9:-3], source[:-12]])
    spi = PhaseSlopeIndex(fs=2048, fmin=1, fmax=200)
    result = spi.multivariate(Data(raw, normalise=False))
    assert (result[np.triu_indices(3, 1)] > 0).all()
    order = [2, 0, 1]
    reordered = spi.multivariate(Data(raw[order], normalise=False))
    np.testing.assert_allclose(reordered, result[np.ix_(order, order)], atol=1e-12)


def test_constant_phase_has_no_slope_and_missing_bins_are_not_bridged():
    c = np.full(8, np.exp(.7j))
    conn = analytic_connection(c, 14)
    np.testing.assert_allclose(conn.phase_slope_index((1, 6))[0, 0, 1], 0, atol=1e-14)
    c[3] = np.nan
    conn = analytic_connection(c, 14)
    assert np.isnan(conn.phase_slope_index((2, 4))).all()


def test_zero_power_only_invalidates_pairs_with_that_channel():
    raw = np.random.default_rng(485).normal(size=(3, 128))
    raw[2] = 0
    result = PhaseSlopeIndex().multivariate(Data(raw, normalise=False))
    assert np.isnan(result[2]).all() and np.isnan(result[:, 2]).all()
    assert np.isfinite(result[0, 1])


def test_adjacent_reduction_bounds_temporary_arrays():
    c = np.exp(1j * np.arange(65) * .1)
    conn = analytic_connection(c, 128)
    conn.block_bytes = 64
    sizes = []
    einsum = np.einsum
    def record(expression, left, right, **kwargs):
        sizes.append(left.nbytes)
        return einsum(expression, left, right, **kwargs)
    with patch('pyspi._spectral.np.einsum', side_effect=record):
        actual = conn.phase_slope_index((1, 63))
    assert len(sizes) > 1 and max(sizes) <= conn.block_bytes
    np.testing.assert_allclose(actual[0, 0, 1], 62 * np.sin(.1), atol=1e-13)


def test_agrees_with_mne_adjacent_formula_on_identical_coherency():
    from mne_connectivity.effective import _compute_psi
    raw = np.random.default_rng(486).normal(size=(3, 128))
    conn = spectral_workspace(Data(raw, normalise=False), 128)
    spectrum = conn._coherency
    # MNE uses open edges; nextafter selects exactly our inclusive [4, 30] bins.
    bounds = [(np.nextafter(4., -np.inf), np.nextafter(30., np.inf))]
    expected, _, _ = _compute_psi(
        SimpleNamespace(shape=spectrum.shape, get_data=lambda: spectrum),
        conn.frequencies, bounds, freq_dim=1,
    )
    np.testing.assert_allclose(conn.phase_slope_index((4, 30)), expected[:, 0],
                               rtol=1e-12, atol=1e-13)


def test_calculator_refuses_unavailable_band_before_computing(tmp_path):
    config = tmp_path / 'psi.yaml'
    config.write_text(yaml.safe_dump({'.statistics.spectral': {'PhaseSlopeIndex': {
        'configs': [{'fs': 2048, 'fmin': 8, 'fmax': 9}],
    }}}))
    raw = np.random.default_rng(487).normal(size=(2, 1024))
    with pytest.raises(ValueError, match='psi_mean_band0008-0009hz: band 8-9 Hz has 1 usable frequencies'):
        Calculator(dataset=Data(raw, normalise=False), configfile=str(config))
    # The estimator itself still refuses rather than fabricating zero.
    spi = PhaseSlopeIndex(fs=2048, fmin=8, fmax=9)
    with pytest.raises(ValueError, match='at least two FFT bins'):
        spi.multivariate(Data(raw, normalise=False))

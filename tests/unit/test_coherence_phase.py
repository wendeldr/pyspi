"""Circular phase summaries checked against geometry and scalar FFT formulas."""

import math
from unittest.mock import patch

import numpy as np
import pytest

from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi.data import Data
from pyspi.statistics.spectral import CoherencePhase


def connection(values):
    values = np.asarray(values, complex)
    nfft = 2 * (len(values)-1)
    conn = SpectralConnectivity(np.zeros((1, 1, 5, nfft, 2), complex),
                                np.fft.fftfreq(nfft, 1/nfft), np.array([0.]))
    spectrum = np.full((1, len(values), 2, 2), np.nan, complex)
    spectrum[0, :, 0, 1], spectrum[0, :, 1, 0] = values, values.conj()
    conn.__dict__["_coherency"] = spectrum
    return conn


@pytest.mark.parametrize("degrees,expected", [([179, -179], 180), ([0, 90], 45),
                                              ([170, -170, 180], 180), ([20, 40], 30)])
def test_mean_follows_circle_including_wrap_boundary(degrees, expected):
    conn = connection(np.exp(1j*np.deg2rad(degrees)))
    result = conn.coherence_phase_mean()[0]
    assert abs(np.angle(np.exp(1j*(result[0, 1]-np.deg2rad(expected))))) < 1e-14
    assert abs(np.angle(np.exp(1j*(result[0, 1]+result[1, 0])))) < 1e-14


@pytest.mark.parametrize("values", [[0, 0], [np.nan, np.nan], [1, -1], [1j, -1j],
                                    np.exp(1j*np.deg2rad([30, 210])), [0, np.nan]])
def test_missing_or_cancelling_directions_are_undefined(values):
    assert np.isnan(connection(values).coherence_phase_mean()).all()


def test_zero_and_missing_bins_are_omitted_without_magnitude_weighting():
    conn = connection([.9, .1j, 0, np.nan])
    assert conn.coherence_phase_mean()[0, 0, 1] == pytest.approx(np.pi/4)
    assert conn.coherence_phase_mean((1, 3))[0, 0, 1] == pytest.approx(np.pi/2)
    assert np.isnan(conn.coherence_phase_mean((2, 3))).all()
    assert np.isnan(conn.coherence_phase_mean((.1, .9))).all()


@pytest.mark.parametrize("magnitude", [1., 1e-20, 1e-200, 1e-300, 1e-320, np.nextafter(0., 1.)])
def test_tiny_nonzero_coherency_still_has_equal_weight(magnitude):
    value = connection([.9, magnitude*1j]).coherence_phase_mean()[0, 0, 1]
    assert value == pytest.approx(np.pi/4)


def test_small_but_resolvable_resultant_is_not_discarded():
    value = connection([1, np.exp(1j*(np.pi-1e-8))]).coherence_phase_mean()[0, 0, 1]
    assert np.isfinite(value)
    assert value == pytest.approx(np.pi/2, abs=1e-8)


@pytest.mark.parametrize("statistic", ["max", "min", "delay", "slope", "rvalue"])
def test_only_circular_mean_is_supported(statistic):
    with pytest.raises(ValueError, match="circular mean"):
        CoherencePhase(statistic=statistic)


@pytest.mark.parametrize("n", [64, 129])
def test_real_data_matches_scalar_circular_reference_and_channel_permutation(n):
    raw = np.random.default_rng(906).normal(size=(4, n))
    data = Data(raw, normalise=False)
    conn = spectral_workspace(data, 128)
    coefficients = sc.Multitaper(raw.T, sampling_frequency=128).fft()[0]
    selected = np.flatnonzero((conn.frequencies >= 3) & (conn.frequencies <= 51))
    spi = CoherencePhase(fs=128, fmin=3, fmax=51)
    expected = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            phases = []
            for f in selected:
                cross = (coefficients[..., f, i]*coefficients[..., f, j].conj()).mean()
                phases.append(math.atan2(cross.imag, cross.real))
            expected[i, j] = math.atan2(math.fsum(map(math.sin, phases)),
                                       math.fsum(map(math.cos, phases)))
    actual = spi.multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    order = [2, 0, 3, 1]
    permuted = spi.multivariate(Data(raw[order], normalise=False))
    np.testing.assert_allclose(permuted, expected[np.ix_(order, order)], atol=1e-13)
    # Force both frequency and row blocking; neither may change the definition.
    with patch.object(conn, "block_bytes", 128):
        blocked = conn.coherence_phase_mean((3, 51))[0]
    np.testing.assert_allclose(blocked, expected, rtol=1e-12, atol=1e-13)
    actual[:] = 123
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-13)


def test_all_bands_share_coherency_and_cache_only_small_summaries():
    data = Data(np.random.default_rng(3).normal(size=(3, 128)), normalise=False)
    with patch.object(SpectralConnectivity, "_mean_cross_spectrum",
                      wraps=SpectralConnectivity._mean_cross_spectrum) as cross:
        for lo, hi in [(0, .5), (0, .25), (.25, .5)]:
            CoherencePhase(fmin=lo, fmax=hi).multivariate(data)
        assert cross.call_count == 1
    conn = spectral_workspace(data, 1)
    with patch.object(conn, "coherence_phase_mean", side_effect=AssertionError("recomputed")):
        CoherencePhase().multivariate(data)

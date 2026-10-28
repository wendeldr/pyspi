"""Shared Fourier endpoints and transform-derived group-delay spacing."""

import numpy as np
import pytest
import yaml

from pyspi._spectral import SpectralConnectivity, spectral_workspace
from pyspi._spectral_compat import spectral_connectivity as sc
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics import spectral


@pytest.mark.parametrize('n', [125, 128])
def test_common_grid_and_transfer_include_dc_and_last_positive_bin(n):
    raw = np.random.default_rng(540).normal(size=(2, n))
    conn = spectral_workspace(Data(raw, normalise=False), 2048)
    transform = sc.Multitaper(raw.T, sampling_frequency=2048)
    coefficients = transform.fft()
    frequencies = np.fft.rfftfreq(coefficients.shape[-2], 1/2048)
    np.testing.assert_array_equal(conn.frequencies, frequencies)
    for f in (0, len(frequencies) - 1):
        x, y = coefficients[0, :, :, f, 0], coefficients[0, :, :, f, 1]
        expected = (x * y.conj()).mean() / np.sqrt((abs(x)**2).mean() * (abs(y)**2).mean())
        np.testing.assert_allclose(conn.coherency()[0, f, 0, 1], expected, atol=1e-13)
        np.testing.assert_allclose(conn.power()[0, f, 0], (abs(x)**2).mean(), atol=1e-13)
    # Factorization identities on the complete two-sided spectrum; independent
    # known-model recovery and causal projection are checked in test_wilson.py.
    factor = conn._minimum_phase_factor
    spectrum = np.einsum("wrkfi,wrkfj->wfij", coefficients, coefficients.conj())/5
    np.testing.assert_allclose(factor @ factor.swapaxes(-1, -2).conj(), spectrum, rtol=1e-9, atol=1e-10)
    lag_zero = np.fft.ifft(factor, axis=1).real[:, 0]
    transfer = factor @ np.linalg.inv(lag_zero)[:, None]
    np.testing.assert_allclose(conn._transfer_function, transfer[:, :len(frequencies)],
                               rtol=1e-10, atol=1e-12)
    for name in ('directed_coherence', 'partial_directed_coherence',
                 'generalized_partial_directed_coherence', 'directed_transfer_function',
                 'direct_directed_transfer_function', 'pairwise_spectral_granger_prediction'):
        assert conn.measure(name).shape == (1, len(frequencies), 2, 2)


@pytest.mark.parametrize('cls,kind', [
    (spectral.CoherenceMagnitude, 'cohmag'), (spectral.CoherencePhase, 'phase'),
    (spectral.ImaginaryCoherence, 'icoh'), (spectral.PhaseLockingValue, 'plv'),
    (spectral.PhaseLagIndex, 'pli'), (spectral.WeightedPhaseLagIndex, 'wpli'),
    (spectral.DebiasedSquaredPhaseLagIndex, 'dspli'),
    (spectral.DebiasedSquaredWeightedPhaseLagIndex, 'dswpli'),
    (spectral.PairwisePhaseConsistency, 'ppc'),
])
@pytest.mark.parametrize('f', [0, 64])
def test_single_endpoint_band_matches_scalar_taper_formula(cls, kind, f):
    raw = np.random.default_rng(541).normal(size=(2, 128))
    coefficients = sc.Multitaper(raw.T, sampling_frequency=128).fft()[0]
    x, y = coefficients[:, :, f, 0].ravel(), coefficients[:, :, f, 1].ravel()
    cross = x * y.conj()
    coherency = cross.mean() / np.sqrt(np.mean(abs(x)**2) * np.mean(abs(y)**2))
    unit = cross / abs(cross)
    imag = cross.imag
    n = cross.size
    with np.errstate(invalid='ignore', divide='ignore'):
        weight = np.mean(abs(imag))
        denominator = abs(imag).sum()**2 - (imag**2).sum()
        expected = {
            'cohmag': abs(coherency)**2, 'phase': np.angle(coherency),
            'icoh': abs(coherency.imag), 'plv': abs(unit.mean()),
            'pli': abs(np.sign(imag).mean()),
            'wpli': abs(imag.mean()/weight) if weight > 0 else np.nan,
            'dspli': (np.sign(imag).sum()**2 - np.count_nonzero(imag)) / (n*(n-1)),
            'dswpli': ((imag.sum()**2 - (imag**2).sum()) / denominator
                      if denominator != 0 else np.nan),
            'ppc': (abs(unit.sum())**2 - n) / (n**2 - n),
        }[kind]
        actual = cls(fs=128, fmin=f, fmax=f).multivariate(Data(raw, normalise=False))
    np.testing.assert_allclose(actual[0, 1], expected, rtol=1e-12, atol=1e-13)


def analytic_delay_connection(fs=100, resolution=6):
    nfft = 100
    frequencies = np.fft.rfftfreq(nfft, 1/fs)
    # Mild curvature makes the fitted result depend on exactly which bins enter.
    phase = .01 * np.arange(frequencies.size) + .0002 * np.arange(frequencies.size)**2
    c = .999 * np.exp(1j * phase)
    conn = SpectralConnectivity(np.zeros((1, 1, 5, nfft, 2), complex),
                                np.fft.fftfreq(nfft, 1/fs), np.array([0.]),
                                frequency_resolution=resolution)
    spectrum = np.full((1, len(c), 2, 2), np.nan, complex)
    spectrum[0, :, 0, 1], spectrum[0, :, 1, 0] = c, c.conj()
    conn.__dict__['_coherency'] = spectrum
    return conn, frequencies, phase


@pytest.mark.parametrize('band', [(0, 12), (6, 24), (0, 48), (1, 19)])
def test_group_delay_regresses_closed_band_at_transform_spacing(band):
    conn, frequencies, phase = analytic_delay_connection()
    indices = np.flatnonzero((frequencies >= band[0]) & (frequencies <= band[1]))[::6]
    x, y = frequencies[indices], phase[indices]
    slope = ((x-x.mean()) * (y-y.mean())).sum() / ((x-x.mean())**2).sum()
    expected_r = np.corrcoef(x, y)[0, 1]
    delay, fitted, rvalue = conn.group_delay(band)
    np.testing.assert_allclose(fitted[0, 0, 1], slope, atol=1e-13)
    np.testing.assert_allclose(delay[0, 0, 1], slope/(2*np.pi), atol=1e-13)
    np.testing.assert_allclose(fitted[0, 1, 0], -slope, atol=1e-13)
    np.testing.assert_allclose(rvalue[0, 0, 1], expected_r, atol=1e-13)
    np.testing.assert_allclose(rvalue[0, 1, 0], -expected_r, atol=1e-13)
    assert np.isnan(delay[0].diagonal()).all()


@pytest.mark.parametrize('scale', [1/100, 3.7, 2048/100])
def test_group_delay_frequency_units_do_not_change_retained_bins(scale):
    base, _, _ = analytic_delay_connection()
    scaled, _, _ = analytic_delay_connection(fs=100*scale, resolution=6*scale)
    expected = base.group_delay((0, 48))
    actual = scaled.group_delay((0, 48*scale))
    for index in (0, 1):
        np.testing.assert_allclose(actual[index]*scale, expected[index], atol=1e-13)
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-13)


@pytest.mark.parametrize('n', [1024, 1025])
def test_group_delay_uses_window_bandwidth_even_with_fft_padding(n):
    raw = np.random.default_rng(542).normal(size=(2, n))
    conn = spectral_workspace(Data(raw, normalise=False), 2048)
    assert conn.frequency_resolution == 6 * 2048 / n
    assert conn.pair(0, 1).frequency_resolution == conn.frequency_resolution


@pytest.mark.parametrize('band', [(1, 4), (4, 8), (8, 12)])
def test_short_eeg_band_has_no_group_delay_estimate(band):
    data = Data(np.random.default_rng(543).normal(size=(2, 1024)), normalise=False)
    with pytest.raises(ValueError, match='at least three frequency bins'):
        spectral.GroupDelay(fs=2048, fmin=band[0], fmax=band[1], statistic='delay').multivariate(data)


def test_calculator_reports_unavailable_group_delay(tmp_path):
    path = tmp_path / 'gd.yaml'
    path.write_text(yaml.safe_dump({'.statistics.spectral': {'GroupDelay': {'configs': [
        dict(fs=2048, fmin=4, fmax=8, statistic='delay')]}}}))
    raw = np.random.default_rng(544).normal(size=(2, 1024))
    calc = Calculator(dataset=Data(raw, normalise=False), configfile=str(path))
    with pytest.warns(UserWarning, match='at least three frequency bins'):
        calc.compute()
    report, = calc.metric_reports.values()
    assert report.status == 'failed'
    assert 'at least three frequency bins' in report.error.message
    assert np.isnan(calc.table.to_numpy()).all()


@pytest.mark.parametrize('method', ['parametric', 'nonparametric'])
def test_granger_zero_lower_bound_includes_dc(method):
    data = Data(np.random.default_rng(545).normal(size=(2, 256)), normalise=False)
    spi = spectral.SpectralGrangerCausality(method=method, order=2, fmin=0, fmax=0)
    values, frequencies = spi._get_cache(data)
    assert frequencies[0] == 0 and spi._fmin == 0
    expected = values[0, 0].copy()
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(spi.multivariate(data), expected, rtol=1e-12, atol=1e-13)
    assert np.isfinite(expected[0, 1]) or np.isfinite(expected[1, 0])
    assert spectral.SpectralGrangerCausality()._fmin == 0
    assert spectral.SpectralGrangerCausality(fs=2048)._fmax == 1024
    with pytest.raises(ValueError):
        spectral.SpectralGrangerCausality(method=method, fmin=-.1)

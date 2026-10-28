"""Independent PAC definition, band ownership, and PEC preprocessing checks."""

import numpy as np
import pytest
from scipy.signal import hilbert

from pyspi._bands import band_inputs
from pyspi.data import Data
from pyspi.statistics.misc import PhaseAmplitudeCoupling, PowerEnvelopeCorrelation


def inputs(*, normalise=True, detrend=True):
    rng = np.random.default_rng(49)
    t = np.arange(1024) / 2048
    phase = np.cos(2*np.pi*np.array([20, 28, 16])[:, None]*t)
    amplitude = (1 + .5*np.cos(2*np.pi*np.array([16, 20, 28])[:, None]*t)) * np.cos(
        2*np.pi*np.array([100, 110, 120])[:, None]*t)
    envelope = rng.normal(size=(3, 1024))
    data = Data(rng.normal(size=(3, 1024)), normalise=normalise, detrend=detrend)
    bands = {(16, 30): phase, (70, 150): amplitude, (30, 70): envelope}
    data.set_filtered_bands(bands, sampling_rate_hz=2048)
    return data, bands


def scalar_pac(phase_samples, amplitude_samples):
    phase = np.angle(hilbert(phase_samples, axis=1))
    amplitude = abs(hilbert(amplitude_samples, axis=1))
    result = np.full((len(phase), len(phase)), np.nan)
    for source in range(len(phase)):
        for target in range(len(phase)):
            if source == target:
                continue
            centered = amplitude[target] - amplitude[target].mean()
            result[source, target] = abs(np.sum(np.exp(1j*phase[source])*centered)) / (
                np.sqrt(len(centered))*np.linalg.norm(centered))
    return result


def test_pac_matches_scalar_prototype_transposed_and_known_modulation():
    data, bands = inputs()
    metric = PhaseAmplitudeCoupling()
    actual = metric.multivariate(data)
    expected = scalar_pac(bands[(16, 30)], bands[(70, 150)])
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert actual[0, 1] == pytest.approx(1/np.sqrt(2), abs=1e-13)
    assert actual[1, 0] < 1e-13
    assert metric.bivariate(data, i=0, j=1) == actual[0, 1]
    assert "directed" in metric.labels and not metric.issigned()


@pytest.mark.parametrize("orth,absolute", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("normalise,detrend", [(False, False), (False, True), (True, False), (True, True)])
def test_banded_pec_reuses_corrected_definition_and_exact_window_preprocessing(orth, absolute, normalise, detrend):
    data, bands = inputs(normalise=normalise, detrend=detrend)
    expected = PowerEnvelopeCorrelation(orth=orth, absolute=absolute).multivariate(
        Data(bands[(30, 70)].copy(), normalise=normalise, detrend=detrend))
    actual = PowerEnvelopeCorrelation(orth=orth, absolute=absolute, band=(30, 70)).multivariate(data)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_permutation_units_and_owned_results():
    data, bands = inputs()
    metric = PhaseAmplitudeCoupling()
    original = metric.multivariate(data)
    original[0, 1] = -100
    assert metric.multivariate(data)[0, 1] > 0
    original = metric.multivariate(data)
    order = np.array([2, 0, 1])
    other = Data(data.to_numpy(realisation=0)[order].copy(), normalise=False)
    other.set_filtered_bands({key: values[order] * np.array([1e-250, 1e250, 1])[:, None]
                             for key, values in bands.items()}, sampling_rate_hz=2048)
    np.testing.assert_allclose(metric.multivariate(other), original[np.ix_(order, order)], atol=2e-14)


def test_band_ownership_replacement_and_data_mutation():
    data, bands = inputs()
    metric = PhaseAmplitudeCoupling()
    expected = metric.multivariate(data)
    np.testing.assert_array_equal(metric.multivariate(data, inplace=False), expected)
    bands[(70, 150)][:] = 0
    np.testing.assert_array_equal(metric.multivariate(data), expected)
    data.set_filtered_bands(bands, sampling_rate_hz=2048)
    assert np.isnan(metric.multivariate(data)).all()
    data.to_numpy()[0, 0, 0] += 1
    with pytest.raises(ValueError, match="no longer match"):
        metric.multivariate(data)
    data.set_data(np.random.default_rng(5).normal(size=(3, 1024)))
    with pytest.raises(ValueError, match="requires continuously filtered"):
        metric.multivariate(data)


def test_transforms_share_only_with_matching_preprocessing():
    data, _ = inputs(normalise=False, detrend=False)
    work = band_inputs(data)
    assert work.envelope((30, 70), preprocess=True) is work.envelope((30, 70), preprocess=False)
    other, _ = inputs(normalise=True, detrend=True)
    work = band_inputs(other)
    assert work.envelope((30, 70), preprocess=True) is not work.envelope((30, 70), preprocess=False)
    before = len(work.workspaces)
    for _ in range(3):
        PowerEnvelopeCorrelation(band=(30, 70), orth=True, absolute=True).multivariate(other)
    assert len(work.workspaces) == before


def test_degenerate_and_nonfinite_inputs_report_missing_not_zero():
    data, bands = inputs()
    bands[(16, 30)][0] = 0
    bands[(70, 150)][2] = np.nan
    bands[(30, 70)][1] = 0
    data.set_filtered_bands(bands, sampling_rate_hz=2048)
    pac = PhaseAmplitudeCoupling()
    assert np.isnan(pac.multivariate(data)[0]).all()
    assert np.isnan(pac.multivariate(data)[:, 2]).all()
    assert pac.diagnostics(data)["reason"][0, 1] == "unresolved_phase"
    assert pac.diagnostics(data)["reason"][1, 2] == "nonfinite_amplitude_input"
    pec = PowerEnvelopeCorrelation(band=(30, 70))
    assert np.isnan(pec.multivariate(data)[1]).all()
    assert np.isfinite(pec.multivariate(data)[0, 2])
    assert pec.diagnostics(data)["reason"][0, 1] == "unusable_band_window_variance"


def test_constant_phase_and_pure_carrier_amplitude_are_unavailable():
    data, bands = inputs()
    bands[(16, 30)][0] = 1
    bands[(70, 150)][1] = np.cos(2*np.pi*100*np.arange(1024)/2048)
    data.set_filtered_bands(bands, sampling_rate_hz=2048)
    metric = PhaseAmplitudeCoupling()
    values = metric.multivariate(data)
    assert np.isnan(values[0]).all() and np.isnan(values[:, 1]).all()
    assert metric.diagnostics(data)["reason"][0, 2] == "unresolved_phase"
    assert metric.diagnostics(data)["reason"][2, 1] == "constant_amplitude"


@pytest.mark.parametrize("band", [(0, 10), (30, 20), (True, 40), (16, np.inf), (16,), "16-30"])
def test_bad_band_definitions_fail(band):
    with pytest.raises(ValueError):
        PowerEnvelopeCorrelation(band=band)


def test_missing_and_incompatible_band_windows_fail_explicitly():
    data = Data(np.random.default_rng(1).normal(size=(3, 100)))
    with pytest.raises(ValueError, match="requires continuously filtered"):
        PowerEnvelopeCorrelation(band=(30, 70)).multivariate(data)
    with pytest.raises(ValueError, match="match Data"):
        data.set_filtered_bands({(30, 70): np.zeros((3, 99))}, sampling_rate_hz=2048)
    with pytest.raises(ValueError, match="Nyquist"):
        data.set_filtered_bands({(30, 70): np.zeros((3, 100))}, sampling_rate_hz=100)
    with pytest.raises(ValueError, match="ordered and nonoverlapping"):
        PhaseAmplitudeCoupling(phase_band=(16, 80))

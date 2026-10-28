"""Shared directed planning, native grids and pair-local numerical diagnostics."""

from unittest.mock import patch

import numpy as np
import pytest
import yaml

from pyspi import _spectral as implementation
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics import spectral


CLASSES = [spectral.DirectedCoherence, spectral.DirectedTransferFunction,
           spectral.PartialDirectedCoherence, spectral.GeneralizedPartialDirectedCoherence,
           spectral.DirectDirectedTransferFunction, spectral.SpectralGrangerCausality]


@pytest.mark.parametrize("cls", CLASSES)
def test_singular_pair_isolated_and_diagnostics_detached(cls):
    raw = np.random.default_rng(62).normal(size=(3, 256))
    raw[2] = raw[0]
    data = Data(raw, normalise=False)
    spi = cls()
    # SGC retains its existing warning when unavailable-frequency fraction is high.
    with np.errstate(invalid="ignore"):
        actual = spi.multivariate(data)
    assert np.isnan(actual[0, 2]) and np.isnan(actual[2, 0])
    assert np.isfinite(actual[0, 1]) and np.isfinite(actual[1, 0])
    expected = cls().multivariate(Data(raw[:2], normalise=False))
    np.testing.assert_allclose(actual[:2, :2], expected, atol=1e-12)
    diag = spi.diagnostics(data)
    assert diag["reason"][0, 2] == "singular_spectrum"
    assert diag["reason"][0, 1] == "estimated"
    assert diag["reconstruction_error"][0, 1] < 1e-10
    diag["reason"][:] = "changed"
    actual[:] = 123
    assert spi.diagnostics(data)["reason"][0, 1] == "estimated"
    np.testing.assert_allclose(spi.multivariate(data)[:2, :2], expected, atol=1e-12)


def test_parametric_pair_failure_keeps_healthy_pairs():
    raw = np.random.default_rng(63).normal(size=(3, 256))
    raw[2] = raw[0]
    data = Data(raw, normalise=False)
    spi = spectral.SpectralGrangerCausality(method="parametric", order=2)
    actual = spi.multivariate(data)
    assert np.isnan(actual[0, 2]) and np.isnan(actual[2, 0])
    assert np.isfinite(actual[0, 1]) and np.isfinite(actual[1, 0])
    expected = spi.multivariate(Data(raw[:2], normalise=False))
    np.testing.assert_allclose(actual[:2, :2], expected, atol=1e-13)
    assert spi.diagnostics(data)["reason"][0, 2] in ("model_fit_failed", "undefined_frequency_values")


@pytest.mark.parametrize("method", ["parametric", "nonparametric"])
def test_unsupported_granger_realisations_raise_instead_of_reporting_successful_nan(method):
    data = Data(np.random.default_rng(69).normal(size=(3, 128, 2)), dim_order="psr", normalise=False)
    spi = spectral.SpectralGrangerCausality(method=method, order=2)
    with pytest.raises(ValueError, match="single realisation"):
        spi.multivariate(data)


def test_native_directed_grid_reuses_tapers_without_fft_speed_padding():
    raw = np.random.default_rng(64).normal(size=(2, 129))
    data = Data(raw, normalise=False)
    ordinary = implementation.spectral_workspace(data, 2048)
    native = implementation.directed_workspace(data, 2048)
    assert ordinary.fourier_coefficients.shape[-2] > 129
    assert native.fourier_coefficients.shape[-2] == 129
    reference = implementation.sc.Multitaper(raw.T, sampling_frequency=2048, n_fft_samples=129)
    np.testing.assert_allclose(native.fourier_coefficients, reference.fft(), atol=1e-14, rtol=1e-12)
    np.testing.assert_array_equal(native.frequencies, np.fft.rfftfreq(129, 1/2048))
    assert implementation.directed_workspace(data, 2048) is native
    assert implementation.spectral_workspace(data, 2048) is ordinary
    values, frequencies = spectral.DirectDirectedTransferFunction(fs=2048)._get_cache(data)
    assert values.shape[1] == len(frequencies) == 65
    np.testing.assert_array_equal(frequencies, native.frequencies)


def test_calculator_plans_only_enabled_families_and_streams_all_bands(tmp_path):
    config = {".statistics.spectral": {
        "DirectedCoherence": {"configs": [{"fmin": 0, "fmax": .25}, {"fmin": .25, "fmax": .5}]},
        "GeneralizedPartialDirectedCoherence": {"configs": [{"fmin": 0, "fmax": .25}]},
        "SpectralGrangerCausality": {"configs": [{"fmin": 0, "fmax": .5}]},
    }}
    path = tmp_path/"directed.yaml"
    path.write_text(yaml.safe_dump(config))
    data = Data(np.random.default_rng(65).normal(size=(4, 128)), normalise=False)
    calculator = Calculator(dataset=data, configfile=str(path))
    with patch.object(implementation, "factorize", wraps=implementation.factorize) as fit:
        with patch.object(implementation.SpectralConnectivity, "direct_directed_transfer_function",
                          side_effect=AssertionError("Unrequested dDTF must not be computed")):
            calculator.compute()
    assert sum(call.args[0].shape[0] for call in fit.call_args_list) == 6
    assert all(report.status == "success" for report in calculator.metric_reports.values())
    for spi in calculator.spis.values():
        fresh = Data(data.to_numpy(realisation=0).copy(), normalise=False)
        np.testing.assert_allclose(calculator.table[spi.identifier].to_numpy(), spi.multivariate(fresh), atol=1e-12)


def test_bivariate_reverse_uses_same_fit_and_mutation_invalidates_it():
    data = Data(np.random.default_rng(66).normal(size=(3, 128)), normalise=False)
    spi = spectral.DirectedCoherence()
    with patch.object(implementation, "factorize", wraps=implementation.factorize) as fit:
        forward = spi.bivariate(data, i=2, j=0)
        reverse = spi.bivariate(data, i=0, j=2)
        assert sum(call.args[0].shape[0] for call in fit.call_args_list) == 1
    matrix = spi.multivariate(data)
    assert forward == pytest.approx(matrix[2, 0]) and reverse == pytest.approx(matrix[0, 2])
    data.to_numpy()[0, :, 0] = np.random.default_rng(67).normal(size=128)
    expected = spi.multivariate(Data(data.to_numpy().copy(), dim_order="psr", normalise=False))
    np.testing.assert_allclose(spi.multivariate(data), expected, atol=1e-12)


def test_empty_band_diagnostics_are_not_overwritten_by_other_band_fit():
    data = Data(np.random.default_rng(68).normal(size=(3, 128)), normalise=False)
    spectral.DirectedCoherence().multivariate(data)
    spi = spectral.DirectedCoherence(fmin=.001, fmax=.002)
    assert np.isnan(spi.multivariate(data)).all()
    reasons = spi.diagnostics(data)["reason"]
    assert reasons[0, 1] == "empty_frequency_band"
    assert reasons[0, 0] == "diagonal"

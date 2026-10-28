"""Integration checks for naming and feasibility guards across the newer work."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from pyspi._naming import parse
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics.basic import Covariance
from pyspi.statistics.distance import DistanceCorrelation, HistogramRelativeEntropy, PairwiseDistance
from pyspi.statistics.misc import PhaseAmplitudeCoupling, PowerEnvelopeCorrelation
from pyspi.statistics.spectral import CoherenceMagnitude, PhaseSlopeIndex, SpectralGrangerCausality


def test_new_candidates_have_parseable_names_and_distinct_settings():
    spis = [HistogramRelativeEntropy(), HistogramRelativeEntropy(pseudocount=1e-9),
            PowerEnvelopeCorrelation(band=(30, 70)),
            PowerEnvelopeCorrelation(band=(30, 70), orth=True, absolute=True),
            PhaseAmplitudeCoupling(), PhaseAmplitudeCoupling(phase_band=(8, 12))]
    assert len({s.identifier for s in spis}) == len(spis)
    for spi in spis:
        parse(spi.identifier)
    assert parse(spis[2].identifier)[2]["band"] == ((30., 70.), "hz")
    assert parse(spis[4].identifier)[2]["phaseband"] == ((16., 30.), "hz")


def test_native_distance_correlation_rejects_undefined_three_sample_case():
    spi = DistanceCorrelation()
    assert spi.input_problem(3, 2)
    assert spi.input_problem(4, 2) is None
    assert np.isnan(spi.multivariate(Data(np.array([[0., 1., 2.], [2., 0., 1.]]), normalise=False))).all()


def test_empirical_reliability_margins_are_not_impossibility_checks():
    assert Covariance(estimator="GraphicalLasso").input_problem(4, 8) is None
    assert SpectralGrangerCausality(method="parametric", order=2).input_problem(10, 2) is None
    # Bray-Curtis and ordinary PEC are defined on these short unnormalized inputs.
    data = Data(np.array([[1., 2., 4.], [2., 5., 1.]]), normalise=False, detrend=False)
    for spi in (PairwiseDistance(metric="braycurtis"), PowerEnvelopeCorrelation()):
        assert spi.input_problem(3, 2) is None
        assert np.isfinite(spi.multivariate(data)[0, 1])


def test_compute_checks_metrics_added_after_loading_before_any_calculation(tmp_path, monkeypatch):
    config = tmp_path / "metrics.yaml"
    config.write_text(yaml.safe_dump({".statistics.basic": {"Covariance": {"configs": [{}]}}}))
    calc = Calculator(dataset=np.random.default_rng(2).normal(size=(2, 1024)), configfile=config)
    bad = PhaseSlopeIndex(fs=2048, fmin=8, fmax=9)
    calc.spis[bad.identifier] = bad
    def forbidden(*args, **kwargs):
        pytest.fail("a metric ran before the invalid selection was rejected")
    for spi in calc.spis.values():
        monkeypatch.setattr(spi, "multivariate", forbidden)
    with pytest.raises(ValueError, match="needs 2"):
        calc.compute()


@pytest.mark.parametrize("samples", [7, 63, 101, 1024])
def test_guard_grid_matches_actual_multitaper_transform(samples):
    from pyspi._spectral import spectral_workspace
    data = Data(np.random.default_rng(4).normal(size=(2, samples)), normalise=False)
    actual = spectral_workspace(data, 2048).frequencies
    expected = CoherenceMagnitude(fs=2048)._band_grid(samples)
    np.testing.assert_array_equal(expected, actual)


def test_wavelet_psi_cannot_be_mistaken_for_valid_band_selection():
    from pyspi.statistics.wavelet import PhaseSlopeIndex as WaveletPSI
    assert "frequency indices select time samples" in WaveletPSI().input_problem(1024, 2)


def test_identifier_numbers_preserve_large_integers_and_zero():
    from pyspi._naming import number
    assert number(2**53+1) != number(2**53)
    assert number(-0.) == number(0.) == "0"
    assert number(np.float32(.1)) != number(.1)


def test_calculator_rejects_duplicate_output_names(tmp_path):
    config = tmp_path / "duplicate.yaml"
    config.write_text(yaml.safe_dump({".statistics.basic": {"Covariance": {"configs": [{}, {}]}}}))
    with pytest.raises(ValueError, match="Duplicate SPI identifier"):
        Calculator(configfile=config)


def test_reviewed_hz_conversion_preserves_all_selected_spectral_values():
    import importlib
    root = Path(__file__).resolve().parents[2]
    selected = yaml.safe_load((root / "pyspi/workflow/reviewed.yaml").read_text())[".statistics.spectral"]
    data = Data(np.random.default_rng(83).normal(size=(3, 1024)), normalise=False)
    module = importlib.import_module("pyspi.statistics.spectral")
    for cls, settings in selected.items():
        for params in settings["configs"]:
            assert params["fs"] == 2048
            normalized = {**params, "fs": 1, "fmin": params.get("fmin", 0)/2048,
                          "fmax": params.get("fmax", 1024)/2048}
            hz, cycles = getattr(module, cls)(**params), getattr(module, cls)(**normalized)
            assert hz.input_problem(1024, 3) is None
            np.testing.assert_allclose(hz.multivariate(data), cycles.multivariate(data), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("cls,params,minimum", [
    ("MutualInfo", {"dyn_corr_excl": 3}, 6),
    ("TimeLaggedMutualInfo", {}, 3),
    ("TimeLaggedMutualInfo", {"dyn_corr_excl": 3}, 7),
    ("TransferEntropy", {"k_history": 2}, 4),
    ("TransferEntropy", {"k_history": 2, "dyn_corr_excl": 3}, 8),
])
def test_kernel_guards_match_native_alignment_and_fixed_exclusion(cls, params, minimum):
    from pyspi.statistics import infotheory
    spi = getattr(infotheory, cls)(estimator="kernel", **params)
    assert spi.input_problem(minimum-1, 2)
    assert spi.input_problem(minimum, 2) is None
    raw = np.random.default_rng(11).normal(size=(2, minimum))
    assert np.isfinite(spi.multivariate(Data(raw, normalise=False))[0, 1])

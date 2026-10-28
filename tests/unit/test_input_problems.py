"""Directed.input_problem: SPIs that cannot be estimated for a window shape are refused before computing."""

import contextlib
import importlib
import io

import numpy as np
import pytest
import yaml

from pyspi.base import input_problems
from pyspi.calculator import Calculator
from pyspi.data import Data

#: (module, class, parameters, shortest window with an estimate). Each limit
#: is derived from the calculation and checked here against computing it.
LIMITS = [
    ("basic", "Covariance", {"estimator": "GraphicalLassoCV"}, 5),
    ("distance", "MultiscaleGraphCorrelation", {}, 5),
    ("distance", "DistanceCorrelation", {}, 4),
    ("distance", "HilbertSchmidtIndependenceCriterion", {}, 3),
    ("distance", "CrossDistanceCorrelation", {"max_lag": 3}, 6),
    ("distance", "CrossMultiscaleGraphCorrelation", {"max_lag": 3}, 8),
    ("causal", "ConvergentCrossMapping", {"embedding_dimension": 2}, 43),
    ("misc", "Cointegration", {"det_order": 0, "k_ar_diff": 2}, 12),
    ("misc", "Cointegration", {"det_order": -1, "k_ar_diff": 2}, 11),
    ("misc", "Cointegration", {"method": "aeg", "statistic": "tstat", "maxlag": 5}, 12),
    ("misc", "Cointegration", {"method": "aeg", "statistic": "tstat", "trend": "ctt", "maxlag": 1}, 5),
    ("spectral", "CoherenceMagnitude", {}, 7),
    ("spectral", "DirectedTransferFunction", {}, 7),
    ("infotheory", "TransferEntropy", {"k_history": 2, "k_tau": 3}, 9),
    ("infotheory", "TransferEntropy", {"l_history": 3, "l_tau": 2}, 11),
    ("infotheory", "TransferEntropy",
     {"estimator": "symbolic", "pattern_length": 4, "pattern_spacing": 2, "prediction_shift": 3}, 10),
    ("infotheory", "CrossmapEntropy", {"history_length": 3}, 8),
    ("infotheory", "CrossmapEntropy", {"history_length": 3, "estimator": "kozachenko"}, 5),
    ("infotheory", "CausalEntropy", {"n": 2}, 6),
    ("infotheory", "DirectedInfo", {"n": 5}, 15),
    ("infotheory", "StochasticInteraction", {"delay": 3}, 8),
    ("infotheory", "TimeLaggedMutualInfo", {}, 4),
    ("infotheory", "MutualInfo", {}, 3),
    ("infotheory", "IntegratedInformation", {"delay": 3}, 7),
]


def _spi(module, cls, params):
    with contextlib.redirect_stdout(io.StringIO()):
        return getattr(importlib.import_module(f"pyspi.statistics.{module}"), cls)(**params)


def _complete(module, cls, params, n_observations, seed):
    """Every off-diagonal value is finite (no error, no NaN or inf pair)."""
    raw = np.random.default_rng(1000 * seed + 7 * n_observations).normal(size=(3, n_observations))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            values = np.asarray(_spi(module, cls, params).multivariate(Data(raw, normalise=True)), dtype=float)
    except Exception:
        return False
    return bool(np.isfinite(values[~np.eye(3, dtype=bool)]).all())


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("module,cls,params,minimum", LIMITS,
                         ids=[f"{c}-{'-'.join(map(str, p.values()))}" for _, c, p, _ in LIMITS])
def test_limit_is_where_the_calculation_becomes_complete(module, cls, params, minimum):
    spi = _spi(module, cls, params)
    problem = spi.input_problem(minimum - 1, 3)
    assert problem is not None and problem.startswith(spi.identifier + ":")
    assert spi.input_problem(minimum, 3) is None
    seeds = range(6)
    assert all(_complete(module, cls, params, minimum, seed) for seed in seeds)
    assert not all(_complete(module, cls, params, minimum - 1, seed) for seed in seeds)


@pytest.mark.parametrize("cls,params,n_observations,refused", [
    ("PhaseSlopeIndex", {"fs": 2048, "fmin": 1, "fmax": 4}, 512, True),    # one 4 Hz bin; PSI needs two
    ("PhaseSlopeIndex", {"fs": 2048, "fmin": 1, "fmax": 4}, 1024, False),  # bins at 2 and 4 Hz
    ("CoherenceMagnitude", {"fs": 2048, "fmin": 1, "fmax": 4}, 256, True),  # 8 Hz grid has nothing in 1-4
    ("CoherenceMagnitude", {"fs": 2048, "fmin": 1, "fmax": 4}, 1024, False),
    ("GroupDelay", {"fs": 2048, "fmin": 1, "fmax": 4}, 1024, True),        # needs 3 points 12 Hz apart
    ("GroupDelay", {"fs": 2048, "fmin": 30, "fmax": 70}, 1024, False),
    ("DirectedCoherence", {"fs": 2048, "fmin": 1, "fmax": 2}, 512, True),   # native grid every 4 Hz
    ("SpectralGrangerCausality", {"fs": 2048, "fmin": 1, "fmax": 1.5, "method": "parametric", "order": 2}, 1024, True),
])
def test_spectral_bands_are_checked_on_the_calculation_grid(cls, params, n_observations, refused):
    spi = _spi("spectral", cls, params)
    assert (spi.input_problem(n_observations, 3) is not None) == refused


def test_wavelet_bands_outside_the_wavelet_grid_and_single_epoch_estimators_are_refused():
    assert _spi("wavelet", "CoherenceMagnitude", {"fs": 2048, "fmin": 30, "fmax": 70}).input_problem(1024, 3)
    assert _spi("wavelet", "CoherenceMagnitude", {"fs": 2048, "fmin": 500, "fmax": 700}).input_problem(1024, 3) is None
    for cls in ("PairwisePhaseConsistency", "DebiasedSquaredWeightedPhaseLagIndex"):
        assert "one epoch" in _spi("wavelet", cls, {}).input_problem(10_000, 3)


@pytest.mark.parametrize("metric,n_observations,n_processes,refused", [
    ("haversine", 2, 3, False), ("haversine", 64, 3, True),
    ("mahalanobis", 4, 8, False), ("mahalanobis", 8, 8, True), ("euclidean", 2, 2, False),
])
def test_pairwise_distance_metrics_with_shape_requirements(metric, n_observations, n_processes, refused):
    spi = _spi("distance", "PairwiseDistance", {"metric": metric})
    assert (spi.input_problem(n_observations, n_processes) is not None) == refused


@pytest.mark.parametrize("metric", ["wminkowski", "precomputed"])
def test_pairwise_distance_metrics_that_never_apply_are_rejected(metric):
    with pytest.raises(ValueError, match="cannot use metric"):
        _spi("distance", "PairwiseDistance", {"metric": metric})


def test_unimplemented_phi_types_are_rejected():
    with pytest.raises(ValueError, match="star.*Geo"):
        _spi("infotheory", "IntegratedInformation", {"phitype": "MI1"})


def test_calculator_refuses_unestimable_spis_before_attaching_data(tmp_path):
    config = tmp_path / "spis.yaml"
    config.write_text(yaml.safe_dump({".statistics.spectral": {"PhaseSlopeIndex": {"configs": [
        {"fs": 2048, "fmin": 1, "fmax": 4}, {"fs": 2048, "fmin": 30, "fmax": 70}]}},
        ".statistics.basic": {"Covariance": {"configs": [{"estimator": "EmpiricalCovariance"}]}}}))
    raw = np.random.default_rng(3).normal(size=(3, 512))
    with pytest.raises(ValueError, match=r"1 SPI\(s\) cannot be estimated from 512-observation") as caught:
        Calculator(dataset=raw, configfile=str(config))
    assert "psi_mean_band0001-0004hz" in str(caught.value) and "band0030-0070hz" not in str(caught.value)

    calc = Calculator(configfile=str(config))
    with pytest.raises(ValueError):
        calc.load_dataset(raw)
    assert not hasattr(calc, "_dataset")
    calc.load_dataset(np.random.default_rng(3).normal(size=(3, 1024)))
    assert calc.dataset.n_observations == 1024


def test_input_problems_lists_one_reason_per_refused_spi():
    spis = [_spi("spectral", "CoherenceMagnitude", {}), _spi("basic", "Covariance", {})]
    assert len(input_problems(spis, 6, 3)) == 1 and input_problems(spis, 7, 3) == []

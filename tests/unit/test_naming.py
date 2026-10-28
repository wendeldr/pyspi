"""SPI identifier grammar: filename-safe fields, parseable parameters and Hz bands."""

import csv
import json
import re
from pathlib import Path

import pytest

from pyspi._naming import compose, duration, frequency_band, number, parameter, parse
from pyspi.statistics.basic import CrossCorrelation
from pyspi.statistics.spectral import CoherenceMagnitude, SignedPhaseLagIndex, SpectralGrangerCausality
from pyspi.statistics.wavelet import PhaseSlopeIndex

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("value,width,expected", [
    (30, 4, "0030"), (1024.0, 4, "1024"), (0.05, 1, "0p05"), (0.5, 4, "0000p5"),
    (0.00048828125, 1, "0p00048828125"), (1e-7, 1, "0p0000001"), (123, 2, "123"),
])
def test_number_is_exact_padded_and_filename_safe(value, width, expected):
    assert number(value, width) == expected


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_number_rejects_values_without_a_safe_spelling(value):
    with pytest.raises((ValueError, TypeError)):
        number(value)


def test_band_uses_hz_with_a_rate_and_cycles_per_sample_without():
    assert frequency_band(30, 70, 2048) == "band0030-0070hz"
    assert frequency_band(0.5, 4, 2048) == "band0000p5-0004hz"
    assert frequency_band(0.25, 0.5, 1) == "band0p25-0p5cyc"


def test_compose_and_parse_round_trip():
    name = compose("xcorr", "sq", "holm", "mean", duration("maxlag", 0.125), parameter("alpha", 0.05, 1))
    assert name == "xcorr_sq_holm_mean_maxlag0125ms_alpha0p05"
    assert parse(name) == ("xcorr", ["sq", "holm", "mean"], {"maxlag": (125.0, "ms"), "alpha": (0.05, "")})
    assert parse("cohmag_mean_band0030-0070hz") == ("cohmag", ["mean"], {"band": ((30.0, 70.0), "hz")})


@pytest.mark.parametrize("fields", [("Mean",), ("max-lag",), ("band0p5.1",), ("a_b",), ("",)])
def test_compose_rejects_fields_outside_the_grammar(fields):
    with pytest.raises(ValueError):
        compose("cohmag", *fields)


def test_spi_identifiers_name_their_settings():
    assert CoherenceMagnitude(fs=2048, fmin=30, fmax=70).identifier == "cohmag_mean_band0030-0070hz"
    assert CoherenceMagnitude(fs=2048).identifier == "cohmag_mean_band0000-1024hz"
    assert SignedPhaseLagIndex(fs=2048, fmin=1, fmax=4, statistic="max").identifier == "pli_signed_max_band0001-0004hz"
    assert PhaseSlopeIndex(fs=2048, fmin=30, fmax=70).identifier == "psi_wavelet_mean_band0030-0070hz"
    assert (SpectralGrangerCausality(fs=2048, fmin=30, fmax=70, method="parametric").identifier
            == "sgc_parametric_autoorder_mean_band0030-0070hz_maxorder50")
    assert CrossCorrelation(statistic="max", sigonly=False, fs=2048).identifier == "xcorr_max_maxlag0125ms"


def _check_family(spis):
    names = [spi.identifier for spi in spis]
    assert len(set(names)) == len(names), names
    for name in names:
        assert re.fullmatch(r"[a-z0-9_-]+", name), name
        parse(name)
    return names


def test_option_values_with_digits_or_none_are_named():
    from sklearn.metrics import pairwise
    from pyspi.statistics.distance import PairwiseDistance
    from pyspi.statistics.infotheory import IntegratedInformation
    from pyspi.statistics.misc import Cointegration, GPModel, InterDependenceScore

    names = _check_family(PairwiseDistance(metric=m) for m in sorted(pairwise._VALID_METRICS))
    assert {"pdist_lone", "pdist_ltwo", "pdist_cityblock"} <= set(names)
    _check_family(GPModel(kernel=k) for k in ("RBF", "Matern", "RationalQuadratic", "ExpSineSquared",
                                               "DotProduct", "WhiteKernel", "ConstantKernel"))
    assert IntegratedInformation(phitype="MI1").identifier == "phi_mione_delay01"
    _check_family(IntegratedInformation(phitype=p, normalization=n) for p in ("star", "Geo", "MI1") for n in (0, 1))
    _check_family(Cointegration(method="aeg", statistic="tstat", trend=t, autolag=a, maxlag=m)
                  for t in ("n", "c", "ct", "ctt") for a in ("aic", "bic", "t-stat", None) for m in (None, 10))
    _check_family(Cointegration(det_order=d, k_ar_diff=k) for d in (-1, 0, 1, 2) for k in (1, 10))
    assert Cointegration(method="aeg", statistic="tstat", maxlag=None).identifier == \
        "coint_aeg_constant_autolagaic_defaultmaxlag_tstat"
    _check_family(InterDependenceScore(pnorm=p) for p in ("max", 1, 2))


def test_options_outside_the_shipped_configs_are_named():
    from pyspi.statistics.basic import Covariance
    from pyspi.statistics.causal import ConvergentCrossMapping
    from pyspi.statistics.distance import Barycenter
    from pyspi.statistics.infotheory import TransferEntropy
    from pyspi.statistics.misc import LinearModel
    from pyspi.statistics.spectral import GroupDelay

    _check_family(GroupDelay(statistic=s) for s in ("delay", "slope", "rvalue"))
    _check_family(Barycenter(mode=m, statistic=s, squared=q) for m in ("euclidean", "dtw", "sgddtw", "softdtw")
                  for s in ("mean", "max", "max_time") for q in (False, True))
    _check_family(Covariance(estimator=e, squared=q) for e in (
        "EmpiricalCovariance", "EllipticEnvelope", "GraphicalLasso", "GraphicalLassoCV", "LedoitWolf",
        "MinCovDet", "OAS", "ShrunkCovariance") for q in (False, True))
    _check_family([LinearModel("Ridge"), LinearModel("Ridge", penalty_scale="mean"), LinearModel("Lasso"),
                   LinearModel("ElasticNet", l1_ratio=0.2), LinearModel("BayesianRidge")])
    _check_family(ConvergentCrossMapping(statistic=s, embedding_dimension=e)
                  for s in ("mean", "max", "diff") for e in (None, 3))
    _check_family([SpectralGrangerCausality(), SpectralGrangerCausality(ignore_nan=False),
                   SpectralGrangerCausality(method="parametric", order=3),
                   SpectralGrangerCausality(method="parametric", max_order=20)])
    _check_family(CrossCorrelation(statistic=s, squared=q, sigonly=g, fs=f)
                  for s in ("mean", "max", "min", "peak") for q in (False, True) for g in (False, True)
                  for f in (1, 2048) if not (s == "peak" and q))
    assert TransferEntropy(estimator="kraskov", dyn_corr_excl=5).identifier == (
        "te_kraskov_neighbours04_theiler05_targethistory01_targetdelay01_sourcehistory01_sourcedelay01")


def test_wavelet_cache_separates_sampling_rates():
    import numpy as np
    from pyspi.data import Data
    from pyspi.statistics.wavelet import CoherenceMagnitude, PhaseSlopeIndex

    raw = np.random.default_rng(5).normal(size=(3, 256))
    for cls in (CoherenceMagnitude, PhaseSlopeIndex):
        shared = Data(raw, normalise=False)
        at_one = cls(fmin=0.1, fmax=0.4).multivariate(shared)
        at_rate = cls(fs=256, fmin=0.1 * 256, fmax=0.4 * 256).multivariate(shared)
        alone = cls(fs=256, fmin=0.1 * 256, fmax=0.4 * 256).multivariate(Data(raw, normalise=False))
        np.testing.assert_array_equal(at_rate, alone)
        np.testing.assert_array_equal(at_one, alone)


def test_every_catalogued_identifier_fits_the_grammar_and_the_mapping():
    names = json.loads((ROOT / "tests/regression/goldens/small/meta.json").read_text())["calculator_order"]
    names += json.loads((ROOT / "tests/regression/goldens/extra/small/meta.json").read_text())["calculator_order"]
    with open(ROOT / "docs/metric_names.csv") as f:
        rows = list(csv.DictReader(f))
    mapped = {row["new_identifier"] for row in rows}
    # One new name per row; an old name maps once per configuration file
    # (the fs=1 full band is cycles per sample in the library configs and
    # 0-1024 Hz in the 2048 Hz workflow preset).
    assert len(mapped) == len(rows)
    for config in {c for row in rows for c in row["configs"].split(";")}:
        olds = [row["old_identifier"] for row in rows if config in row["configs"].split(";")]
        assert len(olds) == len(set(olds)), config
    for name in names:
        assert re.fullmatch(r"[a-z0-9_-]+", name), name
        parse(name)
        assert name in mapped, name

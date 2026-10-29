"""Independent formulas, alignment, conditioning and shared-cache contracts."""

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from sklearn.covariance import LedoitWolf

from pyspi.data import Data
from pyspi.statistics.basic import (FixedLagCorrelation, ThresholdEventExcess,
                                    LogScaleRatio, PartialCorrelation, Precision)
from pyspi.statistics.distance import HistogramRelativeEntropy, HistogramJensenShannon


def data(x):
    return Data(np.array(x, copy=True), normalise=False)


def test_lag_alignment_and_overlap_specific_normalization():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(4, 123))
    x[1, 20:] = x[0, :-20]
    x[:, :20] += np.arange(4)[:, None]*10
    metric = FixedLagCorrelation()
    actual = metric.multivariate(data(x))
    for i in range(4):
        for j in range(4):
            if i != j:
                expected = np.corrcoef(x[i, :-20], x[j, 20:])[0, 1]
                assert actual[i, j] == pytest.approx(expected, abs=1e-14)
    # A separate exact delayed copy verifies the positive-lag source direction.
    x[1, 20:] = x[0, :-20]
    actual = metric.multivariate(data(x))
    assert actual[0, 1] == pytest.approx(1.)
    assert abs(actual[1, 0]) < .5
    assert "9p765625ms" in metric.identifier


def test_event_indicators_with_independent_expectation():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 1024))
    x[:, ::37] += 10
    x[1] *= -5
    for threshold in (1., 3., 4.):
        z = (x-x.mean(axis=1, keepdims=True))/x.std(axis=1, ddof=1, keepdims=True)
        events = abs(z) > threshold
        actual = ThresholdEventExcess(threshold).multivariate(data(x))
        for i in range(4):
            for j in range(i+1, 4):
                expected = np.mean(events[i] & events[j])-np.mean(events[i])*np.mean(events[j])
                assert actual[i, j] == pytest.approx(expected, abs=1e-15)
    x = np.tile(np.arange(10.), (2, 1))
    assert ThresholdEventExcess().multivariate(data(x))[0, 1] == 0.  # no events is valid


def test_log_scale_ratio_extreme_units_and_direction():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(4, 71))*np.array([1e-200, 1e200, 2., 4.])[:, None]
    extended = x.astype(np.longdouble)
    logsd = np.log(extended.std(axis=1))
    expected = logsd[None, :]-logsd[:, None]
    np.fill_diagonal(expected, np.nan)
    actual = LogScaleRatio().multivariate(data(x))
    np.testing.assert_allclose(actual, expected.astype(float), atol=1e-12)
    np.testing.assert_allclose(actual, -actual.T, atol=1e-14)
    assert actual[0, 1] > 900  # no artificial voltage floor, no overflow


def test_partial_correlation_independent_inverse_and_shared_fit(monkeypatch):
    import pyspi.statistics.basic as module
    rng = np.random.default_rng(44)
    x = rng.normal(size=(5, 311))
    x[1] += x[0]
    x[2] += 2*x[0]
    estimator = LedoitWolf().fit(x.T)
    # Inverse covariance via solves, independently of production's precision attribute.
    precision = np.linalg.solve(estimator.covariance_, np.eye(len(x)))
    expected = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    np.fill_diagonal(expected, np.nan)
    calls = []
    original = module._fit_estimator
    def counted(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(module, "_fit_estimator", counted)
    d = data(x)
    before = Precision(estimator="LedoitWolf").multivariate(d)
    metric = PartialCorrelation()
    np.testing.assert_allclose(metric.multivariate(d), expected, atol=1e-13)
    metric.diagnostics(d)
    np.testing.assert_equal(Precision(estimator="LedoitWolf").multivariate(d), before)
    assert calls == ["LedoitWolf"]
    # Same A/B with different conditioning channels is intentionally a different estimator.
    assert not np.isclose(metric.multivariate(data(x[:2]))[0, 1], expected[0, 1])


@pytest.mark.parametrize("pseudocount", [1e-10, np.nextafter(0., 1.), 1e300])
def test_jensen_shannon_independent_formula(pseudocount):
    x = np.random.default_rng(45).normal(size=(5, 103))
    x[0] **= 3
    actual = HistogramJensenShannon(pseudocount=pseudocount).multivariate(data(x))
    counts = np.stack([np.histogram(row, bins=10)[0] for row in x]).astype(np.longdouble)
    counts += np.longdouble(pseudocount)
    p = counts/counts.sum(axis=1, keepdims=True)
    for i in range(5):
        for j in range(i+1, 5):
            m = (p[i]+p[j])/2
            expected = .5*np.sum(p[i]*np.log(p[i]/m)+p[j]*np.log(p[j]/m))
            assert actual[i, j] == pytest.approx(float(expected), abs=2e-13)
            assert 0 <= actual[i, j] <= np.log(2)+1e-14
    if pseudocount == 1e-10:
        assert actual[0, 1] == pytest.approx(jensenshannon(np.array(p[0], float), np.array(p[1], float))**2)


def test_histogram_boundary_semantics_and_shared_counts(monkeypatch):
    import pyspi._histogram as module
    from pyspi._moments import moment_workspace
    edges = np.linspace(0., 1., 11)
    row = np.concatenate([edges, np.nextafter(edges[1:-1], 0), np.nextafter(edges[1:-1], 1)])
    x = np.stack([row, row**2, row**3])
    np.testing.assert_equal(module._counts(x, edges), np.stack([np.histogram(r, bins=edges)[0] for r in x]))
    calls = []
    original = module._counts
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "_counts", counted)
    d = data(x)
    HistogramRelativeEntropy().multivariate(d)
    HistogramJensenShannon().multivariate(d)
    assert len(calls) == 1
    assert moment_workspace(d).arrays.nbytes <= moment_workspace(d).arrays.max_bytes


@pytest.mark.parametrize("metric", [FixedLagCorrelation(3), ThresholdEventExcess(1), LogScaleRatio(),
                                    PartialCorrelation(), HistogramJensenShannon()])
def test_permutation_mutation_outputs_and_invalid_input(metric):
    x = np.random.default_rng(46).normal(size=(4, 121))
    d = data(x)
    expected = metric.multivariate(d)
    permutation = [2, 0, 3, 1]
    np.testing.assert_allclose(metric.multivariate(data(x[permutation])), expected[np.ix_(permutation, permutation)], atol=1e-13)
    actual = metric.multivariate(d)
    actual[:] = 100
    np.testing.assert_allclose(metric.multivariate(d), expected, atol=1e-13)
    assert metric.bivariate(d, i=0, j=1) == pytest.approx(expected[0, 1])
    d.to_numpy()[0, :, 0] **= 3
    changed = metric.multivariate(d)
    assert not np.allclose(changed, expected, equal_nan=True)
    d.to_numpy()[0, :, 0] = 2.
    invalid = metric.multivariate(d)
    assert np.isnan(invalid[0]).all() and np.isnan(invalid[:, 0]).all()
    assert np.isfinite(invalid[1, 2])
    d.to_numpy()[0, 0, 0] = np.nan
    invalid = metric.multivariate(d)
    assert np.isnan(invalid[0]).all() and np.isnan(invalid[:, 0]).all()
    reasons = metric.diagnostics(d)["reason"]
    if isinstance(metric, PartialCorrelation):
        assert np.isnan(invalid).all()
        assert reasons[1, 2] == "nonfinite_conditioning"
    else:
        assert reasons[0, 1] == "nonfinite_input"
        assert np.isfinite(invalid[1, 2])


@pytest.mark.parametrize("kwargs", [{"lag_samples": 0}, {"lag_samples": True}, {"lag_samples": 1.5}, {"fs": 0}, {"fs": np.inf}])
def test_invalid_lag_parameters(kwargs):
    with pytest.raises(ValueError):
        FixedLagCorrelation(**kwargs)


def test_lag_preflight_and_fs_binding(tmp_path):
    from pyspi.workflow._metrics import resolve_metrics, metric_input_problems
    path = tmp_path / "metrics.yaml"
    path.write_text(".statistics.basic:\n  FixedLagCorrelation:\n    configs: [{}]\n")
    metrics = resolve_metrics(path, sampling_rate_hz=1024)
    assert metrics[0].parameters["fs"] == 1024
    assert "19p53125ms" in metrics[0].identifier
    assert metric_input_problems(metrics, 21, 2)
    assert not metric_input_problems(metrics, 22, 2)
    with pytest.raises(ValueError, match="overlapping"):
        FixedLagCorrelation().multivariate(data(np.ones((2, 21))))


def test_lag_undefined_samples_only_affect_directions_using_them():
    x = np.random.default_rng(821).normal(size=(3, 41))
    d = data(x)
    d.to_numpy()[0, -1, 0] = np.nan
    metric = FixedLagCorrelation(3)
    actual = metric.multivariate(d)
    assert np.isfinite(actual[0, 1]) and np.isnan(actual[1, 0])
    assert metric.diagnostics(d)["reason"][1, 0] == "nonfinite_input"
    assert metric.diagnostics(d)["reason"][0, 1] == "estimated"

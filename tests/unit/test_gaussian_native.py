"""Independent Gaussian identities, voltage units and JVM-free integration."""

import copy
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from pyspi._moments import moment_workspace
from pyspi.data import Data
from pyspi.statistics.infotheory import JointEntropy, ConditionalEntropy, MutualInfo, TimeLaggedMutualInfo
from pyspi.statistics.misc import LinearModel


CLASSES = [JointEntropy, ConditionalEntropy, MutualInfo, TimeLaggedMutualInfo]


def data(x):
    return Data(np.array(x, copy=True), normalise=False, detrend=False)


def entropy(observations):
    observations = np.atleast_2d(observations)
    covariance = np.atleast_2d(np.cov(observations, ddof=1))
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    return .5*(len(covariance)*np.log(2*np.pi*np.e)+logdet)


def reference(x, cls):
    result = np.full((len(x), len(x)), np.nan)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            a, b = x[i], x[j]
            if cls is TimeLaggedMutualInfo:
                a, b = a[:-1], b[1:]
            h = entropy([a, b])
            if cls is JointEntropy:
                value = h
            elif cls is ConditionalEntropy:
                value = h-entropy(a)
            else:
                value = entropy(a)+entropy(b)-h
            result[i, j] = value
    return result


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("kind", ["ordinary", "volts", "offset"])
def test_gaussian_native_against_independent_covariance(cls, kind):
    x = np.random.default_rng(1101).normal(size=(4, 201))
    x[1, 1:] += .8*x[0, :-1]
    x[2] += .5*x[0]
    if kind == "volts":
        x *= [[1e-6], [3e-5], [8e-6], [4e-6]]
    elif kind == "offset":
        x = x*[[3], [.4], [2], [1]]+[[100], [-50], [2], [-3]]
    spi, d = cls(), data(x)
    expected = reference(x, cls)
    actual = spi.multivariate(d)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(spi.bivariate(d, i=0, j=1), expected[0, 1], atol=1e-12)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(spi.multivariate(data(x[order])), actual[np.ix_(order, order)], atol=1e-12)
    if cls in (JointEntropy, MutualInfo):
        np.testing.assert_array_equal(actual, actual.T)
    elif cls is ConditionalEntropy and kind != "ordinary":
        assert abs(actual[0, 1]-actual[1, 0]) > .1
    elif cls is TimeLaggedMutualInfo:
        assert actual[0, 1] > actual[1, 0]+.1


@pytest.mark.parametrize("cls", CLASSES)
def test_extreme_units_preserve_information_and_entropy_shift(cls):
    x = np.random.default_rng(1102).normal(size=(3, 120))
    x[1] += x[0]
    factors = np.array([1e-200, -1e200, 1e-6])
    spi = cls()
    ordinary = spi.multivariate(data(x))
    scaled = spi.multivariate(data(x*factors[:, None]))
    shifts = np.log(abs(factors))
    if cls is JointEntropy:
        expected = ordinary+shifts[:, None]+shifts[None, :]
    elif cls is ConditionalEntropy:
        expected = ordinary+shifts[None, :]
    else:
        expected = ordinary
    np.testing.assert_allclose(scaled, expected, atol=2e-12)


def test_gaussian_shares_regression_moments_and_checks_mutation():
    x = np.random.default_rng(1103).normal(size=(4, 80))
    d = data(x)
    LinearModel("Ridge").multivariate(d)
    work = moment_workspace(d)
    gram = work.cross()
    outputs = [cls().multivariate(d) for cls in CLASSES]
    assert work.cross() is gram
    assert sum(key[0] == "cross" for key in work.arrays.entries) == 1
    assert sum(key[0] == "gaussian_pairs" for key in work.arrays.entries) == 2
    outputs[2][0, 1] = 100
    assert MutualInfo().multivariate(d)[0, 1] != 100
    info = MutualInfo().diagnostics(d)
    info["reason"][:] = "changed"
    assert MutualInfo().diagnostics(d)["reason"][0, 1] == "estimated"
    x[1] = 2*x[0]+.1*x[1]
    d.to_numpy()[1, :, 0] = x[1]
    np.testing.assert_allclose(MutualInfo().multivariate(d), reference(x, MutualInfo), atol=1e-12)
    assert moment_workspace(d) is not work


def test_degenerate_inputs_are_local_and_distinguished():
    x = np.random.default_rng(1104).normal(size=(6, 100))
    x[2], x[3] = 7., x[0]
    d = data(x)
    d.to_numpy()[4, 0, 0] = np.nan
    mi = MutualInfo().multivariate(d)
    assert mi[0, 2] == mi[2, 0] == 0.
    assert np.isposinf(mi[0, 3])
    assert np.isnan(mi[4]).all() and np.isnan(mi[:, 4]).all()
    je, ce = JointEntropy().multivariate(d), ConditionalEntropy().multivariate(d)
    assert np.isnan(je[0, 2]) and np.isnan(je[0, 3]) and np.isnan(ce[0, 2])
    assert ce[2, 0] == pytest.approx(entropy(x[0]))
    assert np.isfinite(je[0, 1]) and np.isfinite(ce[0, 1])
    reasons = MutualInfo().diagnostics(d)["reason"]
    assert reasons[0, 2] == "constant_variable"
    assert reasons[0, 3] == "singular_covariance"
    assert reasons[0, 4] == "nonfinite_input"
    for cls in CLASSES:
        too_short = data(np.ones((3, 1)))
        assert np.isnan(cls().multivariate(too_short)).all()
        assert cls().diagnostics(too_short)["reason"][0, 1] == "insufficient_observations"


def test_near_perfect_correlation_uses_residual_not_rounded_one():
    x = np.random.default_rng(1105).normal(size=(2, 500))
    x[1] = 2*x[0]+1e-8*x[1]
    centered = x-x.mean(axis=1, keepdims=True)
    slope = np.linalg.lstsq(centered[0, :, None], centered[1], rcond=None)[0][0]
    residual = centered[1]-slope*centered[0]
    expected = .5*np.log((centered[1] @ centered[1])/(residual @ residual))
    actual = MutualInfo().multivariate(data(x))[0, 1]
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, atol=2e-8)


def test_entropy_redundancy_depends_on_normalization_even_after_averaging():
    rng = np.random.default_rng(1106)
    outputs = {normalise: {cls: [] for cls in CLASSES[:3]} for normalise in (True, False)}
    for _ in range(4):
        x = rng.normal(size=(3, 150))*rng.uniform(.1, 4., size=(3, 1))
        for normalize, results in outputs.items():
            d = Data(x.copy(), normalise=normalize, detrend=True)
            for cls in results:
                results[cls].append(cls().multivariate(d))
    h_unit = .5*np.log(2*np.pi*np.e)
    for normalized, results in outputs.items():
        je, ce, mi = [np.mean(results[cls], axis=0) for cls in CLASSES[:3]]
        mask = ~np.eye(3, dtype=bool)
        if normalized:
            np.testing.assert_allclose((je+mi)[mask], 2*h_unit, atol=1e-14)
            np.testing.assert_allclose((ce+mi)[mask], h_unit, atol=1e-14)
        else:
            assert np.ptp((je+mi)[mask]) > .1 and np.ptp((ce+mi)[mask]) > .1


def test_native_constructor_serialization_and_jvm_independence(tmp_path):
    for cls in CLASSES:
        spi = cls()
        assert not cls.requires_java()
        assert not cls.requires_java(estimator="kernel")
        assert spi._entropy_calc is None
        assert pickle.loads(pickle.dumps(spi)).identifier == spi.identifier
        assert copy.deepcopy(spi).identifier == spi.identifier
        assert spi.issigned() == (cls in (JointEntropy, ConditionalEntropy))
    script = '''
import jpype
import numpy as np
from pathlib import Path
from pyspi.statistics.infotheory import (JointEntropy, ConditionalEntropy, MutualInfo, TimeLaggedMutualInfo,
    TransferEntropy, CrossmapEntropy, CausalEntropy, DirectedInfo, StochasticInteraction)
from pyspi.data import Data
from pyspi.calculator import Calculator
from pyspi.workflow._metrics import resolve_metrics
assert not jpype.isJVMStarted()
d=Data(np.random.default_rng(90).normal(size=(3,40))*1e-6, normalise=False)
for cls in (JointEntropy, ConditionalEntropy, MutualInfo, TimeLaggedMutualInfo):
    assert np.isfinite(cls().multivariate(d)[0,1])
for spi in (TransferEntropy(), CrossmapEntropy(), CausalEntropy(), DirectedInfo(), StochasticInteraction(),
            TransferEntropy(auto_embed_method='MAX_CORR_AIS', k_search_max=3, tau_search_max=2),
            TransferEntropy(estimator='symbolic',pattern_spacing=4,prediction_shift=4)):
    assert np.isfinite(spi.multivariate(d)[0,1])
assert not jpype.isJVMStarted()
path=Path(__import__('sys').argv[1])
path.write_text('.statistics.infotheory:\\n  MutualInfo:\\n    dependencies: [java]\\n    configs:\\n    - estimator: gaussian\\n')
assert resolve_metrics(path)[0].dependencies == ()
calc=Calculator(configfile=path, dataset=d)
assert list(calc.spis)==['mi_gaussian']
calc.compute()
assert not jpype.isJVMStarted()
assert Calculator._optional_dependencies == {}
'''
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-c", script, str(tmp_path/"metrics.yaml")],
                   env=env, check=True, capture_output=True, text=True)


@pytest.mark.java
def test_java_dependency_is_checked_after_native_construction(tmp_path):
    script = '''
import sys, jpype, numpy as np
from pathlib import Path
from pyspi.calculator import Calculator
path=Path(sys.argv[1])
for estimator in ('gaussian', 'kernel', 'kraskov'):
    path.write_text('.statistics.infotheory:\\n  MutualInfo:\\n    configs:\\n    - estimator: '+estimator+'\\n')
    calc=Calculator(configfile=path, dataset=np.random.default_rng(91).normal(size=(2,40)))
    assert jpype.isJVMStarted() == (estimator == 'kraskov')
    calc.compute()
    identifier = {'gaussian':'mi_gaussian','kernel':'mi_kernel_width0p5','kraskov':'mi_kraskov_neighbours04'}[estimator]
    assert list(calc.spis) == [identifier]
    assert np.isfinite(calc.table[identifier].to_numpy()[0,1])
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path/"mixed.yaml")],
                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1"}, check=True,
                   capture_output=True, text=True)

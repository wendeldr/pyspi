"""Unit tests for ``CalculatorFrame`` and ``CorrelationFrame`` in
``pyspi.calculator``.

``CorrelationFrame(<CalculatorFrame>)`` currently raises AttributeError
(``getstatlabels``); that is a known bug owned by ``test_known_bugs.py``, so
the CorrelationFrame methods here are exercised on hand-built frames."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyspi.calculator import CalculatorFrame, CorrelationFrame
from pyspi.data import Data

SEED = 42
FABFOUR = ["cov_EmpiricalCovariance", "spearmanr-sq", "di_gaussian", "pec"]
SHAPES = {"a": (3, 100), "b": (4, 80)}  # name -> (n_processes, n_observations)


@pytest.fixture(scope="module")
def frame(jvm):
    # Module-scoped, so it is set up before the function-scoped autouse
    # ``_skip_java_without_jvm``: skip here instead of building a 3-SPI fabfour.
    if not jvm:
        pytest.skip("JVM unavailable: fabfour includes the JIDT SPI di_gaussian")
    rng = np.random.default_rng(SEED)
    datasets = [Data(rng.standard_normal(SHAPES[n]), dim_order="ps") for n in SHAPES]
    cf = CalculatorFrame(
        datasets=datasets, names=list(SHAPES), labels=[["la"], ["lb"]], subset="fabfour"
    )
    cf.compute()
    return cf, datasets


# --------------------------------------------------------------------------- #
# CorrelationFrame static helpers
# --------------------------------------------------------------------------- #
def test_verify_classes_rejects_subset():
    CorrelationFrame._verify_classes([["a"], ["b", "c"]])  # disjoint: fine

    classes = ["a", ["b"]]
    CorrelationFrame._verify_classes(classes)
    assert classes == [["a"], ["b"]]  # scalars are wrapped in place

    with pytest.raises(AssertionError, match="subset"):
        CorrelationFrame._verify_classes([["a"], ["a", "b"]])


def test_get_group_returns_first_match_or_minus_one():
    classes = [["x"], ["a", "b"], ["a"]]
    assert CorrelationFrame._get_group(["a", "b", "c"], classes, "inst") == 1
    assert CorrelationFrame._get_group(["a"], classes, "inst") == 2
    assert CorrelationFrame._get_group(["c"], classes, "inst") == -1


# --------------------------------------------------------------------------- #
# CalculatorFrame
# --------------------------------------------------------------------------- #
@pytest.mark.java
def test_calculator_frame_computes_fabfour_on_two_datasets(frame):
    cf, datasets = frame
    calcs = [row[0] for row in cf.calculators.values]
    assert cf.n_calculators == 2
    assert [c.name for c in calcs] == list(SHAPES)
    assert [c.labels for c in calcs] == [["la"], ["lb"]]
    for calc, data, (p, _) in zip(calcs, datasets, SHAPES.values()):
        assert calc.dataset is data
        assert list(calc.spis) == FABFOUR
        assert calc.table.shape == (p, p * len(FABFOUR))
        for spi in FABFOUR:
            S = calc.table[spi].to_numpy()
            assert np.isnan(np.diag(S)).all()
            assert np.isfinite(S[~np.eye(p, dtype=bool)]).all(), (calc.name, spi)


@pytest.mark.java
def test_get_correlation_df_without_labels(frame):
    cf, _ = frame
    mdf, shapes = cf.get_correlation_df()

    assert mdf.index.names == ["Dataset", "SPI-1"]
    assert mdf.columns.name == "SPI-2"
    assert list(mdf.columns) == FABFOUR
    assert mdf.shape == (len(SHAPES) * len(FABFOUR), len(FABFOUR))
    calcs = {row[0].name: row[0] for row in cf.calculators.values}
    for name in SHAPES:
        block = mdf.loc[name].astype(float)
        assert list(block.index) == FABFOUR
        np.testing.assert_allclose(np.diag(block.to_numpy()), 1.0)
        np.testing.assert_allclose(block.to_numpy(), block.to_numpy().T)
        # Only cov is signed in fabfour, so every pair goes through the
        # abs-then-Spearman branch of Calculator._get_correlation_df.
        calc = calcs[name]
        assert [s for s in FABFOUR if calc.spis[s].issigned()] == ["cov_EmpiricalCovariance"]
        expected = calc.table.stack().abs().corr(method="spearman")
        np.testing.assert_allclose(
            block.to_numpy(), expected.loc[FABFOUR, FABFOUR].to_numpy(), rtol=1e-12
        )

    assert shapes.index.name == "Dataset"
    assert list(shapes.index) == list(SHAPES)
    assert shapes.loc[:, ["n_processes", "n_observations"]].values.tolist() == [
        list(v) for v in SHAPES.values()
    ]


# --------------------------------------------------------------------------- #
# CorrelationFrame.get_pvalues (on a hand-built float frame)
# --------------------------------------------------------------------------- #
def test_get_pvalues_shape_and_values():
    spis = ["s1", "s2"]
    r = {"a": 0.3, "b": -0.6}
    n_obs = {"a": 100, "b": 80}
    index = pd.MultiIndex.from_product([list(r), spis], names=["Dataset", "SPI-1"])
    values = np.vstack([[[1.0, r[d]], [r[d], 1.0]] for d in r])
    mdf = pd.DataFrame(values, index=index, columns=pd.Index(spis, name="SPI-2"))
    shapes = pd.DataFrame(
        {"n_processes": [3, 4], "n_observations": [n_obs[d] for d in r]},
        index=pd.Index(list(r), name="Dataset"),
    )

    corr = CorrelationFrame()
    corr._mdf = mdf
    corr._shapes = shapes
    with np.errstate(divide="ignore"):  # r = 1 on the diagonal
        pvals = corr.get_pvalues()

    assert pvals.shape == mdf.shape
    assert pvals.index.equals(mdf.index)
    assert pvals.columns.equals(mdf.columns)
    for d in r:
        block = pvals.loc[d].to_numpy()
        n = n_obs[d]
        expected = stats.f.sf(n * r[d] ** 2 / (1 - r[d] ** 2), 1, n - 1)
        np.testing.assert_array_equal(np.diag(block), 0.0)
        np.testing.assert_allclose(block[0, 1], expected, rtol=1e-12)
        np.testing.assert_allclose(block[1, 0], expected, rtol=1e-12)

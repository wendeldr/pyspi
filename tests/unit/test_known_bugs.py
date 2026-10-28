"""Known bugs: documented and locked, not fixed.

Each bug has a pair of tests:

* ``test_<bug>_current`` locks today's (buggy) behaviour, so an optimisation
  that changes it is noticed;
* ``test_<bug>_fixed`` is a strict xfail asserting the behaviour a fix would give. When
  the bug is fixed it XPASSes, which fails the run under ``xfail_strict`` and
  forces both tests (and the goldens) to be updated deliberately.

Inputs are small and seeded; SPI objects are built directly from the
statistics classes.
"""
import importlib

import numpy as np
import pandas as pd
import pytest
import yaml

from pyspi import utils
from pyspi.calculator import Calculator, CalculatorFrame, CorrelationFrame
from pyspi.data import Data

PANDAS_MAJOR = int(pd.__version__.split(".")[0])


def _seeded(p, t, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((p, t))
    x[1, 1:] += 0.6 * x[0, :-1]  # some lagged coupling so outputs are not trivial
    return x


def _data(x):
    return Data(np.array(x, copy=True), dim_order="ps")


def _same(a, b):
    return np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)


def _finite_offdiag(*mats):
    """Guard for the equality locks: equal-because-both-NaN would be vacuous."""
    for m in mats:
        off = ~np.eye(m.shape[0], dtype=bool)
        assert np.isfinite(m[off]).all(), "off-diagonal NaN/inf makes the equality lock vacuous"
    return mats


# --------------------------------------------------------------------------- #
# 1. Symbolic TE dispatch is fixed; independent definitions live in
#    test_symbolic_transfer.py. Keep a small guard against reintroducing it.
# --------------------------------------------------------------------------- #
def test_te_symbolic_fixed():
    from pyspi.statistics import infotheory
    x = _seeded(3, 80, seed=11)
    sym = infotheory.TransferEntropy(estimator="symbolic", pattern_length=3)
    gau = infotheory.TransferEntropy(estimator="gaussian", k_history=1, l_history=1)
    assert sym.identifier.startswith("te_symbolic") and gau.identifier.startswith("gc_gaussian")
    symbolic, gaussian = _finite_offdiag(sym.multivariate(_data(x)), gau.multivariate(_data(x)))
    assert not _same(symbolic, gaussian)


# --------------------------------------------------------------------------- #
# 2. ConvergentCrossMapping(embedding_dimension=None) picks the max E (10)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ccm_outputs():
    causal = importlib.import_module("pyspi.statistics.causal")
    # 70 observations is the minimum-ish length for which E=10 still yields a
    # positive CCM library-size increment.
    x = _seeded(3, 70, seed=12)
    auto = causal.ConvergentCrossMapping(statistic="mean")
    e10 = causal.ConvergentCrossMapping(statistic="mean", embedding_dimension=10)
    return _finite_offdiag(auto.multivariate(_data(x)), e10.multivariate(_data(x)))


@pytest.mark.slow
def test_ccm_auto_embedding_current(ccm_outputs):
    auto, e10 = ccm_outputs
    assert _same(auto, e10)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="known bug: ccm E=None takes EmbedDimension().max()['E'] (max E), not the best-rho E")
def test_ccm_auto_embedding_fixed(ccm_outputs):
    auto, e10 = ccm_outputs
    assert not _same(auto, e10)


# --------------------------------------------------------------------------- #
# 3. SoftDynamicTimeWarping ignores global_constraint
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def softdtw_outputs():
    distance = importlib.import_module("pyspi.statistics.distance")
    x = _seeded(3, 60, seed=13)
    plain = distance.SoftDynamicTimeWarping()
    sakoe = distance.SoftDynamicTimeWarping(global_constraint="sakoe_chiba")
    assert plain.identifier != sakoe.identifier
    return _finite_offdiag(plain.multivariate(_data(x)), sakoe.multivariate(_data(x)))


def test_softdtw_global_constraint_current(softdtw_outputs):
    plain, sakoe = softdtw_outputs
    assert _same(plain, sakoe)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="known bug: SoftDynamicTimeWarping.bivariate never passes global_constraint")
def test_softdtw_global_constraint_fixed(softdtw_outputs):
    plain, sakoe = softdtw_outputs
    assert not _same(plain, sakoe)


# --------------------------------------------------------------------------- #
# 5. Standard PLI now takes magnitude per frequency before band averaging.
# Independent definition tests also distinguish this from abs(band mean).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pli_output():
    spectral = importlib.import_module("pyspi.statistics.spectral")
    assert issubclass(spectral.PhaseLagIndex, importlib.import_module("pyspi.base").Undirected)
    return spectral.PhaseLagIndex().multivariate(_data(_seeded(4, 80, seed=14)))


def test_undirected_spectral_symmetry_current(pli_output):
    A = pli_output
    off = ~np.eye(A.shape[0], dtype=bool)
    assert np.isfinite(A[off]).all()
    assert np.all(A[off] >= 0) and np.any(A[off] > 0)
    np.testing.assert_allclose(A, A.T, rtol=1e-12, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# 6. Calculator duplicate-identifier check is dead code
# --------------------------------------------------------------------------- #
@pytest.fixture
def duplicate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(Calculator, "_optional_dependencies", {"java": False})
    cfg = {".statistics.basic": {"Covariance": {
        "labels": ["undirected", "linear", "signed"],
        "dependencies": None,
        "configs": [{"estimator": "EmpiricalCovariance"}, {"estimator": "EmpiricalCovariance"}],
    }}}
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def test_duplicate_identifiers_current(duplicate_config):
    calc = Calculator(configfile=duplicate_config)
    assert list(calc.spis) == ["cov_EmpiricalCovariance"]


# pytest.fail.Exception is what pytest.raises throws on "DID NOT RAISE"; any
# other error during Calculator construction is reported as a real failure.
@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason="known bug: duplicates are counted on dict keys, which are already unique")
def test_duplicate_identifiers_fixed(duplicate_config):
    with pytest.raises(ValueError, match="Duplicate SPI identifiers"):
        Calculator(configfile=duplicate_config)


# --------------------------------------------------------------------------- #
# 7. Data(procnames=...) is validated but never stored
# --------------------------------------------------------------------------- #
def test_data_procnames_current():
    d = Data(_seeded(3, 20, seed=15), procnames=["a", "b", "c"])
    assert d.procnames == ["proc-0", "proc-1", "proc-2"]


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="known bug: Data.__init__ asserts len(procnames) but never stores them")
def test_data_procnames_fixed():
    d = Data(_seeded(3, 20, seed=15), procnames=["a", "b", "c"])
    assert d.procnames == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# 8. Data.convert_to_numpy('.csv') passes ',' as the dtype
# --------------------------------------------------------------------------- #
@pytest.fixture
def csv_file(tmp_path):
    x = np.random.default_rng(16).standard_normal((3, 10))
    path = tmp_path / "x.csv"
    np.savetxt(path, x, delimiter=",")
    return str(path), x


def test_convert_to_numpy_csv_current(csv_file):
    path, _ = csv_file
    # np.genfromtxt(path, ",") parses "," as a dtype string -> SyntaxError (numpy 1.26).
    with pytest.raises(SyntaxError):
        Data.convert_to_numpy(path)


@pytest.mark.xfail(strict=True, raises=SyntaxError,
                   reason="known bug: np.genfromtxt(data, ',') passes the delimiter as dtype")
def test_convert_to_numpy_csv_fixed(csv_file):
    path, x = csv_file
    np.testing.assert_allclose(Data.convert_to_numpy(path), x)


# --------------------------------------------------------------------------- #
# 9. CorrelationFrame(calc_frame) calls the nonexistent getstatlabels()
# --------------------------------------------------------------------------- #
@pytest.fixture
def fabfour_frame(jvm, statistics_modules):
    if not jvm:
        pytest.skip("JVM unavailable: fabfour includes the JIDT SPI di_gaussian")
    rng = np.random.default_rng(17)
    datasets = [rng.standard_normal((3, 50)), rng.standard_normal((3, 50))]
    cf = CalculatorFrame(datasets=datasets, names=["d0", "d1"], subset="fabfour")
    cf.compute()
    return cf


@pytest.mark.java
def test_correlation_frame_current(fabfour_frame):
    with pytest.raises(AttributeError, match="getstatlabels"):
        CorrelationFrame(fabfour_frame)


@pytest.mark.java
@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="known bug: Calculator._get_correlation_df(with_labels=True) calls self.getstatlabels()")
def test_correlation_frame_fixed(fabfour_frame):
    cfr = CorrelationFrame(fabfour_frame)
    assert cfr.mdf.shape[1] == fabfour_frame.calculators.iloc[0, 0].n_spis


# --------------------------------------------------------------------------- #
# 10. convert_mdf_to_ddf is incompatible with pandas>=3 (found while writing
#     these tests; the pre-pandas-3 pivot shape is checked in test_utils_more.py)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(PANDAS_MAJOR < 3, reason="stack(dropna=...) is only rejected by pandas>=3")
def test_convert_mdf_to_ddf_pandas3_current():
    idx = pd.MultiIndex.from_product([["d0"], ["s1", "s2"]], names=["Dataset", "SPI-1"])
    mdf = pd.DataFrame(np.ones((2, 2)), index=idx, columns=pd.Index(["s1", "s2"], name="SPI-2"))
    with pytest.raises(ValueError, match="dropna must be unspecified"):
        utils.convert_mdf_to_ddf(mdf)

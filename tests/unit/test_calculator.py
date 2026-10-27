"""Unit tests for ``pyspi.calculator.Calculator`` (config loading, dependency
skipping, compute error handling, table layout and dataset preprocessing flags).

These lock current behaviour; they do not assert that it is desirable."""
import numpy as np
import pytest
import yaml

from pyspi.calculator import Calculator
from pyspi.data import Data

SEED = 42

#: Identifier prefixes produced by the JIDT (java-dependent) infotheory SPIs.
JAVA_PREFIXES = ("mi_", "te_", "gc_", "je_", "ce_", "tlmi_", "xme_", "cce_", "di_", "si_")

BASIC_LABELS = ["undirected", "linear", "signed", "multivariate", "contemporaneous"]


def _write_yaml(path, content):
    path.write_text(yaml.safe_dump(content))
    return str(path)


@pytest.fixture
def basic_config(tmp_path):
    """A tiny config using only ``.statistics.basic`` (no JVM needed)."""
    return _write_yaml(
        tmp_path / "basic.yaml",
        {
            ".statistics.basic": {
                "Covariance": {
                    "labels": BASIC_LABELS,
                    "dependencies": None,
                    "configs": [{"estimator": "EmpiricalCovariance"}, {"estimator": "LedoitWolf"}],
                },
                "SpearmanR": {
                    "labels": BASIC_LABELS,
                    "dependencies": None,
                    "configs": [{"squared": False}],
                },
            }
        },
    )


def _randn(p=3, t=100, seed=SEED):
    return np.random.default_rng(seed).standard_normal((p, t))


@pytest.fixture(scope="module")
def computed_fabfour(jvm):
    # ``jvm`` is requested so the JVM starts before ``infotheory`` is imported.
    # Without a JVM, di_gaussian is excluded and the remaining three SPIs are
    # used; the tests on this fixture only check per-SPI layout, so they still run.
    calc = Calculator(dataset=_randn(), subset="fabfour")
    calc.compute()
    return calc


# --------------------------------------------------------------------------- #
# Configuration loading
# --------------------------------------------------------------------------- #
def test_temp_yaml_loads_expected_identifiers(basic_config):
    calc = Calculator(configfile=basic_config)
    assert list(calc.spis) == ["cov_EmpiricalCovariance", "cov_LedoitWolf", "spearmanr"]
    assert calc.n_spis == 3
    assert calc._excluded_spis == []


def test_missing_java_dependency_skips_jidt_spis(jvm, monkeypatch):
    monkeypatch.setattr(Calculator, "_optional_dependencies", {"java": False})
    calc = Calculator()
    ids = list(calc.spis)
    assert not [i for i in ids if i.startswith(JAVA_PREFIXES)]
    # One _excluded_spis entry per skipped config: 39 java SPIs out of 298.
    assert len(calc._excluded_spis) == 39
    assert calc.n_spis == 259
    assert all(deps == ["java"] for _, deps in calc._excluded_spis)
    # Integrated information lives in the infotheory module but has no java dependency.
    assert any(i.startswith("phi_") for i in ids)


def test_type_error_in_config_falls_back_to_single_default_spi(tmp_path):
    # A bad parameter aborts the whole configs loop for that class: the remaining
    # configs are dropped and ONE default-constructed instance is added instead.
    configfile = _write_yaml(
        tmp_path / "bogus.yaml",
        {
            ".statistics.basic": {
                "SpearmanR": {
                    "labels": BASIC_LABELS,
                    "dependencies": None,
                    "configs": [{"bogus": 1}, {"squared": True}],
                }
            }
        },
    )
    calc = Calculator(configfile=configfile)
    assert list(calc.spis) == ["spearmanr"]


@pytest.mark.java  # the counts include the JIDT SPIs, so they only hold with a JVM
def test_subset_counts(jvm):
    counts = {subset: Calculator(subset=subset).n_spis for subset in ("all", "fast", "sonnet", "fabfour")}
    assert counts == {"all": 298, "fast": 230, "sonnet": 14, "fabfour": 4}


def test_invalid_subset_raises():
    with pytest.raises(ValueError, match="does not exist"):
        Calculator(subset="no-such-subset")


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def test_compute_without_dataset_raises(basic_config):
    calc = Calculator(configfile=basic_config)
    with pytest.raises(AttributeError, match="Dataset not loaded"):
        calc.compute()


def test_spi_exception_in_compute_yields_all_nan_block(jvm, monkeypatch):
    # Without a JVM fabfour loses di_gaussian; the check below covers whichever
    # SPIs were loaded, so this test also runs without Java.
    calc = Calculator(dataset=_randn(), subset="fabfour")
    failing = "spearmanr-sq"

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(calc.spis[failing], "multivariate", boom)
    with pytest.warns(UserWarning, match="Caught"):
        calc.compute()

    assert np.isnan(calc.table[failing].to_numpy()).all()
    for spi in calc.spis:
        if spi != failing:
            assert not np.isnan(calc.table[spi].to_numpy()).all(), spi


def test_diagonal_is_nan_after_compute(computed_fabfour):
    p = computed_fabfour.dataset.n_processes
    offdiag = ~np.eye(p, dtype=bool)
    for spi in computed_fabfour.spis:
        S = computed_fabfour.table[spi].to_numpy()
        assert np.isnan(np.diag(S)).all(), spi
        assert np.isfinite(S[offdiag]).all(), spi


def test_table_layout(computed_fabfour):
    calc = computed_fabfour
    table = calc.table
    procnames = [f"proc-{i}" for i in range(3)]
    assert table.columns.names == ["spi", "process"]
    assert list(table.index) == procnames
    assert list(table.columns.get_level_values("spi").unique()) == list(calc.spis)
    assert table.shape == (3, 3 * calc.n_spis)
    for spi in calc.spis:
        block = table[spi]
        assert block.shape == (3, 3)
        assert list(block.columns) == procnames


# --------------------------------------------------------------------------- #
# Dataset preprocessing flags
# --------------------------------------------------------------------------- #
def test_detrend_normalise_flags_forwarded_to_data(basic_config):
    rng = np.random.default_rng(SEED)
    t = np.arange(80)
    x = rng.standard_normal((3, 80)) * np.array([[0.5], [2.0], [5.0]]) + 0.1 * t + 3.0

    raw = Calculator(dataset=x, configfile=basic_config, normalise=False, detrend=False)
    np.testing.assert_array_equal(raw.dataset.to_numpy(squeeze=True), x)

    both = Calculator(dataset=x, configfile=basic_config, detrend=True, normalise=True)
    np.testing.assert_array_equal(
        both.dataset.to_numpy(), Data(x, detrend=True, normalise=True).to_numpy()
    )
    # detrend really was applied (not just the default normalise=True).
    assert not np.allclose(both.dataset.to_numpy(), Data(x, detrend=False, normalise=True).to_numpy())

    # A Data object is used as-is; the calculator's flags are ignored.
    d = Data(x, normalise=False)
    kept = Calculator(dataset=d, configfile=basic_config, detrend=True, normalise=True)
    assert kept.dataset is d
    np.testing.assert_array_equal(kept.dataset.to_numpy(squeeze=True), x)

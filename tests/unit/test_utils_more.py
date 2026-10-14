"""Unit tests for ``pyspi.utils`` helpers not covered by ``tests/test_utils.py``.

One test per concern. Current behaviour is locked, including the misspelled
default ``nan_policy`` of ``normalise`` (documented, not fixed).
"""
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import zscore

from pyspi import utils

PANDAS_MAJOR = int(pd.__version__.split(".")[0])


def test_swap_chars_docstring_example_and_index_order_symmetry():
    assert utils.swap_chars("heLlotHere", 2, 6) == "heHlotLere"
    assert utils.swap_chars("heLlotHere", 6, 2) == "heHlotLere"


def test_strshort_string_list_and_passthrough():
    long = "abcdefghijklmnopqrstuvwxyz"
    # mlength=10 keeps the first 4 chars, an ellipsis and the last 3 chars.
    assert utils.strshort(long, 10) == "abcd...xyz"
    assert utils.strshort("short", 10) == "short"
    assert utils.strshort([long, "short"], 10) == ["abcd...xyz", "short"]


def test_acf_matches_explicit_lag_products():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    z = zscore(x)
    n = z.size
    expected = np.array([np.dot(z[: n - k], z[k:]) for k in range(n)]) / np.dot(z, z)

    pos = utils.acf(x)
    np.testing.assert_allclose(pos, expected, rtol=1e-12, atol=1e-12)
    assert pos[0] == pytest.approx(1.0)

    full = utils.acf(x[None, :], mode="full")  # 2-D input is squeezed
    assert full.shape == (2 * n - 1,)
    np.testing.assert_allclose(full, np.concatenate([expected[:0:-1], expected]), rtol=1e-12, atol=1e-12)


def test_standardise_zero_mean_unit_std_and_constant_input():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(40) * 3.0 + 7.0
    s = utils.standardise(a)
    assert s.mean() == pytest.approx(0.0, abs=1e-12)
    assert s.std(ddof=1) == pytest.approx(1.0, abs=1e-12)

    const = np.full(10, 4.2)
    np.testing.assert_array_equal(utils.standardise(const), const - const.mean())


def test_normalise_explicit_policies():
    clean = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 40.0]])
    np.testing.assert_allclose(
        utils.normalise(clean, nan_policy="propagate"),
        np.array([[0.0, 0.0], [0.5, 1.0 / 3.0], [1.0, 1.0]]),
    )

    with_nan = np.array([[1.0, 10.0], [np.nan, 20.0], [5.0, 40.0]])
    out = utils.normalise(with_nan, nan_policy="omit")
    np.testing.assert_allclose(out, np.array([[0.0, 0.0], [np.nan, 1.0 / 3.0], [1.0, 1.0]]))


def test_normalise_default_policy_is_misspelled_and_raises():
    # Default nan_policy is 'propogate' (sic), which _contains_nan rejects.
    with pytest.raises(ValueError, match="nan_policy must be one of"):
        utils.normalise(np.arange(5.0))


class _FakeDataset:
    n_processes = 3


class _FakeCalc:
    """Minimal stand-in exposing what ``inspect_calc_results`` reads."""

    def __init__(self, blocks):
        procs = [f"proc-{i}" for i in range(3)]
        self.spis = {k: None for k in blocks}
        self.n_spis = len(blocks)
        self.dataset = _FakeDataset()
        cols = pd.MultiIndex.from_product([list(blocks), procs], names=["spi", "process"])
        self.table = pd.DataFrame(np.hstack([blocks[k] for k in blocks]), index=procs, columns=cols)


def test_inspect_calc_results_classifies_nan_partial_and_successful(capsys):
    ok = np.arange(9.0).reshape(3, 3)
    np.fill_diagonal(ok, np.nan)  # diagonal-only NaNs: n_processes NaNs -> successful
    partial = ok.copy()
    partial[0, 1] = np.nan  # one off-diagonal NaN -> more than n_processes NaNs
    all_nan = np.full((3, 3), np.nan)
    calc = _FakeCalc({"spi_ok": ok, "spi_partial": partial, "spi_allnan": all_nan})

    utils.inspect_calc_results(calc)
    out = capsys.readouterr().out

    assert "Total number of SPIs attempted: 3" in out
    assert "Number of SPIs successfully computed: 1 (33.33%)" in out
    nan_section = out.split("[1] SPI(s) produced NaN outputs:")[1]
    partial_section = out.split("[1] SPIs which produced partial NaN outputs:")[1]
    assert "1. spi_allnan" in nan_section.split("partial NaN outputs")[0]
    assert "1. spi_partial" in partial_section
    assert "spi_ok" not in out


def _write_filter_config(path):
    cfg = {
        ".statistics.basic": {
            "Covariance": {
                "labels": ["undirected", "linear", "signed"],
                "dependencies": None,
                "configs": [{"estimator": "EmpiricalCovariance"}],
            },
            "SpearmanR": {
                "labels": ["undirected", "nonlinear", "signed"],
                "dependencies": None,
                "configs": [{"squared": False}, {"squared": True}],
            },
        },
        ".statistics.distance": {
            "PairwiseDistance": {
                "labels": ["unsigned", "linear", "undirected"],
                "dependencies": None,
                "configs": [{"metric": "euclidean"}],
            },
        },
    }
    path.write_text(yaml.dump(cfg, sort_keys=False))
    return cfg


def test_filter_spis_writes_filtered_yaml_in_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "in.yaml"
    cfg = _write_filter_config(cfg_path)

    utils.filter_spis(["linear"], output_name="out", configfile=str(cfg_path))

    with open(tmp_path / "out.yaml") as f:
        got = yaml.load(f, Loader=yaml.FullLoader)
    assert got == {
        ".statistics.basic": {"Covariance": cfg[".statistics.basic"]["Covariance"]},
        ".statistics.distance": cfg[".statistics.distance"],
    }
    assert "Total SPIs Matched: 2 SPI(s)" in capsys.readouterr().out


def test_filter_spis_no_match_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "in.yaml"
    _write_filter_config(cfg_path)
    with pytest.raises(ValueError, match="0 SPIs were found"):
        utils.filter_spis(["no-such-label"], output_name="out", configfile=str(cfg_path))
    assert not (tmp_path / "out.yaml").exists()


# This body only executes on pandas<3. On pandas>=3 (the locked/CI environment)
# it always xfails on the ValueError; the active lock there is
# test_convert_mdf_to_ddf_pandas3_current in test_known_bugs.py.
@pytest.mark.xfail(
    PANDAS_MAJOR >= 3,
    strict=True,
    raises=ValueError,
    reason="known bug: convert_mdf_to_ddf passes stack(dropna=False), rejected by pandas>=3 "
           "(current behaviour locked in test_known_bugs.py)",
)
def test_convert_mdf_to_ddf_pivots_to_spi_pair_by_dataset():
    idx = pd.MultiIndex.from_product([["d0", "d1"], ["s1", "s2", "s3"]], names=["Dataset", "SPI-1"])
    cols = pd.Index(["s1", "s2", "s3"], name="SPI-2")
    mdf = pd.DataFrame(np.arange(18.0).reshape(6, 3), index=idx, columns=cols)
    mdf.iloc[0, 0] = np.nan

    ddf = utils.convert_mdf_to_ddf(mdf)

    assert ddf.shape == (9, 2)  # (SPI-1, SPI-2) pairs x datasets; NaN pair kept
    assert list(ddf.columns) == ["d0", "d1"]
    assert ddf.loc[("s2", "s3"), "d1"] == 14.0
    assert np.isnan(ddf.loc[("s1", "s1"), "d0"])

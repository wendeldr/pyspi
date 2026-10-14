"""Golden-value regression tests for working SPIs that ``config.yaml`` leaves out.

``tests/regression/extra_config.yaml`` lists nine wavelet classes from
``pyspi/statistics/wavelet.py`` (cohmag, phase, icoh, plv, ppc, pli, dspli,
wspli, dwspli) that run today but are absent from ``pyspi/config.yaml``, each
with the six band/statistic configs ``PhaseSlopeIndex`` uses there. They all
go through ``mne._get_cache`` and its per-``Data`` ``data.mne[measure]`` cache,
which no SPI in the default config exercises.

Like the main suite these are change detectors, not correctness proofs: the
goldens in ``goldens/extra/small/`` record what the code produced when they
were generated, and a failure means an output changed. Several of these
snapshots are degenerate on the single-epoch input (for example ``cohmag`` and
``plv`` are 1 everywhere, ``ppc`` and ``dspli`` are ``-inf`` off the diagonal);
they are locked as-is.

Regenerate only for an intentional change::

    .venv/bin/python tests/regression/generate_goldens.py --size small --repeat 2 \\
        --configfile tests/regression/extra_config.yaml --out tests/regression/goldens/extra

The parameter is named ``extra_identifier`` (not ``identifier``) on purpose:
``tests/conftest.py`` looks ``identifier`` up in the ``--pyspi-size`` golden
set, which does not contain these SPIs. Markers are attached here instead.
The data come from this set's own ``meta.json`` dataset spec (the ``small``
window), independent of ``--pyspi-size``.
"""
import numpy as np
import pytest

from regression import harness, spi_policy

EXTRA_SET = "extra/small"
EXTRA_CONFIG = harness.TESTS_DIR / "regression" / "extra_config.yaml"

def _params():
    harness.require_golden(EXTRA_SET)
    meta = harness.load_meta(EXTRA_SET)
    params = []
    for ident in meta["calculator_order"]:
        entry = meta["spis"][ident]
        marks = []
        if spi_policy.is_slow(ident, entry.get("time_s")):
            marks.append(pytest.mark.slow)
        if "java" in entry.get("deps", []):
            marks.append(pytest.mark.java)
        if spi_policy.is_stochastic(ident):
            marks.append(pytest.mark.stochastic)
        if entry.get("status") in ("all_nan", "error"):
            marks.append(pytest.mark.nan_golden)
        params.append(pytest.param(ident, marks=marks, id=ident))
    return params


@pytest.fixture(scope="module")
def extra_meta():
    return harness.load_meta(EXTRA_SET)


@pytest.fixture(scope="module")
def extra_values():
    return harness.load_values(EXTRA_SET)


@pytest.fixture(scope="module")
def extra_raw(extra_meta):
    return harness.load_raw(extra_meta["dataset"])


@pytest.fixture
def extra_data(extra_raw, extra_meta):
    """A fresh ``Data`` per test so the ``data.mne`` cache never leaks between tests."""
    return harness.make_data(extra_raw, extra_meta["dataset"])


def test_extra_set_matches_config(extra_meta):
    """The stored set was generated from extra_config.yaml on the small window."""
    assert extra_meta["dataset"] == harness.size_spec("small")
    entries, by_id = harness.catalogue(EXTRA_CONFIG)
    assert [e["identifier"] for e in entries] == extra_meta["calculator_order"]
    assert set(by_id) == set(extra_meta["spis"])  # meta.json is written with sorted keys
    assert len(by_id) == 54
    assert all(e["cls"] != "PhaseSlopeIndex" for e in entries)
    from regression.generate_goldens import catalogue_mismatches
    for ident, entry in by_id.items():
        assert not catalogue_mismatches(extra_meta["spis"][ident], entry), ident


@pytest.mark.regression
@pytest.mark.parametrize("extra_identifier", _params())
def test_extra_spi_matches_golden(extra_identifier, extra_meta, extra_values, extra_data, seed_all):
    entry = extra_meta["spis"][extra_identifier]
    out = harness.compute_recorded(dict(entry, identifier=extra_identifier),
                                   extra_data, extra_meta["seed"], seed_all)
    harness.assert_spi_allclose(out, extra_values[extra_identifier], extra_identifier)


@pytest.mark.regression
@pytest.mark.parametrize("with_callback", [False, True], ids=["default", "callback"])
def test_extra_calculator_matches_goldens(extra_meta, extra_values, extra_raw, seed_all, with_callback):
    """One Calculator pass over extra_config.yaml on a single ``Data``.

    The six band variants of each class reuse one ``data.mne[measure]`` entry
    and the nine classes share the ``data.mne`` dict, so this exercises the
    cache hits that the per-SPI tests (fresh ``Data`` each) never reach.
    """
    from pyspi.calculator import Calculator

    data = harness.make_data(extra_raw, extra_meta["dataset"])
    calc = Calculator(dataset=data, configfile=str(EXTRA_CONFIG))
    assert list(calc.spis) == extra_meta["calculator_order"]
    seed_all(extra_meta["seed"])
    harness.compute_calculator_with_reports(calc, with_callback)

    measures = {spi.measure for spi in calc.spis.values()}
    assert len(measures) == 9
    assert set(data.mne) == measures

    failures = []
    for ident in extra_meta["spis"]:
        actual = calc.table[ident].to_numpy()
        expected = extra_values[ident]
        try:
            harness.assert_spi_allclose(actual, expected, ident)
        except AssertionError:
            max_abs, max_rel = harness.max_differences(actual, expected)
            nan_diff = int((np.isnan(actual) != np.isnan(expected)).sum())
            failures.append(f"{ident}: max_abs={max_abs:.3g} max_rel={max_rel:.3g} nan_mask_diff={nan_diff}")
    assert not failures, "Calculator table differs from the extra goldens:\n" + "\n".join(failures)

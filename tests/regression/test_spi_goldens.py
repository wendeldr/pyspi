"""Golden-value regression tests for every SPI in ``config.yaml``.

These are change detectors, not correctness proofs: the goldens record what the
code produced when they were generated (see ``goldens/<name>/meta.json``), and
a failure means an output changed. Regenerate only for an intentional change
(``tests/README.md``).

* ``test_spi_matches_golden[<identifier>]`` -- one test per SPI at the
  selected size (``--pyspi-size``), each on a fresh ``Data`` so caches never
  leak between SPIs.
* ``test_preprocessing_variant_matches_golden[<identifier>-n?d?]`` -- a
  representative subset under all four ``(normalise, detrend)`` combinations.
* ``test_calculator_table_matches_goldens_fast`` -- one ``Calculator`` pass
  (shared ``Data`` caches) over every SPI that is not ``slow`` at the selected
  size, compared to the per-SPI goldens. Default tier; it is what notices a
  cache-keying change (two SPIs reading each other's cached spectra), which the
  per-SPI tests cannot see because each one runs on a fresh ``Data``.
* ``test_calculator_table_matches_goldens`` -- the same over the full
  ``config.yaml`` (``slow``).
* ``test_catalogue_matches_golden_meta`` -- the config as loaded now (module,
  class, constructor params, dependencies, labels per identifier) equals the
  recorded catalogue, so a parameter edit that keeps the identifier (for
  example ``terms`` of ``ids``) is reported even though the per-SPI tests
  rebuild each SPI from the recorded params.
* ``test_spi_catalogue[...]`` -- subset sizes, order and labels.

The ``slow``/``java``/``stochastic``/``nan_golden`` markers are added per
identifier by ``tests/conftest.py`` from the golden ``meta.json``; the
Calculator and catalogue tests carry ``java`` explicitly (without a JVM the
Calculator loads a smaller catalogue).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from regression import harness, spi_policy


def pytest_generate_tests(metafunc):
    if "identifier" in metafunc.fixturenames:
        size = metafunc.config.getoption("--pyspi-size")
        harness.require_golden(size)
        idents = list(harness.load_meta(size)["spis"])
        metafunc.parametrize("identifier", idents, ids=str)
    if "preproc_key" in metafunc.fixturenames:
        harness.require_golden("preproc")
        keys = list(harness.load_meta("preproc")["variants"])
        metafunc.parametrize("preproc_key", keys, ids=[k.replace("::", "-") for k in keys])


def _compute(entry, data, seed, seed_all):
    """Build, seed and compute, enforcing the recorded success/error identity."""
    return harness.compute_recorded(entry, data, seed, seed_all)


# --------------------------------------------------------------------------- #
# Per-SPI regression
# --------------------------------------------------------------------------- #
@pytest.mark.regression
def test_spi_matches_golden(identifier, golden_meta, golden_values, spi_data, seed_all):
    entry = golden_meta["spis"][identifier]
    out = _compute(entry, spi_data, golden_meta["seed"], seed_all)
    harness.assert_spi_allclose(out, golden_values[identifier], identifier)


# --------------------------------------------------------------------------- #
# Preprocessing-variant regression
# --------------------------------------------------------------------------- #
@pytest.mark.regression
def test_preprocessing_variant_matches_golden(preproc_key, preproc_meta, preproc_values, seed_all):
    identifier, normalise, detrend = spi_policy.split_preproc_key(preproc_key)
    spec = dict(preproc_meta["dataset"], normalise=normalise, detrend=detrend)
    data = harness.make_data(harness.load_raw(spec), spec)
    # Build from the SPI entry; the status that decides error handling is the variant's.
    entry = {**preproc_meta["spis"][identifier], **preproc_meta["variants"][preproc_key]}
    out = _compute(entry, data, preproc_meta["seed"], seed_all)
    harness.assert_spi_allclose(out, preproc_values[preproc_key], identifier)


@pytest.mark.parametrize("invariant_spi", ["spearmanr", "kendalltau", "dcorr"])
def test_preproc_fixture_normalise_invariant(invariant_spi, preproc_values):
    """Rank-based and per-variable scale-invariant SPIs ignore ``normalise``.

    Guards the preprocessing fixture: if these differed, the n1/n0 inputs would
    not be what the variant keys claim.
    """
    n1 = preproc_values[spi_policy.preproc_key(invariant_spi, True, False)]
    n0 = preproc_values[spi_policy.preproc_key(invariant_spi, False, False)]
    harness.assert_spi_allclose(n1, n0, invariant_spi)


def test_preproc_fixture_normalise_changes_covariance(preproc_values):
    """On the distorted fixture, z-scoring must change the empirical covariance."""
    ident = "cov_EmpiricalCovariance"
    n1 = preproc_values[spi_policy.preproc_key(ident, True, False)]
    n0 = preproc_values[spi_policy.preproc_key(ident, False, False)]
    assert not np.allclose(n1, n0, equal_nan=True)


# --------------------------------------------------------------------------- #
# Calculator end-to-end (shared caches)
# --------------------------------------------------------------------------- #
def _package_config():
    """``config.yaml`` of the imported ``pyspi`` package (what ``Calculator()`` loads)."""
    import pyspi
    return Path(pyspi.__file__).resolve().parent / "config.yaml"


def _write_fast_config(src, dst, golden_meta):
    """Write ``src`` minus every config whose identifier is ``slow`` (or has no golden).

    Identifiers come from :func:`harness.iter_config_entries`, i.e. the SPIs are
    instantiated exactly as ``Calculator._load_yaml`` would. A class built
    without params (empty ``configs`` or the ``TypeError`` fallback) is copied
    verbatim when its single identifier is kept. Classes left with no configs,
    and classes whose dependencies are unmet, are dropped. Module and class
    order is preserved, so the shared caches fill in the same order as in a
    full Calculator pass. Returns the kept identifiers in Calculator order.
    """
    with open(src) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)

    def keep(ident):
        entry = golden_meta["spis"].get(ident)
        return entry is not None and not spi_policy.is_slow(ident, entry.get("time_s"))

    kept_params, kept_verbatim, kept_idents = {}, set(), []
    for entry, _spi in harness.iter_config_entries(src, harness.optional_dependencies()):
        if not keep(entry["identifier"]):
            continue
        kept_idents.append(entry["identifier"])
        key = (entry["module"], entry["cls"])
        if entry["params"] is None:
            kept_verbatim.add(key)
        else:
            kept_params.setdefault(key, []).append(entry["params"])

    out = {}
    for module_name, classes in yf.items():
        for cls_name, spec in classes.items():
            key = (module_name, cls_name)
            if key in kept_verbatim:
                new_spec = spec
            elif key in kept_params:
                new_spec = dict(spec, configs=kept_params[key])
            else:
                continue
            out.setdefault(module_name, {})[cls_name] = new_spec
    with open(dst, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return kept_idents


def _calculator_mismatches(calc, golden_meta, golden_values, identifiers):
    """One line per identifier whose Calculator output differs from its golden."""
    failures = []
    for ident in identifiers:
        if spi_policy.matches_any(ident, spi_policy.CALC_E2E_SKIP):
            continue
        actual = calc.table[ident].to_numpy()
        expected = golden_values[ident]
        try:
            harness.assert_spi_allclose(actual, expected, ident)
        except AssertionError:
            max_abs, max_rel = harness.max_differences(actual, expected)
            nan_diff = int((np.isnan(actual) != np.isnan(expected)).sum())
            failures.append(f"{ident}: max_abs={max_abs:.3g} max_rel={max_rel:.3g} nan_mask_diff={nan_diff}")
    return failures


@pytest.mark.java
@pytest.mark.regression
@pytest.mark.parametrize("with_callback", [False, True], ids=["default", "callback"])
def test_calculator_table_matches_goldens_fast(golden_meta, golden_values, raw_dataset, seed_all, tmp_path, with_callback):
    """Calculator pass (shared caches) over the non-slow SPIs only."""
    from pyspi.calculator import Calculator

    configfile = tmp_path / "fast_goldens_config.yaml"
    kept = _write_fast_config(_package_config(), configfile, golden_meta)
    assert kept, "no non-slow SPIs left to run"

    data = harness.make_data(raw_dataset, golden_meta["dataset"])
    calc = Calculator(dataset=data, configfile=str(configfile))
    assert list(calc.spis) == kept
    seed_all(golden_meta["seed"])
    harness.compute_calculator_with_reports(calc, with_callback)

    failures = _calculator_mismatches(calc, golden_meta, golden_values, kept)
    assert not failures, "Calculator table differs from per-SPI goldens:\n" + "\n".join(failures)


@pytest.mark.slow
@pytest.mark.java
@pytest.mark.regression
@pytest.mark.parametrize("with_callback", [False, True], ids=["default", "callback"])
def test_calculator_table_matches_goldens(golden_meta, golden_values, raw_dataset, seed_all, with_callback):
    from pyspi.calculator import Calculator

    data = harness.make_data(raw_dataset, golden_meta["dataset"])
    calc = Calculator(dataset=data)
    seed_all(golden_meta["seed"])
    harness.compute_calculator_with_reports(calc, with_callback)

    failures = _calculator_mismatches(calc, golden_meta, golden_values, golden_meta["spis"])
    assert not failures, "Calculator table differs from per-SPI goldens:\n" + "\n".join(failures)


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
_CATALOGUE_FIELDS = ("module", "cls", "params", "deps")


def _json_roundtrip(value):
    """Normalise a value the way ``meta.json`` stored it (tuples become lists, ...)."""
    return json.loads(json.dumps(value))


@pytest.mark.java
def test_catalogue_matches_golden_meta(golden_meta):
    """The loaded config equals the recorded catalogue, field by field.

    The per-SPI tests rebuild every SPI from the params recorded in
    ``meta.json``, so on their own they cannot notice a ``config.yaml`` edit
    that leaves the identifier unchanged.
    """
    from pyspi.calculator import Calculator

    configfile = _package_config()
    entries, by_id = harness.catalogue(configfile, harness.optional_dependencies())
    calc = Calculator(configfile=str(configfile))

    golden_spis = golden_meta["spis"]
    assert set(golden_spis) == set(calc.spis)
    # The harness must mirror what the Calculator actually loaded.
    assert list(by_id) == list(calc.spis)
    for ident, spi in calc.spis.items():
        assert type(spi).__name__ == by_id[ident]["cls"], ident

    diffs = []
    for ident, entry in by_id.items():
        recorded = golden_spis[ident]
        for field in _CATALOGUE_FIELDS:
            now = _json_roundtrip(entry[field])
            if now != recorded[field]:
                diffs.append(f"{ident}.{field}: config={now!r} golden={recorded[field]!r}")
        if sorted(set(entry["labels"])) != sorted(set(recorded["labels"])):
            diffs.append(f"{ident}.labels: config={sorted(set(entry['labels']))} "
                         f"golden={sorted(set(recorded['labels']))}")
    assert not diffs, "config.yaml differs from the recorded catalogue:\n" + "\n".join(diffs)


@pytest.mark.parametrize(
    "subset, expected_n",
    [("all", 285), ("fast", 217), ("sonnet", 14), ("fabfour", 4)],
)
def test_spi_catalogue(subset, expected_n, golden_meta, jvm):
    from pyspi.calculator import Calculator

    if not jvm:
        pytest.skip("JVM unavailable: Calculator would exclude the java SPIs")

    calc = Calculator(subset=subset)
    assert len(calc.spis) == expected_n
    if subset == "all":
        assert list(calc.spis) == golden_meta["calculator_order"]
        for ident, spi in calc.spis.items():
            assert sorted(set(spi.labels)) == sorted(set(golden_meta["spis"][ident]["labels"])), ident
    else:
        missing = [i for i in calc.spis if i not in golden_meta["spis"]]
        assert not missing, f"{subset} identifiers not in the full catalogue: {missing}"

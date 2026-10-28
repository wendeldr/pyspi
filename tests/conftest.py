"""Root pytest configuration for the pyspi suite.

Tiers
-----
* default: ``pytest`` — small golden size, no ``slow`` tests.
* ``pytest --runslow`` — also runs the slow SPI tests.
* ``pytest --pyspi-size=full --runslow`` — the complete run (7 x 100 inputs, every test).

Fixture ordering matters: the JVM must be running before any
``pyspi.statistics.infotheory`` SPI is instantiated (the module itself imports
without it), and ``pyspi.statistics.misc`` reseeds numpy on import, so
``seed_all`` depends on ``statistics_modules``.
"""
import os

import pytest

from regression import harness, spi_policy

# The storage-format suite has its own pytest configuration; pyproject's
# ``--ignore=tests/storage`` only applies from the repo root, so also ignore it
# when pytest is started from inside tests/.
collect_ignore = ["storage"]


# --------------------------------------------------------------------------- #
# Options and markers
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    parser.addoption(
        "--pyspi-size",
        choices=["small", "full"],
        default=os.environ.get("PYSPI_TEST_SIZE", "small"),
        help="Golden dataset size for the SPI regression tests (env: PYSPI_TEST_SIZE).",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked slow.",
    )
    parser.addoption(
        "--require-java", action="store_true", default=False,
        help="Fail instead of skipping when the JIDT JVM is unavailable.",
    )


def _golden_meta_or_none(size):
    if harness.golden_exists(size):
        return harness.load_meta(size)
    return None


def _str_param(callspec, name):
    """A string-valued parametrisation argument, else None.

    When a golden set is missing, ``test_spi_goldens.py`` parametrises with an
    empty list and pytest emits one placeholder item whose value is a NOTSET
    sentinel (the test then skips); it must not reach ``spi_policy``.
    """
    if callspec is None:
        return None
    value = callspec.params.get(name)
    return value if isinstance(value, str) else None


def pytest_collection_modifyitems(config, items):
    size = config.getoption("--pyspi-size")
    size_meta = _golden_meta_or_none(size)
    size_spis = size_meta["spis"] if size_meta else {}
    preproc_meta = _golden_meta_or_none("preproc")
    preproc_spis = preproc_meta["spis"] if preproc_meta else {}
    preproc_variants = preproc_meta.get("variants", {}) if preproc_meta else {}

    for item in items:
        callspec = getattr(item, "callspec", None)
        identifier = _str_param(callspec, "identifier")
        preproc_key = _str_param(callspec, "preproc_key")
        if identifier is not None:
            # Per-SPI tests: the selected size's meta.
            entry = size_spis.get(identifier, {})
            time_s, status = entry.get("time_s"), entry.get("status")
            deps = entry.get("deps", [])
        elif preproc_key is not None:
            # Preprocessing variants: that variant's own time/status, the SPI's deps.
            identifier = spi_policy.split_preproc_key(preproc_key)[0]
            variant = preproc_variants.get(preproc_key, {})
            time_s, status = variant.get("time_s"), variant.get("status")
            deps = preproc_spis.get(identifier, {}).get("deps", [])
        else:
            continue

        if spi_policy.is_slow(identifier, time_s):
            item.add_marker(pytest.mark.slow)
        if "java" in deps:
            item.add_marker(pytest.mark.java)
        if spi_policy.is_stochastic(identifier):
            item.add_marker(pytest.mark.stochastic)
        if status in ("all_nan", "error"):
            item.add_marker(pytest.mark.nan_golden)

    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="slow: needs --runslow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# --------------------------------------------------------------------------- #
# Session fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def pyspi_size(pytestconfig):
    return pytestconfig.getoption("--pyspi-size")


@pytest.fixture(scope="session")
def jvm(pytestconfig):
    """Start the JIDT JVM once per process. Returns True when it is running.

    Does not skip by itself: tests that need Java carry the ``java`` marker
    (or request ``jvm`` and check it) and are skipped by ``_skip_java_without_jvm``.
    """
    available = bool(harness.ensure_jvm())
    if not available and pytestconfig.getoption("--require-java"):
        pytest.fail("JIDT JVM is required by --require-java but is unavailable")
    return available


@pytest.fixture(scope="session")
def statistics_modules(jvm):
    """All ``pyspi.statistics.*`` modules, imported after the JVM is up.

    ``infotheory`` imports without a JVM, but its classes cannot be
    instantiated without the JVM, so it is left out and the ``java`` tests skip.
    """
    modules = list(harness.STATISTICS_MODULES)
    if not jvm:
        modules = [m for m in modules if m != "infotheory"]
    return harness.import_statistics_modules(modules)


@pytest.fixture(autouse=True)
def _skip_java_without_jvm(request):
    """Skip ``java``-marked tests when the JVM is unavailable."""
    if request.node.get_closest_marker("java") is not None:
        if not request.getfixturevalue("jvm"):
            pytest.skip("JVM unavailable: cannot run JIDT-based SPIs")


@pytest.fixture
def seed_all(statistics_modules):
    """Callable seeding numpy/random/torch; safe only after the statistics modules are imported."""
    return harness.seed_all


@pytest.fixture(scope="session")
def golden_meta(pyspi_size):
    harness.require_golden(pyspi_size)
    return harness.load_meta(pyspi_size)


@pytest.fixture(scope="session")
def golden_values(pyspi_size, golden_meta):
    return harness.load_values(pyspi_size)


@pytest.fixture(scope="session")
def raw_dataset(golden_meta):
    """Raw (processes x observations) array for the selected size, before preprocessing."""
    return harness.load_raw(golden_meta["dataset"])


@pytest.fixture
def spi_data(raw_dataset, golden_meta):
    """A fresh ``Data`` per test so per-SPI caches never leak between tests."""
    return harness.make_data(raw_dataset, golden_meta["dataset"])


@pytest.fixture(scope="session")
def preproc_meta():
    harness.require_golden("preproc")
    return harness.load_meta("preproc")


@pytest.fixture(scope="session")
def preproc_values(preproc_meta):
    return harness.load_values("preproc")


# --------------------------------------------------------------------------- #
# Hygiene
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _restore_optional_deps():
    """``Calculator._optional_dependencies`` is a class-level cache that some
    legacy tests overwrite (including a stale ``'octave'`` key) and that the
    legacy ``test_calc.py`` tests expect to be already populated. Fill it the
    way ``Calculator.__init__`` does if it is empty, and restore that clean
    copy after every test so one test's mutation cannot change which SPIs
    another test's Calculator loads."""
    from pyspi.calculator import Calculator
    from pyspi.utils import check_optional_deps
    if not Calculator._optional_dependencies:
        Calculator._optional_dependencies = check_optional_deps()
    clean = dict(Calculator._optional_dependencies)
    yield
    Calculator._optional_dependencies = dict(clean)

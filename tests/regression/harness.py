"""Shared machinery for the golden-value regression suite.

Used by both the pytest tests and ``generate_goldens.py`` so that an SPI is
built, fed, seeded and compared in exactly one way.

Conventions
-----------
* A *catalogue entry* is a dict ``{identifier, module, cls, params, labels,
  deps}`` describing how ``Calculator._load_yaml`` instantiated one SPI.
  ``params`` is ``None`` when the class was constructed without arguments
  (empty ``configs`` or the ``TypeError`` fallback in ``_load_yaml``).
* A *dataset spec* is a dict ``{file, dim_order, n_processes, n_observations,
  normalise, detrend, transform}`` where ``transform`` is ``None`` or
  ``{scale, offset, slope}`` (per-process lists) applied as
  ``x[i] * scale[i] + offset[i] + slope[i] * t`` to the raw, truncated
  (processes x observations) array before it is handed to ``Data``.
* Golden sets live under ``tests/regression/goldens/<name>/`` as
  ``values.npz`` (one float64 ``(p, p)`` array per key) and ``meta.json``.
"""
import copy
import functools
import importlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyspi
import yaml

from regression import spi_policy

TESTS_DIR = Path(__file__).resolve().parents[1]
#: Only used as the ``cwd`` of ``git rev-parse`` for provenance.
REPO_ROOT = TESTS_DIR.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
#: The imported package, so the config and data files are the ones
#: ``Calculator`` reads even if another ``pyspi`` shadows the repo checkout.
PACKAGE_DIR = Path(pyspi.__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"

STATISTICS_MODULES = [
    "basic",
    "distance",
    "causal",
    "infotheory",
    "spectral",
    "wavelet",
    "misc",
]

#: Packages whose versions are recorded in ``meta.json``.
TRACKED_PACKAGES = [
    "numpy", "scipy", "scikit-learn", "pandas", "statsmodels", "hyppo",
    "tslearn", "pyEDM", "jpype1", "mne", "spectral-connectivity", "nitime",
    "sktime", "cdt", "torch", "pyyaml",
]

SIZE_SPECS = {
    "full": dict(file="cml7.npy", dim_order="sp", n_processes=None, n_observations=None,
                 normalise=True, detrend=False, transform=None),
    "small": dict(file="cml7.npy", dim_order="sp", n_processes=5, n_observations=64,
                  normalise=True, detrend=False, transform=None),
}

#: Deterministic per-process affine distortion for the preprocessing variants.
#: Raw cml7 is near unit scale and trend-free, so without this the
#: ``normalise``/``detrend`` variants would be almost indistinguishable.
PREPROC_TRANSFORM = dict(
    scale=[0.5, 2.0, 5.0, 0.2, 10.0, 1.0, 3.0],
    offset=[0.0, 1.0, -2.0, 5.0, 0.5, -1.0, 2.0],
    slope=[0.0, 0.05, -0.02, 0.1, 0.01, -0.05, 0.03],
)


# --------------------------------------------------------------------------- #
# Runtime set-up
# --------------------------------------------------------------------------- #
def ensure_jvm():
    """Start the JIDT JVM if possible. Returns True when a JVM is running.

    ``pyspi.statistics.infotheory`` imports without a JVM, but its classes
    cannot be instantiated without it, so this must run before any of them is
    built. The JVM cannot be restarted in-process, so never shut it down.
    """
    from pyspi.utils import is_jpype_jvm_available
    return bool(is_jpype_jvm_available())


def import_statistics_modules(modules=STATISTICS_MODULES):
    """Import the statistics modules (after :func:`ensure_jvm`).

    Importing ``pyspi.statistics.misc`` calls ``np.random.seed(1717)`` as a
    side effect, which is why :func:`seed_all` must run after this.
    """
    return {m: importlib.import_module(f"pyspi.statistics.{m}") for m in modules}


def seed_all(seed=spi_policy.SEED):
    """Seed every RNG an SPI might touch. Call right before ``multivariate``."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # torch absent or unusable; nothing to seed
        pass


def optional_dependencies():
    """The dependency flags ``Calculator`` uses, computed the same way."""
    from pyspi.utils import check_optional_deps
    return check_optional_deps()


# --------------------------------------------------------------------------- #
# Catalogue: mirror Calculator._load_yaml exactly
# --------------------------------------------------------------------------- #
def iter_config_entries(configfile=None, optional_deps=None):
    """Instantiate every SPI in a config, mirroring ``Calculator._load_yaml``.

    Yields ``(entry, spi)`` pairs in the order the Calculator would register
    them. Identifier collisions are *not* resolved here (the Calculator keeps
    the last one); callers that build a dict get the same last-wins result.

    The mirrored quirks: a class whose ``dependencies`` are unmet is skipped
    entirely; a ``TypeError`` raised while iterating ``configs`` (including
    ``configs`` being ``None``) aborts that class's loop and adds one
    default-constructed instance.
    """
    configfile = Path(configfile) if configfile is not None else DEFAULT_CONFIG
    if optional_deps is None:
        optional_deps = optional_dependencies()
    with open(configfile) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)

    for module_name in yf:
        module = importlib.import_module(module_name, "pyspi")
        for fcn in yf[module_name]:
            spec = yf[module_name][fcn]
            deps = spec.get("dependencies")
            if deps is not None:
                if not all(optional_deps.get(dep, False) for dep in deps):
                    continue
            cls = getattr(module, fcn)
            try:
                for params in spec.get("configs"):
                    spi = cls(**params)
                    yield _entry(module_name, fcn, params, deps, spi), spi
            except TypeError:
                spi = cls()
                yield _entry(module_name, fcn, None, deps, spi), spi


def _entry(module_name, cls_name, params, deps, spi):
    return dict(
        identifier=spi.identifier,
        module=module_name,
        cls=cls_name,
        params=copy.deepcopy(params) if params is not None else None,
        labels=list(spi.labels),
        deps=list(deps) if deps else [],
    )


def catalogue(configfile=None, optional_deps=None):
    """``(entries_in_order, dict_by_identifier)`` with last-wins on collisions."""
    entries = []
    by_id = {}
    spis = []
    for entry, spi in iter_config_entries(configfile, optional_deps):
        entries.append(entry)
        by_id[entry["identifier"]] = entry
        spis.append(spi)
    # Labels are re-read after the whole walk: SpearmanR/KendallTau extend a
    # class-level list in place (known bug), so an instance's labels depend on
    # which siblings have been constructed since. Recording the final state
    # matches what ``Calculator().spis[...].labels`` reports.
    for entry, spi in zip(entries, spis):
        entry["labels"] = sorted(set(spi.labels))
    return entries, by_id


def duplicate_identifiers(entries):
    seen, dups = set(), []
    for e in entries:
        if e["identifier"] in seen and e["identifier"] not in dups:
            dups.append(e["identifier"])
        seen.add(e["identifier"])
    return dups


def build_spi(entry):
    """Instantiate one SPI from a catalogue entry.

    ``params is None`` records that ``_load_yaml`` default-constructed the
    class. Otherwise the recorded params are passed as-is with no fallback, so
    a renamed or removed constructor argument fails loudly instead of silently
    building a default instance.
    """
    module = importlib.import_module(entry["module"], "pyspi")
    cls = getattr(module, entry["cls"])
    return cls() if entry["params"] is None else cls(**entry["params"])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def size_spec(name):
    return copy.deepcopy(SIZE_SPECS[name])


def preproc_spec(normalise, detrend, base="small"):
    spec = size_spec(base)
    spec["normalise"] = bool(normalise)
    spec["detrend"] = bool(detrend)
    spec["transform"] = copy.deepcopy(PREPROC_TRANSFORM)
    return spec


def load_raw(spec):
    """Raw (processes x observations) float64 array for a dataset spec, before preprocessing."""
    arr = np.load(DATA_DIR / spec["file"]).astype(np.float64)
    if spec["dim_order"] == "sp":
        arr = arr.T
    elif spec["dim_order"] != "ps":
        raise ValueError(f"Unsupported dim_order {spec['dim_order']!r}")
    if spec.get("n_processes") is not None:
        arr = arr[: spec["n_processes"]]
    if spec.get("n_observations") is not None:
        arr = arr[:, : spec["n_observations"]]
    transform = spec.get("transform")
    if transform:
        p, t = arr.shape
        scale = np.asarray(transform["scale"][:p], dtype=np.float64)[:, None]
        offset = np.asarray(transform["offset"][:p], dtype=np.float64)[:, None]
        slope = np.asarray(transform["slope"][:p], dtype=np.float64)[:, None]
        if scale.shape[0] != p:
            raise ValueError("transform arrays shorter than the number of processes")
        arr = arr * scale + offset + slope * np.arange(t, dtype=np.float64)[None, :]
    return np.ascontiguousarray(arr)


def make_data(raw, spec):
    """A fresh ``pyspi.data.Data`` with the spec's preprocessing flags.

    ``raw`` is already (processes x observations) and truncated, so the
    result matches ``Data(file, dim_order=..., n_processes=..., n_observations=...)``
    within rounding (~1e-15; memory layout differs): ``Data.set_data``
    truncates before it detrends and z-scores.
    """
    from pyspi.data import Data
    return Data(np.array(raw, copy=True), dim_order="ps",
                normalise=spec["normalise"], detrend=spec["detrend"])


# --------------------------------------------------------------------------- #
# Compute and compare
# --------------------------------------------------------------------------- #
def compute_spi(spi, data):
    """``spi.multivariate(data)`` post-processed exactly like ``Calculator.compute``."""
    S = spi.multivariate(data)
    S = np.array(S, dtype=np.float64, copy=True)
    np.fill_diagonal(S, np.nan)
    return S


def status_of(S):
    """``ok | partial_nan | all_nan`` using ``inspect_calc_results``'s rule."""
    n = S.shape[0]
    nan_count = int(np.isnan(S).sum())
    if nan_count == S.size:
        return "all_nan"
    if nan_count > n:
        return "partial_nan"
    return "ok"


def assert_spi_allclose(actual, expected, identifier, rtol=None, atol=None):
    if rtol is None or atol is None:
        prtol, patol = spi_policy.tolerance_for(identifier)
        rtol = prtol if rtol is None else rtol
        atol = patol if atol is None else atol
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{identifier}: shape {actual.shape} != {expected.shape}"
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected),
                                  err_msg=f"{identifier}: NaN mask differs")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True,
                               err_msg=f"{identifier}: values differ (rtol={rtol}, atol={atol})")


def max_differences(actual, expected):
    """``(max_abs, max_rel)`` over entries that differ, ignoring positions that
    are NaN in both or equal in both (so equal infinities do not give NaN)."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    both = ~(np.isnan(a) & np.isnan(e)) & ~(a == e)
    if not both.any():
        return 0.0, 0.0
    diff = np.abs(a[both] - e[both])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.abs(e[both])
    rel = rel[np.isfinite(rel)]
    return float(np.nanmax(diff)) if diff.size else 0.0, float(rel.max()) if rel.size else 0.0


# --------------------------------------------------------------------------- #
# Golden files
# --------------------------------------------------------------------------- #
def golden_dir(name):
    return GOLDEN_DIR / name


def golden_exists(name):
    d = golden_dir(name)
    return (d / "values.npz").is_file() and (d / "meta.json").is_file()


@functools.lru_cache(maxsize=None)
def load_meta(name):
    with open(golden_dir(name) / "meta.json") as f:
        return json.load(f)


def load_values(name):
    """Dict ``{key: ndarray}`` (loaded eagerly so the npz handle is closed)."""
    with np.load(golden_dir(name) / "values.npz") as z:
        return {k: z[k] for k in z.files}


def save_golden(name, values, meta):
    d = golden_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    for key in values:
        if "/" in key or key.startswith("."):
            raise ValueError(f"unsafe npz key {key!r}")
    np.savez_compressed(d / "values.npz", **values)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")


def environment_info():
    """Provenance recorded in ``meta.json``."""
    from importlib import metadata as md
    pkgs = {}
    for p in TRACKED_PACKAGES:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    try:
        import jpype
        java = str(jpype.java.lang.System.getProperty("java.version")) if jpype.isJVMStarted() else None
    except Exception:
        java = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    try:
        pyspi_version = md.version("pyspi")
    except md.PackageNotFoundError:
        pyspi_version = None
    return dict(
        python=sys.version.split()[0],
        platform=sys.platform,
        java=java,
        pyspi_commit=commit,
        pyspi_version=pyspi_version,
        packages=pkgs,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
    )

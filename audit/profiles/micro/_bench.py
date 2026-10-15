"""Shared helpers for ``audit/profiles/micro/opp_<opportunity_id>.py`` micro-benchmarks.

Project venv only (``.venv/bin/python``). A micro-benchmark times the current pyspi
call (the baseline) against a candidate implementation that lives INSIDE the
micro-benchmark script (never in ``pyspi/``) on the profiler's synthetic data, and
records how far the candidate's output is from the baseline's. The baseline is
today's output; agreement with it says only that the candidate reproduces the
current behaviour, not that either value is mathematically right.

Typical use::

    import sys; sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/profiles/micro")
    import _bench as B
    raw, make = B.data("8x1024")                     # (P, T) raw array, fresh-Data factory
    S_base, t_base = B.baseline("spearmanr", make)   # seeded pyspi call, min wall over repeats
    t_cand, S_cand = B.timeit(lambda: candidate(make()))
    B.write("rank_corr-1", {"8x1024": B.result(S_base, t_base, S_cand, t_cand)}, notes="...")

``data(shape)``: shapes as in profile_units (``7x100`` is the golden cml7 set,
``PxT`` synthetic AR(2) + delayed pink sources, seed 20261014, detrend + z-score
through ``Data``). ``make()`` returns a fresh ``Data`` each call, so caches do not
carry across repeats unless the script shares one ``Data`` deliberately.
"""
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/media/dan/Data/git/pyspi")
MICRO = REPO / "audit" / "profiles" / "micro"
for p in (REPO / "tests", REPO / "audit" / "profiles"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import profile_units as pu  # noqa: E402  (synthetic(), data_factory(), parse_shape(), load_units())
from regression import harness as h  # noqa: E402

SPI_SEED = 42
_ENTRIES = None


def _entries():
    """identifier -> build entry: config catalogue, then extra_config, then units.yaml params."""
    global _ENTRIES
    if _ENTRIES is None:
        ents = {}
        for unit in pu.load_units(REPO / "audit" / "units.yaml"):
            for ident in unit["identifiers"]:
                params = (unit.get("identifier_params") or {}).get(ident)
                if params is not None and unit.get("pyspi_class") and unit.get("module"):
                    ents[ident] = dict(identifier=ident, module=f".statistics.{unit['module']}",
                                       cls=unit["pyspi_class"], params=json.loads(json.dumps(params)))
        for e in h.catalogue(pu.EXTRA_CONFIG)[0]:
            ents[e["identifier"]] = e
        for e in h.catalogue()[0]:
            ents[e["identifier"]] = e
        _ENTRIES = ents
    return _ENTRIES


def jvm():
    """Start the JIDT JVM (needed before building any infotheory SPI). Returns True if running."""
    return bool(h.ensure_jvm())


def spi(identifier):
    """A freshly built SPI whose ``identifier`` is asserted to match."""
    entry = _entries()[identifier]
    if entry["module"].endswith("infotheory"):
        jvm()
    s = h.build_spi(entry)
    assert s.identifier == identifier, (s.identifier, identifier)
    return s


def data(shape):
    """``(raw, make)`` for a profile_units shape name such as ``8x1024``."""
    return pu.data_factory(h, pu.parse_shape(shape))


def timeit(fn, repeats=3, cutoff_s=30.0):
    """``(min_wall_s, last_result, walls)``; stops repeating once a call exceeds ``cutoff_s``."""
    walls, out = [], None
    for _ in range(max(1, repeats)):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
        if walls[-1] > cutoff_s:
            break
    return min(walls), out, walls


def baseline(identifier, make, repeats=3, cutoff_s=30.0):
    """Seeded ``harness.compute_spi`` on a fresh ``Data`` per repeat: ``(S, min_wall_s)``.

    One untimed warm-up call first (numba compilation, JVM class loading, imports).
    """
    s = spi(identifier)
    h.seed_all(SPI_SEED)
    h.compute_spi(s, make())

    def call():
        d = make()
        sp = spi(identifier)
        h.seed_all(SPI_SEED)
        return h.compute_spi(sp, d)
    wall, S, _ = timeit(call, repeats, cutoff_s)
    return S, wall


def diff(base, cand):
    """max_abs, max_rel (``harness.max_differences``), nan_mask_equal, n_entries, bit_identical."""
    a = np.asarray(base, dtype=np.float64)
    b = np.asarray(cand, dtype=np.float64)
    if a.shape != b.shape:
        return dict(max_abs_diff=None, max_rel_diff=None, nan_mask_equal=False,
                    n_entries=int(a.size), bit_identical=False,
                    shape_mismatch=[list(a.shape), list(b.shape)])
    max_abs, max_rel = h.max_differences(b, a)
    nan_eq = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    bit = nan_eq and bool(np.array_equal(a[~np.isnan(a)].view(np.uint64),
                                         b[~np.isnan(b)].view(np.uint64)))
    return dict(max_abs_diff=max_abs, max_rel_diff=max_rel, nan_mask_equal=nan_eq,
                n_entries=int(a.size), bit_identical=bit)


def result(S_base, t_base, S_cand, t_cand, **extra):
    """One shape's record: timings, speedup and the diff of candidate vs baseline output."""
    r = dict(baseline_s=t_base, candidate_s=t_cand,
             speedup=(t_base / t_cand) if t_cand else None, **diff(S_base, S_cand))
    r.update(extra)
    return r


def golden(identifier):
    """The 7x100 golden snapshot array for ``identifier`` (``None`` if absent).

    A snapshot of current pyspi output, used only to confirm a candidate reproduces it.
    """
    vals = h.load_values("full")
    return vals.get(identifier)


def write(opportunity_id, shapes, notes="", **extra):
    """Write ``opp_<opportunity_id>.json`` next to the scripts and return its path."""
    doc = dict(schema=1, opportunity_id=opportunity_id, shapes=shapes, notes=notes,
               script=f"audit/profiles/micro/opp_{opportunity_id}.py",
               env=dict(python=platform.python_version(), numpy=np.__version__,
                        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                        cpu_count=os.cpu_count()),
               written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    path = MICRO / f"opp_{opportunity_id}.json"
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)
    return path

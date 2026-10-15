#!/usr/bin/env python
"""Timing-protocol runner for the pyspi optimisation audit.

Reads ``audit/units.yaml`` (or fabricates a 3-unit catalogue with ``--demo``) and
writes one JSON record per (unit, shape, mode) to ``--out`` (default
``audit/profiles``)::

    .venv/bin/python audit/profiles/profile_units.py --all --shapes 7x100,8x256,8x1024,32x1024,118x1024
    .venv/bin/python audit/profiles/profile_units.py --units spectral.cohmag --profile --shapes 8x1024
    .venv/bin/python audit/profiles/profile_units.py --summarise

Process layout: the parent never imports pyspi or starts a JVM. The catalogue
is dumped by one ``--dump-catalogue`` subprocess; every (family, shape) at
P <= 32 and every (unit, shape) [fresh] / (family, shape) [family-warm] at
P > 32 runs in its own ``--worker`` subprocess, so peak RSS and JVM state are
attributable. Workers start the JVM only when the family contains an
infotheory SPI (``--jvm always`` overrides). Each worker runs in its own
process group under a timeout of ``budget_s * n_calls + 120`` s (``n_calls``
counts warm-up calls and the maximum number of timed repeats); on expiry the
whole group is killed and every record the worker had not written yet gets
``status = measure_mode = "timeout"`` with ``timeout_s``.

Units and SPIs: ``units.yaml`` units with ``source: infrastructure`` are
skipped. Each identifier's SPI entry comes from the unit's own catalogue
(``config``: pyspi/config.yaml, ``extra``: tests/regression/extra_config.yaml)
or, failing that, any catalogue; identifiers in no catalogue (e.g. the
``source: retained`` units) are built from the unit's ``identifier_params`` +
``pyspi_class`` + ``module`` and the worker asserts ``spi.identifier`` equals
the expected identifier. ``entry_sources`` in every record says which applied.
Family order is catalogue order (config, then extra); non-catalogue
identifiers follow the catalogue ones in ``units.yaml`` order.

Peak RSS is per record: VmHWM is reset through ``/proc/self/clear_refs``
before each measurement (``hwm_reset`` says whether that worked);
``process_peak_rss_mb`` is the worker's lifetime peak (``ru_maxrss_mb`` is
recorded too but follows the reset hiwater mark). ``peak_rss_delta_mb`` =
VmHWM - VmRSS at the start, clamped at 0; the kernel batches RSS counters per
CPU, so deltas below ~10 MB are not resolved.

Timing: ``harness.seed_all(42)`` then ``harness.compute_spi`` around
``perf_counter``; SPI construction and ``Data`` construction are untimed.
Every worker first makes one untimed call of every identifier it will time
(each on its own fresh 8x1024 ``Data``; for a family-warm job that is every
identifier of the family) so numba/JVM/pyEDM first-call costs are excluded;
the cold time is ``t_cold_s`` (the representative's) and, in family-warm
records, ``increments_s[<id>].t_cold_s``. Repeats: ``--repeats`` (default 3)
at P <= 32, 1 above. An identifier is not re-timed once one of its timed
calls (normally the first) exceeded 30 s; the cutoff is per identifier, the
others keep their repeats, and ``repeats`` is recorded per identifier in
family-warm (``increments_s[<id>].repeats``). If ``wall_cv``
> 0.25 after >= 3 repeats at P <= 32, repeats are extended to 6 (subject to
the same 30 s rule); ``repeats_extended`` says so and ``wall_cv`` is the final
value. Every timed call records ``harness.status_of(S)``: ``result_status``
is the worst (``all_nan`` > ``partial_nan`` > ``ok``) over the record's calls,
so a fast all-NaN failure is visible; family-warm records also carry it per
identifier.

Family-warm: the family's identifiers run in family order on one ``Data`` per
pass, and ``increments_s[<id>]`` (min/median over passes, ``repeats`` per
identifier) is the primary data. A family can hold several cache groups
(pyspi's spectral cache is keyed per measure, so phase / cohmag / icoh each
pay their own cross-spectral computation, and the family's very first measure
pays it twice because ``data.spectral_mv`` is first created without the key),
so no single "shared cost" is recorded: ``first_identifier`` /
``first_identifier_increment_s`` are family-level (the first identifier in
family order and its increment), ``unit_first_identifier`` /
``unit_first_increment_s`` are the first of the unit's own identifiers in
family order, from which a unit's cache-group cost can be read off. A unit's
``wall_s`` is the min over passes of the sum of its identifiers' increments
(passes where all of them were timed), and ``t_per_pair_ms`` is that
summed-increment ``wall_s`` divided by ``n_pairs``; ``repeats`` is the number
of such passes. If an identifier is not re-timed (a timing > 30 s;
``not_retimed`` maps it to that pass), later identifiers of its cache group
may absorb its cache cost in later passes; ``min_s`` is unaffected but
``median_s`` and ``cv`` may be inflated.

``measure_mode`` is one of ``measured``, ``extrapolated-pairs-from-32``,
``extrapolated-power-from-32``, ``extrapolated-T`` (the projection applied a
T-exponent measured from 8x256/8x1024 other than 1), ``infeasible``,
``skipped-budget``, ``error``, ``timeout``. Skip records carry the projection
that gated them in a nested ``fit`` object (``fit.measure_mode`` says which
fit produced it). Power fits with t32 < t8 are clamped flat at max(t8, t32)
(``fit_clamped``).

Profile ownership heuristic (``--profile``): cProfile only creates frames for
Python functions and ``builtin_function_or_method`` objects. ``owner`` is
``pyspi`` for files under the pyspi package, ``harness`` for files under
``tests/``, ``third-party:<pkg>`` for files under ``site-packages/<pkg>``,
``python`` otherwise (stdlib, generated code). A builtin entry (file ``~``:
numpy/scipy C functions, ...) gets the owner of the caller that spent the most
cumulative time calling it. Every Python-code frame (``.py`` files, frozen
modules, ``<string>``) is transparent; ``opaque_share`` is the fraction of
total self time (tottime) in builtin entries. Calls into callables that
cProfile does not see at all (numba dispatchers, JPype Java methods,
Cython/C-extension types' ``__call__``, ufuncs invoked through operators) are
folded into the *calling Python frame's* tottime, so they count as transparent
time of that caller: ``opaque_share`` is a lower bound on time spent outside
Python.

``--summarise`` writes ``summary.json``: per unit and mode the measured walls,
``wall_cv`` per shape, the 8x1024/32x1024 fit and ``projected_window_s_118``
(measured at 118x1024, else from the fit, else null with a ``reason``).
``machine_factor`` (fresh only) divides the 7x100 wall by the golden
``time_s``; ``golden_environment_note`` says why it conflates machine and
software environment.
"""
from __future__ import annotations

import argparse, fnmatch, gc, hashlib, json, math, os, resource, signal, statistics
import subprocess, sys, tempfile, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
DEFAULT_UNITS = REPO / "audit" / "units.yaml"
DEFAULT_OUT = REPO / "audit" / "profiles"
EXTRA_CONFIG = REPO / "tests" / "regression" / "extra_config.yaml"
GOLDEN_META = REPO / "tests" / "regression" / "goldens" / "full" / "meta.json"

SPI_SEED = 42
SYNTH_SEED = 20261014
SYNTH_BURN_IN = 200
SYNTH_MAX_DELAY = 5
AR_COEFS = (1.5, -0.8)
PINK_EXPONENT = 0.5        # amplitude ~ f^-0.5  =>  power ~ 1/f (pink)
# Warm-up shape (one untimed call per identifier before timing; gives t_cold_s). Overridable via
# PROFILE_WARMUP_SHAPE for units whose 8x1024 call is too slow to warm up with (causal.ccm.*);
# every record stores the shape it used in ``warmup_shape``.
WARMUP_SHAPE = os.environ.get("PROFILE_WARMUP_SHAPE", "8x1024")
REPEAT_CUTOFF_S = 30.0     # an identifier whose first timing exceeds this is not re-timed
NOISY_CV = 0.25            # wall_cv above this after >= 3 repeats at P <= SMALL_P ...
NOISY_MAX_REPEATS = 6      # ... extends repeats up to this many
SMALL_P = 32               # P <= SMALL_P: repeats, one subprocess per (family, shape)
WINDOWS_PER_PATIENT = 600
TIMEOUT_SLACK_S = 120.0    # worker timeout = budget_s * n_calls + this
MODES = ("fresh", "family-warm")
MEASURE_MODES = ("measured", "extrapolated-pairs-from-32", "extrapolated-power-from-32",
                 "extrapolated-pairs-from-8", "extrapolated-T", "infeasible", "skipped-budget",
                 "error", "timeout")
RESULT_STATUS_RANK = {"ok": 0, "partial_nan": 1, "all_nan": 2}
# Multivariate units whose cost is a Python loop over channel pairs inside the delegate, so the
# pairs-only projection from 8x1024 applies with this basis (summarise only; records keep
# pair_basis "multivariate"). spectral.sgc.nonparametric: one Wilson factorisation per unordered
# pair (spectral_connectivity/connectivity.py:722-743).
PAIR_LOOP_BASIS = {"spectral.sgc.nonparametric": "unordered"}

# Every timing record (worker-written, skip, error, timeout) carries these keys.
COMMON_KEYS = (
    "schema", "unit_id", "family", "mode", "pair_basis", "representative", "identifiers",
    "entry_sources", "shape", "P", "T", "data_kind", "data_sha256", "edf", "n_pairs",
    "status", "error", "traceback_tail", "reason", "measure_mode", "fit", "timeout_s",
    "t_cold_s", "warmup_shape", "warmup_error", "repeats", "repeats_planned",
    "repeats_extended", "wall_s", "wall_median_s", "wall_cv", "walls_s", "t_per_pair_ms",
    "result_status", "peak_rss_mb", "baseline_rss_mb", "peak_rss_delta_mb", "ru_maxrss_mb",
    "process_peak_rss_mb", "hwm_reset", "budget_s", "jvm_requested", "jvm_started",
    "threads", "env", "git_commit", "pyspi_dirty", "started_at", "finished_at", "spi_seed",
    "synth_seed")
MODE_KEYS = {
    "fresh": ("cpu_s", "cpu_over_wall", "result_statuses"),
    "family-warm": ("family_identifiers", "increments_s", "first_identifier",
                    "first_identifier_increment_s", "unit_first_identifier",
                    "unit_first_increment_s", "family_total_s", "peak_rss_scope",
                    "not_retimed"),
    "profile": (),
}


def log(msg):
    print(f"[profile_units] {msg}", file=sys.stderr, flush=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def git_pyspi_dirty():
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "pyspi"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return None


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def worst_status(statuses):
    vals = [s for s in statuses if s is not None]
    return max(vals, key=lambda s: RESULT_STATUS_RANK.get(s, 3)) if vals else None


def record_skeleton(unit, shape, mode, entry_sources=None, **known):
    """Every key a timing record of ``mode`` carries, unknown ones ``None``.

    Used by the worker (``base``) and by the parent's skip / error / timeout
    records so that all records of a mode share one schema.
    """
    rec = dict.fromkeys(COMMON_KEYS + MODE_KEYS.get(mode, ()))
    idents = list(unit["identifiers"]) if mode == "family-warm" else [unit["representative"]]
    rec.update(schema=1, unit_id=unit["unit_id"], family=unit["family"], mode=mode,
               pair_basis=unit["pair_basis"], representative=unit["representative"],
               identifiers=idents,
               entry_sources={i: (entry_sources or {}).get(i) for i in idents},
               shape=shape["name"], P=shape["P"], T=shape["T"], data_kind=shape["kind"],
               edf=shape.get("edf"), n_pairs=n_pairs(shape["P"], unit["pair_basis"]),
               warmup_shape=WARMUP_SHAPE, spi_seed=SPI_SEED,
               synth_seed=SYNTH_SEED if shape["kind"] == "synthetic" else None)
    rec.update(known)
    return rec


# ---- Shapes and data -------------------------------------------------------
def parse_shape(name):
    """``7x100`` is the golden cml7 dataset; ``edfPxT`` a real EDF window; else synthetic."""
    kind = "edf" if name.startswith("edf") else ("golden" if name == "7x100" else "synthetic")
    p, t = (int(v) for v in name.removeprefix("edf").split("x"))
    return dict(name=name, kind=kind, P=p, T=t)


def shape_key(shape):
    return (shape["P"] * shape["T"], shape["P"])


def above_8x1024(shape):
    return shape["P"] > 8 or shape["T"] > 1024


def synthetic(P, T, seed=SYNTH_SEED):
    """Per-channel AR(2) plus K = max(1, P//4) delayed pink sources (see module doc)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    n, k, d = T + SYNTH_BURN_IN, max(1, P // 4), SYNTH_MAX_DELAY
    ar = lfilter([1.0], [1.0, -AR_COEFS[0], -AR_COEFS[1]], rng.standard_normal((P, n)), axis=1)
    ar /= ar.std(axis=1, keepdims=True)
    white = rng.standard_normal((k, n + d))
    freqs = np.fft.rfftfreq(n + d)
    gain = np.zeros_like(freqs)
    gain[1:] = freqs[1:] ** -PINK_EXPONENT
    src = np.fft.irfft(np.fft.rfft(white, axis=1) * gain, n=n + d, axis=1)
    src /= src.std(axis=1, keepdims=True)
    mix = rng.standard_normal((P, k)) / math.sqrt(k)
    delays = rng.integers(0, d + 1, size=P)
    shared = np.stack([mix[c] @ src[:, d - delays[c]: d - delays[c] + n] for c in range(P)])
    return np.ascontiguousarray((ar + shared)[:, SYNTH_BURN_IN:], dtype=np.float64)


def sha256_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def load_edf_window(path, n_channels, window_s, start_s, tmpdir):
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    sf = float(raw.info["sfreq"])
    i0, n = int(round(start_s * sf)), int(round(window_s * sf))
    names = raw.ch_names[:n_channels]
    arr = np.ascontiguousarray(raw.get_data(picks=names, start=i0, stop=i0 + n), dtype=np.float64)
    del raw
    shape = parse_shape(f"edf{arr.shape[0]}x{arr.shape[1]}")
    npy = Path(tmpdir) / f"{shape['name']}.npy"
    np.save(npy, arr)
    shape.update(npy=str(npy), edf=dict(path=str(path), sfreq=sf, start_s=start_s,
                                        window_s=window_s, channels=names))
    return shape


def data_factory(h, shape):
    """``(raw_array, make)`` where ``make()`` returns a fresh ``pyspi.data.Data``."""
    from pyspi.data import Data
    if shape["kind"] == "golden":
        dspec = h.size_spec("full")
        raw = h.load_raw(dspec)
        return raw, lambda: h.make_data(raw, dspec)
    raw = np.load(shape["npy"]) if shape["kind"] == "edf" else synthetic(shape["P"], shape["T"])
    return raw, lambda: Data(raw.copy(), dim_order="ps", normalise=True, detrend=True)


# ---- Scaling model ---------------------------------------------------------
def n_pairs(P, basis):
    return {"ordered": P * (P - 1), "unordered": P * (P - 1) // 2}.get(basis)


def fit_scaling(t8, t32, basis):
    """``t = t_fixed + n_pairs*t_pair`` (bivariate) or ``t = a*P^b`` (multivariate).

    ``fit_clamped`` is true when the raw fit was replaced: flat at max(t8, t32)
    when the time fell from 8 to 32 channels (both models), or pinned through
    the 32 point when the pairs fit had a negative intercept.
    """
    p_exp = math.log(t32 / t8) / math.log(4) if t8 > 0 and t32 > 0 else None
    if n_pairs(8, basis) is not None:
        n8, n32 = n_pairs(8, basis), n_pairs(32, basis)
        t_pair = (t32 - t8) / (n32 - n8)
        t_fixed = t8 - n8 * t_pair
        clamped = None
        if t_pair <= 0:                  # noise-dominated: flat in P
            t_fixed, t_pair, clamped = max(t8, t32), 0.0, "flat"
        elif t_fixed < 0:                # super-linear in pairs: pin through the 32 point
            t_fixed, t_pair, clamped = 0.0, t32 / n32, "pinned-through-32"
        return dict(model="pairs", t_fixed=t_fixed, t_pair=t_pair, P_exponent=p_exp,
                    fit_clamped=clamped is not None, clamp=clamped,
                    measure_mode="extrapolated-pairs-from-32")
    if p_exp is None:
        return None
    if p_exp < 0:                        # t32 < t8: clamp the exponent at 0 (flat)
        return dict(model="power", a=max(t8, t32), b=0.0, P_exponent=p_exp, fit_clamped=True,
                    clamp="flat", measure_mode="extrapolated-power-from-32")
    return dict(model="power", a=t8 / 8 ** p_exp, b=p_exp, P_exponent=p_exp, fit_clamped=False,
                clamp=None, measure_mode="extrapolated-power-from-32")


def project(fit, P, basis, T=1024, t_exp=None):
    """``(seconds, measure_mode)`` of ``fit`` evaluated at ``P`` x ``T``.

    The T scaling uses ``t_exp`` (8x256 -> 8x1024 exponent) when known, else 1;
    ``measure_mode`` becomes ``extrapolated-T`` only when an exponent other
    than 1 was actually applied (T != 1024).
    """
    if fit is None:
        return None, None
    t = (fit["t_fixed"] + n_pairs(P, basis) * fit["t_pair"] if fit["model"] == "pairs"
         else fit["a"] * P ** fit["b"])
    applied = T != 1024 and t_exp is not None and t_exp != 1.0
    t *= (T / 1024) ** (t_exp if applied else 1.0)
    return t, ("extrapolated-T" if applied else fit["measure_mode"])


def projection_fit(fit, P, basis, T, t_exp):
    """The nested ``fit`` object of a skip record: the fit plus its projection."""
    if fit is None:
        return None
    p, mode = project(fit, P, basis, T, t_exp)
    return {**fit, "measure_mode": mode, "p_fit_measure_mode": fit["measure_mode"],
            "projected_wall_s": p, "projected_shape": f"{P}x{T}",
            "T_exponent": t_exp, "T_exponent_applied": mode == "extrapolated-T"}


# ---- Worker side -----------------------------------------------------------
def _proc_status_kb(key):
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


_PROCESS_PEAK_KB = 0


class RssWindow:
    """Peak RSS over a block: reset VmHWM, remember VmRSS, read VmHWM at the end.

    Resetting VmHWM also resets the kernel's hiwater mark that ``ru_maxrss``
    reads, so the worker's lifetime peak is tracked here (``process_peak_rss_mb``).
    The kernel batches RSS counters per CPU, so VmRSS/VmHWM lag by up to ~10 MB:
    ``peak_rss_delta_mb`` (clamped at 0) does not resolve allocations below that.
    """

    def __init__(self):
        global _PROCESS_PEAK_KB
        gc.collect()
        _PROCESS_PEAK_KB = max(_PROCESS_PEAK_KB, _proc_status_kb("VmHWM") or 0)
        try:
            Path("/proc/self/clear_refs").write_text("5")
            self.reset = True
        except OSError:
            self.reset = False
        self.base = _proc_status_kb("VmRSS")

    def stop(self):
        global _PROCESS_PEAK_KB
        hwm = _proc_status_kb("VmHWM")
        _PROCESS_PEAK_KB = max(_PROCESS_PEAK_KB, hwm or 0)
        mb = lambda kb: None if kb is None else round(kb / 1024, 1)
        return dict(peak_rss_mb=mb(hwm), baseline_rss_mb=mb(self.base),
                    peak_rss_delta_mb=(mb(max(0, hwm - self.base))
                                       if hwm is not None and self.base is not None else None),
                    ru_maxrss_mb=mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    process_peak_rss_mb=mb(_PROCESS_PEAK_KB), hwm_reset=self.reset)


def thread_info():
    info = dict(omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                affinity_cpus=len(os.sched_getaffinity(0)),
                os_threads=_proc_status_kb("Threads"))
    try:
        import numba
        info["numba_threads"] = numba.get_num_threads()
    except Exception:
        info["numba_threads"] = None
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = [{k: d.get(k) for k in ("user_api", "internal_api", "num_threads")}
                               for d in threadpool_info()]
    except Exception:
        info["threadpools"] = None
    if "torch" in sys.modules:
        info["torch_threads"] = sys.modules["torch"].get_num_threads()
    try:
        import jpype
        info["jvm_running"] = bool(jpype.isJVMStarted())
    except Exception:
        info["jvm_running"] = False
    return info


def build_checked(h, entry):
    """``harness.build_spi`` plus a check that the SPI has the expected identifier."""
    spi = h.build_spi(entry)
    want = entry.get("identifier")
    if want is not None and spi.identifier != want:
        raise AssertionError(f"built SPI identifier {spi.identifier!r} != expected {want!r} "
                             f"(entry_source={entry.get('entry_source')})")
    return spi


def timed_call(h, entry, data):
    """``(wall_s, cpu_s, result_status)`` of one seeded ``compute_spi`` call."""
    spi = build_checked(h, entry)
    gc.collect()
    h.seed_all(SPI_SEED)
    c0, t0 = time.process_time(), time.perf_counter()
    S = h.compute_spi(spi, data)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return wall, cpu, h.status_of(S)


def timing_stats(walls, cpus=()):
    if not walls:
        return dict(wall_s=None, wall_median_s=None, wall_cv=None, walls_s=[])
    mean = statistics.fmean(walls)
    cv = statistics.pstdev(walls) / mean if len(walls) > 1 and mean > 0 else None
    out = dict(wall_s=min(walls), wall_median_s=statistics.median(walls), wall_cv=cv,
               walls_s=list(walls))
    if cpus:
        out.update(cpu_s=min(cpus), cpu_over_wall=sum(cpus) / sum(walls) if sum(walls) else None)
    return out


def noisy(walls, target, small):
    """Extend repeats to NOISY_MAX_REPEATS? (>= 3 repeats done at P <= SMALL_P, cv > NOISY_CV)."""
    if not small or len(walls) < 3 or len(walls) != target or target >= NOISY_MAX_REPEATS:
        return False
    cv = timing_stats(walls)["wall_cv"]
    return cv is not None and cv > NOISY_CV


def error_fields(exc):
    if exc is None:
        return dict(status="ok", error=None)
    tb = traceback.format_exception(exc)
    return dict(status="error", error=repr(exc), traceback_tail="".join(tb[-3:])[-2000:])


def run_fresh(h, spec, unit, make, n_rep, small):
    entry = spec["entries"][unit["representative"]]
    walls, cpus, statuses, exc = [], [], [], None
    target, extended = n_rep, False
    rss = RssWindow()
    try:
        while len(walls) < target:
            w, c, st = timed_call(h, entry, make())
            walls.append(w)
            cpus.append(c)
            statuses.append(st)
            if w > REPEAT_CUTOFF_S:       # a timing over 30 s is not repeated
                break
            if noisy(walls, target, small):
                target, extended = NOISY_MAX_REPEATS, True
    except Exception as e:
        exc = e
    return dict(identifiers=[unit["representative"]], repeats=len(walls), repeats_planned=n_rep,
                repeats_extended=extended, result_status=worst_status(statuses),
                result_statuses=statuses, **timing_stats(walls, cpus), **rss.stop(),
                **error_fields(exc))


def unit_walls(runs, mine):
    """Per pass, the sum of ``mine``'s increments (passes where all of them were timed)."""
    return [sum(r[i] for i in mine) for r in runs if mine and all(r[i] is not None for i in mine)]


def run_family(h, spec, make, n_rep, small, unit_id_sets):
    """Run the family's identifiers in family order on one Data per pass.

    ``runs[k][id]`` is the increment of ``id`` in pass k, ``None`` if it raised
    or was not re-timed (an earlier timing exceeded REPEAT_CUTOFF_S). Passes are
    extended to NOISY_MAX_REPEATS when any unit in ``unit_id_sets`` is noisy.
    """
    ids = spec["family_identifiers"]
    runs, errors, statuses, not_retimed = [], {}, {}, {}
    target, extended = n_rep, False
    rss = RssWindow()
    while len(runs) < target and any(i not in not_retimed for i in ids):
        data, inc = make(), {}
        for ident in ids:
            inc[ident] = None
            if ident in not_retimed:
                continue
            try:
                w, _c, st = timed_call(h, spec["entries"][ident], data)
            except Exception as exc:
                errors.setdefault(ident, repr(exc))
                continue
            inc[ident] = w
            statuses.setdefault(ident, []).append(st)
            if w > REPEAT_CUTOFF_S:
                not_retimed[ident] = len(runs)        # 0-based pass of the slow timing
        runs.append(inc)
        if any(noisy(unit_walls(runs, mine), target, small) for mine in unit_id_sets):
            target, extended = NOISY_MAX_REPEATS, True
    return dict(runs=runs, errors=errors, statuses=statuses, not_retimed=not_retimed,
                extended=extended, rss=rss.stop())


def family_record(unit, ids, fam, n_rep, cold):
    runs, errors = fam["runs"], fam["errors"]
    mine = [i for i in ids if i in set(unit["identifiers"])]
    per_id = {}
    for i in ids:
        vals = [r[i] for r in runs if r[i] is not None]
        c = cold.get(i, {})
        per_id[i] = dict(min_s=min(vals) if vals else None,
                         median_s=statistics.median(vals) if vals else None,
                         cv=timing_stats(vals)["wall_cv"], repeats=len(vals),
                         result_status=worst_status(fam["statuses"].get(i, [])),
                         t_cold_s=c.get("t_cold_s"), cold_result_status=c.get("result_status"),
                         warmup_error=c.get("error"), not_retimed=i in fam["not_retimed"],
                         error=errors.get(i))
    walls = unit_walls(runs, mine)
    complete = [r for r in runs if all(r[i] is not None for i in ids if i not in errors)]
    totals = [sum(v for v in r.values() if v is not None) for r in complete]
    errs = {i: errors[i] for i in mine if i in errors}
    rec = dict(identifiers=mine, family_identifiers=ids, increments_s=per_id,
               first_identifier=ids[0], first_identifier_increment_s=per_id[ids[0]]["min_s"],
               unit_first_identifier=mine[0] if mine else None,
               unit_first_increment_s=per_id[mine[0]]["min_s"] if mine else None,
               family_total_s=min(totals) if totals else None,
               repeats=len(walls), repeats_planned=n_rep, repeats_extended=fam["extended"],
               not_retimed={i: k for i, k in fam["not_retimed"].items()},
               result_status=worst_status(s for i in mine for s in fam["statuses"].get(i, [])),
               peak_rss_scope="family", **timing_stats(walls), **fam["rss"])
    if errs:
        rec.update(status="error", error="; ".join(f"{k}: {v}" for k, v in errs.items()))
    else:
        rec.update(status="ok", error=None)
    return rec


def worker_main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    os.environ["OMP_NUM_THREADS"] = "1"
    from regression import harness as h
    jvm = bool(h.ensure_jvm()) if spec["jvm"] else False
    h.import_statistics_modules(sorted({e["module"].rsplit(".", 1)[-1]
                                        for e in spec["entries"].values()}))
    shape, out = spec["shape"], Path(spec["out"])
    raw, make = data_factory(h, shape)
    sha = sha256_array(raw)
    small = shape["P"] <= SMALL_P
    n_rep = spec["repeats"] if small else 1
    sources = {i: e.get("entry_source") for i, e in spec["entries"].items()}

    # Warm-up: one untimed call of every identifier this worker times (all of the
    # family's for a family-warm job), each on a fresh 8x1024 Data -> t_cold_s.
    _, make_warm = (raw, make) if shape["name"] == WARMUP_SHAPE else data_factory(
        h, parse_shape(WARMUP_SHAPE))
    cold = {}
    for ident, entry in spec["entries"].items():
        try:
            w, _c, st = timed_call(h, entry, make_warm())
            cold[ident] = dict(t_cold_s=w, result_status=st, error=None)
        except Exception as exc:
            cold[ident] = dict(t_cold_s=None, result_status=None, error=repr(exc))

    env, commit, dirty = h.environment_info(), git_commit(), git_pyspi_dirty()

    def base(unit, mode, started):
        rec = record_skeleton(unit, shape, mode, sources)
        warm_errs = {i: cold[i]["error"] for i in rec["identifiers"]
                     if cold.get(i, {}).get("error")}
        rec.update(data_sha256=sha, t_cold_s=cold.get(unit["representative"], {}).get("t_cold_s"),
                   warmup_error="; ".join(f"{k}: {v}" for k, v in warm_errs.items()) or None,
                   measure_mode="measured", budget_s=spec["budget_s"],
                   timeout_s=spec.get("timeout_s"), jvm_requested=spec["jvm"], jvm_started=jvm,
                   threads=thread_info(), env=env, git_commit=commit, pyspi_dirty=dirty,
                   started_at=started, finished_at=now())
        return rec

    def finish(rec):
        npairs = rec["n_pairs"]
        # family-warm: wall_s is the unit's summed increments, so this is that sum / n_pairs
        rec["t_per_pair_ms"] = (rec["wall_s"] * 1e3 / npairs
                                if npairs and rec.get("wall_s") is not None else None)
        if rec["status"] != "ok":
            rec["measure_mode"] = "error"
        rec["finished_at"] = now()
        write_json(record_path(out, rec["unit_id"], shape["name"], rec["mode"]), rec)

    if spec["task"] == "profile":
        unit = spec["units"][0]
        started = now()
        rec = base(unit, "profile", started)
        rec.update(profile_one(h, spec, unit, make, out, shape))
        if rec["status"] != "ok":
            rec["measure_mode"] = "error"
        rec["finished_at"] = now()
        write_json(out / f"{unit['unit_id']}__{shape['name']}__profile.json", rec)
        return 0

    by_id = {u["unit_id"]: u for u in spec["units"]}
    for uid in spec["fresh_units"]:
        started = now()
        res = run_fresh(h, spec, by_id[uid], make, n_rep, small)
        finish({**base(by_id[uid], "fresh", started), **res})
    if spec["family_warm"]:
        started = now()
        fids = spec["family_identifiers"]
        mine_sets = [[i for i in fids if i in set(by_id[uid]["identifiers"])]
                     for uid in spec["family_warm_write"]]
        fam = run_family(h, spec, make, n_rep, small, mine_sets)
        for unit in spec["units"]:
            if unit["unit_id"] in spec["family_warm_write"]:
                finish({**base(unit, "family-warm", started),
                        **family_record(unit, fids, fam, n_rep, cold)})
    return 0


# ---- Profiling -------------------------------------------------------------
TESTS_DIR = str(REPO / "tests")


def owner_of(file, pkg_dir):
    if file == "~":
        return "builtin"
    if file.startswith(pkg_dir + "/"):
        return "pyspi"
    if file.startswith(TESTS_DIR + "/"):
        return "harness"
    if "/site-packages/" in file:
        top = file.split("/site-packages/", 1)[1].split("/", 1)[0]
        return "third-party:" + top.split(".")[0]
    return "python"


def short_path(file, pkg_dir):
    if file.startswith(pkg_dir + "/"):
        return "pyspi/" + file[len(pkg_dir) + 1:]
    if "/site-packages/" in file:
        return file.split("/site-packages/", 1)[1]
    return file


def analyse_profile(prof, pkg_dir):
    import pstats
    st = pstats.Stats(prof)
    total = st.total_tt or float("nan")
    rows, opaque, opaque_by, self_by = [], 0.0, {}, {}
    for (file, line, func), (_cc, nc, tt, ct, callers) in st.stats.items():
        raw_owner = owner_of(file, pkg_dir)
        owner = raw_owner
        if file == "~" and callers:
            top_caller = max(callers, key=lambda k: callers[k][3])
            owner = owner_of(top_caller[0], pkg_dir)
        transparent = file != "~"           # every Python-code frame (.py, frozen, <string>)
        if not transparent:
            opaque += tt
            opaque_by[owner] = opaque_by.get(owner, 0.0) + tt
        self_by[owner] = self_by.get(owner, 0.0) + tt
        if "_lsprof.Profiler" not in func:
            rows.append(dict(function=f"{short_path(file, pkg_dir)}:{func}", line=line, ncalls=nc,
                             tottime_s=tt, cumtime_s=ct, owner=owner, opaque=not transparent))
    rows.sort(key=lambda r: -r["cumtime_s"])
    frac = lambda d: {k: v / total for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
    return dict(total_tt_s=st.total_tt, top10_cumulative=rows[:10], opaque_share=opaque / total,
                opaque_share_by_owner=frac(opaque_by), self_time_share_by_owner=frac(self_by),
                opaque_heuristic="tottime of builtin entries (file '~'); every Python frame "
                                 "is transparent; tests/ frames are owner 'harness'; "
                                 "builtins attributed to their dominant caller's owner; "
                                 "numba/JPype/C-type calls hidden in caller tottime (lower bound)")


def profile_one(h, spec, unit, make, out, shape):
    import cProfile
    entry = spec["entries"][unit["representative"]]
    data, spi = make(), build_checked(h, entry)
    prof = cProfile.Profile()
    h.seed_all(SPI_SEED)
    rss, exc, status = RssWindow(), None, None
    t0 = time.perf_counter()
    try:
        S = prof.runcall(h.compute_spi, spi, data)
    except Exception as e:
        exc = e
    wall = time.perf_counter() - t0
    if exc is None:
        status = h.status_of(S)
    pstats_path = out / f"{unit['unit_id']}__{shape['name']}.pstats"
    prof.dump_stats(pstats_path)
    return dict(identifiers=[unit["representative"]], wall_profiled_s=wall, repeats=1,
                result_status=status, pstats=str(pstats_path), **rss.stop(), **error_fields(exc),
                **analyse_profile(prof, str(h.PACKAGE_DIR)))


# ---- Parent side: catalogue, units, orchestration --------------------------
def kill_group(proc):
    """SIGKILL a worker's whole process group (it was started in its own session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def record_path(out, unit_id, shape_name, mode):
    return Path(out) / f"{unit_id}__{shape_name}__{mode}.json"


def dump_catalogue(path):
    from regression import harness as h
    cat = {"config": h.catalogue()[0], "extra": h.catalogue(EXTRA_CONFIG)[0]}
    write_json(path, cat)
    return 0


def load_catalogue(tmpdir, out):
    path = Path(tmpdir) / "catalogue.json"
    with open(out / "_logs" / "catalogue.log", "w") as fh:
        rc = subprocess.run([sys.executable, __file__, "--dump-catalogue", str(path)], cwd=REPO,
                            stdout=fh, stderr=subprocess.STDOUT,
                            env=dict(os.environ, OMP_NUM_THREADS="1")).returncode
    if rc != 0:
        sys.exit(f"catalogue dump failed (rc={rc}); see {out / '_logs' / 'catalogue.log'}")
    cat = {}
    for src, entries in read_json(path).items():
        cat[src] = {}
        for i, e in enumerate(entries):     # last wins on duplicates, like Calculator
            cat[src][e["identifier"]] = {**e, "order": (0 if src == "config" else 1, i)}
    return cat


def demo_units(cat):
    c = cat["config"]

    def ids(pred, limit):
        return [i for i, e in sorted(c.items(), key=lambda kv: kv[1]["order"]) if pred(i, e)][:limit]
    specs = [
        ("demo.cov", "basic", "Covariance", "demo_basic", "multivariate",
         ids(lambda i, e: e["cls"] == "Covariance" and "EmpiricalCovariance" in i, 4),
         "cov_EmpiricalCovariance"),
        ("demo.spearmanr", "basic", "SpearmanR", "demo_basic", "unordered",
         ids(lambda i, e: e["cls"] == "SpearmanR", 4), "spearmanr"),
        ("demo.cohmag", "spectral", "CoherenceMagnitude", "demo_spectral", "multivariate",
         ids(lambda i, e: e["cls"] == "CoherenceMagnitude" and "_mean_" in i, 3),
         "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5"),
    ]
    return [dict(unit_id=u, module=m, pyspi_class=cls, method_or_estimator=None,
                 hyperparameters={}, source="config", identifiers=i, representative=rep,
                 family=fam, in_eeg_selection=True, pair_basis=pb)
            for u, m, cls, fam, pb, i, rep in specs]


def load_units(path):
    import yaml
    doc = yaml.safe_load(Path(path).read_text())
    if doc.get("schema") != 1:
        sys.exit(f"{path}: unsupported schema {doc.get('schema')!r}")
    return [u for u in doc["units"] if u.get("source") != "infrastructure" and u.get("identifiers")]


def check_units(units, cat):
    """Resolve every identifier to a build entry: ``(units, {unit_id: {identifier: entry}})``.

    Preference: the unit's own catalogue, then any catalogue, then the unit's
    own ``identifier_params`` + ``pyspi_class`` + ``module`` from units.yaml
    (``entry_source: "units.yaml"``; the worker asserts ``spi.identifier``).
    Catalogue entries keep catalogue order; units.yaml-built ones get order
    ``(2, k)`` so they follow the catalogue identifiers within a family. A unit
    that cannot be built at all is dropped with a warning, not fatal.
    """
    kept, resolved, k = [], {}, 0
    for u in units:
        if u["representative"] not in u["identifiers"]:
            sys.exit(f"{u['unit_id']}: representative {u['representative']} not in identifiers")
        src, ents, built, other, bad = u.get("source"), {}, [], [], []
        for ident in u["identifiers"]:
            hit = next((c for c in [src, *cat] if c in cat and ident in cat[c]), None)
            if hit is not None:
                ents[ident] = {**cat[hit][ident], "entry_source": hit}
                if hit != src:
                    other.append(f"{ident} ({hit})")
                continue
            params = (u.get("identifier_params") or {}).get(ident)
            if params is None or not u.get("pyspi_class") or not u.get("module"):
                bad.append(ident)
                continue
            ents[ident] = dict(identifier=ident, module=f".statistics.{u['module']}",
                               cls=u["pyspi_class"], params=json.loads(json.dumps(params)),
                               labels=list(u.get("labels") or []), deps=list(u.get("deps") or []),
                               entry_source="units.yaml", order=(2, k))
            k += 1
            built.append(ident)
        if bad:
            log(f"WARNING {u['unit_id']}: {bad} are in no catalogue and units.yaml lacks "
                f"identifier_params/pyspi_class/module to build them; unit dropped")
            continue
        if other:
            log(f"{u['unit_id']} (source {src}): taken from another catalogue: {other}")
        if built:
            log(f"{u['unit_id']} (source {src}): {built} in no catalogue; built from units.yaml "
                f"identifier_params with {u['pyspi_class']} from .statistics.{u['module']} "
                f"(identifier asserted in the worker; ordered after catalogue identifiers)")
        kept.append(u)
        resolved[u["unit_id"]] = ents
    return kept, resolved


def matches(unit_id, patterns):
    return any(fnmatch.fnmatchcase(unit_id, p) for p in patterns)


def split_list(values):
    return [x for v in (values or []) for x in v.split(",") if x]


class Runner:
    def __init__(self, args, units, entries, tmpdir):
        self.args, self.ent, self.tmp, self.out = args, entries, Path(tmpdir), Path(args.out)
        self.units = units
        self.families = {}
        for u in units:
            self.families.setdefault(u["family"], []).append(u)
        self.infeasible = split_list(args.infeasible)
        self.commit = git_commit()

    # -- helpers ----------------------------------------------------------- #
    def entry(self, unit, ident):
        return self.ent[unit["unit_id"]][ident]

    def sources(self, unit):
        return {i: e.get("entry_source") for i, e in self.ent[unit["unit_id"]].items()}

    def family_ids(self, fam):
        ids = {}
        for u in self.families[fam]:
            for i in u["identifiers"]:
                ids[i] = self.entry(u, i)["order"]
        return sorted(ids, key=ids.get)

    def entries_for(self, units, idents):
        src = {i: u for u in units for i in u["identifiers"]}
        return {i: {k: v for k, v in self.entry(src[i], i).items() if k != "order"}
                for i in idents}

    def needs_jvm(self, entries):
        return self.args.jvm == "always" or any(e["module"].endswith("infotheory")
                                                for e in entries.values())

    def need(self, unit, shape, mode):
        return self.args.force or not record_path(self.out, unit["unit_id"], shape["name"],
                                                  mode).exists()

    def rec(self, unit, shape_name, mode):
        return read_json(record_path(self.out, unit["unit_id"], shape_name, mode))

    def log_record(self, rec):
        wall = rec.get("wall_s") if rec.get("wall_s") is not None else rec.get("wall_profiled_s")
        wall = f"{wall:.4f}s" if isinstance(wall, (int, float)) else "-"
        extra = rec.get("error") or rec.get("reason") or ""
        log(f"{rec['unit_id']:<32} {rec['shape']:<11} {rec['mode']:<12} {rec['status']:<15} "
            f"wall={wall} {str(extra)[:120]}")

    def skip_record(self, unit, shape, mode, status, reason, fit=None):
        """``status`` is ``infeasible`` | ``skipped-budget``; ``measure_mode`` equals it and the
        projection that gated the run (if any) is the nested ``fit`` object."""
        if not self.need(unit, shape, mode):
            return
        stamp = now()
        rec = record_skeleton(unit, shape, mode, self.sources(unit), status=status, reason=reason,
                              measure_mode=status, fit=fit, budget_s=self.args.budget_s,
                              git_commit=self.commit, started_at=stamp, finished_at=stamp)
        write_json(record_path(self.out, unit["unit_id"], shape["name"], mode), rec)
        self.log_record(rec)

    # -- gating ------------------------------------------------------------ #
    def over_budget_before(self, units, shape):
        """Name of a smaller-shape record whose timed call exceeded the budget, if any."""
        for u in units:
            for path in self.out.glob(f"{u['unit_id']}__*__*.json"):
                r = read_json(path)
                if not r or r.get("mode") not in MODES or "P" not in r:
                    continue
                if shape_key(r) >= shape_key(shape):
                    continue
                # single timed calls only: fresh wall, or this unit's own increments
                times = ([v.get("min_s") for i, v in (r.get("increments_s") or {}).items()
                          if i in u["identifiers"]] if r["mode"] == "family-warm"
                         else [r.get("wall_s")])
                if r.get("status") in ("skipped-budget", "timeout") or any(
                        t is not None and t > self.args.budget_s for t in times):
                    return path.name
        return None

    def t_exponent(self, unit, mode):
        r256, r1024 = self.rec(unit, "8x256", mode), self.rec(unit, "8x1024", mode)
        if r256 and r1024 and r256.get("wall_s") and r1024.get("wall_s"):
            return math.log(r1024["wall_s"] / r256["wall_s"]) / math.log(4)
        return None

    def gate(self, units, shape, mode):
        """``None`` to run, else ``(status, reason, extra)`` applied to every unit."""
        bad = [u["unit_id"] for u in units if matches(u["unit_id"], self.infeasible)]
        if bad and above_8x1024(shape):
            return "infeasible", f"--infeasible: {bad}", {}
        prior = self.over_budget_before(units, shape)
        if prior:
            return "skipped-budget", f"over budget at smaller shape ({prior})", {}
        if shape["P"] <= SMALL_P:
            return None
        projections = {}
        for u in units:
            r8, r32 = self.rec(u, "8x1024", mode), self.rec(u, "32x1024", mode)
            if not (r8 and r32 and r8.get("status") == "ok" == r32.get("status")):
                return "skipped-budget", "no ok 8x1024 and 32x1024 records to project from", {}
            t_exp = self.t_exponent(u, mode)
            if mode == "fresh":
                f = fit_scaling(r8["wall_s"], r32["wall_s"], u["pair_basis"])
                projections[u["unit_id"]] = projection_fit(f, shape["P"], u["pair_basis"],
                                                           shape["T"], t_exp)
            else:                       # family-warm: project every identifier's increment
                for i in u["identifiers"]:
                    v, w = r8["increments_s"].get(i, {}), r32["increments_s"].get(i, {})
                    if v.get("min_s") and w.get("min_s"):
                        f = fit_scaling(v["min_s"], w["min_s"], u["pair_basis"])
                        projections[i] = projection_fit(f, shape["P"], u["pair_basis"],
                                                        shape["T"], t_exp)
        worst = max((p["projected_wall_s"] for p in projections.values() if p), default=None)
        if worst is not None and worst <= self.args.budget_s:
            return None
        return ("skipped-budget", f"projected {worst:.3g}s > budget {self.args.budget_s}s"
                if worst is not None else "no projection possible",
                dict(projections=projections))

    # -- launching --------------------------------------------------------- #
    def timeout_for(self, spec):
        """``budget_s * n_calls + TIMEOUT_SLACK_S``: warm-up calls plus the most timed repeats."""
        reps = 1 if spec["shape"]["P"] > SMALL_P else max(self.args.repeats, NOISY_MAX_REPEATS)
        calls = len(spec["entries"])                                  # warm-up
        if spec["task"] == "profile":
            calls += 1
        else:
            calls += reps * (len(spec["fresh_units"])
                             + (len(spec["family_identifiers"]) if spec["family_warm"] else 0))
        return self.args.budget_s * calls + TIMEOUT_SLACK_S

    def launch(self, tag, spec, expected):
        spec["timeout_s"] = timeout = self.timeout_for(spec)
        spec_path = self.tmp / f"{tag}.json"
        spec_path.write_text(json.dumps(spec))
        log_path = self.out / "_logs" / f"{tag}.log"
        t0, started = time.time(), now()
        log(f"launch {tag} (timeout {timeout:.0f}s)")
        timed_out = False
        with open(log_path, "w") as fh:
            proc = subprocess.Popen([sys.executable, __file__, "--worker", str(spec_path)],
                                    cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS="1"),
                                    start_new_session=True)          # own process group
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(proc)
                rc = proc.returncode
                log(f"{tag}: worker timed out after {timeout:.0f}s; process group killed")
            except BaseException:
                kill_group(proc)
                raise
        for unit, mode, path in expected:
            rec = read_json(path)
            if rec is None or path.stat().st_mtime < t0 - 1:
                tail = log_path.read_text()[-1500:]
                if timed_out:
                    known = dict(status="timeout", measure_mode="timeout",
                                 error=f"worker killed after the {timeout:.0f}s timeout "
                                       f"before writing this record")
                else:
                    known = dict(status="error", measure_mode="error",
                                 error=f"worker exited rc={rc} without writing this record")
                rec = record_skeleton(unit, spec["shape"], mode, self.sources(unit), **known,
                                      timeout_s=timeout, budget_s=self.args.budget_s,
                                      jvm_requested=spec["jvm"], git_commit=self.commit,
                                      started_at=started, finished_at=now(),
                                      log=str(log_path), log_tail=tail)
                write_json(path, rec)
            self.log_record(rec)

    def worker_spec(self, task, shape, units, idents):
        entries = self.entries_for(units, idents)
        return dict(task=task, shape=shape, units=units, entries=entries, out=str(self.out),
                    repeats=self.args.repeats, budget_s=self.args.budget_s,
                    jvm=self.needs_jvm(entries))

    def run_timing(self, shape, fam, selected):
        fam_units = self.families[fam]
        fresh = []
        if "fresh" in self.args.modes:
            for u in selected:
                if not self.need(u, shape, "fresh"):
                    continue
                g = self.gate([u], shape, "fresh")
                if g:
                    self.skip_record(u, shape, "fresh", g[0], g[1],
                                     fit=g[2].get("projections", {}).get(u["unit_id"]))
                else:
                    fresh.append(u)
        fw_write = []
        if "family-warm" in self.args.modes and any(self.need(u, shape, "family-warm")
                                                    for u in selected):
            g = self.gate(fam_units, shape, "family-warm")
            if g:
                for u in fam_units:
                    mine = {k: v for k, v in g[2].get("projections", {}).items()
                            if k in u["identifiers"]}
                    fit = None
                    if mine:
                        whole = set(mine) == set(u["identifiers"]) and all(mine.values())
                        modes = {v["measure_mode"] for v in mine.values() if v}
                        fit = dict(model="sum-of-identifier-increment-fits",
                                   measure_mode=(modes.pop() if len(modes) == 1
                                                 else "mixed" if modes else None),
                                   projected_wall_s=(sum(v["projected_wall_s"] for v in mine.values())
                                                     if whole else None),
                                   per_identifier=mine)
                    self.skip_record(u, shape, "family-warm", g[0], g[1], fit=fit)
            else:
                fw_write = [u for u in fam_units if self.need(u, shape, "family-warm")]
        fids = self.family_ids(fam)
        jobs = []                    # (tag, units-in-process, fresh ids, family-warm?)
        if shape["P"] <= SMALL_P:
            if fresh or fw_write:
                jobs.append((f"time__{fam}__{shape['name']}", fam_units, fresh, bool(fw_write)))
        else:
            jobs += [(f"time__{u['unit_id']}__{shape['name']}", [u], [u], False) for u in fresh]
            if fw_write:
                jobs.append((f"time__{fam}__{shape['name']}__family-warm", fam_units, [], True))
        for tag, units, fr, fw in jobs:
            warm_units = list({u["unit_id"]: u for u in (fr + (fam_units if fw else []))}.values())
            idents = fids if fw else [u["representative"] for u in fr]
            spec = self.worker_spec("time", shape, warm_units, idents)
            spec.update(fresh_units=[u["unit_id"] for u in fr], family_warm=fw,
                        family_identifiers=fids, family_warm_write=[u["unit_id"] for u in fw_write])
            expected = [(u, "fresh", record_path(self.out, u["unit_id"], shape["name"], "fresh"))
                        for u in fr]
            if fw:
                expected += [(u, "family-warm", record_path(self.out, u["unit_id"], shape["name"],
                                                            "family-warm")) for u in fw_write]
            self.launch(tag, spec, expected)

    def run_profile(self, shape, unit):
        path = self.out / f"{unit['unit_id']}__{shape['name']}__profile.json"
        if path.exists() and not self.args.force:
            return
        timing = self.rec(unit, shape["name"], "fresh")
        if (timing and timing.get("status") != "ok") or (shape["P"] > SMALL_P and not timing):
            log(f"{unit['unit_id']:<32} {shape['name']:<11} profile      skipped "
                f"(fresh record status {timing and timing.get('status')})")
            return
        spec = self.worker_spec("profile", shape, [unit], [unit["representative"]])
        self.launch(f"profile__{unit['unit_id']}__{shape['name']}", spec,
                    [(unit, "profile", path)])

    def run(self, selected, shapes):
        sel_ids = {u["unit_id"] for u in selected}
        for shape in shapes:
            for fam, fam_units in self.families.items():
                chosen = [u for u in fam_units if u["unit_id"] in sel_ids]
                if not chosen:
                    continue
                if self.args.profile:
                    for u in chosen:
                        self.run_profile(shape, u)
                else:
                    self.run_timing(shape, fam, chosen)


# ---- Summary ---------------------------------------------------------------
def summarise(out):
    meta = read_json(GOLDEN_META) or {}
    golden = {k: v.get("time_s") for k, v in meta.get("spis", {}).items()}
    recs = {}
    for path in sorted(Path(out).glob("*__*__*.json")):
        r = read_json(path)
        if r and r.get("mode") in MODES:
            recs.setdefault(r["unit_id"], {}).setdefault(r["mode"], {})[r["shape"]] = r
    ok = lambda r: r is not None and r.get("status") == "ok" and r.get("wall_s") is not None
    units, factors, run_env = {}, [], None
    for uid, modes in sorted(recs.items()):
        any_rec = next(iter(next(iter(modes.values())).values()))
        basis = any_rec["pair_basis"]
        entry = dict(family=any_rec.get("family"), pair_basis=basis,
                     representative=any_rec.get("representative"), modes={})
        for mode, by_shape in modes.items():
            g = lambda s: by_shape.get(s)
            r7, r256, r8, r32, r118 = g("7x100"), g("8x256"), g("8x1024"), g("32x1024"), g("118x1024")
            run_env = run_env or next((v.get("env") for v in by_shape.values() if v.get("env")),
                                      None)
            s = dict(measured={k: v.get("wall_s") for k, v in sorted(by_shape.items())},
                     statuses={k: v.get("status") for k, v in sorted(by_shape.items())},
                     measure_modes={k: v.get("measure_mode") for k, v in sorted(by_shape.items())},
                     result_statuses={k: v.get("result_status") for k, v in sorted(by_shape.items())},
                     wall_cv={k: v.get("wall_cv") for k, v in sorted(by_shape.items())},
                     repeats={k: v.get("repeats") for k, v in sorted(by_shape.items())})
            fit = fit_scaling(r8["wall_s"], r32["wall_s"], basis) if ok(r8) and ok(r32) else None
            s["fit"] = fit
            s["P_exponent"] = fit and fit.get("P_exponent")
            s["T_exponent"] = (math.log(r8["wall_s"] / r256["wall_s"]) / math.log(4)
                               if ok(r8) and ok(r256) and r256["wall_s"] > 0 else None)
            s["reason"] = None
            if ok(r118):
                s["projected_window_s_118"], s["measure_mode"] = r118["wall_s"], "measured"
            elif fit:
                s["projected_window_s_118"], s["measure_mode"] = project(
                    fit, 118, basis, 1024, s["T_exponent"])
            elif ok(r8) and n_pairs(8, PAIR_LOOP_BASIS.get(uid, basis)):
                # Units never run above 8x1024 (infeasible list): scale the 8x1024 wall by the
                # pair count alone (t_fixed assumed 0). Accurate when the per-pair cost
                # dominates; an overestimate by up to t_fixed * (n118/n8 - 1) otherwise.
                pb = PAIR_LOOP_BASIS.get(uid, basis)
                s["projected_window_s_118"] = r8["wall_s"] * n_pairs(118, pb) / n_pairs(8, pb)
                s["measure_mode"] = "extrapolated-pairs-from-8"
                s["reason"] = ("pairs-only scaling from the 8x1024 wall (no 32x1024 record); "
                               "t_fixed assumed 0")
                if pb != basis:
                    s["reason"] += (f"; {pb} pair count of the delegate's pair loop "
                                    f"(pair_basis {basis}, PAIR_LOOP_BASIS)")
            else:
                s["projected_window_s_118"], s["measure_mode"] = None, None
                s["reason"] = ("no 118x1024 measurement and no ok 8x1024 + 32x1024 records to "
                               "fit (no extrapolation from 8x1024 alone)")
            p = s["projected_window_s_118"]
            s["projected_patient_s"] = p * WINDOWS_PER_PATIENT if p is not None else None
            ids = (r7 or {}).get("identifiers") or [entry["representative"]]
            gold = [golden.get(i) for i in ids]
            s["time_s_7x100_golden"] = sum(gold) if gold and None not in gold else None
            # fresh only: a family-warm wall (summed increments) is not comparable to the
            # goldens' per-SPI time_s
            s["machine_factor"] = (r7["wall_s"] / s["time_s_7x100_golden"]
                                   if mode == "fresh" and ok(r7) and s["time_s_7x100_golden"]
                                   else None)
            if s["machine_factor"]:
                factors.append(s["machine_factor"])
            entry["modes"][mode] = s
        units[uid] = entry
    genv = meta.get("environment") or {}
    renv = run_env or {}
    note = (f"Goldens were generated under Python {genv.get('python')} / NumPy "
            f"{(genv.get('packages') or {}).get('numpy')} ({GOLDEN_META.relative_to(REPO)} "
            f"'environment'); these records come from the current venv (Python "
            f"{renv.get('python', sys.version.split()[0])} / NumPy "
            f"{(renv.get('packages') or {}).get('numpy', np.__version__)}), so machine_factor "
            f"conflates machine speed with software-environment differences. It is computed "
            f"for fresh mode only (family-warm: null).")
    summary = dict(schema=1, generated_at=now(), git_commit=git_commit(), out=str(out),
                   windows_per_patient=WINDOWS_PER_PATIENT, measure_modes=list(MEASURE_MODES),
                   golden_meta=str(GOLDEN_META), golden_generated=meta.get("generated"),
                   golden_environment=meta.get("environment"), golden_environment_note=note,
                   machine_factor_median=statistics.median(factors) if factors else None,
                   units=units)
    write_json(Path(out) / "summary.json", summary)
    log(f"summary: {len(units)} units -> {Path(out) / 'summary.json'}")


# ---- CLI -------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = ap.add_argument_group("selection")
    sel.add_argument("--units", action="append", help="unit ids or globs, comma separated")
    sel.add_argument("--families", action="append", help="family names, comma separated")
    sel.add_argument("--all", action="store_true")
    sel.add_argument("--units-file", default=str(DEFAULT_UNITS))
    sel.add_argument("--demo", action="store_true",
                     help="fabricate a 3-unit catalogue (cov, spearmanr, cohmag)")
    ap.add_argument("--shapes", default="7x100,8x256,8x1024,32x1024,118x1024")
    ap.add_argument("--modes", default="fresh,family-warm")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=120.0)
    ap.add_argument("--infeasible", action="append",
                    help="unit ids/globs never run above 8x1024, comma separated")
    ap.add_argument("--profile", action="store_true", help="cProfile pass instead of timing")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="overwrite existing records")
    ap.add_argument("--jvm", choices=("auto", "always"), default="auto",
                    help="auto: start the JVM only in workers running infotheory SPIs")
    ap.add_argument("--edf", help="EDF file; adds shape edf<P>x<T>")
    ap.add_argument("--edf-channels", type=int, default=118)
    ap.add_argument("--edf-window-s", type=float, default=0.5)
    ap.add_argument("--edf-start-s", type=float, default=60.0)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--dump-catalogue", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.modes = split_list([args.modes])
    if bad := set(args.modes) - set(MODES):
        ap.error(f"unknown modes {bad}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    if args.dump_catalogue:
        return dump_catalogue(args.dump_catalogue)
    out = Path(args.out)
    (out / "_logs").mkdir(parents=True, exist_ok=True)
    want_run = args.all or args.units or args.families
    if want_run:
        if not args.demo and not Path(args.units_file).exists():
            sys.exit(f"{args.units_file} not found (use --demo for a fabricated catalogue)")
        with tempfile.TemporaryDirectory(prefix="profile_units_") as tmp:
            cat = load_catalogue(tmp, out)
            units = demo_units(cat) if args.demo else load_units(args.units_file)
            units, entries = check_units(units, cat)
            fams, pats = set(split_list(args.families)), split_list(args.units)
            selected = [u for u in units if args.all or u["family"] in fams
                        or matches(u["unit_id"], pats)]
            if not selected:
                sys.exit("no units selected")
            shapes = [parse_shape(s) for s in split_list([args.shapes])]
            if args.edf:
                shapes.append(load_edf_window(args.edf, args.edf_channels, args.edf_window_s,
                                              args.edf_start_s, tmp))
            shapes.sort(key=shape_key)
            log(f"{len(selected)} units, shapes {[s['name'] for s in shapes]}, "
                f"{'profile' if args.profile else 'modes ' + ','.join(args.modes)}")
            Runner(args, units, entries, tmp).run(selected, shapes)
    if args.summarise:
        summarise(out)
    if not (want_run or args.summarise):
        sys.exit("nothing to do: pass --units/--families/--all and/or --summarise")
    return 0


if __name__ == "__main__":
    sys.exit(main())

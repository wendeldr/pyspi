"""Probes for the spectral_psi_gd family (PROJECT venv; called by pyspi_outputs.py).

Every function has the probe signature ``fn(x, run) -> ndarray``; ``x`` is the raw
dataset (P, T) after the manifest's process selection and ``run`` is the manifest
entry.  Band and class parameters live in ``run["probe"]`` (``cls`` = "psi" | "gd",
``fmin``, ``fmax``) so they are part of the run's cache key.

* ``eeg_band``  constructs ``PhaseSlopeIndex`` / ``GroupDelay`` with an EEG band
  (pyspi/workflow/retained185.yaml) that is not a config identifier, runs it
  through ``harness.compute_spi`` and, like ``Calculator.compute``
  (pyspi/calculator.py), turns an exception into an all-NaN matrix.  The
  exception text (or None) is written to ``out/exchange/spectral_psi_gd/<run>.json``
  for the reference side, which decides the qualitative outcome.
* ``embed``     pair (x, 2 processes) stacked with the 5 iid_gaussian channels of
  the same T; returns the pair's 2x2 block of the 7-channel matrix.
* ``perm``      A(X[perm])[inv, inv] with perm = reversed channel order.
* ``m5``        REVIEW.md M5 example: the library's own ``_inner_combination`` on
  four unit coherencies; returns [Im all-pairs value].
* ``longdelay`` x0 = delayed_noise(T=4096) process 0; x_k(t) = x0(t + 200 - d_k)
  + 0.1 * iid_gaussian row k, d = LONG_DELAYS, t < T (non-circular, T from the
  run); pyspi PSI or GD on that matrix.

Module level imports only numpy so the comparison venv can import
``build_longdelay`` and ``M5_PHASES`` without pyspi.
"""
import json
import sys
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parent.parent
EXCHANGE = CHECKS / "out" / "exchange" / "spectral_psi_gd"
M5_PHASES = [0.6925794593, -0.3522172326, 2.7390760913, 2.2241494312]   # docs/storage_comparison/REVIEW.md:174
LONG_DELAYS = (0, 20, 45, 60, 100)


def _exchange_path(run_id):
    return EXCHANGE / (run_id.replace("|", "__").replace("/", "_") + ".json")


def build_longdelay(base, noise, T, delays=LONG_DELAYS, offset=200, noise_sd=0.1):
    """x_k(t) = base(t + offset - d_k) + noise_sd * noise[k, t], t = 0..T-1."""
    rows = [base[offset - d: offset - d + T] + noise_sd * noise[k, :T] for k, d in enumerate(delays)]
    return np.vstack(rows)


def _make_data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _spi(probe):
    from pyspi.statistics.spectral import GroupDelay, PhaseSlopeIndex
    if probe["cls"] == "psi":
        return PhaseSlopeIndex(fmin=probe["fmin"], fmax=probe["fmax"])
    return GroupDelay(fmin=probe["fmin"], fmax=probe["fmax"], statistic="delay")


def _compute(x, run):
    sys.path.insert(0, str(CHECKS.parent.parent / "tests"))
    from regression import harness
    return harness.compute_spi(_spi(run["probe"]), _make_data(x, run["path"]))


def eeg_band(x, run):
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    spi = _spi(run["probe"])
    rec = {"run_id": run["run_id"], "identifier": spi.identifier, "error": None}
    try:
        S = _compute(x, run)
    except Exception as exc:                       # Calculator.compute: whole matrix NaN
        rec["error"] = repr(exc)
        S = np.full((x.shape[0], x.shape[0]), np.nan)
    rec["all_nan"] = bool(np.isnan(S).all())
    _exchange_path(run["run_id"]).write_text(json.dumps(rec, indent=1) + "\n")
    return S


def embed(x, run):
    sys.path.insert(0, str(CHECKS))
    import make_data
    noise = np.asarray(make_data.load("iid_gaussian", x.shape[1])["x"], dtype=np.float64)
    S = _compute(np.vstack([x, noise]), run)
    return S[:2, :2]


def perm(x, run):
    p = np.arange(x.shape[0])[::-1]
    inv = np.argsort(p)
    S = _compute(x[p], run)
    return S[np.ix_(inv, inv)]


def m5(x, run):
    from pyspi import _spectral_compat  # noqa: F401  (pyspi's NumPy-2 shim, as in normal use)
    from spectral_connectivity.connectivity import _inner_combination
    C = np.exp(1j * np.asarray(M5_PHASES)).reshape(-1, 1, 1)      # (n_freq, 1, 1), axis -3 = frequency
    return np.atleast_1d(np.asarray(_inner_combination(C).imag, dtype=float).ravel()[0])


def longdelay(x, run):
    sys.path.insert(0, str(CHECKS))
    import make_data
    T = int(run["T"])
    base = np.asarray(make_data.load("delayed_noise", 4096)["x"][0], dtype=np.float64)
    noise = np.asarray(make_data.load("iid_gaussian", T)["x"], dtype=np.float64)
    return _compute(build_longdelay(base, noise, T), run)


def shares(x, run):
    """[first result shares memory with data.spectral_mv (0/1),
        first-result mutation visible on the 2nd call (0/1),
        second-result mutation visible on the 3rd call (0/1)].
    The first insert into a fresh Data is keyed by self.measure and later
    lookups use self.key (spectral.py:73-81), so the first result is a view of
    an entry that is never read again; from the second call on, results are
    views of the self.key entry that later calls return."""
    data = _make_data(x, run["path"])
    spi = _spi(run["probe"])
    S1 = spi.multivariate(data)
    arrays = []
    for v in data.spectral_mv.values():
        for a in (v if isinstance(v, tuple) else (v,)):
            if isinstance(a, np.ndarray):
                arrays.append(a)
    share = any(np.shares_memory(S1, a) for a in arrays)
    S1[0, 1] = 12345.0
    S2 = spi.multivariate(data)
    vis2 = S2[0, 1] == 12345.0
    S2[0, 1] = 54321.0
    S3 = spi.multivariate(data)
    return np.array([float(share), float(vis2), float(S3[0, 1] == 54321.0)])


def spy(x, run):
    """Record the keyword arguments of every call pyspi makes to the library
    method; returns [n_calls, first_call_n_kwargs, fmin, fmax, frequency_resolution]
    of the last call (NaN when absent)."""
    import spectral_connectivity as sc
    name = "phase_slope_index" if run["probe"]["cls"] == "psi" else "group_delay"
    orig = getattr(sc.Connectivity, name)
    calls = []

    def wrapper(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return orig(self, *args, **kwargs)

    setattr(sc.Connectivity, name, wrapper)
    try:
        _spi(run["probe"]).multivariate(_make_data(x, run["path"]))
    finally:
        setattr(sc.Connectivity, name, orig)
    last = calls[-1] if calls else {}
    foi = last.get("frequencies_of_interest") or [np.nan, np.nan]
    return np.array([len(calls), len(calls[0]) if calls else -1, foi[0], foi[1],
                     last.get("frequency_resolution", np.nan)], dtype=float)

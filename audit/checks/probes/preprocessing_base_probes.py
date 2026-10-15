"""Probes for the preprocessing_base family (project venv; called by pyspi_outputs.py).

Units: preproc.detrend, preproc.zscore, preproc.two_array_api, base.pair_handling.
These units are infrastructure, so every check needs pyspi intermediates that the plain
SPI path does not store: the array that ``Data`` holds after preprocessing, the array an
SPI receives through the two-array API, call orders, and matrices computed under
different visiting orders. Each probe returns one numpy array (stored as ``value`` by
pyspi_outputs.py). Probes that need to hand non-numeric bookkeeping (identifier lists,
exception texts) to the reference write a JSON sidecar to
``audit/checks/out/exchange/preprocessing_base/<run_id with '|' -> '__'>.json``.

Signature of every probe: ``fn(x, run)`` with x the raw dataset (P, T) after the run's
process selection.
"""
import contextlib
import importlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parents[1]
EXCH = CHECKS / "out" / "exchange" / "preprocessing_base"
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(CHECKS.parents[1] / "tests"))


# --------------------------------------------------------------------------- helpers
def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _sidecar(run, payload):
    EXCH.mkdir(parents=True, exist_ok=True)
    (EXCH / (run["run_id"].replace("|", "__") + ".json")).write_text(json.dumps(payload, indent=1, default=str))


def _units():
    import pyspi_outputs as po
    return po.load_units()


def _entry(identifier):
    import pyspi_outputs as po
    _, by_id = _units()
    return po.entry_for(identifier, by_id)


def _build(identifier):
    from regression import harness
    with _quiet():
        return harness.build_spi(_entry(identifier))


def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    with _quiet():
        if path == "kernel":
            return Data(x, dim_order="ps", normalise=False, detrend=False)
        if path == "pipeline":
            return Data(x, dim_order="ps")
        if path == "pipeline-detrend":
            return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _mv(spi, data):
    with _quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.array(np.real(spi.multivariate(data)), dtype=np.float64, copy=True)
    np.fill_diagonal(S, np.nan)
    return S


def _config_ids_of_class(identifier):
    """Every config identifier whose unit has the same module and pyspi_class, in units.yaml order."""
    doc, by_id = _units()
    u0, _ = by_id[identifier]
    out = []
    for u in doc["units"]:
        if u.get("source") == "config" and u["module"] == u0["module"] and u["pyspi_class"] == u0["pyspi_class"]:
            out.extend(u["identifiers"])
    return out


def _seed(seed=42):
    from regression import harness
    harness.seed_all(seed)


# =========================================================================== preproc.detrend / zscore
def data_array(x, run):
    """``Data(x, ...).to_numpy(squeeze=True)`` for the run's path, with optional truncation.

    run tag 'trunc' (run_id contains '|trunc') builds ``Data(x, detrend=True,
    n_processes=P-1, n_observations=T-7)`` to expose the truncate -> detrend -> z-score order.
    run path 'detrend-only' is encoded by the tag '|detonly': ``Data(x, normalise=False, detrend=True)``.
    """
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    rid = run["run_id"]
    with _quiet():
        if "|trunc" in rid:
            d = Data(x, dim_order="ps", detrend=True, normalise=True,
                     n_processes=x.shape[0] - 1, n_observations=x.shape[1] - 7)
        elif "|detonly" in rid:
            d = Data(x, dim_order="ps", normalise=False, detrend=True)
        else:
            return _data(x, run["path"]).to_numpy(squeeze=True)
    return d.to_numpy(squeeze=True)


def historical_order_spis(x, run):
    """cov_EmpiricalCovariance and spearmanr on the HISTORICAL order detrend(zscore(x, ddof=1))
    (scipy, as the historical patient stores did; EEG_OPTIMIZATION_NOTES.md:42-43) and on the
    CURRENT order Data(x, detrend=True). Returns (2 identifiers, 2 orders [current, historical], P, P)."""
    from scipy.signal import detrend
    from scipy.stats import zscore
    hist = detrend(zscore(np.asarray(x, float), axis=1, ddof=1), axis=1)
    out = []
    for ident in ("cov_EmpiricalCovariance", "spearmanr"):
        cur = _mv(_build(ident), _data(x, "pipeline-detrend"))
        his = _mv(_build(ident), _data(hist, "kernel"))
        out.append([cur, his])
    return np.array(out)


FLAT_REPLACEMENTS = [("const 0.0", lambda t: np.zeros_like(t)),
                     ("const 3.0", lambda t: np.full_like(t, 3.0)),
                     ("const 1e-3", lambda t: np.full_like(t, 1e-3)),
                     ("linear 0.37t+1.1", lambda t: 0.37 * t + 1.1)]


def flat_channel(x, run):
    """Row 1 of x replaced by a constant or an exact ramp. For each replacement returns
    [raised_with_detrend (1/0), std_ddof1 of the stored row (NaN if raised), number of unique
    values of the stored row, lag-1 autocorrelation of the stored row, raised_without_detrend (1/0)].
    Shape (4, 5)."""
    from pyspi.data import Data
    T = x.shape[1]
    t = np.arange(T, dtype=float)
    rows, msgs = [], []
    for name, f in FLAT_REPLACEMENTS:
        y = np.array(x, float, copy=True)
        y[1] = f(t)
        rec = [np.nan] * 5
        try:
            with _quiet():
                d = Data(y, dim_order="ps", detrend=True, normalise=True)
            z = d.to_numpy(squeeze=True)[1]
            rec[0] = 0.0
            rec[1] = float(np.std(z, ddof=1))
            rec[2] = float(len(np.unique(z)))
            zc = z - z.mean()
            rec[3] = float(zc[:-1] @ zc[1:] / (zc @ zc))
            msgs.append([name, "accepted"])
        except Exception as exc:  # expected for some replacements
            rec[0] = 1.0
            msgs.append([name, repr(exc)])
        try:
            with _quiet():
                Data(y, dim_order="ps", detrend=False, normalise=True)
            rec[4] = 0.0
        except Exception:
            rec[4] = 1.0
        rows.append(rec)
    _sidecar(run, {"replacements": [n for n, _ in FLAT_REPLACEMENTS], "messages": msgs})
    return np.array(rows)


def error_paths(x, run):
    """Degenerate-input behaviour of Data.set_data. Returns a vector:
    [0] constant row, detrend=False, normalise=True: raised (1/0)
    [1]   ... message mentions 'NaN' (1/0)
    [2]   ... message mentions 'variance' or 'constant' (1/0)
    [3] NaN sample in row 2, normalise=True: raised (1/0)
    [4] same NaN input, scipy zscore(nan_policy='omit') alone leaves finite values in the other rows (1/0)
    [5] inf sample, normalise=False, detrend=True: accepted (1/0)
    [6]   ... inf still present in the stored array (1/0)
    [7]   ... OLS slope of stored row 0 (non-zero = row was NOT detrended)
    [8] same inf input with normalise=True: raised (1/0)
    """
    from pyspi.data import Data
    from scipy.stats import zscore
    out = np.full(9, np.nan)
    texts = {}
    y = np.array(x, float, copy=True)
    y[1] = 2.5
    try:
        with _quiet():
            Data(y, dim_order="ps")
        out[0] = 0
    except Exception as exc:
        out[0] = 1
        m = str(exc)
        texts["constant"] = repr(exc)
        out[1] = float("NaN" in m)
        out[2] = float(("variance" in m.lower()) or ("constant" in m.lower()))
    y = np.array(x, float, copy=True)
    y[2, 5] = np.nan
    try:
        with _quiet():
            Data(y, dim_order="ps")
        out[3] = 0
    except Exception as exc:
        out[3] = 1
        texts["nan"] = repr(exc)
    zz = zscore(y, axis=1, nan_policy="omit", ddof=1)
    out[4] = float(np.isfinite(np.delete(zz, 2, axis=0)).all())
    y = np.array(x, float, copy=True)
    y[0] = y[0] + 0.05 * np.arange(y.shape[1])
    y[1, 3] = np.inf
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            d = Data(y, dim_order="ps", normalise=False, detrend=True)
        out[5] = 1
        z = d.to_numpy(squeeze=True)
        out[6] = float(np.isinf(z).any())
        t = np.arange(z.shape[1], dtype=float)
        tc = t - t.mean()
        out[7] = float(tc @ (z[0] - z[0].mean()) / (tc @ tc))
    except Exception as exc:
        out[5] = 0
        texts["inf_norm_false"] = repr(exc)
    texts["inf_stdout"] = buf.getvalue()[-300:]
    try:
        with _quiet():
            Data(y, dim_order="ps", normalise=True, detrend=True)
        out[8] = 0
    except Exception as exc:
        out[8] = 1
        texts["inf_norm_true"] = repr(exc)
    _sidecar(run, texts)
    return out


# =========================================================================== preproc.two_array_api
class _Capture:
    """Toy SPI built on pyspi's own parse decorators: stores the array the SPI receives."""

    def __init__(self):
        from pyspi.base import Directed, parse_bivariate, parse_multivariate

        class Cap(Directed):
            identifier = "capture"
            seen = None

            @parse_bivariate
            def bivariate(self, data, i=None, j=None):
                type(self).seen = (data.to_numpy(squeeze=True).copy(), i, j)
                return 0.0

            @parse_multivariate
            def multivariate(self, data):
                type(self).seen = (data.to_numpy(squeeze=True).copy(), None, None)
                return np.zeros((data.n_processes, data.n_processes))

        self.cls = Cap


def capture_bivariate(x, run):
    """Array seen by ``spi.bivariate(x[0], x[1])`` (two raw 1-D arrays). Shape (2, T)."""
    c = _Capture().cls()
    with _quiet():
        c.bivariate(np.array(x[0], float), np.array(x[1], float))
    seen, i, j = type(c).seen
    _sidecar(run, {"i": i, "j": j})
    return seen


def capture_multivariate_list(x, run):
    """Array seen by ``spi.multivariate([x[0], ..., x[P-1]])`` (list of raw 1-D arrays). Shape (P, T)."""
    c = _Capture().cls()
    with _quiet():
        c.multivariate([np.array(r, float) for r in x])
    return type(c).seen[0]


def _zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def two_array_spi(x, run):
    """Values of run['identifier'] through three input paths on raw x[0], x[1]:
    [[bivariate(x0, x1),  Data path A[0, 1],  bivariate(z(x0), z(x1))],
     [bivariate(x1, x0),  Data path A[1, 0],  bivariate(z(x1), z(x0))]]
    For classes that override multivariate (no direct bivariate), the list path
    multivariate([x0, x1]) replaces bivariate. Shape (2, 3)."""
    ident = run["identifier"]
    a, b = np.array(x[0], float), np.array(x[1], float)
    spi = _build(ident)
    A = _mv(_build(ident), _data(np.vstack([a, b]), "pipeline"))
    out = np.full((2, 3), np.nan)
    has_biv = "bivariate" in {k for c in type(spi).__mro__ for k in c.__dict__ if c.__name__ not in ("Directed", "Undirected")}
    with _quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if has_biv:
            out[0] = [_build(ident).bivariate(a, b), A[0, 1], _build(ident).bivariate(_zs(a), _zs(b))]
            out[1] = [_build(ident).bivariate(b, a), A[1, 0], _build(ident).bivariate(_zs(b), _zs(a))]
        else:
            L = np.array(_build(ident).multivariate([a, b]), float)
            Lz = np.array(_build(ident).multivariate([_zs(a), _zs(b)]), float)
            out[0] = [L[0, 1], A[0, 1], Lz[0, 1]]
            out[1] = [L[1, 0], A[1, 0], Lz[1, 0]]
    _sidecar(run, {"identifier": ident, "path_used": "bivariate" if has_biv else "multivariate(list)"})
    return out


def unset_ij(x, run):
    """For every config identifier in the module named by the run_id tag ('|mod-<module>'), call
    ``spi.bivariate(Data(x))`` (P >= 3) with i = j = None. Per identifier returns
    [outcome (0 raised, 1 finite number, 2 non-finite number / array), n_python_warnings,
    n_logging_records (level >= WARNING, captured by a handler on the root logger)]."""
    import logging

    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records = []

        def emit(self, record):
            self.records.append(f"{record.levelname}: {record.getMessage()}"[:160])

    doc, by_id = _units()
    mod = run["run_id"].split("|mod-")[1].split("|")[0]
    ids = [i for u in doc["units"] if u.get("source") == "config" and u["module"] == mod for i in u["identifiers"]]
    out, info = [], []
    for ident in ids:
        spi = _build(ident)
        with _quiet():
            d = _data(x, "pipeline")
        lh = _ListHandler()
        logging.getLogger().addHandler(lh)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                with _quiet():
                    v = spi.bivariate(d)
                v = np.asarray(v, dtype=float) if np.ndim(v) == 0 or np.size(v) else np.asarray(np.nan)
                code = 1.0 if (np.ndim(v) == 0 and np.isfinite(v)) else 2.0
                info.append([ident, "returned", repr(np.asarray(v).ravel()[:3].tolist())])
            except Exception as exc:
                code = 0.0
                info.append([ident, "raised", repr(exc)[:200]])
        logging.getLogger().removeHandler(lh)
        msgs = [str(m.message)[:120] for m in w]
        info[-1].append(msgs[:3])
        info[-1].append(lh.records[:3])
        out.append([code, float(len(w)), float(len(lh.records))])
    _sidecar(run, {"module": mod, "identifiers": ids, "info": info})
    return np.array(out)


# =========================================================================== base.pair_handling
def toy_pairs(x, run):
    """Toy Directed and Undirected subclasses whose bivariate returns 10 i + j and logs calls.
    Returns (3, P, P): [A_directed, A_undirected, C] where C[i, j] is the 0-based position of
    the call bivariate(i, j) in the Undirected loop (NaN if never called)."""
    from pyspi.base import Directed, Undirected, parse_bivariate
    P = x.shape[0]
    calls = []

    class D(Directed):
        @parse_bivariate
        def bivariate(self, data, i=None, j=None):
            return 10.0 * i + j

    class U(Undirected):
        @parse_bivariate
        def bivariate(self, data, i=None, j=None):
            calls.append((i, j))
            return 10.0 * i + j

    d = _data(x, run["path"])
    Ad = _mv(D(), d)
    Au = _mv(U(), d)
    C = np.full((P, P), np.nan)
    for k, (i, j) in enumerate(calls):
        C[i, j] = k
    _sidecar(run, {"undirected_calls": calls, "n_calls": len(calls)})
    return np.array([Ad, Au, C])


def call_order_real(x, run):
    """Call log of run['identifier'] (a Directed-loop or Undirected-mirror SPI) under multivariate:
    a subclass forwards to the original bivariate and records (i, j). Returns (2, P, P):
    [A, C] with C[i, j] the call position."""
    spi = _build(run["identifier"])
    base = type(spi)
    calls = []

    class Logged(base):
        def bivariate(self, data, *a, **k):
            calls.append((k.get("i"), k.get("j")))
            return base.bivariate(self, data, *a, **k)

    spi.__class__ = Logged
    A = _mv(spi, _data(x, run["path"]))
    P = x.shape[0]
    C = np.full((P, P), np.nan)
    for n, (i, j) in enumerate(calls):
        C[i, j] = n
    _sidecar(run, {"identifier": run["identifier"], "calls": calls})
    return np.array([A, C])


def class_matrices(x, run):
    """multivariate of every config identifier in run['identifier']'s class (same module and
    pyspi_class in units.yaml), each on a fresh Data, seeded 42 before each. Shape (n, P, P)."""
    ids = _config_ids_of_class(run["identifier"])
    mats = []
    for ident in ids:
        _seed(42)
        try:
            mats.append(_mv(_build(ident), _data(x, run["path"])))
        except Exception:
            mats.append(np.full((x.shape[0], x.shape[0]), np.nan))
    _sidecar(run, {"identifiers": ids})
    return np.array(mats)


def upper_only(x, run):
    """For every identifier of run['identifier']'s class: [A_loop, A_upper] where A_loop is the
    stored multivariate output (fresh Data) and A_upper is built on a second fresh Data from the
    calls bivariate(i, j) with i < j only (visiting order (0,1), (0,2), ...), mirrored. Shape (n, 2, P, P)."""
    ids = _config_ids_of_class(run["identifier"])
    P = x.shape[0]
    out = []
    for ident in ids:
        _seed(42)
        A = _mv(_build(ident), _data(x, run["path"]))
        _seed(42)
        spi = _build(ident)
        d = _data(x, run["path"])
        B = np.full((P, P), np.nan)
        with _quiet(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(P):
                for j in range(i + 1, P):
                    B[i, j] = B[j, i] = spi.bivariate(d, i=i, j=j)
        out.append([A, B])
    _sidecar(run, {"identifiers": ids})
    return np.array(out)


CACHE_GROUPS = {
    "xcorr": [("basic", "CrossCorrelation")],
    "coint": [("misc", "Cointegration")],
    "covprec": [("basic", "Covariance"), ("basic", "Precision")],
    "spectral_coh": [("spectral", "CoherenceMagnitude"), ("spectral", "CoherencePhase"),
                     ("spectral", "ImaginaryCoherence")],
    "spectral_sync": [("spectral", "PhaseLockingValue"), ("spectral", "PhaseLagIndex"),
                      ("spectral", "WeightedPhaseLagIndex")],
    "barycenter": [("distance", "Barycenter")],
    "ccm": [("causal", "ConvergentCrossMapping")],
    "je_ce_gaussian": [("infotheory", "JointEntropy"), ("infotheory", "ConditionalEntropy"),
                       ("infotheory", "CausalEntropy")],
}


def cache_order(x, run):
    """Variants of one cache-sharing group computed on ONE shared Data (as Calculator does),
    seeded 42 once before the sequence, in config order and in reversed order.
    Group from the run_id tag '|grp-<name>'. Returns (2, n, P, P) = [forward, reversed (re-aligned to
    forward order)]."""
    grp = run["run_id"].split("|grp-")[1].split("|")[0]
    doc, _ = _units()
    ids = [i for (mod, cls) in CACHE_GROUPS[grp] for u in doc["units"]
           if u.get("source") == "config" and u["module"] == mod and u["pyspi_class"] == cls
           for i in u["identifiers"]]
    res = {}
    for order_name, order in (("forward", ids), ("reversed", ids[::-1])):
        d = _data(x, run["path"])
        _seed(42)
        res[order_name] = {}
        for ident in order:
            try:
                res[order_name][ident] = _mv(_build(ident), d)
            except Exception:
                res[order_name][ident] = np.full((x.shape[0], x.shape[0]), np.nan)
    _sidecar(run, {"group": grp, "identifiers": ids})
    return np.array([[res["forward"][i] for i in ids], [res["reversed"][i] for i in ids]])


def rng_sensitivity(x, run):
    """bary_sgddtw_mean on a fresh Data after seed 42, and after seed 42 followed by one
    np.random.random() draw (as if an earlier SPI had consumed one number). Shape (2, P, P)."""
    out = []
    for extra in (0, 1):
        _seed(42)
        if extra:
            np.random.random()
        out.append(_mv(_build(run["identifier"]), _data(x, run["path"])))
    return np.array(out)


def direction(x, run):
    """Direction table entry for run['identifier'] on the run's dataset/path. Returns (4, P, P):
    [A, A - A.T, B, B - B.T] with A = multivariate (stored orientation) and B[i, j] = an isolated
    bivariate(i, j) on a fresh Data (NaN when the class has no pair-level bivariate)."""
    ident = run["identifier"]
    _seed(42)
    A = _mv(_build(ident), _data(x, run["path"]))
    P = x.shape[0]
    B = np.full((P, P), np.nan)
    spi = _build(ident)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            _seed(42)
            try:
                with _quiet(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    B[i, j] = float(np.real(_build(ident).bivariate(_data(x, run["path"]), i=i, j=j)))
            except Exception:
                B[i, j] = np.nan
    return np.array([A, A - A.T, B, B - B.T])


def univariate_raw(x, run):
    """A toy SPI method decorated with pyspi's parse_univariate, called with a raw 1-D array x[0].
    Returns [raised (1/0), returned_value_or_NaN]; the exception text goes to the sidecar."""
    from pyspi.base import parse_univariate

    class U:
        @parse_univariate
        def f(self, data, i=None):
            return float(np.mean(data.to_numpy(squeeze=True)[i]))

    try:
        with _quiet():
            v = U().f(np.array(x[0], float))
        _sidecar(run, {"error": None})
        return np.array([0.0, v])
    except Exception as exc:
        _sidecar(run, {"error": repr(exc)})
        return np.array([1.0, np.nan])

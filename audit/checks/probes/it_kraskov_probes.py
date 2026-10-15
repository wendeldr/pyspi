"""Probes for the it_kraskov family (PROJECT venv: pyspi + JIDT via JPype).

Every public function has the signature ``fn(x, run)`` required by
``audit/checks/pyspi_outputs.py``: ``x`` is the raw dataset array (P, T) after the
optional process selection and ``run`` the manifest entry (its ``identifier`` names
the SPI the probe builds, its ``path`` the preprocessing).  The returned array is
stored as the pyspi-side ``value``; the reference side (refs/it_kraskov.py, comparison
venv, no JIDT) documents the layout of each array next to the check that reads it.

The probes expose what the plain SPI path cannot: the cached DCE window matrix
(``data.theiler`` holds the raw float before ``int()``), JIDT's DYN_CORR_EXCL
semantics on the nearest-neighbour searchers themselves, NUM_THREADS / NOISE_SEED
variations, the embedding JIDT's MAX_CORR_AIS search chose for every ordered pair,
the fixed-embedding TE recomputed with that embedding, the full ordered (unmirrored)
MI matrix, and the pickle / deepcopy / CalculatorFrame copy paths.
"""
import contextlib
import copy
import io
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parents[1]
REPO = CHECKS.parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(CHECKS))

from regression import harness  # noqa: E402

EMB_PROPS = ("k_HISTORY", "k_TAU", "l_HISTORY", "l_TAU")


# ----------------------------------------------------------------------------- helpers
def _jvm():
    harness.ensure_jvm()
    harness.import_statistics_modules()


def _data(x, path="pipeline"):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if path == "kernel":
            return Data(x, dim_order="ps", normalise=False, detrend=False)
        if path == "pipeline":
            return Data(x, dim_order="ps")
        if path == "pipeline-detrend":
            return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _entry(identifier):
    import yaml
    doc = yaml.safe_load((CHECKS.parent / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u["identifiers"]:
            return dict(identifier=identifier, module=f".statistics.{u['module']}", cls=u["pyspi_class"],
                        params=u["identifier_params"][identifier], labels=u["labels"], deps=u["deps"])
    raise KeyError(identifier)


def _spi(identifier):
    return harness.build_spi(_entry(identifier))


def _compute(spi, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return harness.compute_spi(spi, data)


def _num(v):
    """getProperty result -> float (NaN for None / 'NONE' / non-numeric)."""
    if v is None:
        return np.nan
    try:
        return float(str(v))
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------- theiler window
def theiler_raw(x, run):
    """data.theiler after pyspi's AUTO rule (infotheory.py _set_theiler_window), diagonal NaN.

    The cached matrix holds the RAW value 2 * dot(acf_i, acf_j); int() is applied only when
    the property is set.  The SPI's calculator receives DYN_CORR_EXCL = int(raw[0, 1]) here.
    """
    _jvm()
    data = _data(x, run.get("path", "pipeline"))
    spi = _spi(run["identifier"] or "mi_kraskov_NN-4_DCE")
    spi._set_theiler_window(data, 0, 1)
    W = np.array(data.theiler, dtype=np.float64, copy=True)
    np.fill_diagonal(W, np.nan)
    return W


def dce_semantics(x, run):
    """JIDT DYN_CORR_EXCL semantics on a ramp (x unused).

    rows W = 0..6, columns:
      0 W
      1 UnivariateNearestNeighbourSearcher.countPointsStrictlyWithinR(t=15, r=1e9, W) on 0..31
      2 KdTree (2-D) countPointsStrictlyWithinR(t=15, r=1e9, W)
      3 smallest |s - t| among the K=4 nearest neighbours KdTree.findKNearestNeighbours(4, 15, W)
        of the ramp (the nearest admissible neighbours are those just outside the window)
      4 1 if MutualInfoCalculatorMultiVariateKraskov1 (k=4, DYN_CORR_EXCL=W) raises at N = 4 + 2W
      5 1 if it computes at N = 5 + 2W
      6 MI value of the KSG1 calculator with DYN_CORR_EXCL=W on (u0, u1) below (T=40)
    u0, u1: deterministic test series (sin and cos of a quadratic phase) so the reference
    can recompute the MI under |s-t| <= W and |s-t| < W exclusion.
    """
    _jvm()
    import jpype as jp
    U = jp.JPackage("infodynamics.utils")
    K = jp.JPackage("infodynamics.measures.continuous.kraskov")
    ramp = np.arange(32, dtype=np.float64)
    ramp2 = np.stack([ramp, 2.0 * ramp], axis=1)
    uni = U.UnivariateNearestNeighbourSearcher(jp.JArray(jp.JDouble, 1)(ramp))
    kd = U.KdTree(jp.JArray(jp.JDouble, 2)(ramp2))
    u0, u1 = semantics_series()
    rows = []
    for W in range(7):
        c1 = int(uni.countPointsStrictlyWithinR(15, 1e9, W))
        c2 = int(kd.countPointsStrictlyWithinR(15, 1e9, W))
        pq = kd.findKNearestNeighbours(4, 15, W)
        idx = []
        while not pq.isEmpty():
            idx.append(int(pq.poll().sampleIndex))
        dmin = float(min(abs(i - 15) for i in idx))
        flags = []
        for N in (4 + 2 * W, 5 + 2 * W):
            c = K.MutualInfoCalculatorMultiVariateKraskov1()
            c.setProperty("k", "4")
            c.setProperty("NOISE_SEED", "42")
            c.setProperty("DYN_CORR_EXCL", str(W))
            c.initialise(1, 1)
            try:
                c.setObservations(jp.JArray(jp.JDouble, 1)(u0[:N]), jp.JArray(jp.JDouble, 1)(u1[:N]))
                c.computeAverageLocalOfObservations()
                flags.append(0.0)
            except Exception:
                flags.append(1.0)
        c = K.MutualInfoCalculatorMultiVariateKraskov1()
        c.setProperty("k", "4")
        c.setProperty("NOISE_SEED", "42")
        c.setProperty("DYN_CORR_EXCL", str(W))
        c.initialise(1, 1)
        c.setObservations(jp.JArray(jp.JDouble, 1)(u0), jp.JArray(jp.JDouble, 1)(u1))
        mi = float(c.computeAverageLocalOfObservations())
        rows.append([W, c1, c2, dmin, flags[0], 1.0 - flags[1], mi])
    return np.array(rows, dtype=np.float64)


def semantics_series():
    """Deterministic tie-free pair used by dce_semantics (duplicated in refs/it_kraskov.py)."""
    t = np.arange(40, dtype=np.float64)
    ph = 0.37 * t + 0.011 * t ** 2
    return np.sin(ph) + 0.1 * np.cos(3.1 * t), np.cos(1.3 * ph) + 0.05 * t / 40.0


# ----------------------------------------------------------------------------- estimator probes
def direct_jidt(x, run):
    """JIDT KSG1 MI and CMI calculators driven directly with pyspi's properties on raw rows.

    value[0, w] = MI(x0; x1) with DYN_CORR_EXCL = W_LIST[w]
    value[1, w] = CMI(x0; x1 | x2) with DYN_CORR_EXCL = W_LIST[w]
    (k=4, NOISE_SEED=42, NORMALISE default true, jitter default 1e-8)
    """
    _jvm()
    import jpype as jp
    K = jp.JPackage("infodynamics.measures.continuous.kraskov")
    out = np.full((2, len(W_LIST)), np.nan)
    a, b, c = (np.ascontiguousarray(x[i], dtype=np.float64) for i in range(3))
    for w, W in enumerate(W_LIST):
        mi = K.MutualInfoCalculatorMultiVariateKraskov1()
        mi.setProperty("k", "4")
        mi.setProperty("NOISE_SEED", "42")
        mi.setProperty("DYN_CORR_EXCL", str(W))
        mi.initialise(1, 1)
        mi.setObservations(jp.JArray(jp.JDouble, 1)(a), jp.JArray(jp.JDouble, 1)(b))
        out[0, w] = mi.computeAverageLocalOfObservations()
        cm = K.ConditionalMutualInfoCalculatorMultiVariateKraskov1()
        cm.setProperty("k", "4")
        cm.setProperty("NOISE_SEED", "42")
        cm.setProperty("DYN_CORR_EXCL", str(W))
        cm.initialise(1, 1, 1)
        cm.setObservations(jp.JArray(jp.JDouble, 1)(a), jp.JArray(jp.JDouble, 1)(b), jp.JArray(jp.JDouble, 1)(c))
        out[1, w] = cm.computeAverageLocalOfObservations()
    return out


W_LIST = (0, 1, 2, 5, 20, 85)


def thread_props(x, run):
    """[availableProcessors, NUM_THREADS property of a fresh MI Kraskov calculator,
    NUM_THREADS of a fresh conditional-MI Kraskov calculator, BIAS_CORRECTION is None (MI),
    BIAS_CORRECTION is None (TE), NOISE_SEED of a fresh calculator is 'NONE',
    NOISE_SEED after pyspi's _setup]."""
    _jvm()
    import jpype as jp
    K = jp.JPackage("infodynamics.measures.continuous.kraskov")
    ncpu = int(jp.java.lang.Runtime.getRuntime().availableProcessors())
    mi = K.MutualInfoCalculatorMultiVariateKraskov1()
    cm = K.ConditionalMutualInfoCalculatorMultiVariateKraskov1()
    te = K.TransferEntropyCalculatorKraskov()
    fresh_seed = str(mi.getProperty("NOISE_SEED"))
    spi = _spi("mi_kraskov_NN-4")
    return np.array([ncpu, _num(mi.getProperty("NUM_THREADS")), _num(cm.getProperty("NUM_THREADS")),
                     float(spi._calc.getProperty("BIAS_CORRECTION") is None),
                     float(_spi("te_kraskov_NN-4_k-1_kt-1_l-1_lt-1")._calc.getProperty("BIAS_CORRECTION") is None),
                     float(fresh_seed.upper() == "NONE"), _num(spi._calc.getProperty("NOISE_SEED")),
                     float(te.getProperty("NUM_THREADS") is None)], dtype=np.float64)


def num_threads(x, run):
    """(3, P, P): [default NUM_THREADS, NUM_THREADS=1, default again (repeat)] for run['identifier']."""
    _jvm()
    out = []
    for nt in (None, "1", None):
        data = _data(x, run.get("path", "pipeline"))
        spi = _spi(run["identifier"])
        if nt is not None:
            spi._calc.setProperty("NUM_THREADS", nt)
        out.append(_compute(spi, data))
    return np.stack(out)


def seeds(x, run):
    """(3, P, P): run['identifier'] with NOISE_SEED = 42 (pyspi), 43, 44."""
    _jvm()
    out = []
    for s in ("42", "43", "44"):
        data = _data(x, run.get("path", "pipeline"))
        spi = _spi(run["identifier"])
        spi._calc.setProperty("NOISE_SEED", s)
        out.append(_compute(spi, data))
    return np.stack(out)


def ordered_mi(x, run):
    """Full ordered matrix bivariate(i, j) for all i != j (Directed.multivariate, no mirror)."""
    _jvm()
    from pyspi.base import Directed
    data = _data(x, run.get("path", "pipeline"))
    spi = _spi(run["identifier"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = Directed.multivariate(spi, data)
    A = np.array(A, dtype=np.float64)
    np.fill_diagonal(A, np.nan)
    return A


# ----------------------------------------------------------------------------- auto-embedding
def auto_embedding(x, run):
    """(P, P, 5): [TE value, k_HISTORY, k_TAU, l_HISTORY, l_TAU] that JIDT's MAX_CORR_AIS search
    chose for each ordered pair (i -> j), read with getProperty right after bivariate(i, j).
    NaN rows where the pair returned NaN (the properties are stale then)."""
    _jvm()
    data = _data(x, run.get("path", "pipeline"))
    spi = _spi(run["identifier"])
    P = x.shape[0]
    out = np.full((P, P, 5), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                v = spi.bivariate(data, i=i, j=j)
                if np.isfinite(v):
                    out[i, j] = [v] + [_num(spi._calc.getProperty(p)) for p in EMB_PROPS]
                else:
                    out[i, j, 0] = v
    return out


def auto_fixed_repro(x, run):
    """(2, P, P): [auto-embedding TE A[i, j], fixed-embedding pyspi TE with the (k, kt, l, lt)
    the auto search chose for that pair (TransferEntropy(estimator='kraskov', k_history=k,
    k_tau=kt, l_history=l, l_tau=lt[, dyn_corr_excl='AUTO']) on the same Data object)]."""
    _jvm()
    from pyspi.statistics.infotheory import TransferEntropy
    data = _data(x, run.get("path", "pipeline"))
    spi = _spi(run["identifier"])
    dce = "_DCE" in run["identifier"]
    P = x.shape[0]
    out = np.full((2, P, P), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                v = spi.bivariate(data, i=i, j=j)
                out[0, i, j] = v
                if not np.isfinite(v):
                    continue
                k, kt, l, lt = (int(_num(spi._calc.getProperty(p))) for p in EMB_PROPS)
                fx = TransferEntropy(estimator="kraskov", prop_k=4, k_history=k, k_tau=kt, l_history=l,
                                     l_tau=lt, dyn_corr_excl="AUTO" if dce else None)
                out[1, i, j] = fx.bivariate(data, i=i, j=j)
    return out


# ----------------------------------------------------------------------------- copy paths
COPY_PROPS = ("k", "NOISE_SEED", "k_HISTORY", "l_HISTORY", "AUTO_EMBED_METHOD")


def copy_paths(x, run):
    """(4, P, P): [Calculator, CalculatorFrame, copy.deepcopy(spi), pickle round trip] for
    run['identifier'] on the same data (pipeline path).  Calculator and CalculatorFrame are
    built from a one-SPI config file (CalculatorFrame.init_from_list deep-copies the
    Calculator, calculator.py:556-563)."""
    _jvm()
    import yaml
    from pyspi.calculator import Calculator, CalculatorFrame
    ident = run["identifier"]
    e = _entry(ident)
    cfg = {e["module"]: {e["cls"]: {"labels": e["labels"], "dependencies": e["deps"] or [],
                                     "configs": [e["params"]]}}}
    out = []
    with tempfile.TemporaryDirectory() as td, warnings.catch_warnings(), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        cf = Path(td) / "one.yaml"
        cf.write_text(yaml.safe_dump(cfg))
        xx = np.array(x, dtype=np.float64, copy=True)
        c = Calculator(dataset=xx, configfile=str(cf))
        c.compute()
        out.append(np.array(c.table[ident].to_numpy(), dtype=np.float64))
        f = CalculatorFrame(datasets=[np.array(x, dtype=np.float64, copy=True)], names=["a"], labels=[["l"]],
                            configfile=str(cf))
        f.compute()
        fc = f.calculators.iloc[0, 0]
        out.append(np.array(fc.table[ident].to_numpy(), dtype=np.float64))
    spi = _spi(ident)
    out.append(_compute(copy.deepcopy(spi), _data(x)))
    out.append(_compute(pickle.loads(pickle.dumps(spi)), _data(x)))
    out = np.stack(out)
    for a in out:
        np.fill_diagonal(a, np.nan)
    return out


def copy_props(x, run):
    """(3, 5): getProperty read-back of COPY_PROPS on [original, deepcopy, pickle] calculators
    (numbers; NOISE_SEED 'NONE' -> NaN; AUTO_EMBED_METHOD MAX_CORR_AIS -> 1, NONE -> 0)."""
    _jvm()
    spi = _spi(run["identifier"])
    rows = []
    for o in (spi, copy.deepcopy(spi), pickle.loads(pickle.dumps(spi))):
        r = []
        for p in COPY_PROPS:
            v = o._calc.getProperty(p)
            if p == "AUTO_EMBED_METHOD":
                r.append(1.0 if str(v) == "MAX_CORR_AIS" else 0.0 if str(v).upper() == "NONE" else np.nan)
            else:
                r.append(_num(v))
        rows.append(r)
    return np.array(rows, dtype=np.float64)


# ----------------------------------------------------------------------------- review additions
def _copy_layers(x, ident, module, cls, labels, deps, params, spi_factory):
    """(4, P, P): [Calculator, CalculatorFrame, copy.deepcopy(spi), pickle round trip] for one SPI
    given by explicit constructor params (one-SPI config file, as copy_paths)."""
    import yaml
    from pyspi.calculator import Calculator, CalculatorFrame
    cfg = {module: {cls: {"labels": labels, "dependencies": deps or [], "configs": [params]}}}
    out = []
    with tempfile.TemporaryDirectory() as td, warnings.catch_warnings(), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        cf = Path(td) / "one.yaml"
        cf.write_text(yaml.safe_dump(cfg))
        c = Calculator(dataset=np.array(x, dtype=np.float64, copy=True), configfile=str(cf))
        c.compute()
        out.append(np.array(c.table[ident].to_numpy(), dtype=np.float64))
        f = CalculatorFrame(datasets=[np.array(x, dtype=np.float64, copy=True)], names=["a"], labels=[["l"]],
                            configfile=str(cf))
        f.compute()
        out.append(np.array(f.calculators.iloc[0, 0].table[ident].to_numpy(), dtype=np.float64))
    spi = spi_factory()
    out.append(_compute(copy.deepcopy(spi), _data(x)))
    out.append(_compute(pickle.loads(pickle.dumps(spi)), _data(x)))
    out = np.stack(out)
    for a in out:
        np.fill_diagonal(a, np.nan)
    return out


K6_PARAMS = {"estimator": "kraskov", "prop_k": 6}


def _mi_k6():
    from pyspi.statistics.infotheory import MutualInfo
    return MutualInfo(**K6_PARAMS)


def copy_paths_mi_k6(x, run):
    """(4, P, P): copy_paths for MutualInfo(estimator='kraskov', prop_k=6) = mi_kraskov_NN-6
    (not a config identifier; exposes whether the copies keep prop_k)."""
    _jvm()
    e = _entry("mi_kraskov_NN-4")
    spi = _mi_k6()
    assert spi.identifier == "mi_kraskov_NN-6", spi.identifier
    return _copy_layers(x, spi.identifier, e["module"], e["cls"], e["labels"], e["deps"], dict(K6_PARAMS), _mi_k6)


def copy_props_mi_k6(x, run):
    """(3, 5): COPY_PROPS read-back on [original, deepcopy, pickle] of MutualInfo(kraskov, prop_k=6)."""
    _jvm()
    spi = _mi_k6()
    rows = []
    for o in (spi, copy.deepcopy(spi), pickle.loads(pickle.dumps(spi))):
        r = []
        for p in COPY_PROPS:
            v = o._calc.getProperty(p)
            if p == "AUTO_EMBED_METHOD":
                r.append(1.0 if str(v) == "MAX_CORR_AIS" else 0.0 if str(v).upper() == "NONE" else np.nan)
            else:
                r.append(_num(v))
        rows.append(r)
    return np.array(rows, dtype=np.float64)


def time_reversed(x, run):
    """(2, P, P): [A(x), A(x reversed in time)] for run['identifier'] (fresh SPI and Data each)."""
    _jvm()
    out = []
    for xx in (x, x[:, ::-1]):
        out.append(_compute(_spi(run["identifier"]), _data(np.ascontiguousarray(xx), run.get("path", "pipeline"))))
    return np.stack(out)


def guard_window_N(identifier, T):
    """Number of KSG samples N for the SPIs whose guard probe is defined (k = 4)."""
    if identifier.startswith("tlmi_kraskov") or identifier.startswith("te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1"):
        return T - 1
    raise KeyError(identifier)


def guard_numeric(x, run):
    """(2, P, P): run['identifier'] rebuilt with a NUMERIC dyn_corr_excl (infotheory.py:217-220 path)
    at W = Wc - 1 and W = Wc, where Wc = smallest W with N <= 4 + 2W (JIDT guard), N = T - 1."""
    _jvm()
    e = _entry(run["identifier"])
    T = x.shape[1]
    N = guard_window_N(run["identifier"], T)
    Wc = int(np.ceil((N - 4) / 2.0))
    out = []
    for W in (Wc - 1, Wc):
        params = dict(e["params"])
        params["dyn_corr_excl"] = W
        spi = harness.build_spi(dict(e, params=params))
        out.append(_compute(spi, _data(x, run.get("path", "pipeline"))))
    return np.stack(out)

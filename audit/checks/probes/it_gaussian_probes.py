"""Probes for the it_gaussian family (PROJECT venv: pyspi + JIDT via JPype).

Each public function has the signature ``fn(x, run)`` required by
``audit/checks/pyspi_outputs.py``: ``x`` is the raw dataset array (P, T) after
the optional process selection, ``run`` the manifest entry.  The returned array
is stored as the pyspi-side ``value``.  The JVM is already started by
``pyspi_outputs.py`` (harness.ensure_jvm()); every probe calls it again, which
is a no-op, so the module also works stand-alone.

Pure helpers that build the constructed datasets (``shift_series``,
``indep_pair``) and the task list of the direct-JIDT probe (``jidt_tasks``)
are duplicated verbatim in ``refs/it_gaussian.py`` so the reference never
imports this module (and therefore never imports JIDT).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parents[1]
REPO = CHECKS.parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(CHECKS))

from regression import harness  # noqa: E402

H1 = 0.5 * (1.0 + np.log(2.0 * np.pi))   # Gaussian entropy of a unit-variance variable (nats)

FAMILY_IDENTIFIERS = [
    "je_gaussian", "ce_gaussian", "mi_gaussian", "tlmi_gaussian", "cce_gaussian", "di_gaussian",
    "si_gaussian_k-1", "xme_gaussian_k1", "xme_gaussian_k10", "gc_gaussian_k-1_kt-1_l-1_lt-1",
    "gc_gaussian_k-max-10_tau-max-2", "te_symbolic_k-1_kt-1_l-1_lt-1", "te_symbolic_k-10_kt-1_l-1_lt-1",
]


# ----------------------------------------------------------------------------- helpers
def _jvm():
    harness.ensure_jvm()
    harness.import_statistics_modules()


def _data(x, normalise=True, detrend=False):
    from pyspi.data import Data
    return Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps",
                normalise=normalise, detrend=detrend)


def _spi(identifier):
    """Build a configured SPI exactly as pyspi_outputs.py does (units.yaml identifier_params)."""
    import yaml
    doc = yaml.safe_load((CHECKS.parent / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u["identifiers"]:
            entry = dict(identifier=identifier, module=f".statistics.{u['module']}", cls=u["pyspi_class"],
                         params=u["identifier_params"][identifier], labels=u["labels"], deps=u["deps"])
            return harness.build_spi(entry)
    raise KeyError(identifier)


def _matrix(spi, data):
    return harness.compute_spi(spi, data)


# ------------------------------------------------- constructed data (duplicated in refs)
def shift_series(u, d):
    """Circular shift y(t) = u(t - d)."""
    return np.roll(np.asarray(u, dtype=np.float64), d)


def shift_dataset(x):
    """[u, u(t-9) + 0.1 v1, u(t-10) + 0.1 v2] from three iid_gaussian rows (deterministic)."""
    u, v1, v2 = x[0], x[1], x[2]
    return np.stack([u, shift_series(u, 9) + 0.1 * v1, shift_series(u, 10) + 0.1 * v2])


def jidt_tasks(P):
    """Deterministic task list for the direct-JIDT probe on the lag matrix M (columns =
    process p at lag L for p in range(P), L in 0..4).  Each task is (kind, A, B, C) with
    column-index tuples; kind in {'H', 'MI', 'CMI'}."""
    ncol = 5 * P
    tasks = []
    for d in range(1, min(10, ncol) + 1):
        tasks.append(("H", tuple(range(d)), (), ()))
    tasks.append(("H", tuple(range(0, ncol, 2))[:10], (), ()))
    for d in range(1, min(6, ncol - 1) + 1):
        tasks.append(("MI", (0,), tuple(range(1, 1 + d)), ()))
    tasks.append(("MI", (0, 1), (ncol - 1, ncol - 2), ()))
    for d in range(1, min(5, ncol - 2) + 1):
        tasks.append(("CMI", (0,), (1,), tuple(range(2, 2 + d))))
    tasks.append(("CMI", (0, 2), (1,), (ncol - 1,)))
    return tasks


def lag_matrix(x, maxlag=4):
    """(T - maxlag, 5P) matrix of x_p(t - L), columns ordered p-major."""
    x = np.asarray(x, dtype=np.float64)
    P, T = x.shape
    cols = [x[p, maxlag - L:T - L] for p in range(P) for L in range(maxlag + 1)]
    return np.stack(cols, axis=1)


# ----------------------------------------------------------------------------- probes
def xme_k2(x, run):
    """CrossmapEntropy(history_length=2, estimator='gaussian') on the default pipeline."""
    _jvm()
    from pyspi.statistics.infotheory import CrossmapEntropy
    return _matrix(CrossmapEntropy(history_length=2, estimator="gaussian"), _data(x))


def xme_shift_stack(x, run):
    """xme_gaussian_k1, history_length=2 and xme_gaussian_k10 on the constructed shift data
    [u, u(t-9)+0.1v1, u(t-10)+0.1v2]; returns (3, 3, 3) stacked [k1, k2, k10]."""
    _jvm()
    from pyspi.statistics.infotheory import CrossmapEntropy
    X = shift_dataset(x)
    out = []
    for k in (1, 2, 10):
        out.append(_matrix(CrossmapEntropy(history_length=k, estimator="gaussian"), _data(X)))
    return np.stack(out)


def cce_n10(x, run):
    _jvm()
    from pyspi.statistics.infotheory import CausalEntropy
    return _matrix(CausalEntropy(n=10, estimator="gaussian"), _data(x))


def di_n10(x, run):
    _jvm()
    from pyspi.statistics.infotheory import DirectedInfo
    return _matrix(DirectedInfo(n=10, estimator="gaussian"), _data(x))


def configured_history(x, run):
    """[_n of the configured cce_gaussian, _n of the configured di_gaussian]."""
    _jvm()
    return np.array([float(_spi("cce_gaussian")._n), float(_spi("di_gaussian")._n)])


def indep_pair(x_var2, x_iid):
    """[iid_gaussian row 0 (white source), var2_bivariate row 0 (AR(2) target)] - no coupling."""
    return np.stack([np.asarray(x_iid[0], dtype=np.float64), np.asarray(x_var2[0], dtype=np.float64)])


def di_independent_pair(x, run):
    """di_gaussian on [iid_gaussian row 0, var2_bivariate row 0] at the run's T (default pipeline)."""
    _jvm()
    import make_data
    X = indep_pair(x, make_data.load("iid_gaussian", int(run["T"]))["x"])
    return _matrix(_spi("di_gaussian"), _data(X))


def te_auto_params(x, run):
    """(P, P, 4) array of the embedding JIDT selected for each ordered pair (i = source,
    j = target) under gc_gaussian_k-max-10_tau-max-2: [k_HISTORY, k_TAU, l_HISTORY, l_TAU],
    read with getProperty after bivariate(i, j).  Diagonal NaN."""
    _jvm()
    spi = _spi("gc_gaussian_k-max-10_tau-max-2")
    data = _data(x)
    P = x.shape[0]
    out = np.full((P, P, 4), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            spi.bivariate(data, i=i, j=j)
            out[i, j] = [float(str(spi._calc.getProperty(p))) for p in ("k_HISTORY", "k_TAU", "l_HISTORY", "l_TAU")]
    return out


def te_symbolic_monotone(x, run):
    """te_symbolic_k-1 on x and on exp(x) (a strictly monotone map, which leaves ordinal
    patterns unchanged); returns (2, P, P)."""
    _jvm()
    a = _matrix(_spi("te_symbolic_k-1_kt-1_l-1_lt-1"), _data(x))
    b = _matrix(_spi("te_symbolic_k-1_kt-1_l-1_lt-1"), _data(np.exp(x)))
    return np.stack([a, b])


def est_entropy_cache(x, run):
    """Univariate Gaussian entropies cached in data.entropy[('gaussian',)] after ce_gaussian."""
    _jvm()
    data = _data(x)
    _matrix(_spi("ce_gaussian"), data)
    return np.asarray(data.entropy[("gaussian",)], dtype=np.float64)


def est_direct_jidt(x, run):
    """Drive the JIDT Gaussian entropy / MI / CMI calculators directly (pyspi's own calculator
    objects, properties as pyspi sets them) on column subsets of the raw lag matrix."""
    _jvm()
    import jpype as jp
    from pyspi.statistics.infotheory import MutualInfo, JointEntropy
    M = lag_matrix(x)
    ent = JointEntropy(estimator="gaussian")._entropy_calc
    mi = MutualInfo(estimator="gaussian")._calc
    cmi = jp.JPackage("infodynamics.measures.continuous.gaussian").ConditionalMutualInfoCalculatorMultiVariateGaussian()
    cmi.setProperty("BIAS_CORRECTION", "false")
    D2 = lambda a: jp.JArray(jp.JDouble, 2)(np.ascontiguousarray(a))
    vals = []
    for kind, A, B, C in jidt_tasks(x.shape[0]):
        if kind == "H":
            ent.initialise(len(A))
            ent.setObservations(D2(M[:, list(A)]))
            vals.append(ent.computeAverageLocalOfObservations())
        elif kind == "MI":
            mi.initialise(len(A), len(B))
            mi.setObservations(D2(M[:, list(A)]), D2(M[:, list(B)]))
            vals.append(mi.computeAverageLocalOfObservations())
        else:
            cmi.initialise(len(A), len(B), len(C))
            cmi.setObservations(D2(M[:, list(A)]), D2(M[:, list(B)]), D2(M[:, list(C)]))
            vals.append(cmi.computeAverageLocalOfObservations())
    return np.asarray(vals, dtype=np.float64)


def est_normalise(x, run):
    """[entropy calc NORMALISE, MI calc NORMALISE, TE calc NORMALISE (1 = 'true'),
    H(10 x0) - H(x0) from the entropy calculator on raw x0, H(x0) raw]."""
    _jvm()
    import jpype as jp
    from pyspi.statistics.infotheory import MutualInfo, TransferEntropy, JointEntropy
    ent = JointEntropy(estimator="gaussian")._entropy_calc
    mi = MutualInfo(estimator="gaussian")._calc
    te = TransferEntropy(estimator="gaussian")._calc

    def flag(c):
        v = c.getProperty("NORMALISE")
        return np.nan if v is None else float(str(v).lower() == "true")

    def H(v):
        ent.initialise(1)
        ent.setObservations(jp.JArray(jp.JDouble, 1)(np.asarray(v, dtype=np.float64)))
        return float(ent.computeAverageLocalOfObservations())

    h0 = H(x[0])
    return np.array([flag(ent), flag(mi), flag(te), H(10.0 * x[0]) - h0, h0])


def degenerate_contract(x, run):
    """Behaviour of every Gaussian SPI of the family on input with a bit-identical pair
    (shared_envelope rows 0 and 2), default pipeline.  Row per FAMILY_IDENTIFIERS entry:
    [raised (1/0), n_posinf_offdiag, n_nan_offdiag, value at (0, 2), raised NonPositiveDefinite (1/0)].
    A final extra row runs a real pyspi Calculator restricted to je_gaussian + mi_gaussian and
    reports [je table all-NaN (1/0), mi table n_posinf, mi table n_nan offdiag, je report failed (1/0), 0]."""
    _jvm()
    P = x.shape[0]
    off = ~np.eye(P, dtype=bool)
    rows = []
    for ident in FAMILY_IDENTIFIERS:
        try:
            S = _matrix(_spi(ident), _data(x))
            rows.append([0.0, float(np.isposinf(S[off]).sum()), float(np.isnan(S[off]).sum()), float(S[0, 2]), 0.0])
        except Exception as exc:  # expected for the entropy-based SPIs
            rows.append([1.0, np.nan, np.nan, np.nan, float("NonPositiveDefinite" in repr(exc))])
    from pyspi.calculator import Calculator
    cfg = (".statistics.infotheory:\n"
           "  JointEntropy:\n    labels: [undirected]\n    dependencies: [java]\n    configs:\n      - estimator: gaussian\n"
           "  MutualInfo:\n    labels: [undirected]\n    dependencies: [java]\n    configs:\n      - estimator: gaussian\n")
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "it_gaussian_degenerate.yaml"
        cf.write_text(cfg)
        calc = Calculator(dataset=np.array(x, copy=True), configfile=str(cf))
        calc.compute()
        je = np.asarray(calc.table["je_gaussian"], dtype=np.float64)
        mi = np.asarray(calc.table["mi_gaussian"], dtype=np.float64)
        rep = calc.metric_reports.get("je_gaussian")
        failed = float(getattr(rep, "status", "") == "failed")
    rows.append([float(np.isnan(je).all()), float(np.isposinf(mi[off]).sum()), float(np.isnan(mi[off]).sum()), failed, 0.0])
    return np.asarray(rows, dtype=np.float64)


def instance_labels(x, run):
    """[has 'linear', has 'symbolic', has 'nonlinear'] of the instance labels of
    te_symbolic_k-1, te_symbolic_k-10 and gc_gaussian_k-1 (rows in that order)."""
    _jvm()
    rows = []
    for ident in ("te_symbolic_k-1_kt-1_l-1_lt-1", "te_symbolic_k-10_kt-1_l-1_lt-1", "gc_gaussian_k-1_kt-1_l-1_lt-1"):
        lab = set(_spi(ident).labels)
        rows.append([float("linear" in lab), float("symbolic" in lab), float("nonlinear" in lab)])
    return np.asarray(rows)


def est_jar_provenance(x, run):
    """[superclass of gaussian.EntropyCalculatorMultiVariateGaussian is
    infodynamics.measures.continuous.EntropyCalculatorMultiVariateCommon (1/0),
    that class loads from the bundled jar (1/0),
    entropy calculator getProperty('BIAS_CORRECTION') is None (1/0),
    entropy calculator getProperty('NOISE_SEED') as set by pyspi (value or NaN)]."""
    _jvm()
    import jpype as jp
    from pyspi.statistics.infotheory import JointEntropy
    cls = jp.JClass("infodynamics.measures.continuous.gaussian.EntropyCalculatorMultiVariateGaussian")
    sup = str(cls.class_.getSuperclass().getName())
    try:
        jp.JClass("infodynamics.measures.continuous.EntropyCalculatorMultiVariateCommon")
        has_common = 1.0
    except Exception:
        has_common = 0.0
    ent = JointEntropy(estimator="gaussian")._entropy_calc
    bc = ent.getProperty("BIAS_CORRECTION")
    ns = ent.getProperty("NOISE_SEED")
    try:
        ns_v = float(str(ns))
    except Exception:
        ns_v = np.nan
    return np.array([float(sup == "infodynamics.measures.continuous.EntropyCalculatorMultiVariateCommon"),
                     has_common, float(bc is None), ns_v])


# ------------------------------------------------- review follow-up (scaled_offset, cce null)
def _cached_univariate_entropies(x, normalise):
    """data.entropy[('gaussian',)] after ce_gaussian on Data(x, normalise=normalise)."""
    data = _data(x, normalise=normalise)
    _matrix(_spi("ce_gaussian"), data)
    return np.asarray(data.entropy[("gaussian",)], dtype=np.float64)


def est_scaled_shift(x, run):
    """Per-channel shift of pyspi's cached univariate Gaussian entropy with normalise=False:
    H(scaled_offset row p) - H(bivariate_gaussian row p) (same draw, x = scale*D2 + offset).
    Returns (4,) in process order (scales 0.2, 2, 5, 10)."""
    _jvm()
    import make_data
    d2 = make_data.load("bivariate_gaussian", int(run["T"]))["x"]
    return _cached_univariate_entropies(x, False) - _cached_univariate_entropies(d2, False)


def norm_invariance(x, run):
    """mi_gaussian, tlmi_gaussian and gc_gaussian_k-1 on the same raw array with
    Data(normalise=True) and Data(normalise=False); returns (2, 3, P, P) stacked
    [[mi, tlmi, gc1] with z-scoring, [mi, tlmi, gc1] without]."""
    _jvm()
    ids = ("mi_gaussian", "tlmi_gaussian", "gc_gaussian_k-1_kt-1_l-1_lt-1")
    out = []
    for norm in (True, False):
        out.append(np.stack([_matrix(_spi(i), _data(x, normalise=norm)) for i in ids]))
    return np.stack(out)


def je_scaled_shift(x, run):
    """je_gaussian with normalise=False on scaled_offset minus je_gaussian with
    normalise=False on the unscaled bivariate_gaussian rows (same draw); (P, P)."""
    _jvm()
    import make_data
    d2 = make_data.load("bivariate_gaussian", int(run["T"]))["x"]
    return _matrix(_spi("je_gaussian"), _data(x, normalise=False)) - _matrix(_spi("je_gaussian"), _data(d2, normalise=False))

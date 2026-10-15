"""Probes for the it_kozachenko family (PROJECT venv: pyspi + JIDT via JPype).

Each public function has the signature ``fn(x, run)`` required by
``audit/checks/pyspi_outputs.py``: ``x`` is the raw dataset array (P, T) after the
optional process selection, ``run`` the manifest entry.  The returned array is stored as
the pyspi-side ``value``.  ``pyspi_outputs.py`` starts the JVM; every probe calls
``harness.ensure_jvm()`` again (a no-op), so the module also works stand-alone.

The pure helpers that build design matrices and constructed datasets (``design``,
``nn_configs``, ``shift_dataset``, ``indep_pair``, ``DLIST``, ``SCALES``) are duplicated
verbatim in ``refs/it_kozachenko.py`` so that the reference never imports this module
(and therefore never imports JIDT).

The estimator probes drive pyspi's OWN entropy calculator object
(``JointEntropy(estimator='kozachenko')._entropy_calc``, i.e. a JIDT
``EntropyCalculatorMultiVariateKozachenko`` with the properties pyspi's ``_setup`` sets:
BIAS_CORRECTION=false, NOISE_SEED=42).  ``NOISE_LEVEL_TO_ADD='0'`` switches the 1e-8
jitter off where a run tag says ``noiseoff``.
"""
import sys
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parents[1]
REPO = CHECKS.parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(CHECKS))

from regression import harness  # noqa: E402

DLIST = (1, 2, 3, 5, 6, 10)
SCALES = (0.1, 10.0, 1000.0)
KOZ_IDENTIFIERS = ["je_kozachenko", "ce_kozachenko", "cce_kozachenko", "di_kozachenko", "si_kozachenko_k-1",
                   "xme_kozachenko_k1", "xme_kozachenko_k10"]


# ----------------------------------------------------------------------------- helpers
def _jvm():
    harness.ensure_jvm()
    harness.import_statistics_modules()


def _tag(run):
    return run["run_id"].split("|")[-1]


def _data(x, normalise=True, detrend=False):
    from pyspi.data import Data
    return Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps", normalise=normalise, detrend=detrend)


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


def _calc(noise_off=False):
    """pyspi's own Kozachenko entropy calculator (properties as pyspi sets them)."""
    from pyspi.statistics.infotheory import JointEntropy
    c = JointEntropy(estimator="kozachenko")._entropy_calc
    if noise_off:
        c.setProperty("NOISE_LEVEL_TO_ADD", "0")
    return c


def _H(calc, X):
    import jpype as jp
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim == 1:
        X = X[:, None]
    calc.initialise(X.shape[1])
    calc.setObservations(jp.JArray(jp.JDouble, 2)(X))
    return float(calc.computeAverageLocalOfObservations())


# ------------------------------------------------- pure helpers (duplicated in refs)
def design(x, d):
    """(T - maxlag, d) design matrix: columns x_p(t - L) for L = 0, 1, ... (outer) and
    p = 0..P-1 (inner), first d columns, maxlag = ceil(d / P) - 1.  Raw values."""
    x = np.asarray(x, dtype=np.float64)
    P, T = x.shape
    maxlag = -(-d // P) - 1
    cols = [x[p, maxlag - L:T - L] for L in range(maxlag + 1) for p in range(P)]
    return np.stack(cols[:d], axis=1)


def nn_configs(T):
    """(d, N, scale) triples of the nearest-neighbour routine probe."""
    out = []
    for d in (1, 2, 3, 5, 10):
        for N in (2000, 2001, T - 9):
            out.append((d, N, 1.0))
    out += [(2, T - 9, 10.0), (10, T - 9, 0.05)]
    return out


def shift_dataset(x):
    """[u, u(t-9) + 0.1 v1, u(t-10) + 0.1 v2] from three iid_gaussian rows (circular shift)."""
    u, v1, v2 = (np.asarray(x[k], dtype=np.float64) for k in range(3))
    return np.stack([u, np.roll(u, 9) + 0.1 * v1, np.roll(u, 10) + 0.1 * v2])


def indep_pair(x_var2, x_iid):
    """[iid_gaussian row 0 (white source), var2_bivariate row 0 (AR(2) target)] - no coupling."""
    return np.stack([np.asarray(x_iid[0], dtype=np.float64), np.asarray(x_var2[0], dtype=np.float64)])


# ----------------------------------------------------------------------------- estimator probes
def est_direct(x, run):
    """H of design(x, d) for d in DLIST from pyspi's own calculator; tag containing
    'noiseoff' sets NOISE_LEVEL_TO_ADD=0, otherwise the default 1e-8 jitter (seed 42)."""
    _jvm()
    c = _calc(noise_off="noiseoff" in _tag(run))
    return np.array([_H(c, design(x, d)) for d in DLIST])


def nn_routines(x, run):
    """JIDT EuclideanUtils nearest-neighbour routines called directly on scale * design(x, d)[:N]
    for every nn_configs(T) triple.  Concatenated [naive, dispatch] per triple, plus fast for
    N <= 2001 (fast at N > 2001 is skipped: O(N^2) memory)."""
    _jvm()
    import jpype as jp
    EU = jp.JClass("infodynamics.utils.EuclideanUtils")
    out = []
    for d, N, s in nn_configs(x.shape[1]):
        X = s * design(x, d)[:N]
        J = jp.JArray(jp.JDouble, 2)(np.ascontiguousarray(X))
        out.append(np.array(EU.computeMinEuclideanDistancesNaive(J)))
        out.append(np.array(EU.computeMinEuclideanDistances(J)))
        if N <= 2001:
            out.append(np.array(EU.computeMinEuclideanDistancesFast(J)))
    return np.concatenate(out)


def scale_law_naive(x, run):
    """As scale_law but on X = design(x, d)[:T - 9] (more than 2000 rows, so JIDT's naive
    nearest-neighbour routine is used): H(c X) - H(X), noise off, d in (1, 2, 5), c in SCALES."""
    _jvm()
    c0 = _calc(noise_off=True)
    out = []
    for d in (1, 2, 5):
        X = design(x, d)[:x.shape[1] - 9]
        h = _H(c0, X)
        for s in SCALES:
            out.append(_H(c0, s * X) - h)
    return np.array(out)


def spi_noiseoff(x, run):
    """The run's identifier on the default pipeline with NOISE_LEVEL_TO_ADD='0' set on the SPI's
    own entropy calculator (removes JIDT's 1e-8 jitter; everything else unchanged)."""
    _jvm()
    spi = _spi(run["identifier"])
    spi._entropy_calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    return _matrix(spi, _data(x))


def scale_law(x, run):
    """H(c X) - H(X), noise off, X = design(x, d)[:2000], for d in (1, 2, 5) (outer) and c in SCALES."""
    _jvm()
    c0 = _calc(noise_off=True)
    out = []
    for d in (1, 2, 5):
        X = design(x, d)[:2000]
        h = _H(c0, X)
        for s in SCALES:
            out.append(_H(c0, s * X) - h)
    return np.array(out)


def determinism(x, run):
    """[H, H again, H a third time] on one calculator and H on a fresh calculator, default noise,
    X = design(x, 2)."""
    _jvm()
    X = design(x, 2)
    c = _calc()
    a = [_H(c, X) for _ in range(3)]
    return np.array(a + [_H(_calc(), X)])


def ties(x, run):
    """Row 0 rounded to 0.5 steps: [H default noise, H noise off, H of the continuous row default noise]."""
    _jvm()
    xr = np.round(np.asarray(x[0], dtype=np.float64) * 2.0) / 2.0
    return np.array([_H(_calc(), xr), _H(_calc(noise_off=True), xr), _H(_calc(), x[0])])


def properties(x, run):
    """[NORMALISE reads 'true', NOISE_SEED value, BIAS_CORRECTION is None, superclass is
    EntropyCalculatorMultiVariateCommon, NOISE_LEVEL_TO_ADD value, H(10 x0) - H(x0) noise off,
    H(10 x0) - H(x0) default noise]."""
    _jvm()
    import jpype as jp
    c = _calc()
    cls = jp.JClass("infodynamics.measures.continuous.kozachenko.EntropyCalculatorMultiVariateKozachenko")
    sup = str(cls.class_.getSuperclass().getName())

    def prop(name):
        v = c.getProperty(name)
        return None if v is None else str(v)

    seed = prop("NOISE_SEED")
    lvl = prop("NOISE_LEVEL_TO_ADD")
    c0 = _calc(noise_off=True)
    x0 = np.asarray(x[0], dtype=np.float64)
    return np.array([float(str(prop("NORMALISE")).lower() == "true"),
                     float(seed) if seed is not None else np.nan,
                     float(prop("BIAS_CORRECTION") is None),
                     float(sup == "infodynamics.measures.continuous.EntropyCalculatorMultiVariateCommon"),
                     float(lvl) if lvl is not None else np.nan,
                     _H(c0, 10.0 * x0) - _H(c0, x0),
                     _H(_calc(), 10.0 * x0) - _H(_calc(), x0)])


def labels(x, run):
    """[has 'linear', has 'nonlinear'] of the instance labels of every kozachenko identifier
    (rows in KOZ_IDENTIFIERS order)."""
    _jvm()
    rows = []
    for ident in KOZ_IDENTIFIERS:
        lab = set(_spi(ident).labels)
        rows.append([float("linear" in lab), float("nonlinear" in lab)])
    return np.asarray(rows)


# ----------------------------------------------------------------------------- SPI-level probes
def order_check(x, run):
    """For je (tag je-*) or si (tag si-*): [A[0,1], A[1,0] of the pipeline matrix,
    bivariate(0, 1) on a fresh Data, bivariate(1, 0) on another fresh Data]."""
    _jvm()
    ident = "je_kozachenko" if _tag(run).startswith("je") else "si_kozachenko_k-1"
    A = _matrix(_spi(ident), _data(x))
    b01 = _spi(ident).bivariate(_data(x), i=0, j=1)
    b10 = _spi(ident).bivariate(_data(x), i=1, j=0)
    return np.array([A[0, 1], A[1, 0], float(b01), float(b10)])


def noise_sensitivity(x, run):
    """je_kozachenko pipeline matrix with the default 1e-8 jitter and with NOISE_LEVEL_TO_ADD=1e-6;
    returns (2, P, P)."""
    _jvm()
    a = _matrix(_spi("je_kozachenko"), _data(x))
    spi = _spi("je_kozachenko")
    spi._entropy_calc.setProperty("NOISE_LEVEL_TO_ADD", "1e-6")
    b = _matrix(spi, _data(x))
    return np.stack([a, b])


def je_scaled_shift(x, run):
    """je_kozachenko with normalise=False on scaled_offset minus je_kozachenko with normalise=False
    on the unscaled bivariate_gaussian rows (same draw); (P, P)."""
    _jvm()
    import make_data
    d2 = make_data.load("bivariate_gaussian", int(run["T"]))["x"]
    return _matrix(_spi("je_kozachenko"), _data(x, normalise=False)) - _matrix(_spi("je_kozachenko"), _data(d2, normalise=False))


def xme_k2(x, run):
    """CrossmapEntropy(history_length=2, estimator='kozachenko') on the default pipeline."""
    _jvm()
    from pyspi.statistics.infotheory import CrossmapEntropy
    return _matrix(CrossmapEntropy(history_length=2, estimator="kozachenko"), _data(x))


def xme_shift(x, run):
    """xme_kozachenko_k1 and xme_kozachenko_k10 on shift_dataset(x); returns (2, 3, 3)."""
    _jvm()
    X = shift_dataset(x)
    return np.stack([_matrix(_spi("xme_kozachenko_k1"), _data(X)), _matrix(_spi("xme_kozachenko_k10"), _data(X))])


def xme10_threshold(x, run):
    """xme_kozachenko_k10 bivariate(0 -> 1) on the first T' samples of rows 0, 1 (each truncation
    z-scored by its own Data), T' = 2008..2014, i.e. 1998..2004 rows per KL estimate (the JIDT
    switch to the naive routine is at > 2000 rows, so T' >= 2011)."""
    _jvm()
    out = []
    for Tp in range(2008, 2015):
        out.append(float(_spi("xme_kozachenko_k10").bivariate(_data(x[:2, :Tp]), i=0, j=1)))
    return np.array(out)


def cce_n10(x, run):
    _jvm()
    from pyspi.statistics.infotheory import CausalEntropy
    return _matrix(CausalEntropy(n=10, estimator="kozachenko"), _data(x))


def di_n10(x, run):
    _jvm()
    from pyspi.statistics.infotheory import DirectedInfo
    return _matrix(DirectedInfo(n=10, estimator="kozachenko"), _data(x))


def configured_n(x, run):
    """[_n of the configured cce_kozachenko, _n of the configured di_kozachenko]."""
    _jvm()
    return np.array([float(_spi("cce_kozachenko")._n), float(_spi("di_kozachenko")._n)])


def di_indep(x, run):
    """di_kozachenko on indep_pair(var2_bivariate x, iid_gaussian x) at the run's T (default pipeline)."""
    _jvm()
    import make_data
    X = indep_pair(x, make_data.load("iid_gaussian", int(run["T"]))["x"])
    return _matrix(_spi("di_kozachenko"), _data(X))


# ----------------------------------------------------------------------------- copy (CalculatorFrame) path
def _unit_entry(identifier):
    import yaml
    doc = yaml.safe_load((CHECKS.parent / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u["identifiers"]:
            return dict(identifier=identifier, module=f".statistics.{u['module']}", cls=u["pyspi_class"],
                        params=u["identifier_params"][identifier], labels=u["labels"], deps=u["deps"])
    raise KeyError(identifier)


def _config_yaml(identifier):
    """Single-SPI pyspi config file text for ``identifier`` (units.yaml parameters)."""
    e = _unit_entry(identifier)
    params = e["params"] or {}
    lines = [f"{e['module']}:", f"  {e['cls']}:", "    labels: [l]", "    dependencies: [java]", "    configs:"]
    first = True
    for k, v in params.items():
        lines.append(("      - " if first else "        ") + f"{k}: {v}")
        first = False
    if first:
        lines.append("      - {}")
    return "\n".join(lines) + "\n"


def _prop(calc, name):
    """getProperty as float: nan when unknown/None, -1 for a non-numeric string such as 'NONE'."""
    try:
        v = calc.getProperty(name)
    except Exception:
        return np.nan
    if v is None:
        return np.nan
    try:
        return float(str(v))
    except ValueError:
        return -1.0


def copy_props(x, run):
    """Per kozachenko identifier (rows in KOZ_IDENTIFIERS order): NOISE_SEED of _entropy_calc on
    [original, copy.deepcopy, pickle round trip] and NOISE_LEVEL_TO_ADD on the same three
    ('NONE' -> -1). Returns (7, 6)."""
    import copy
    import pickle
    _jvm()
    rows = []
    for ident in KOZ_IDENTIFIERS:
        spi = _spi(ident)
        dc = copy.deepcopy(spi)
        pk = pickle.loads(pickle.dumps(spi))
        objs = (spi, dc, pk)
        rows.append([_prop(o._entropy_calc, "NOISE_SEED") for o in objs]
                    + [_prop(o._entropy_calc, "NOISE_LEVEL_TO_ADD") for o in objs])
    return np.array(rows, dtype=np.float64)


def frame(x, run):
    """[Calculator, CalculatorFrame A, CalculatorFrame B, pickle round trip] matrices of the run's
    identifier on the same data (default pipeline). CalculatorFrame.init_from_list deep-copies the
    base Calculator (pyspi/calculator.py:559); A and B are two separate CalculatorFrame objects; the
    pickle slice computes with pickle.loads(pickle.dumps(spi)) on fresh pipeline Data. Returns (4, P, P)."""
    import contextlib
    import io
    import pickle
    import tempfile
    import warnings
    _jvm()
    from pyspi.calculator import Calculator, CalculatorFrame
    ident = run["identifier"]
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "cfg.yaml"
        cfg.write_text(_config_yaml(ident))
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = Calculator(dataset=np.array(x, copy=True), configfile=str(cfg))
            c.compute()
            fa = CalculatorFrame(datasets=[np.array(x, copy=True)], names=["a"], labels=[["l"]], configfile=str(cfg))
            fa.compute()
            fb = CalculatorFrame(datasets=[np.array(x, copy=True)], names=["b"], labels=[["l"]], configfile=str(cfg))
            fb.compute()
    A = c.table[ident].to_numpy(dtype=np.float64)
    FA = fa.calculators.iloc[0, 0].table[ident].to_numpy(dtype=np.float64)
    FB = fb.calculators.iloc[0, 0].table[ident].to_numpy(dtype=np.float64)
    spi2 = pickle.loads(pickle.dumps(c.spis[ident]))
    with contextlib.redirect_stdout(io.StringIO()):
        C = _matrix(spi2, _data(x))
    np.fill_diagonal(C, np.nan)
    return np.stack([A, FA, FB, C])

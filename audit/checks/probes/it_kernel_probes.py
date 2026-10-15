"""Probes for the it_kernel family (PROJECT venv: pyspi + JIDT via JPype).

Each public function has the signature ``fn(x, run)`` required by
``audit/checks/pyspi_outputs.py``: ``x`` is the raw dataset array (P, T) after the
optional process selection, ``run`` the manifest entry. The returned array is stored
as the pyspi-side ``value``. Every probe calls ``harness.ensure_jvm()`` (a no-op when
the JVM is already up), so the module also works stand-alone.

The probes expose what the plain SPI path cannot: the CalculatorFrame (deepcopy) and
pickle paths, JIDT property read-back, direct JIDT kernel-calculator calls on a lag
matrix (dimensions 1..10, both widths), tie-grid probes with NORMALISE=false, the
FORCE_KERNEL_COMPARE_TO_ALL bin-search switch, raw-array ``bivariate(x, y)`` calls,
non-config parameters (history_length=2, n=10, TE l/tau variants) and 1-D cached
entropies. Pure helpers that the reference must reproduce (``lag_matrix``,
``direct_tasks``, ``grid_*``, ``indep_pair``) are duplicated verbatim in
``refs/it_kernel.py`` so the reference never imports this module (and therefore
never imports JIDT).
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

KERNEL_IDS = [
    "je_kernel_W-0.5", "ce_kernel_W-0.5", "cce_kernel_W-0.5", "xme_kernel_W-0.5_k1",
    "xme_kernel_W-0.5_k10", "di_kernel_W-0.5", "si_kernel_W-0.5_k-1",
    "mi_kernel_W-0.25", "tlmi_kernel_W-0.25", "te_kernel_W-0.25_k-1",
]


# ----------------------------------------------------------------------------- helpers
def _jvm():
    harness.ensure_jvm()
    harness.import_statistics_modules()



def _data(x, normalise=True, detrend=False):
    from pyspi.data import Data
    with contextlib.redirect_stdout(io.StringIO()):
        return Data(np.array(x, dtype=np.float64, copy=True), dim_order="ps",
                    normalise=normalise, detrend=detrend)


def _unit_entry(identifier):
    import yaml
    doc = yaml.safe_load((CHECKS.parent / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u["identifiers"]:
            return dict(identifier=identifier, module=f".statistics.{u['module']}", cls=u["pyspi_class"],
                        params=u["identifier_params"][identifier], labels=u["labels"], deps=u["deps"])
    raise KeyError(identifier)


def _spi(identifier):
    """Build a configured SPI exactly as pyspi_outputs.py does (units.yaml identifier_params)."""
    return harness.build_spi(_unit_entry(identifier))


def _matrix(spi, data):
    with contextlib.redirect_stdout(io.StringIO()):
        return harness.compute_spi(spi, data)


def _prop(calc, name):
    """getProperty as float (nan when the class does not know the property)."""
    try:
        v = calc.getProperty(name)
    except Exception:
        return np.nan
    if v is None:
        return np.nan
    s = str(v)
    if s.lower() == "true":
        return 1.0
    if s.lower() == "false":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return -1.0          # a non-numeric string such as 'NONE'


def _kpkg():
    import jpype as jp
    return jp.JPackage("infodynamics.measures.continuous.kernel")


def _J1(a):
    import jpype as jp
    return jp.JArray(jp.JDouble, 1)(np.ascontiguousarray(a, dtype=np.float64))


def _J2(a):
    import jpype as jp
    return jp.JArray(jp.JDouble, 2)(np.ascontiguousarray(a, dtype=np.float64))


# ------------------------------------------------- pure helpers (duplicated in refs)
def lag_matrix(x, maxlag=4):
    """(T - maxlag, 5P) matrix of x_p(t - L), columns ordered p-major."""
    x = np.asarray(x, dtype=np.float64)
    P, T = x.shape
    return np.stack([x[p, maxlag - L:T - L] for p in range(P) for L in range(maxlag + 1)], axis=1)


def direct_tasks(P):
    """Task list of the direct-JIDT probe. ('H', cols, width) entropies of d = 1..10 columns of
    the lag matrix; ('MI', colsA, colsB, width); ('TE', src_row, targ_row, k, width) on raw rows."""
    ncol = 5 * P
    tasks = []
    for w in (0.5, 0.25):
        for d in range(1, min(10, ncol) + 1):
            tasks.append(("H", tuple(range(d)), (), w))
        tasks.append(("MI", (0,), (5 % ncol,), w))
        tasks.append(("MI", (0, 1), (ncol - 1, ncol - 2), w))
        tasks.append(("TE", (0,), (1 % P,), w, 1))
        tasks.append(("TE", (1 % P,), (0,), w, 1))
        tasks.append(("TE", (0,), (1 % P,), w, 2))
    return tasks


GRID = [0.0, 1.0, 2.0, 3.0, 10.0]
GRID_B = [0.0, 2.0, 1.0, 3.0, 4.0]
GRID2 = [[a, b] for a in (0.0, 1.0, 2.0) for b in (0.0, 2.0, 5.0)]
TE_S = [0.0, 1, 2, 1, 0, 3, 2, 1, 0, 1]
TE_T = [1.0, 0, 1, 2, 1, 0, 3, 2, 1, 0]
TENTH_GRID_N = 31          # values k * 0.1, k = 0..30


def grid_tasks():
    """(kind, args, width) for the NORMALISE=false tie-grid probe (absolute widths)."""
    q = list(np.arange(TENTH_GRID_N) * 0.1)
    return [
        ("H", GRID, 1.0 - 1e-6), ("H", GRID, 1.0), ("H", GRID, 1.0 + 1e-6), ("H", GRID, 2.0),
        ("H", GRID2, 1.0 - 1e-6), ("H", GRID2, 1.0), ("H", GRID2, 2.0),
        ("H", [0.0, 100.0], 0.5),
        ("MI", (GRID, GRID), 1.0), ("MI", (GRID, GRID_B), 1.0), ("MI", (GRID, GRID_B), 1.0 - 1e-6),
        ("TE", (TE_S, TE_T), 1.0), ("TE", (TE_S, TE_T), 1.0 - 1e-6),
        ("H", q, 0.1), ("H", q, 0.1 + 1e-12),
    ]


def indep_pair(x_var2, x_iid):
    """[iid_gaussian row 0, var2_bivariate row 0]: white source, AR(2) target, independent."""
    return np.stack([np.asarray(x_iid[0], dtype=np.float64), np.asarray(x_var2[0], dtype=np.float64)])


# ----------------------------------------------------------------------------- SPI-level probes
def sym_fresh(x, run):
    """Directed fill D[i,j] = spi.bivariate(fresh pipeline Data, i, j) for every i != j (no cache,
    no Undirected mirror), to test symmetry of the estimator itself."""
    _jvm()
    spi = _spi(run["identifier"])
    P = x.shape[0]
    D = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                with contextlib.redirect_stdout(io.StringIO()):
                    D[i, j] = spi.bivariate(_data(x), i=i, j=j)
    return D


def _config_yaml(identifier):
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


def frame(x, run):
    """[Calculator, CalculatorFrame, pickle round trip] matrices of one kernel SPI (same data,
    same single-SPI config). CalculatorFrame.init_from_list deep-copies the base Calculator
    (pyspi/calculator.py:556-563); the pickle slice computes with pickle.loads(pickle.dumps(spi))
    on fresh pipeline Data. Returns (3, P, P)."""
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
            f = CalculatorFrame(datasets=[np.array(x, copy=True)], names=["a"], labels=[["l"]], configfile=str(cfg))
            f.compute()
    A = c.table[ident].to_numpy(dtype=np.float64)
    B = f.calculators.iloc[0, 0].table[ident].to_numpy(dtype=np.float64)
    spi2 = pickle.loads(pickle.dumps(c.spis[ident]))
    C = _matrix(spi2, _data(x))
    return np.stack([A, B, C])


def copy_props(x, run):
    """Per kernel identifier (rows in KERNEL_IDS order): KERNEL_WIDTH of _entropy_calc on the
    original / copy.deepcopy / pickle round trip, the same for _calc (nan when absent), and
    NOISE_SEED of _entropy_calc on the original and the deepcopy (42 -> 42.0, 'NONE' -> -1).
    Returns (10, 8)."""
    _jvm()
    rows = []
    for ident in KERNEL_IDS:
        spi = _spi(ident)
        dc = copy.deepcopy(spi)
        pk = pickle.loads(pickle.dumps(spi))
        r = [_prop(o._entropy_calc, "KERNEL_WIDTH") for o in (spi, dc, pk)]
        r += [_prop(o._calc, "KERNEL_WIDTH") if hasattr(o, "_calc") else np.nan for o in (spi, dc, pk)]
        r += [_prop(spi._entropy_calc, "NOISE_SEED"), _prop(dc._entropy_calc, "NOISE_SEED")]
        rows.append(r)
    return np.array(rows, dtype=np.float64)


def props(x, run):
    """Property read-back on the three JIDT kernel classes as pyspi configures them, and
    perturbation of the entropy/MI value on z-scored x[0], x[1].

    Returns [ent NOISE_SEED, ent NOISE_LEVEL_TO_ADD, ent BIAS_CORRECTION,
             mi  NOISE_SEED, mi  NOISE_LEVEL_TO_ADD, mi  BIAS_CORRECTION,
             te  NOISE_SEED, te  NOISE_LEVEL_TO_ADD, te  BIAS_CORRECTION,
             ent KERNEL_WIDTH, mi KERNEL_WIDTH, te KERNEL_WIDTH, ent NORMALISE, te NORMALISE,
             dH(seed only), dH(noise 1e-8 + seed 42), dH(BIAS_CORRECTION true), dH(noise 0.01 + seed 42),
             dMI(seed only), dMI(noise 1e-8 + seed 42), dMI(BIAS_CORRECTION true), dMI(noise 0.01 + seed 42)]
    (nan = property unknown to the class)."""
    _jvm()
    je, mi, te = _spi("je_kernel_W-0.5"), _spi("mi_kernel_W-0.25"), _spi("te_kernel_W-0.25_k-1")
    ent, mic, tec = je._entropy_calc, mi._calc, te._calc
    out = []
    for c in (ent, mic, tec):
        out += [_prop(c, "NOISE_SEED"), _prop(c, "NOISE_LEVEL_TO_ADD"), _prop(c, "BIAS_CORRECTION")]
    out += [_prop(ent, "KERNEL_WIDTH"), _prop(mic, "KERNEL_WIDTH"), _prop(tec, "KERNEL_WIDTH"),
            _prop(ent, "NORMALISE"), _prop(tec, "NORMALISE")]
    z = _data(x).to_numpy(squeeze=True)
    K = _kpkg()

    def H(extra):
        c = K.EntropyCalculatorMultiVariateKernel()
        c.setProperty("KERNEL_WIDTH", "0.5")
        for k, v in extra:
            c.setProperty(k, v)
        c.initialise(1)
        c.setObservations(_J1(z[0]))
        return c.computeAverageLocalOfObservations()

    def M(extra):
        c = K.MutualInfoCalculatorMultiVariateKernel()
        c.setProperty("KERNEL_WIDTH", "0.25")
        for k, v in extra:
            c.setProperty(k, v)
        c.initialise(1, 1)
        c.setObservations(_J1(z[0]), _J1(z[1]))
        return c.computeAverageLocalOfObservations()

    for f in (H, M):
        base = f([])
        out += [f([("NOISE_SEED", "42")]) - base,
                f([("NOISE_LEVEL_TO_ADD", "1e-8"), ("NOISE_SEED", "42")]) - base,
                f([("BIAS_CORRECTION", "true")]) - base,
                f([("NOISE_LEVEL_TO_ADD", "0.01"), ("NOISE_SEED", "42")]) - base]
    return np.array(out, dtype=np.float64)


def direct_jidt(x, run):
    """Direct JIDT kernel calculators (default NORMALISE=true) on the raw lag matrix / rows:
    entropy of d = 1..10 columns, MI 1+1 and 2+2 columns, TE k = 1, 2 (both directions for k = 1),
    each at KERNEL_WIDTH 0.5 and 0.25 (bits). Task list: direct_tasks(P)."""
    _jvm()
    K = _kpkg()
    M = lag_matrix(x)
    out = []
    for t in direct_tasks(x.shape[0]):
        kind, A, B, w = t[0], t[1], t[2], t[3]
        if kind == "H":
            c = K.EntropyCalculatorMultiVariateKernel()
            c.setProperty("KERNEL_WIDTH", str(w))
            c.initialise(len(A))
            c.setObservations(_J2(M[:, list(A)]))
        elif kind == "MI":
            c = K.MutualInfoCalculatorMultiVariateKernel()
            c.setProperty("KERNEL_WIDTH", str(w))
            c.initialise(len(A), len(B))
            c.setObservations(_J2(M[:, list(A)]), _J2(M[:, list(B)]))
        else:
            c = K.TransferEntropyCalculatorKernel()
            c.setProperty("KERNEL_WIDTH", str(w))
            c.setProperty("k_HISTORY", str(t[4]))
            c.initialise()
            c.setObservations(_J1(x[A[0]]), _J1(x[B[0]]))
        out.append(c.computeAverageLocalOfObservations())
    return np.array(out, dtype=np.float64)


def tie_grid(x, run):
    """JIDT kernel calculators with NORMALISE=false (absolute widths) on small grids whose box
    counts change exactly at |dx| = w (grid_tasks()); x is unused (harness only)."""
    _jvm()
    K = _kpkg()
    out = []
    for kind, args, w in grid_tasks():
        if kind == "H":
            X = np.asarray(args, dtype=np.float64)
            X = X[:, None] if X.ndim == 1 else X
            c = K.EntropyCalculatorMultiVariateKernel()
            c.setProperty("NORMALISE", "false")
            c.initialise(X.shape[1], float(w))
            c.setObservations(_J2(X))
        elif kind == "MI":
            c = K.MutualInfoCalculatorMultiVariateKernel()
            c.setProperty("NORMALISE", "false")
            c.initialise(1, 1, float(w))
            c.setObservations(_J1(args[0]), _J1(args[1]))
        else:
            c = K.TransferEntropyCalculatorKernel()
            c.setProperty("NORMALISE", "false")
            c.initialise(1, float(w))
            c.setObservations(_J1(args[0]), _J1(args[1]))
        out.append(c.computeAverageLocalOfObservations())
    return np.array(out, dtype=np.float64)


def entropy_1d(x, run):
    """pyspi's cached univariate kernel entropies (JIDTBase._compute_entropy, the H(x_i) term of
    ce) on pipeline Data, at kernel_width 0.5 (configured) and 0.25. Returns (2, P), bits."""
    _jvm()
    from pyspi.statistics.infotheory import JointEntropy
    P = x.shape[0]
    out = np.full((2, P), np.nan)
    for r, w in enumerate((0.5, 0.25)):
        spi = JointEntropy(estimator="kernel", kernel_width=w)
        d = _data(x)
        for i in range(P):
            out[r, i] = spi._compute_entropy(d, i=i)
    return out


def force_all(x, run):
    """MI(x0; x1) and TE(x0 -> x1), TE(x1 -> x0) at width 0.25 on the z-scored rows with
    FORCE_KERNEL_COMPARE_TO_ALL false (bin search, default) and true. Returns
    [mi_bins, mi_all, te01_bins, te01_all, te10_bins, te10_all]."""
    _jvm()
    K = _kpkg()
    z = _data(x).to_numpy(squeeze=True)
    out = []
    for force in ("false", "true"):
        c = K.MutualInfoCalculatorMultiVariateKernel()
        c.setProperty("KERNEL_WIDTH", "0.25")
        c.setProperty("FORCE_KERNEL_COMPARE_TO_ALL", force)
        c.initialise(1, 1)
        c.setObservations(_J1(z[0]), _J1(z[1]))
        out.append(c.computeAverageLocalOfObservations())
    te = []
    for (s, t) in ((0, 1), (1, 0)):
        for force in ("false", "true"):
            c = K.TransferEntropyCalculatorKernel()
            c.setProperty("KERNEL_WIDTH", "0.25")
            c.setProperty("FORCE_KERNEL_COMPARE_TO_ALL", force)
            c.initialise(1)
            c.setObservations(_J1(z[s]), _J1(z[t]))
            te.append(c.computeAverageLocalOfObservations())
    return np.array(out + te, dtype=np.float64)


RAW_IDS = ["je_kernel_W-0.5", "ce_kernel_W-0.5", "mi_kernel_W-0.25", "tlmi_kernel_W-0.25", "te_kernel_W-0.25_k-1"]


def raw_array(x, run):
    """spi.bivariate(x_a, x_b) with raw arrays (rows 2, 3 of the dataset; parse_bivariate wraps them
    with Data().add_process, pyspi/base.py:27-47, data.py:219-240) vs spi.bivariate(Data(x), i=2, j=3)
    on pipeline Data, for je, ce, mi, tlmi, te. Returns [raw, data] per identifier (10 values)."""
    _jvm()
    out = []
    for ident in RAW_IDS:
        with contextlib.redirect_stdout(io.StringIO()):
            out.append(_spi(ident).bivariate(np.array(x[2], copy=True), np.array(x[3], copy=True)))
            out.append(_spi(ident).bivariate(_data(x), i=2, j=3))
    return np.array(out, dtype=np.float64)


TE_VARIANTS = [(1, 1, 1), (3, 1, 1), (1, 4, 1), (1, 1, 2), (3, 4, 2)]   # (l_history, k_tau, l_tau)


def te_forwarding(x, run):
    """TransferEntropy(estimator='kernel', kernel_width=0.25, k_history=1, l_history=l, k_tau=kt,
    l_tau=lt) for TE_VARIANTS on pipeline Data. Returns (5, P, P)."""
    _jvm()
    from pyspi.statistics.infotheory import TransferEntropy
    out = []
    for l, kt, lt in TE_VARIANTS:
        spi = TransferEntropy(estimator="kernel", kernel_width=0.25, k_history=1, l_history=l, k_tau=kt, l_tau=lt)
        out.append(_matrix(spi, _data(x)))
    return np.stack(out)


def te_forwarding_flags(x, run):
    """[identifier == 'te_kernel_W-0.25_k-1' for every TE_VARIANTS entry (count),
        getProperty('k_HISTORY') of the configured SPI,
        setProperty raises on TransferEntropyCalculatorKernel for l_HISTORY, k_TAU, l_TAU (1/0 each),
        auto_embed: identifier ends with '_k-max-4' (1/0), getProperty('k_HISTORY') after auto_embed,
        auto_embed value minus fixed k=1 value on pipeline Data (A[0,1])]."""
    _jvm()
    from pyspi.statistics.infotheory import TransferEntropy
    n_ok = 0
    for l, kt, lt in TE_VARIANTS:
        spi = TransferEntropy(estimator="kernel", kernel_width=0.25, k_history=1, l_history=l, k_tau=kt, l_tau=lt)
        n_ok += spi.identifier == "te_kernel_W-0.25_k-1"
    conf = _spi("te_kernel_W-0.25_k-1")
    out = [float(n_ok), _prop(conf._calc, "k_HISTORY")]
    K = _kpkg()
    for name in ("l_HISTORY", "k_TAU", "l_TAU"):
        c = K.TransferEntropyCalculatorKernel()
        try:
            c.setProperty(name, "2")
            out.append(0.0)
        except Exception:
            out.append(1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        auto = TransferEntropy(estimator="kernel", kernel_width=0.25, auto_embed_method="MAX_CORR_AIS", k_search_max=4)
    out.append(float(auto.identifier.endswith("_k-max-4")))
    out.append(_prop(auto._calc, "k_HISTORY"))
    a = _matrix(auto, _data(x))
    b = _matrix(conf, _data(x))
    out.append(float(a[0, 1] - b[0, 1]))
    return np.array(out, dtype=np.float64)


def xme_k2(x, run):
    """CrossmapEntropy(history_length=2, estimator='kernel') (width 0.5) on pipeline Data."""
    _jvm()
    from pyspi.statistics.infotheory import CrossmapEntropy
    return _matrix(CrossmapEntropy(history_length=2, estimator="kernel"), _data(x))


def cce_n10(x, run):
    """CausalEntropy(n=10, estimator='kernel') (documented history length) on pipeline Data."""
    _jvm()
    from pyspi.statistics.infotheory import CausalEntropy
    return _matrix(CausalEntropy(n=10, estimator="kernel"), _data(x))


def di_n10(x, run):
    """DirectedInfo(n=10, estimator='kernel') on pipeline Data."""
    _jvm()
    from pyspi.statistics.infotheory import DirectedInfo
    return _matrix(DirectedInfo(n=10, estimator="kernel"), _data(x))


def configured_n(x, run):
    """[cce_kernel_W-0.5._n, di_kernel_W-0.5._n] as built from the config parameters."""
    _jvm()
    return np.array([float(_spi("cce_kernel_W-0.5")._n), float(_spi("di_kernel_W-0.5")._n)])


def di_indep_pair(x, run):
    """di_kernel_W-0.5 on [iid_gaussian row 0, var2_bivariate row 0] (independent white source,
    AR(2) target) at the run's T; x is the var2_bivariate array."""
    _jvm()
    import make_data
    X = indep_pair(x, make_data.load("iid_gaussian", int(run["T"]))["x"])
    return _matrix(_spi("di_kernel_W-0.5"), _data(X))

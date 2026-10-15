"""Probes for the ids family (PROJECT venv; called by pyspi_outputs.py as fn(x, run)).

Unit misc.ids = pyspi.statistics.misc.InterDependenceScore (misc.py:178-200), which calls the
vendored pyspi/lib/ids/numpy_dependence.py:compute_IDS_numpy.

Top-level code is numpy-only on purpose: refs/ids.py (COMPARISON venv, no pyspi) loads this file
by path to regenerate the constructed "hidden dependence" sample and to read the probe tables.
Every pyspi import happens inside a function.

Probe functions
  ids_param        InterDependenceScore with non-config parameters parsed from the run tag
                   ("bw0.25", "bw1.0", "pnorm1", "pnorm2"); path from the run.
  ids_transformed  pyspi IDS on a transformed copy of x: "signflip" (row 1 negated),
                   "identical" ([x0, x0, -x0, 2.5 x0 - 1]), "minmax08" (each row min-max scaled to
                   [0, 8], the paper's reported preprocessing, then the kernel path).
  hidden_dependence  pyspi IDS on a constructed DEPENDENT pair whose population IDS-max at k = 6 is 0
                   (see hidden_construction); the dataset x is ignored, T and seed come from the run.
  pnorm_calculator Calculator with a non-default pnorm (3 and "2") at P = 4 and P = 6: what is
                   stored, report status and the exception class (pnorm codes in PNORM_CASES).
  rng_import       fresh interpreter: global numpy RNG draws before / after importing
                   pyspi.lib.ids and after computing IDS (see RNG_IMPORT_KEYS).
  rng_calculator   fresh interpreters: what Calculator construction does to a prior user seed,
                   and whether a second construction in the same process reseeds.
  rng_cross_spi    fresh interpreters: cov_MinCovDet (sklearn MinCovDet, random_state=None, global
                   RNG) computed by a Calculator with and without ids in the config, user seed set
                   before or after construction.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

# ------------------------------------------------------------------ shared tables (read by refs/ids.py)
PNORM_CASES = [(3, 4), (3, 6), ("2", 4), ("2", 6)]          # (pnorm, P)
PNORM_FIELDS = ["table_block_all_nan", "report_failed", "err_ValueError", "err_AssertionError",
                "direct_ndim"]
RNG_IMPORT_KEYS = ["draw_seed123_no_import", "draw_after_import_seeded123", "state_eq_seed1717_after_import",
                   "draw_after_ids_compute", "ids_diff_seed1_vs_seed2", "draw_after_import_pyspi_calculator_only"]
RNG_CALC_KEYS = ["draw_after_first_Calculator_seeded123", "state_eq_seed1717_after_first_Calculator",
                 "draw_after_second_Calculator_seeded5"]
RNG_CROSS_KEYS = ["mcd_with_ids_seed1_before", "mcd_with_ids_seed2_before", "mcd_noids_seed1717_before",
                  "mcd_noids_seed1_before", "mcd_noids_seed2_before",
                  "mcd_idsfirst_seed7_after", "mcd_noids_seed7_after"]
HIDDEN_OMEGA = 8.0          # g0(x) = cos(8 x) before orthogonalisation (lower omega: sup|g|/||g|| too large)
HIDDEN_EPS_FRACTION = 0.95  # eps = 0.95 / sup|g|^2 keeps the joint density positive


# ------------------------------------------------------------------ hidden dependence construction (numpy only)
def _feats(x, terms=6):
    """phi_k(x) = exp(-x^2/2) x^k / sqrt(k!) (pyspi default bandwidth 0.5 = paper B = 1)."""
    x = np.asarray(x, float)
    return np.stack([np.exp(-0.5 * x**2) * x**k / math.sqrt(math.factorial(k)) for k in range(terms)])


def hidden_construction(n_nodes=150):
    """Bounded g with E[g] = 0, E[g phi_k] = 0 (k = 0..5) and E[g^2] = 1 under N(0, 1).

    Joint density p(x, y) = phi(x) phi(y) (1 + eps g(x) g(y)): marginals stay N(0, 1), every
    Cov(phi_k(X), phi_l(Y)) = eps E[g phi_k] E[g phi_l] = 0, so the population IDS_p is 0 for every p,
    while Corr(g(X), g(Y)) = eps > 0 (dependent).  Projection by weighted least squares on
    probabilists' Gauss-Hermite nodes.  Returns (coef, scale, eps, sup_g, ortho_resid)."""
    from numpy.polynomial.hermite_e import hermegauss
    t, w = hermegauss(n_nodes)
    w = w / w.sum()
    Bm = np.vstack([np.ones_like(t), _feats(t)])                  # 7 x n basis
    g0 = np.cos(HIDDEN_OMEGA * t)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq((Bm * sw).T, g0 * sw, rcond=None)
    r = g0 - coef @ Bm
    scale = float(np.sqrt(np.sum(w * r**2)))
    gn = r / scale
    ortho = float(np.max(np.abs((Bm * w) @ gn)))
    grid = np.linspace(-14.0, 14.0, 280001)
    sup_g = float(np.max(np.abs(hidden_g(grid, coef, scale))))
    eps = HIDDEN_EPS_FRACTION / sup_g**2
    return coef, scale, eps, sup_g, ortho


def hidden_g(x, coef, scale):
    x = np.asarray(x, float)
    xf = x.ravel()
    Bm = np.vstack([np.ones_like(xf), _feats(xf)])
    return ((np.cos(HIDDEN_OMEGA * xf) - coef @ Bm) / scale).reshape(x.shape)


def hidden_sample(T, seed):
    """Rejection sample of T pairs from phi(x) phi(y)(1 + eps g(x) g(y)); returns (2, T)."""
    coef, scale, eps, sup_g, _ = hidden_construction()
    rng = np.random.default_rng(int(seed) * 7919 + 17)
    bound = 1.0 + eps * sup_g**2
    out = []
    n = 0
    while n < T:
        xy = rng.standard_normal((2, 4 * T))
        acc = (1.0 + eps * hidden_g(xy[0], coef, scale) * hidden_g(xy[1], coef, scale)) / bound
        keep = rng.uniform(size=4 * T) < acc
        out.append(xy[:, keep])
        n += int(keep.sum())
    return np.concatenate(out, axis=1)[:, :T]


# ------------------------------------------------------------------ helpers (pyspi imported lazily)
def _data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _tag(run):
    return run["run_id"].split("|")[-1]


def _nandiag(S):
    S = np.array(S, dtype=np.float64, copy=True)
    np.fill_diagonal(S, np.nan)
    return S


def _ids(x, path, **kw):
    from pyspi.statistics.misc import InterDependenceScore
    spi = InterDependenceScore(**kw)
    assert spi.identifier == "ids"
    return _nandiag(spi.multivariate(_data(x, path)))


def ids_param(x, run):
    tag = _tag(run)
    kw = {}
    if tag.startswith("eq6-bw"):
        kw["bandwidth"] = float(tag[len("eq6-bw"):])
    elif tag.startswith("pnorm"):
        kw["pnorm"] = int(tag[len("pnorm"):])
    else:
        raise ValueError(tag)
    return _ids(x, run.get("path", "kernel"), **kw)


def transform_input(x, tag):
    """Shared with refs/ids.py (numpy only)."""
    x = np.array(x, dtype=np.float64, copy=True)
    if tag == "signflip":
        x[1] = -x[1]
        return x
    if tag == "identical":
        x0 = x[0]
        return np.stack([x0, x0.copy(), -x0, 2.5 * x0 - 1.0])
    if tag == "minmax08":
        lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
        return 8.0 * (x - lo) / (hi - lo)
    raise ValueError(tag)


def ids_transformed(x, run):
    return _ids(transform_input(x, _tag(run)), run.get("path", "pipeline"))


def hidden_dependence(x, run):
    return _ids(hidden_sample(run["T"], run.get("seed", 42)), run.get("path", "kernel"))


def pnorm_calculator(x, run):
    """x: (4, T).  P = 6 adds x0^2 and x1^2 as rows.  Returns (len(PNORM_CASES), len(PNORM_FIELDS))."""
    import warnings
    from pyspi.calculator import Calculator
    from pyspi.lib.ids.dependence import compute_IDS
    x = np.asarray(x, float)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for pnorm, P in PNORM_CASES:
            xx = x if P == 4 else np.vstack([x, x[:2] ** 2])
            assert xx.shape[0] == P
            cfg = os.path.join(td, "cfg.yaml")
            pn = f"'{pnorm}'" if isinstance(pnorm, str) else str(pnorm)
            with open(cfg, "w") as f:
                f.write(".statistics.misc:\n  InterDependenceScore:\n    configs:\n"
                        f"      - terms: 6\n        pnorm: {pn}\n        bandwidth: 0.5\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calc = Calculator(dataset=xx, configfile=cfg)
                calc.compute()
            block = calc.table["ids"].to_numpy()
            rep = calc.metric_reports["ids"]
            err = str(rep.error or "")
            z = (xx - xx.mean(1, keepdims=True)) / xx.std(1, ddof=1, keepdims=True)
            direct = compute_IDS(z.T, num_terms=6, p_norm=pnorm, bandwidth_term=0.5)
            rows.append([float(np.isnan(block).all()), float(rep.status == "failed"),
                         float("ValueError" in err), float("AssertionError" in err), float(np.ndim(direct))])
    return np.array(rows)


# ------------------------------------------------------------------ RNG side effects (fresh interpreters)
_RNG_IMPORT_SRC = r"""
import json, sys, numpy as np
x = np.load(sys.argv[1])
out = {}
np.random.seed(123); out["draw_seed123_no_import"] = float(np.random.random())
np.random.seed(123)
import pyspi.calculator  # package + Calculator module only: does not import the statistics modules
out["draw_after_import_pyspi_calculator_only"] = float(np.random.random())
np.random.seed(123)
import pyspi.lib.ids.numpy_dependence as nd
st = np.random.get_state()
np.random.seed(1717); ref = np.random.get_state(); np.random.set_state(st)
out["state_eq_seed1717_after_import"] = float(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                                                 for a, b in zip(st, ref)))
out["draw_after_import_seeded123"] = float(np.random.random())
z = ((x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)).T
from pyspi.lib.ids.dependence import compute_IDS
A = compute_IDS(z, num_terms=6, p_norm="max", bandwidth_term=0.5)
out["draw_after_ids_compute"] = float(np.random.random())
np.random.seed(1); A1 = compute_IDS(z); np.random.seed(2); A2 = compute_IDS(z)
out["ids_diff_seed1_vs_seed2"] = float(np.max(np.abs(A1 - A2)))
print("JSON" + json.dumps(out))
"""

_RNG_CALC_SRC = r"""
import json, sys, numpy as np, warnings
x = np.load(sys.argv[1]); cfg = sys.argv[2]
warnings.simplefilter("ignore")
from pyspi.calculator import Calculator
out = {}
np.random.seed(123)
Calculator(dataset=x, configfile=cfg)
st = np.random.get_state()
np.random.seed(1717); ref = np.random.get_state(); np.random.set_state(st)
out["state_eq_seed1717_after_first_Calculator"] = float(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                                                           for a, b in zip(st, ref)))
out["draw_after_first_Calculator_seeded123"] = float(np.random.random())
np.random.seed(5)
Calculator(dataset=x, configfile=cfg)
out["draw_after_second_Calculator_seeded5"] = float(np.random.random())
print("JSON" + json.dumps(out))
"""

_RNG_CROSS_SRC = r"""
import json, sys, numpy as np, warnings
x = np.load(sys.argv[1]); cfg = sys.argv[2]; seed = int(sys.argv[3]); when = sys.argv[4]
warnings.simplefilter("ignore")
from pyspi.calculator import Calculator
if when == "before":
    np.random.seed(seed)
calc = Calculator(dataset=x, configfile=cfg)
if when == "after":
    np.random.seed(seed)
calc.compute()
v = calc.table["cov_MinCovDet"].to_numpy()
print("JSON" + json.dumps({"mcd": float(np.nansum(np.abs(v)))}))
"""

_CFG_MCD = ".statistics.basic:\n  Covariance:\n    configs:\n      - estimator: MinCovDet\n"
_CFG_IDS = (".statistics.misc:\n  InterDependenceScore:\n    configs:\n"
            "      - terms: 6\n        pnorm: 'max'\n        bandwidth: 0.5\n")


def _fresh(src, *args):
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "probe.py")
        with open(script, "w") as f:
            f.write(src)
        env = dict(os.environ)
        r = subprocess.run([sys.executable, script, *[str(a) for a in args]], capture_output=True,
                           text=True, cwd=td, env=env, timeout=300)
    lines = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"fresh interpreter failed rc={r.returncode}: {r.stderr[-1500:]}")
    return json.loads(lines[-1][4:])


def _save_x(td, x):
    p = os.path.join(td, "x.npy")
    np.save(p, np.asarray(x, float))
    return p


def rng_import(x, run):
    with tempfile.TemporaryDirectory() as td:
        out = _fresh(_RNG_IMPORT_SRC, _save_x(td, x))
    return np.array([out[k] for k in RNG_IMPORT_KEYS])


def rng_calculator(x, run):
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "cfg.yaml")
        with open(cfg, "w") as f:
            f.write(_CFG_IDS)
        out = _fresh(_RNG_CALC_SRC, _save_x(td, x), cfg)
    return np.array([out[k] for k in RNG_CALC_KEYS])


def rng_cross_spi(x, run):
    """cov_MinCovDet summary sum|A| (off-diagonal) under several seeding / config variants."""
    with tempfile.TemporaryDirectory() as td:
        xp = _save_x(td, x)
        c_with = os.path.join(td, "with_ids.yaml")     # basic before misc (pyspi config.yaml order)
        c_first = os.path.join(td, "ids_first.yaml")   # misc before basic: ids computed before MinCovDet
        c_no = os.path.join(td, "no_ids.yaml")
        with open(c_with, "w") as f:
            f.write(_CFG_MCD + _CFG_IDS)
        with open(c_first, "w") as f:
            f.write(_CFG_IDS + _CFG_MCD)
        with open(c_no, "w") as f:
            f.write(_CFG_MCD)
        spec = {"mcd_with_ids_seed1_before": (c_with, 1, "before"),
                "mcd_with_ids_seed2_before": (c_with, 2, "before"),
                "mcd_noids_seed1717_before": (c_no, 1717, "before"),
                "mcd_noids_seed1_before": (c_no, 1, "before"),
                "mcd_noids_seed2_before": (c_no, 2, "before"),
                "mcd_idsfirst_seed7_after": (c_first, 7, "after"),
                "mcd_noids_seed7_after": (c_no, 7, "after")}
        out = {k: _fresh(_RNG_CROSS_SRC, xp, *spec[k])["mcd"] for k in RNG_CROSS_KEYS}
    return np.array([out[k] for k in RNG_CROSS_KEYS])

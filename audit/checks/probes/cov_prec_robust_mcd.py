"""cov_prec_robust probes (PROJECT venv): expose the sklearn MinCovDet / EllipticEnvelope fit that
pyspi caches in ``data.covariance[estimator]`` (pyspi/statistics/basic.py:31-44) and run the
seed / RNG-stream studies the plain SPI path cannot express.

Every function has the pyspi_outputs probe signature ``fn(x, run)``; ``x`` is the raw (P, T)
dataset and ``run["path"]`` selects the pyspi preprocessing exactly as pyspi_outputs does.
The estimator ("MinCovDet" or "EllipticEnvelope") and the kind ("cov" -> Covariance,
"prec" -> Precision) are read from ``run["unit_id"]`` (basic.<kind>.<estimator>).
pyspi_outputs calls ``harness.seed_all(run["seed"])`` immediately before the probe.

* ``fit_state``   - one SPI call exactly as the plain SPI run (same seed, same RNG consumption:
                    the fit happens in ``_from_cache`` before anything else), then the fitted
                    attributes copied BEFORE ``multivariate`` NaNs the cached diagonal.
                    Returns a flat float vector (layout below, decoded by refs/cov_prec_robust.py).
* ``seeds``       - for s = 0..NSEEDS-1: ``np.random.seed(s)``, fresh Data, the SPI. Returns an
                    (NSEEDS, P*P + 2) array: row s = [SPI output (P*P, NaN diagonal),
                    log det(raw_covariance_), raw support size]. ``fresh-process`` runs are
                    equivalent to unseeded draws from OS entropy, so a seed sweep stands in for
                    the "20 fresh-process Calculator runs" of the record.
* ``determinism`` - (3, P, P): [A1, A2, A3]. A1 and A2 are two calls each preceded by
                    harness.seed_all(seed) on a fresh Data; A3 is seed_all(seed), then the OTHER
                    robust estimator's SPI (another global-RNG-consuming SPI, as in a Calculator
                    pass where cov_EllipticEnvelope precedes cov_MinCovDet), then this SPI on the
                    same Data object.
* ``robust_corr`` - (P, P): C_ij / sqrt(C_ii C_jj) of the cached reweighted covariance_ (NaN
                    diagonal); pyspi NaNs the diagonal of its output, so the diagonal is read from
                    the fit before that happens.

fit_state layout (all float64, flat):
    [T, P, h_raw, n_reweighted,
     raw_support (T, 0/1), support_ (T, 0/1), raw_location_ (P), raw_covariance_ (P*P),
     location_ (P), covariance_ (P*P, full diagonal), precision_ (P*P, full diagonal),
     dist_ (T), input z (P*T, the (P, T) array the estimator saw, row-major),
     SPI output (P*P, NaN diagonal)]
"""
import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "audit" / "checks"))

from regression import harness  # noqa: E402

NSEEDS = 20


def _kind_est(run):
    _, kind, est = run["unit_id"].split(".")
    return kind, est


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


def _spi(kind, est, squared=False):
    from pyspi.statistics.basic import Covariance, Precision
    cls = Covariance if kind == "cov" else Precision
    return cls(estimator=est, squared=squared)


def _out(spi, data):
    return harness.compute_spi(spi, data)


def fit_state(x, run):
    kind, est = _kind_est(run)
    data = _data(x, run["path"])
    spi = _spi(kind, est)
    fit = spi._from_cache(data)             # the fit multivariate() would make (same RNG draws)
    snap = {k: copy.deepcopy(getattr(fit, k)) for k in
            ("raw_support_", "support_", "raw_location_", "raw_covariance_", "location_",
             "covariance_", "precision_", "dist_")}
    S = _out(spi, data)                     # served from the cache: no further RNG use
    z = data.to_numpy(squeeze=True)
    P, T = z.shape
    parts = [np.array([T, P, snap["raw_support_"].sum(), snap["support_"].sum()], float),
             snap["raw_support_"].astype(float), snap["support_"].astype(float),
             snap["raw_location_"], snap["raw_covariance_"].ravel(), snap["location_"],
             snap["covariance_"].ravel(), snap["precision_"].ravel(), snap["dist_"],
             z.ravel(), S.ravel()]
    return np.concatenate([np.asarray(p, float).ravel() for p in parts])


def seeds(x, run):
    kind, est = _kind_est(run)
    rows = []
    for s in range(NSEEDS):
        np.random.seed(s)
        data = _data(x, run["path"])
        spi = _spi(kind, est)
        fit = spi._from_cache(data)
        raw_logdet = np.linalg.slogdet(fit.raw_covariance_)[1]
        hraw = float(fit.raw_support_.sum())
        S = _out(spi, data)
        rows.append(np.concatenate([S.ravel(), [raw_logdet, hraw]]))
    return np.array(rows)


def determinism(x, run):
    kind, est = _kind_est(run)
    other = "EllipticEnvelope" if est == "MinCovDet" else "MinCovDet"
    seed = int(run.get("seed", 42))
    out = []
    for _ in range(2):
        harness.seed_all(seed)
        out.append(_out(_spi(kind, est), _data(x, run["path"])))
    harness.seed_all(seed)
    data = _data(x, run["path"])
    _out(_spi(kind, other), data)           # other robust SPI first: consumes the global RNG
    out.append(_out(_spi(kind, est), data))
    return np.stack(out)


def robust_corr(x, run):
    kind, est = _kind_est(run)
    data = _data(x, run["path"])
    spi = _spi(kind, est)
    fit = spi._from_cache(data)
    C = np.array(fit.covariance_, dtype=np.float64, copy=True)
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    np.fill_diagonal(R, np.nan)
    _out(spi, data)
    return R


# ---------------------------------------------------------------- fresh-process reproducibility
_FRESH_SCRIPT = r'''
import json, sys
import numpy as np
mode, xpath, cfg, out = sys.argv[1:5]
x = np.load(xpath)
from pyspi.calculator import Calculator
if mode == "full":
    # default config.yaml: _load_yaml imports every statistics module, including pyspi.statistics.misc, whose
    # import runs np.random.seed(1717) (pyspi/lib/ids/numpy_dependence.py:11-12); only the .statistics.basic
    # SPIs (computed first, in config order) are then kept, so the RNG sequence up to them is the full-config one
    calc = Calculator(dataset=x)
    for k in [k for k, v in calc._spis.items() if type(v).__module__ != "pyspi.statistics.basic"]:
        del calc._spis[k]
else:
    calc = Calculator(dataset=x, configfile=cfg)   # robust SPIs only: misc never imported, RNG unseeded
vals = {}
calc.compute(on_metric_complete=lambda rep, v: vals.__setitem__(rep.identifier, None if v is None else np.asarray(v).tolist()))
json.dump(vals, open(out, "w"))
'''

_BASIC_CFG = """.statistics.basic:
  Covariance:
    labels: [undirected, linear, signed, multivariate, contemporaneous]
    dependencies:
    configs:
      - estimator: EllipticEnvelope
      - estimator: MinCovDet
      - estimator: EllipticEnvelope
        squared: True
      - estimator: MinCovDet
        squared: True
  Precision:
    labels: [undirected, linear, signed, multivariate, contemporaneous]
    dependencies:
    configs:
      - estimator: EllipticEnvelope
      - estimator: MinCovDet
      - estimator: EllipticEnvelope
        squared: True
      - estimator: MinCovDet
        squared: True
"""


def fresh_process(x, run):
    """(2, P, P): the unit's unsquared identifier from two FRESH interpreter processes, each running a
    Calculator. Tag ``freshfull``: default config (misc imported -> np.random.seed(1717)), basic block only.
    Tag ``freshbasic``: a config with only the 8 robust SPIs (no misc import, global RNG seeded by the OS)."""
    import json
    import subprocess
    import tempfile
    kind, est = _kind_est(run)
    ident = f"{kind}_{est}"
    mode = "full" if run["run_id"].endswith("freshfull") else "basic"
    exch = REPO / "audit" / "checks" / "out" / "exchange" / "cov_prec_robust"
    exch.mkdir(parents=True, exist_ok=True)
    outs = []
    with tempfile.TemporaryDirectory(dir=exch) as td:
        td = Path(td)
        np.save(td / "x.npy", np.asarray(x, float))
        (td / "basic.yaml").write_text(_BASIC_CFG)
        (td / "run.py").write_text(_FRESH_SCRIPT)
        for k in range(2):
            o = td / f"out{k}.json"
            subprocess.run([sys.executable, str(td / "run.py"), mode, str(td / "x.npy"), str(td / "basic.yaml"), str(o)],
                           check=True, capture_output=True, text=True, cwd=str(td))
            outs.append(np.array(json.loads(o.read_text())[ident], dtype=float))
    return np.stack(outs)


def aliasing(x, run):
    """[returned_is_cached_attribute, cached_diag_all_nan_after_call, other_attribute_diag_finite,
    write_through_to_cache, next_call_returns_edited_value] for the unit's unsquared SPI (basic.py:46-54)."""
    kind, est = _kind_est(run)
    data = _data(x, run["path"])
    spi = _spi(kind, est)
    S = spi.multivariate(data)                       # raw return, before any Calculator post-processing
    fit = data.covariance[est]
    attr = fit.covariance_ if kind == "cov" else fit.precision_
    other = fit.precision_ if kind == "cov" else fit.covariance_
    flags = [S is attr, bool(np.isnan(np.diag(attr)).all()), bool(np.isfinite(np.diag(other)).all())]
    S[0, 1] = 123.0
    flags.append(bool(attr[0, 1] == 123.0))
    flags.append(bool(spi.multivariate(data)[0, 1] == 123.0))
    return np.array(flags, dtype=float)

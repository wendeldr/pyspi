"""Project-venv probes for the coint family (Cointegration, pyspi/statistics/misc.py:22-103).

Each function has the probe signature fn(x, run) -> ndarray (pyspi_outputs.run_probe). x is the
raw (P, T) dataset after optional process selection; run is the manifest entry.

calc        Calculator.compute (pyspi/calculator.py:295, :327-339) on a config holding all eleven
            Cointegration configs; returns (1 + P, P): row 0 = [failed, err_code, all_offdiag_nan,
            n_failed_of_11] (err_code 1 'Singular matrix', 2 'not positive definite', 3 other,
            0 none), rows 1..P = the stored table block of run['identifier'].
onefit      counts statsmodels coint_johansen calls (misc.coint_johansen) while computing the
            max_eig_stat and then the trace_stat identifier of the run's Johansen config on ONE Data;
            returns [calls after first SPI, calls after second, P(P-1)/2,
            max|second - second-on-fresh-Data|].
collinear   pyspi SPI (run['identifier']) on a 2-channel kernel-path Data [z, 3 z + 1], z one
            Data-normalised channel of {cointegrated, iid_gaussian, trending, shared_envelope} x
            T {256, 1024} x channel {0, 1}; rows [ds_idx, T, ch, code, value, eig1, eig2]
            (code 0 returned, 1 LinAlgError 'Singular matrix', 2 'not positive definite',
            3 other; eig from coint_johansen on pyspi's column order [3z+1, z]; NaN for aeg).
cplx        counts complex128 results of numpy.linalg.eig and ComplexWarnings while the run's
            Johansen SPI runs on {cointegrated, iid_gaussian, trending, var2_bivariate,
            delayed_noise} x {256, 1024} (pipeline path); returns [n_eig_calls, n_complex_dtype,
            max|imag eigenvalue|, n_ComplexWarning, n_other_warnings].
meta        [has 'nonlinear' label, has 'linear' label, issigned(), code of
            Cointegration(method='Johansen').bivariate, code of Cointegration(method='AEG').bivariate]
            (code 0 no error, 1 AttributeError naming _det_order, 2 other error).
api         2-channel Data (pipeline path): [bivariate(D) (i=0, j=1), multivariate(D')[0,1],
            multivariate(D')[1,0], multivariate(D'')[0,1] after bivariate(D'') on the same Data].
crit        statsmodels' critical-value tables for the run's test (Johansen r = 0 trace / max-eig rows;
            MacKinnon 2010 AEG values at T in 64, 256, 1024, 4096).
size        null Monte Carlo through pyspi: SIZE_REPS independent random-walk pairs, pipeline Data,
            A[0, 1] per replicate (both Johansen statistics of the config, or the aeg t-statistic).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parent.parent
REPO = CHECKS.parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(CHECKS))
EXCH = CHECKS / "out" / "exchange" / "coint"


def _entry(identifier):
    import yaml
    doc = yaml.safe_load((CHECKS.parent / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u["identifiers"]:
            return dict(identifier=identifier, module=f".statistics.{u['module']}", cls=u["pyspi_class"],
                        params=u["identifier_params"][identifier], labels=u["labels"], deps=u["deps"])
    raise KeyError(identifier)


def _spi(identifier):
    from regression import harness
    return harness.build_spi(_entry(identifier))


def _err_code(msg):
    if "Singular matrix" in msg:
        return 1
    if "not positive definite" in msg:
        return 2
    return 3


def calc(x, run):
    from pyspi.calculator import Calculator
    EXCH.mkdir(parents=True, exist_ok=True)
    cfg = EXCH / "calc_config_coint.yaml"
    src = (REPO / "pyspi" / "config.yaml").read_text().splitlines()
    i0 = next(k for k, l in enumerate(src) if l.strip() == "Cointegration:")
    i1 = next(k for k in range(i0 + 1, len(src)) if src[k].startswith("  # Power envelope"))
    cfg.write_text("\n".join([".statistics.misc:"] + src[i0:i1]) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Calculator(dataset=np.asarray(x, float), configfile=str(cfg))
        c.compute()
    P = x.shape[0]
    rep = c.metric_reports[run["identifier"]]
    M = c.table[run["identifier"]].to_numpy(dtype=float)
    off = ~np.eye(P, dtype=bool)
    n_failed = sum(r.status == "failed" for r in c.metric_reports.values())
    head = np.full(P, np.nan)
    head[:4] = [float(rep.status == "failed"), 0.0 if rep.error is None else _err_code(rep.error.message),
                float(np.isnan(M[off]).all()), float(n_failed)]
    return np.vstack([head, M])


def onefit(x, run):
    from pyspi.data import Data
    import pyspi.statistics.misc as misc
    ident_tr = run["identifier"]
    ident_me = ident_tr.replace("trace_stat", "max_eig_stat")
    orig = misc.coint_johansen
    n = [0]

    def counting(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    misc.coint_johansen = counting
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            D = Data(np.asarray(x, float), dim_order="ps")
            _spi(ident_me).multivariate(D)
            c1 = n[0]
            A2 = _spi(ident_tr).multivariate(D)
            c2 = n[0]
            B2 = _spi(ident_tr).multivariate(Data(np.asarray(x, float), dim_order="ps"))
    finally:
        misc.coint_johansen = orig
    P = x.shape[0]
    off = ~np.eye(P, dtype=bool)
    return np.array([c1, c2, P * (P - 1) / 2, float(np.abs(A2[off] - B2[off]).max())])


def collinear(x, run):
    from pyspi.data import Data
    import make_data
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    spi_params = _entry(run["identifier"])["params"]
    rows = []
    for di, ds in enumerate(["cointegrated", "iid_gaussian", "trending", "shared_envelope"]):
        for T in (256, 1024):
            X = np.asarray(make_data.load(ds, T)["x"], float)
            Z = Data(X.copy(), dim_order="ps").to_numpy(squeeze=True)
            for ch in (0, 1):
                z = Z[ch]
                Y = np.vstack([z, 3 * z + 1])
                code, val = 0, np.nan
                e1 = e2 = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        S = _spi(run["identifier"]).multivariate(Data(Y.copy(), dim_order="ps", normalise=False, detrend=False))
                        val = float(S[0, 1])
                    except Exception as exc:  # noqa: BLE001 - the outcome is the measurement
                        code = _err_code(str(exc))
                    if spi_params["method"] == "johansen":
                        try:
                            r = coint_johansen(Y[[1, 0]].T, det_order=spi_params["det_order"],
                                               k_ar_diff=spi_params["k_ar_diff"])
                            e1, e2 = float(np.real(r.eig[0])), float(np.real(r.eig[1]))
                        except Exception:  # noqa: BLE001
                            pass
                rows.append([di, T, ch, code, val, e1, e2])
    return np.array(rows, float)


def cplx(x, run):
    from pyspi.data import Data
    import make_data
    orig = np.linalg.eig
    st = dict(n=0, nc=0, imag=0.0)

    def wrapped(a):
        w, v = orig(a)
        st["n"] += 1
        if np.iscomplexobj(w):
            st["nc"] += 1
            st["imag"] = max(st["imag"], float(np.abs(np.imag(w)).max()))
        return w, v

    ncw = nother = 0
    np.linalg.eig = wrapped
    try:
        for ds in ["cointegrated", "iid_gaussian", "trending", "var2_bivariate", "delayed_noise"]:
            for T in (256, 1024):
                X = np.asarray(make_data.load(ds, T)["x"], float)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _spi(run["identifier"]).multivariate(Data(X, dim_order="ps"))
                for ww in w:
                    if issubclass(ww.category, np.exceptions.ComplexWarning):
                        ncw += 1
                    else:
                        nother += 1
    finally:
        np.linalg.eig = orig
    return np.array([st["n"], st["nc"], st["imag"], ncw, nother], float)


def meta(x, run):
    from pyspi.data import Data
    from pyspi.statistics.misc import Cointegration
    c = Cointegration()
    out = [float("nonlinear" in c.labels), float("linear" in c.labels), float(c.issigned())]
    D = Data(np.asarray(x, float)[:2], dim_order="ps")
    for m in ("Johansen", "AEG"):
        try:
            Cointegration(method=m).bivariate(D)
            out.append(0.0)
        except AttributeError as exc:
            out.append(1.0 if "_det_order" in str(exc) else 2.0)
        except Exception:  # noqa: BLE001
            out.append(2.0)
    return np.array(out)


def api(x, run):
    from pyspi.data import Data
    X = np.asarray(x, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = _spi(run["identifier"]).bivariate(Data(X.copy(), dim_order="ps"))
        M = _spi(run["identifier"]).multivariate(Data(X.copy(), dim_order="ps"))
        D3 = Data(X.copy(), dim_order="ps")
        _spi(run["identifier"]).bivariate(D3)
        M3 = _spi(run["identifier"]).multivariate(D3)
    return np.array([b, M[0, 1], M[1, 0], M3[0, 1]], float)


def crit(x, run):
    """Critical values statsmodels attaches to the run's test (literature tables, not computed from x).

    Johansen identifiers: coint_johansen(...).cvt[0] and .cvm[0] (r = 0 row, [90%, 95%, 99%]) from
    statsmodels/tsa/coint_tables.py c_sjt / c_sja (LeSage's tables after MacKinnon, Haug & Michelis);
    returns shape (2, 3) = [trace row, max-eig row].
    AEG identifiers: statsmodels.tsa.adfvalues.mackinnoncrit(N=2, regression=trend, nobs=T-1) for
    T in (64, 256, 1024, 4096) (the call stattools.coint makes, _stattools.py:2711-2713, MacKinnon 2010
    response surface); returns shape (4, 3) = [1%, 5%, 10%] per T."""
    p = _entry(run["identifier"])["params"]
    if p["method"] == "johansen":
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = coint_johansen(np.asarray(x, float)[:2].T, det_order=p["det_order"], k_ar_diff=p["k_ar_diff"])
        return np.vstack([r.cvt[0], r.cvm[0]])
    from statsmodels.tsa.adfvalues import mackinnoncrit
    return np.vstack([mackinnoncrit(N=2, regression=p["trend"], nobs=T - 1) for T in (64, 256, 1024, 4096)])


SIZE_REPS = 400


def size_seed(T):
    return 20261015 + int(T)


def size(x, run):
    """Null Monte Carlo through pyspi itself: SIZE_REPS pairs of independent Gaussian random walks of
    length run['T'] (np.random.default_rng(size_seed(T)), walks = cumsum(standard_normal((2, T)), axis=1)
    drawn rep by rep), each wrapped in a pipeline Data (z-score) and passed to the SPI's multivariate;
    the stored value is A[0, 1] (P = 2: the first call is bivariate(i=1, j=0), so y0 = walk 1 for aeg).
    Johansen identifiers: columns [max_eig_stat, trace_stat] from the two SPIs of the config on the
    same Data; aeg: one column. x is ignored (the draws are simulated here and re-simulated by the
    reference with the same generator)."""
    from pyspi.data import Data
    ident = run["identifier"]
    p = _entry(ident)["params"]
    idents = [ident] if p["method"] == "aeg" else [ident.replace("trace_stat", "max_eig_stat"),
                                                    ident.replace("max_eig_stat", "trace_stat")]
    spis = [_spi(i) for i in idents]
    T = int(run["T"])
    rng = np.random.default_rng(size_seed(T))
    out = np.full((SIZE_REPS, len(spis)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(SIZE_REPS):
            w = np.cumsum(rng.standard_normal((2, T)), axis=1)
            D = Data(w.copy(), dim_order="ps")
            for c, s in enumerate(spis):
                out[r, c] = s.multivariate(D)[0, 1]
    return out

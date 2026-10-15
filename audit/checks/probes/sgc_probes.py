"""Probes for the sgc family (PROJECT venv: pyspi, spectral_connectivity 1.1.2, nitime 0.12.1).

Each function is called as ``fn(x, run)`` by ``audit/checks/pyspi_outputs.py``
(x = raw dataset array (P, T) after optional process selection) and returns
the array stored under ``value`` in ``out/pyspi/<run_id>.npz``.  Where an SPI is
needed it is built from ``run["identifier"]`` (a catalogue identifier of
audit/units.yaml) exactly as pyspi_outputs does, so every probe exercises the
configured instance.  Preprocessing goes through ``pyspi.data.Data`` with the
run's ``path`` (kernel = raw, pipeline = z-score ddof=1, pipeline-detrend =
linear detrend then z-score).

Outputs (shapes; refs/sgc.py relies on these layouts):

* ``cache``      pyspi's per-frequency cache ``F[0]`` from ``SPI._get_cache``:
                 (n_freq, P, P), [f, a, b] = per-bin value pyspi reduces into
                 A[a, b].  Nonparametric: (T+1)//2 bins k/T (library grid);
                 parametric: 513 points labelled k/1024.
* ``lib_csm``    two-sided taper-mean cross-spectral matrix of the library
                 Multitaper/Connectivity pyspi builds (spectral.py:69-71),
                 real/imag stacked: (2, n_fft, P, P).
* ``exact_lib``  spectral_connectivity's pairwise_spectral_granger_prediction fed
                 with an EXACT two-sided VAR spectrum through P pseudo-tapers
                 X_m(f) = sqrt(P) H(f) L e_m (taper mean = H Sigma H^H); truth
                 (A_k, Sigma) from make_data; n_fft = run["T"].  (n_fft//2, P, P).
* ``nitime_model`` nitime GrangerAnalyzer fit per pair (i < j), rows
                 [i, j, order, ecov(4, row-major), coef(order*4, row-major)]
                 NaN-padded to 7 + 4*48 columns.
* ``orders``     nitime's chosen order per pair: (P, P), symmetric, NaN diag.
* ``maxorder``   nitime fit_model(x0, x1, order=None, max_order=m) for
                 m = 2..9: returned order, or -1 when it raises ValueError.  (8,)
* ``grid``       (2, 513): row 0 = GA.frequencies (labels), row 1 = the
                 frequencies nitime evaluates, w/(2 pi) from
                 nitime.algorithms.spectral.freq_response (scipy.signal.freqz).
* ``outcome``    error-path record: [code_direct, nanfrac_catch_always,
                 code_delegate, fail_i, fail_j, n_regex_filters, nanfrac_calculator];
                 codes 0 ok, 1 AttributeError, 2 TypeError, 3 ValueError
                 (incl. LinAlgError), 4 other.  nanfrac_calculator comes from a REAL
                 pyspi Calculator run with a one-SPI config file.  Messages go to
                 out/exchange/sgc/<sanitised run_id>.json.
* ``sweep``      outcome code for every dataset of data/MANIFEST.json at run["T"]
                 plus constructed EEG-like inputs, (n_cases, 3):
                 [code_direct, nanfrac_of_output, n_processes]; code 5 = pyspi.data.Data
                 rejected the input before the SPI ran; names/messages in the exchange JSON.
* ``affine``     pyspi matrix on a per-channel transform of x with the run's path
                 (tag token ``affine`` = x*[2,0.5]+[3,-5], ``scale`` = x*[2,0.5]); (P, P).
* ``ntot``       200 borderline VAR(2) draws generated in the probe (x unused, T = run["T"]),
                 (200, 6): [a2, nitime order, sum(z), pyspi A[1,0], pyspi A[0,1], sum(raw)].
"""
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, "/media/dan/Data/git/pyspi/tests")
import make_data  # noqa: E402

EXCH = CHECKS / "out" / "exchange" / "sgc"
SCALE = np.array([2.0, 0.5])
OFFSET = np.array([3.0, -5.0])


# ------------------------------------------------------------------ helpers
def _entry(run):
    import pyspi_outputs as po
    _, by_id = po.load_units()
    return po.entry_for(run["identifier"], by_id)


def _spi(run):
    from regression import harness
    return harness.build_spi(_entry(run))


def _data(x, path):
    import pyspi_outputs as po
    return po.make_pyspi_data(x, path)


def _code(exc):
    if exc is None:
        return 0
    if isinstance(exc, AttributeError):
        return 1
    if isinstance(exc, TypeError):
        return 2
    if isinstance(exc, ValueError):
        return 3
    return 4


def _sidecar(run, obj):
    EXCH.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run["run_id"])
    (EXCH / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")


def _n_regex_filters():
    return sum(1 for f in warnings.filters if f[1] is not None)


def _ga(z, spi):
    import nitime.analysis as nta
    import nitime.timeseries as ts
    return nta.GrangerAnalyzer(ts.TimeSeries(z, sampling_interval=1), order=spi._order,
                               max_order=spi._max_order)


# ------------------------------------------------------------------ probes
def cache(x, run):
    spi = _spi(run)
    F, _ = spi._get_cache(_data(x, run.get("path", "pipeline")))
    return np.asarray(F[0], dtype=np.float64)


def lib_csm(x, run):
    from pyspi._spectral_compat import spectral_connectivity as sc
    z = np.transpose(_data(x, run.get("path", "pipeline")).to_numpy(squeeze=True))
    conn = sc.Connectivity.from_multitaper(sc.Multitaper(z, sampling_frequency=1))
    S = np.asarray(conn._expectation_cross_spectral_matrix())
    S = S.reshape(S.shape[-3:])
    return np.stack([S.real, S.imag])


def exact_lib(x, run):
    from pyspi._spectral_compat import spectral_connectivity as sc
    ds = make_data.load(run["dataset"], run["T"])
    if "A1" in ds:
        A_list, Sigma = [ds["A1"], ds["A2"]], ds["Sigma"]
    else:
        A_list, Sigma = [ds["A"]], ds["Sigma_E"]
    n = int(run["T"])
    f = np.fft.fftfreq(n)
    P = Sigma.shape[0]
    M = np.broadcast_to(np.eye(P, dtype=complex), (n, P, P)).copy()
    for k, A in enumerate(A_list, start=1):
        M -= np.asarray(A)[None] * np.exp(-2j * np.pi * f * k)[:, None, None]
    H = np.linalg.inv(M)
    L = np.linalg.cholesky(Sigma)
    X = np.stack([np.sqrt(float(P)) * (H @ L[:, m]) for m in range(P)])   # (P tapers, n, P)
    conn = sc.Connectivity(fourier_coefficients=X[None, None])            # (1, 1, K, n, P)
    gc = np.asarray(conn.pairwise_spectral_granger_prediction())
    return gc.reshape(gc.shape[-3:])


def nitime_model(x, run):
    spi = _spi(run)
    GA = _ga(_data(x, run.get("path", "pipeline")).to_numpy(squeeze=True), spi)
    rows = []
    for (i, j) in sorted(GA.ij):
        coef, ecov, order = GA.model_coef[i, j], GA.error_cov[i, j], GA.order[i, j]
        r = np.full(7 + 4 * 48, np.nan)
        r[:3] = (i, j, order)
        r[3:7] = np.asarray(ecov).ravel()
        c = np.asarray(coef).ravel()
        r[7:7 + c.size] = c
        rows.append(r)
    return np.array(rows)


def orders(x, run):
    spi = _spi(run)
    z = _data(x, run.get("path", "pipeline")).to_numpy(squeeze=True)
    GA = _ga(z, spi)
    P = z.shape[0]
    O = np.full((P, P), np.nan)
    for (i, j), o in GA.order.items():
        O[i, j] = O[j, i] = o
    return O


def maxorder(x, run):
    from nitime.analysis.granger import fit_model
    z = _data(x, run.get("path", "pipeline")).to_numpy(squeeze=True)
    out = []
    for m in range(2, 10):
        try:
            o = fit_model(z[0], z[1], order=None, max_order=m)[0]
        except ValueError:
            o = -1
        out.append(o)
    return np.array(out, dtype=float)


def grid(x, run):
    import nitime.analysis as nta
    import nitime.timeseries as ts
    from nitime.algorithms.spectral import freq_response
    z = _data(x, "pipeline").to_numpy(squeeze=True)
    GA = nta.GrangerAnalyzer(ts.TimeSeries(z, sampling_interval=1), order=1)
    w, _ = freq_response(np.r_[1.0, 0.5], n_freqs=1024)
    return np.stack([np.asarray(GA.frequencies, dtype=float), np.asarray(w) / (2 * np.pi)])


def _delegate_outcome(z, spi):
    """Call the delegate directly: nitime fit_model per pair (granger.py:119-125) or the
    spectral_connectivity pipeline (spectral.py:69-73)."""
    if getattr(spi, "_method", "") == "parametric":
        from nitime.analysis.granger import fit_model
        P = z.shape[0]
        for i in range(P):
            for j in range(i + 1, P):
                try:
                    fit_model(z[i], z[j], order=spi._order, max_order=spi._max_order)
                except Exception as exc:  # recorded, not hidden
                    return exc, i, j
        return None, -1, -1
    from pyspi._spectral_compat import spectral_connectivity as sc
    try:
        sc.Connectivity.from_multitaper(sc.Multitaper(z.T, sampling_frequency=1)) \
            .pairwise_spectral_granger_prediction()
    except Exception as exc:  # noqa: BLE001
        return exc, -1, -1
    return None, -1, -1


def _calculator_nanfrac(x, run, rec):
    """Real pyspi Calculator with a one-SPI config (calculator.py:325-339 except Exception)."""
    import yaml
    from pyspi.calculator import Calculator
    e = _entry(run)
    params = dict(e["params"])
    cfg = {e["module"]: {e["cls"]: {"labels": list(e["labels"]), "dependencies": None,
                                    "configs": [params]}}}
    path = run.get("path", "pipeline")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "sgc_one.yaml"
        f.write_text(yaml.safe_dump(cfg))
        calc = Calculator(dataset=np.array(x, dtype=float), configfile=str(f),
                          normalise=path != "kernel", detrend=path == "pipeline-detrend")
        calc.compute()
    ident = list(calc.spis)[0]
    A = np.asarray(calc.table[ident].to_numpy(), dtype=float)
    off = ~np.eye(A.shape[0], dtype=bool)
    rec["calculator_identifier"] = ident
    try:
        rec["calculator_report"] = [str(r) for r in getattr(calc, "metric_reports", [])][:3]
    except Exception:  # noqa: BLE001
        pass
    return float(np.isnan(A[off]).mean())


def outcome(x, run):
    from regression import harness
    path = run.get("path", "pipeline")
    rec = {"n_regex_filters": _n_regex_filters()}
    # (1) direct call under the process's current (default) warnings filters
    exc_direct = None
    try:
        harness.compute_spi(_spi(run), _data(x, path))
    except Exception as exc:  # noqa: BLE001 - this is the observation
        exc_direct = exc
    rec["direct"] = repr(exc_direct)
    # (2) a real Calculator
    try:
        nan_calc = _calculator_nanfrac(x, run, rec)
    except Exception as exc:  # noqa: BLE001
        nan_calc = -1.0
        rec["calculator_exception"] = repr(exc)
    # (3) the handler's intended path: catch_warnings(record=True) + simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            S = np.array(_spi(run).multivariate(_data(x, path)), dtype=float)
            off = ~np.eye(S.shape[0], dtype=bool)
            nan_catch = float(np.isnan(S[off]).mean())
            rec["catch_always_warnings"] = [str(m.message)[:300] for m in w][-5:]
        except Exception as exc:  # noqa: BLE001
            nan_catch = -1.0
            rec["catch_always_exception"] = repr(exc)
    # (4) the delegate called directly
    exc_fit, fi, fj = _delegate_outcome(_data(x, path).to_numpy(squeeze=True), _spi(run))
    rec["delegate"] = repr(exc_fit)
    _sidecar(run, rec)
    return np.array([_code(exc_direct), nan_catch, _code(exc_fit), fi, fj,
                     rec["n_regex_filters"], nan_calc], dtype=float)


def _constructed_cases(T):
    """EEG-like degenerate inputs built from iid_gaussian (first 3 of 5 x T)."""
    base = make_data.load("iid_gaussian", T)["x"][:3].copy()
    flat = base.copy(); flat[2] = 0.0
    dup = base.copy(); dup[2] = base[0]
    scaled = base.copy(); scaled[2] = -3.0 * base[0] + 7.0
    near = base.copy(); near[2] = base[0] + 1e-9 * np.random.default_rng(5).standard_normal(T)
    near6 = base.copy(); near6[2] = base[0] + 1e-6 * np.random.default_rng(6).standard_normal(T)
    return [("eeg:flat_channel", flat), ("eeg:duplicate_channel", dup), ("eeg:scaled_copy", scaled),
            ("eeg:near_duplicate_1e-9", near), ("eeg:near_duplicate_1e-6", near6)]


def sweep_cases(T):
    man = json.loads((CHECKS / "data" / "MANIFEST.json").read_text())
    names = sorted({f["name"] for f in man["files"] if int(f["T"]) == int(T)})
    cases = [(nm, make_data.load(nm, T)["x"]) for nm in names]
    return cases + _constructed_cases(T)


def sweep(x, run):
    from regression import harness
    rows, rec = [], {}
    for name, xx in sweep_cases(int(run["T"])):
        spi = _spi(run)
        harness.seed_all(42)
        try:
            data = _data(xx, run.get("path", "pipeline"))
        except Exception as exc:  # noqa: BLE001 - Data() itself rejects the input (not an SPI failure)
            rows.append([5, 1.0, xx.shape[0]])
            rec[name] = "Data(): " + repr(exc)[:300]
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                S = harness.compute_spi(spi, data)
            off = ~np.eye(S.shape[0], dtype=bool)
            rows.append([0, float(np.isnan(S[off]).mean()), xx.shape[0]])
            rec[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            rows.append([_code(exc), 1.0, xx.shape[0]])
            rec[name] = repr(exc)[:300]
    _sidecar(run, {"cases": rec, "order": list(rec)})
    return np.array(rows, dtype=float)


def affine(x, run):
    from regression import harness
    tag = run["run_id"].split("|")[-1]
    y = np.array(x, dtype=float, copy=True)
    s = SCALE[: y.shape[0], None]
    y = y * s + (OFFSET[: y.shape[0], None] if "affine" in tag else 0.0)
    return harness.compute_spi(_spi(run), _data(y, run.get("path", "kernel")))


# ------------------------------------------------------------------ borderline-order draws (ntot)
# Kept byte-identical with refs/sgc.py:_ntot_draws (both venvs run numpy 2.5.4; the probe also returns
# a per-draw checksum so the reference can verify it regenerated the same data).
NTOT_R = 200


def _ntot_draws(T, seed, R=NTOT_R, burn=300):
    """Bivariate VAR(2) with a weak second lag: A1 = [[0.4, 0], [0.3, 0.3]],
    A2 = a2 [[1, 0], [-1, 1]], a2 ~ U(0.02, 0.25), unit noise; x0 -> x1 at lag 1, x0 -> x1 at lag 2 (negative)."""
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.4, 0.0], [0.3, 0.3]])
    for _ in range(R):
        a2 = rng.uniform(0.02, 0.25)
        A2 = a2 * np.array([[1.0, 0.0], [-1.0, 1.0]])
        e = rng.standard_normal((T + burn, 2))
        x = np.zeros((T + burn, 2))
        for t in range(2, T + burn):
            x[t] = A1 @ x[t - 1] + A2 @ x[t - 2] + e[t]
        yield a2, x[burn:].T.copy()


def ntot(x, run):
    """x is ignored apart from run["T"]; returns (R, 6): [a2, nitime order chosen by the configured SPI's
    GrangerAnalyzer (order=None, max_order=50; -1 = ValueError), checksum sum(z) of the pyspi-preprocessed
    draw, pyspi A[1,0], pyspi A[0,1] (NaN when the SPI raised), checksum sum(raw draw)]."""
    spi = _spi(run)
    rows = []
    for a2, xr in _ntot_draws(run["T"], run["seed"]):
        d = _data(xr, run.get("path", "pipeline"))
        z = d.to_numpy(squeeze=True)
        try:
            o = int(_ga(z, spi).order[(0, 1)])
        except ValueError:
            o = -1
        try:
            A = np.asarray(spi.multivariate(d), dtype=float)
            a10, a01 = A[1, 0], A[0, 1]
        except Exception:  # noqa: BLE001
            a10 = a01 = np.nan
        rows.append([a2, o, float(z.sum()), a10, a01, float(xr.sum())])
    return np.array(rows, dtype=float)

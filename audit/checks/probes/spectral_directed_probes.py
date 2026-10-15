"""Probes for the spectral_directed family (PROJECT venv).

Units: spectral.dtf, spectral.dcoh, spectral.pdcoh, spectral.gpdcoh, spectral.ddtf
and the infrastructure unit spectral.wilson_factorisation.  Every function has the
probe signature ``fn(x, run)``: ``x`` is the raw dataset array (P, T) after the
manifest's process selection, ``run`` the manifest entry.  They build exactly the
objects pyspi builds for one ordered pair in
``pyspi/statistics/spectral.py:106-126`` (``sc.Multitaper(z, sampling_frequency=1)``
on the 2-channel array ``[x_i, x_j]`` with time on axis 0, then
``sc.Connectivity.from_multitaper(m)``, ``sc`` imported through
``pyspi._spectral_compat``) and export intermediates of spectral_connectivity 1.1.2
that the plain SPI path does not expose: the two-sided multitaper cross-spectral
matrix (CSM), the Wilson factor's transfer function H and noise covariance Sigma,
the per-bin directed measures before band aggregation, iteration counts and
factorisation residuals.

Conventions shared with ``refs/spectral_directed.py``:

* The selected processes (manifest ``processes``) ARE the pair, in order:
  ``processes = [i, j]`` gives the array pyspi factorises for A[i, j]; the library
  entry [0, 1] of the per-bin measure is what pyspi aggregates into A[i, j].
* Preprocessing follows ``run["path"]``: ``kernel`` = raw array; ``pipeline`` =
  ``Data(x)`` (z-score, ddof=1); ``pipeline-detrend`` = ``Data(x, detrend=True)``.
* Complex arrays are returned as ``np.stack([real, imag])`` (compare.py reads float64).
* The tag (last ``|`` field of the run_id) is ``kind[-opt][:stat-band][~pIJ]``.
* ``perbin`` additionally saves the library's own CSM (two-sided), H (non-negative
  bins), Sigma and frequencies of the same pair to
  ``out/exchange/spectral_directed/<run_id with | -> __>.npz`` so that the reference can
  replicate each measure's formula from the library's own factors.
"""
import json
from pathlib import Path

import numpy as np
import yaml

from pyspi._spectral_compat import spectral_connectivity as sc
import spectral_connectivity.connectivity as scc
import spectral_connectivity.minimum_phase_decomposition as mpd

CHECKS = Path(__file__).resolve().parent.parent
AUDIT = CHECKS.parent
EXCHANGE = CHECKS / "out" / "exchange" / "spectral_directed"

MEASURE = {"dtf": "directed_transfer_function", "dcoh": "directed_coherence",
           "pdcoh": "partial_directed_coherence", "gpdcoh": "generalized_partial_directed_coherence",
           "ddtf": "direct_directed_transfer_function"}
UNITS = ("dtf", "dcoh", "pdcoh", "gpdcoh", "ddtf")
BAND_SUFFIX = {"full": "fmin-0_fmax-0-5", "low": "fmin-0_fmax-0-25", "high": "fmin-0-25_fmax-0-5"}


# ----------------------------------------------------------------------------- helpers
def tag(run):
    return run["run_id"].split("|")[-1]


def kind(run):
    return tag(run).split(":")[0].split("~")[0]


def unit_short(run):
    return run["unit_id"].split(".")[1]


def safe_name(rid):
    return rid.replace("|", "__").replace(":", "_").replace("~", "_")


def prepare(x, run):
    """(P, T) float64 array as pyspi's Data hands it to the spectral class."""
    x = np.asarray(x, dtype=np.float64)
    path = run.get("path", "kernel")
    if path == "kernel":
        return x.copy()
    return np.asarray(make_data(x, path).to_numpy(squeeze=True), dtype=np.float64)


def make_data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def connectivity(z, n_fft=None):
    """(Multitaper, Connectivity) exactly as spectral.py:112-114 builds them; z is (P, T)."""
    kw = {} if n_fft is None else {"n_fft_samples": int(n_fft)}
    m = sc.Multitaper(np.transpose(z), sampling_frequency=1, **kw)
    return m, sc.Connectivity.from_multitaper(m)


def _ri(a):
    return np.stack([np.real(a), np.imag(a)])


class _CountWilson:
    """Counts library Wilson runs and the iterations of each run (patches the module
    globals that connectivity.py and minimum_phase_decomposition.py look up at call time)."""

    def __enter__(self):
        self.runs, self.iters, self.last_converged = 0, [], None
        self._orig_mpd = scc.minimum_phase_decomposition
        self._orig_chk = mpd._check_convergence
        outer = self

        def chk(current, old, tolerance=1e-8):
            r = outer._orig_chk(current, old, tolerance)
            if outer.iters:
                outer.iters[-1] += 1
            outer.last_converged = bool(np.all(r))
            return r

        def wrapped(*a, **k):
            outer.runs += 1
            outer.iters.append(0)
            return outer._orig_mpd(*a, **k)

        scc.minimum_phase_decomposition = wrapped
        mpd._check_convergence = chk
        return self

    def __exit__(self, *exc):
        scc.minimum_phase_decomposition = self._orig_mpd
        mpd._check_convergence = self._orig_chk
        return False


def build_spi(identifier):
    """The SPI object for a config identifier, from audit/units.yaml (as pyspi_outputs does)."""
    import importlib
    doc = yaml.safe_load((AUDIT / "units.yaml").read_text())
    for u in doc["units"]:
        if identifier in u["identifiers"]:
            mod = importlib.import_module(f"pyspi.statistics.{u['module']}")
            params = u["identifier_params"][identifier]
            cls = getattr(mod, u["pyspi_class"])
            return cls() if params is None else cls(**params)
    raise KeyError(identifier)


def identifier_for(unit, stat, band):
    return f"{unit}_multitaper_{stat}_fs-1_{BAND_SUFFIX[band]}"


# ----------------------------------------------------------------------------- library intermediates
def csm(x, run):
    """Library two-sided CSM of the selected processes: (2, n_fft, P, P) = [Re, Im]."""
    _, c = connectivity(prepare(x, run))
    return _ri(c._expectation_cross_spectral_matrix()[0])


def transfer(x, run):
    """Library Wilson transfer function on the non-negative bins: (2, nf, P, P) = [Re, Im]."""
    _, c = connectivity(prepare(x, run))
    return _ri(c._transfer_function[0])


def sigma(x, run):
    """Library Wilson noise covariance Sigma = A0 A0^T: (P, P)."""
    _, c = connectivity(prepare(x, run))
    return np.asarray(c._noise_covariance[0], dtype=np.float64)


def conv(x, run):
    """[n_iter, converged (0/1), max|Psi Psi^H - S| / max|S| (two-sided),
    max|H Sigma H^H - S| / max|S| (non-negative bins)] for the library's own Wilson run
    on the library CSM (same call as Connectivity._minimum_phase_factor)."""
    _, c = connectivity(prepare(x, run))
    S = c._expectation_cross_spectral_matrix()
    with _CountWilson() as cw:
        psi = scc.minimum_phase_decomposition(S)
    n_iter, converged = cw.iters[-1], cw.last_converged
    psi0, S0 = psi[0], S[0]
    rec2 = np.abs(psi0 @ np.conj(np.swapaxes(psi0, -1, -2)) - S0).max() / np.abs(S0).max()
    H = scc._estimate_transfer_function(psi)[0]
    Sg = scc._estimate_noise_covariance(psi)[0]
    nf = (S0.shape[0] + 1) // 2
    Hn = H[:nf]
    recn = np.abs(Hn @ Sg @ np.conj(np.swapaxes(Hn, -1, -2)) - S0[:nf]).max() / np.abs(S0).max()
    return np.array([n_iter, float(converged), rec2, recn], dtype=np.float64)


def perbin(x, run):
    """Library per-bin directed measure of the run's unit on the pair, before band
    aggregation: (nf, P, P) = getattr(Connectivity, measure)()[0] (spectral.py:117).  Also saves
    the same Connectivity's CSM, H, Sigma and frequencies to the exchange directory."""
    _, c = connectivity(prepare(x, run))
    val = np.asarray(getattr(c, MEASURE[unit_short(run)])()[0], dtype=np.float64)
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXCHANGE / f"{safe_name(run['run_id'])}.npz",
                        S=c._expectation_cross_spectral_matrix()[0], H=c._transfer_function[0],
                        Sigma=np.asarray(c._noise_covariance[0]), freqs=np.asarray(c.frequencies))
    return val


# ----------------------------------------------------------------------------- exact spectrum
def _truth(run):
    import sys
    sys.path.insert(0, str(CHECKS))
    import make_data as md
    d = md.load(run["dataset"], run["T"])
    return [np.asarray(d["A1"]), np.asarray(d["A2"])], np.asarray(d["Sigma"])


def _exact_S(run, n=None):
    """Exact two-sided S = H Sigma H^H of the dataset's VAR(2) on fftfreq(n) (all channels)."""
    A, Sigma = _truth(run)
    n = int(run["T"] if n is None else n)
    f = np.fft.fftfreq(n)
    P = Sigma.shape[0]
    Af = np.broadcast_to(np.eye(P, dtype=complex), (n, P, P)).copy()
    for k, Ak in enumerate(A, start=1):
        Af -= Ak[None] * np.exp(-2j * np.pi * f * k)[:, None, None]
    H = np.linalg.inv(Af)
    return H @ Sigma @ np.conj(np.swapaxes(H, -1, -2)), f


class _ExactConnectivity(sc.Connectivity):
    """Connectivity whose CSM is a given exact spectrum (only factorisation + formula under test)."""

    def __init__(self, S, f):
        n, P, _ = S.shape
        super().__init__(fourier_coefficients=np.zeros((1, 1, 1, n, P), dtype=complex), frequencies=f)
        self._S_exact = S[None]

    def _expectation_cross_spectral_matrix(self, fcn=None, dtype=None):
        return self._S_exact


def _tol(run):
    return 1e-12 if "tol12" in tag(run) else 1e-8


def exact_transfer(x, run):
    """Library Wilson H from the exact VAR spectrum (n = T bins), non-negative bins, [Re, Im]."""
    S, _ = _exact_S(run)
    psi = scc.minimum_phase_decomposition(S[None], tolerance=_tol(run))
    H = scc._estimate_transfer_function(psi)[0]
    return _ri(H[: (S.shape[0] + 1) // 2])


def exact_sigma(x, run):
    """Library Wilson Sigma from the exact VAR spectrum."""
    S, _ = _exact_S(run)
    psi = scc.minimum_phase_decomposition(S[None], tolerance=_tol(run))
    return np.asarray(scc._estimate_noise_covariance(psi)[0], dtype=np.float64)


def exact_perbin(x, run):
    """Library per-bin measure of the run's unit computed from the exact VAR spectrum
    (all channels of the dataset; for var2_chain this is the 3-channel multivariate model)."""
    S, f = _exact_S(run)
    c = _ExactConnectivity(S, f)
    return np.asarray(getattr(c, MEASURE[unit_short(run)])()[0], dtype=np.float64)


def ddtf_scale(x, run):
    """Library ddtf on the exact var2 spectrum at n = 64, 256, 1024, 4096: band mean over the
    strict full band (0 < f < 0.5) of entries [1,0] and [0,1] -> (4, 2)."""
    out = []
    for n in (64, 256, 1024, 4096):
        S, f = _exact_S(run, n)
        c = _ExactConnectivity(S, f)
        v = c.direct_directed_transfer_function()[0]
        fr = np.asarray(c.frequencies)[: v.shape[0]]
        sel = (fr > 0.0) & (fr < 0.5)
        out.append([v[sel, 1, 0].mean(), v[sel, 0, 1].mean()])
    return np.array(out)


# ----------------------------------------------------------------------------- pipeline-level probes
def asym(x, run):
    """pyspi SPI matrix A (identifier from the manifest) minus its transpose: the sign at [i, j]
    says whether A[i, j] > A[j, i].  Both orientations are in the stored matrix."""
    spi = build_spi(run["identifier"])
    A = np.asarray(spi.multivariate(make_data(x, run.get("path", "pipeline"))), dtype=np.float64)
    np.fill_diagonal(A, np.nan)
    return A - A.T


def ddtf_slope(x, run):
    """pyspi ddtf mean full band on the run's dataset (pipeline path) at T = 64, 256, 1024, 4096:
    (4, P, P) matrices (the run's own T is ignored)."""
    import sys
    sys.path.insert(0, str(CHECKS))
    import make_data as md
    out = []
    for T in (64, 256, 1024, 4096):
        xx = np.asarray(md.load(run["dataset"], T)["x"], dtype=np.float64)
        if run.get("processes") is not None:
            xx = xx[list(run["processes"])]
        spi = build_spi("ddtf_multitaper_mean_fs-1_fmin-0_fmax-0-5")
        A = np.asarray(spi.multivariate(make_data(xx, run.get("path", "pipeline"))), dtype=np.float64)
        np.fill_diagonal(A, np.nan)
        out.append(A)
    return np.stack(out)


def nfft2(x, run):
    """Zero-padding counterfactual: [rec(n_fft=T), rec(n_fft=2T), dtf[1,0] strict full-band mean at
    n_fft=T, at 2T, dtf[0,1] at T, at 2T] with rec = max|Psi Psi^H - S| / max|S| (two-sided)."""
    z = prepare(x, run)
    T = z.shape[1]
    rec, d10, d01 = [], [], []
    for n in (T, 2 * T):
        _, c = connectivity(z, n_fft=n if n != T else None)
        S = c._expectation_cross_spectral_matrix()
        psi = scc.minimum_phase_decomposition(S)[0]
        rec.append(np.abs(psi @ np.conj(np.swapaxes(psi, -1, -2)) - S[0]).max() / np.abs(S[0]).max())
        v = c.directed_transfer_function()[0]
        fr = np.asarray(c.frequencies)[: v.shape[0]]
        sel = (fr > 0) & (fr < 0.5)
        d10.append(v[sel, 1, 0].mean())
        d01.append(v[sel, 0, 1].mean())
    return np.array(rec + d10 + d01)


def order(x, run):
    """Channel-order dependence: max over bins |L[0,1] on [x_i, x_j] - L[1,0] on [x_j, x_i]| for the
    five measures (dtf, dcoh, pdcoh, gpdcoh, ddtf) -> (5,)."""
    z = prepare(x, run)
    _, c1 = connectivity(z)
    _, c2 = connectivity(z[::-1].copy())
    out = []
    for u in UNITS:
        a = getattr(c1, MEASURE[u])()[0]
        b = getattr(c2, MEASURE[u])()[0]
        out.append(np.abs(a[:, 0, 1] - b[:, 1, 0]).max())
    return np.array(out)


def calls(x, run):
    """Library Wilson runs per pyspi call (fresh Data per SPI): [dtf, dcoh, pdcoh, gpdcoh, ddtf] for
    the mean-full identifier, then the total for all six dtf identifiers on ONE Data (cache shared),
    then max|value(uncached) - value(one shared factorisation per pair)| over the five units."""
    path = run.get("path", "pipeline")
    counts = []
    for u in UNITS:
        with _CountWilson() as cw:
            build_spi(identifier_for(u, "mean", "full")).multivariate(make_data(x, path))
        counts.append(cw.runs)
    d = make_data(x, path)
    with _CountWilson() as cw:
        for stat in ("mean", "max"):
            for band in ("full", "low", "high"):
                build_spi(identifier_for("dtf", stat, band)).multivariate(d)
    counts.append(cw.runs)
    z = prepare(x, run)
    P = z.shape[0]
    worst = 0.0
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            _, c = connectivity(z[[i, j]])
            psi = scc.minimum_phase_decomposition(c._expectation_cross_spectral_matrix())

            class Shared(type(c)):
                @property
                def _minimum_phase_factor(self):
                    return psi
            c2 = Shared(fourier_coefficients=c.fourier_coefficients, expectation_type=c.expectation_type,
                        frequencies=c._frequencies, time=c.time)
            for u in UNITS:
                a = getattr(c, MEASURE[u])()[0]
                b = getattr(c2, MEASURE[u])()[0]
                worst = max(worst, float(np.abs(a - b).max()))
    counts.append(worst)
    return np.array(counts, dtype=np.float64)


def singular(x, run):
    """Degenerate pair behaviour.  Returns the matrix pyspi's Calculator would store for the
    identifier (all NaN if multivariate raises, calculator.py compute) and writes, to
    out/exchange/spectral_directed/<run>.json, the exception text of the whole-matrix call and of
    every ordered pair evaluated on its own (fresh Data per pair)."""
    path = run.get("path", "pipeline")
    rec = {"whole_error": None, "pair_errors": {}, "pair_values": {}}
    try:
        A = np.asarray(build_spi(run["identifier"]).multivariate(make_data(x, path)), dtype=np.float64)
        np.fill_diagonal(A, np.nan)
    except Exception as exc:                       # Calculator.compute: whole matrix NaN
        rec["whole_error"] = repr(exc)
        A = np.full((x.shape[0], x.shape[0]), np.nan)
    P = x.shape[0]
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            try:
                v = float(build_spi(run["identifier"]).bivariate(make_data(x, path), i=i, j=j))
                rec["pair_values"][f"{i},{j}"] = v
            except Exception as exc:
                rec["pair_errors"][f"{i},{j}"] = repr(exc)
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    (EXCHANGE / f"{safe_name(run['run_id'])}.json").write_text(json.dumps(rec, indent=1))
    return A

"""Probes for the wavelet_psi family (PROJECT venv; called by pyspi_outputs.py).

Unit ``wavelet.psi`` = ``pyspi.statistics.wavelet.PhaseSlopeIndex`` (six config
identifiers, mean/max x full/low/high band).  Every function has the probe
signature ``fn(x, run) -> ndarray``; ``x`` is the raw dataset (P, T) after the
manifest's process selection and ``run`` the manifest entry.  The path
(kernel | pipeline | pipeline-detrend) is honoured through ``_make_data``.

* ``tsel``    axis-selection map at the pyspi wrapper level.  Computes the PSI
              once (fills ``data.mne_psi``), then for every time index t of the
              cached (N, N, 1, T) result adds 100 to ``psi[..., t]`` (lower
              triangle), puts the modified array back in the cache and calls
              ``multivariate`` for each of the six identifiers.  Returns
              (6, T) = max |A_perturbed - A| over off-diagonal entries.  The
              library is not touched; only the stored-value selection is probed.
* ``xsel``    input-level dependence horizon (use the kernel path).  For t0 =
              0..63 replaces x[:, t0:] by 10 x independent N(0, 1) noise (seeded
              by t0), recomputes on a fresh Data and returns (6, 64) = max
              |A(x') - A(x)| per identifier.
* ``libinfo`` [psi.shape (4 values), freqs_computed[0] (10 values)] of
              ``mne_connectivity.phase_slope_index`` exactly as pyspi calls it.
* ``rawpsi``  the library's dense PSI(t) as pyspi caches it, (N, N, T) real.
* ``cohmod``  |C(f, t)| of the library's single-epoch coherency
              (``spectral_connectivity_epochs(method='cohy')`` with the same
              arguments ``phase_slope_index`` forwards, effective.py:177-198),
              lower-triangle pairs, shape (n_pairs, 10, T).
* ``nyq``     Im C(f = 0.5, t) / |C(f = 0.5, t)| for the lower-triangle pairs,
              shape (n_pairs, T).
* ``perm``    A(X[::-1])[inv, inv] for the run's identifier (channel reversal
              mapped back to the original order).
* ``winperm`` input-window perturbation at the pyspi level.  The window
              ``run["probe"]["window"] = [a, b]`` (b null = T) of every channel
              is REVERSED in time (deterministic; the multiset of values, hence
              the channel mean and SD used by z-scoring, is unchanged).  Returns
              (2, 6, P, P) = [six identifiers on x, six identifiers on the
              perturbed x], through the run's path.
* ``nullmc``  pyspi on ``n`` independent draws of two iid N(0, 1) channels of
              length T (numpy default_rng(run["probe"]["seed"] + T)), pipeline
              path; returns (6, 2) = [mean, SD (ddof=1)] of the stored A[1, 0]
              per identifier (null distribution of the stored statistic).
* ``tailpert`` tail-sample perturbations through the run's path (review item:
              do samples t >= 37 reach the stored value through z-scoring?).
              Returns (4, 6, P, P) = six identifiers on [x, x with x[:, 37:] +=
              0.5 (channel means move), x with x[0, 37:] *= 3 (mean and SD of
              channel 0 move), x with x[:, 37:] = m + 3 (x[:, 37:] - m), m the
              per-channel tail mean (channel mean fixed, SD moves)].
* ``fscache`` cache-key probe: PhaseSlopeIndex mean (fmin 0, fmax 0.5) with
              fs = 1, then fs = 2 on the SAME Data, then fs = 2 on a fresh Data.
              Returns (3, P, P).
* ``short``   the six identifiers on x[:, :20] (T < 25: MNE's default fmin = 5/T
              drops wavelet frequencies), shape (6, P, P).  An identifier whose
              ``multivariate`` raises (np.nanmax of an empty selection) becomes an
              all-NaN matrix as in ``Calculator.compute``; the exception texts and
              the cached frequency vector go to out/exchange/wavelet_psi/<run>.json.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parent.parent
EXCHANGE = CHECKS / "out" / "exchange" / "wavelet_psi"
FREQS = np.linspace(0.2, 0.5, 10)
IDS = [  # (statistic, fmin, fmax) in manifest order
    ("mean", 0, 0.5), ("mean", 0, 0.25), ("mean", 0.25, 0.5),
    ("max", 0, 0.5), ("max", 0, 0.25), ("max", 0.25, 0.5),
]


def _make_data(x, path):
    from pyspi.data import Data
    x = np.array(x, dtype=np.float64, copy=True)
    if path == "kernel":
        return Data(x, dim_order="ps", normalise=False, detrend=False)
    if path == "pipeline":
        return Data(x, dim_order="ps")
    if path == "pipeline-detrend":
        return Data(x, dim_order="ps", normalise=True, detrend=True)
    raise ValueError(path)


def _spis():
    from pyspi.statistics.wavelet import PhaseSlopeIndex
    return [PhaseSlopeIndex(fs=1, fmin=fmin, fmax=fmax, statistic=st) for st, fmin, fmax in IDS]


def _spi_for(identifier):
    for s in _spis():
        if s.identifier == identifier:
            return s
    raise KeyError(identifier)


def _off(A):
    m = ~np.eye(A.shape[0], dtype=bool)
    return A[m]


def _all6(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [np.array(s.multivariate(data), dtype=float) for s in _spis()]


def tsel(x, run):
    data = _make_data(x, run["path"])
    base = _all6(data)
    psi0 = np.array(data.mne_psi["psi"], copy=True)
    freq = data.mne_psi["freq"]
    N, T = psi0.shape[0], psi0.shape[-1]
    lower = np.tril(np.ones((N, N), dtype=bool), -1)
    out = np.zeros((len(IDS), T))
    for t in range(T):
        p = psi0.copy()
        p[lower, :, t] += 100.0
        data.mne_psi = dict(psi=p, freq=freq)
        for k, A in enumerate(_all6(data)):
            out[k, t] = np.nanmax(np.abs(_off(A) - _off(base[k])))
    data.mne_psi = dict(psi=psi0, freq=freq)
    return out


def xsel(x, run):
    x = np.asarray(x, dtype=np.float64)
    base = _all6(_make_data(x, run["path"]))
    n = 64
    out = np.zeros((len(IDS), n))
    for t0 in range(n):
        rng = np.random.default_rng(1000 + t0)
        xp = x.copy()
        xp[:, t0:] = 10.0 * rng.standard_normal((x.shape[0], x.shape[1] - t0))
        for k, A in enumerate(_all6(_make_data(xp, run["path"]))):
            out[k, t0] = np.nanmax(np.abs(_off(A) - _off(base[k])))
    return out


def _lib_conn(x, run):
    import mne_connectivity as mnec
    data = _make_data(x, run["path"])
    z = np.moveaxis(data.to_numpy(), 2, 0)
    return z, mnec


def libinfo(x, run):
    z, mnec = _lib_conn(x, run)
    con = mnec.phase_slope_index(data=z, mode="cwt_morlet", sfreq=1, mt_adaptive=True,
                                 cwt_freqs=FREQS, verbose="WARNING")
    psi = con.get_data(output="dense")
    fc = np.asarray(con.attrs["freqs_computed"][0], dtype=float)
    return np.r_[np.asarray(psi.shape, dtype=float), fc]


def rawpsi(x, run):
    data = _make_data(x, run["path"])
    _all6(data)
    return np.real(np.asarray(data.mne_psi["psi"]))[:, :, 0, :]


def _cohy(x, run):
    z, mnec = _lib_conn(x, run)
    con = mnec.spectral_connectivity_epochs(z, method="cohy", mode="cwt_morlet", sfreq=1,
                                            fmin=None, fmax=np.inf, faverage=False,
                                            mt_adaptive=True, cwt_freqs=FREQS, cwt_n_cycles=7,
                                            verbose="WARNING")
    C = con.get_data(output="dense")                        # (N, N, n_f, T)
    N = C.shape[0]
    ii, jj = np.tril_indices(N, -1)
    return C[ii, jj], np.asarray(con.freqs)


def cohmod(x, run):
    C, _ = _cohy(x, run)
    return np.abs(C)


def nyq(x, run):
    C, f = _cohy(x, run)
    k = int(np.argmin(np.abs(f - 0.5)))
    c = C[:, k, :]
    return np.imag(c) / np.abs(c)


def perm(x, run):
    sys.path.insert(0, str(CHECKS.parent.parent / "tests"))
    from regression import harness
    p = np.arange(x.shape[0])[::-1]
    inv = np.argsort(p)
    S = harness.compute_spi(_spi_for(run["probe"]["identifier"]), _make_data(np.asarray(x)[p], run["path"]))
    return S[np.ix_(inv, inv)]


def _quiet_all6(x, path):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        data = _make_data(x, path)
    return np.stack(_all6(data))


def winperm(x, run):
    x = np.asarray(x, dtype=np.float64)
    a, b = run["probe"]["window"]
    b = x.shape[1] if b is None else int(b)
    xp = x.copy()
    xp[:, a:b] = x[:, a:b][:, ::-1]
    return np.stack([_quiet_all6(x, run["path"]), _quiet_all6(xp, run["path"])])


def nullmc(x, run):
    T = int(np.asarray(x).shape[1])
    n = int(run["probe"]["n"])
    rng = np.random.default_rng(int(run["probe"]["seed"]) + T)
    vals = np.empty((len(IDS), n))
    for r in range(n):
        A = _quiet_all6(rng.standard_normal((2, T)), "pipeline")
        vals[:, r] = A[:, 1, 0]
    return np.stack([vals.mean(axis=1), vals.std(axis=1, ddof=1)], axis=1)


def short(x, run):
    """Like Calculator.compute, an exception turns the identifier's matrix all-NaN;
    the exception text per identifier goes to out/exchange/wavelet_psi/<run>.json."""
    import json
    xs = np.asarray(x, dtype=np.float64)[:, :20]
    data = _make_data(xs, run["path"])
    out, errs = [], {}
    for s in _spis():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                A = np.array(s.multivariate(data), dtype=float)
            errs[s.identifier] = None
        except Exception as exc:                      # Calculator.compute: whole matrix NaN
            errs[s.identifier] = repr(exc)
            A = np.full((xs.shape[0], xs.shape[0]), np.nan)
        np.fill_diagonal(A, np.nan)
        out.append(A)
    errs["freqs_computed"] = np.asarray(data.mne_psi["freq"]).tolist()
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    (EXCHANGE / (run["run_id"].replace("|", "__").replace("/", "_") + ".json")).write_text(
        json.dumps(errs, indent=1) + "\n")
    return np.stack(out)


TAIL0 = 37


def _tail_variants(x):
    x = np.asarray(x, dtype=np.float64)
    a = x.copy()
    a[:, TAIL0:] += 0.5
    b = x.copy()
    b[0, TAIL0:] *= 3.0
    c = x.copy()
    m = x[:, TAIL0:].mean(axis=1, keepdims=True)
    c[:, TAIL0:] = m + 3.0 * (x[:, TAIL0:] - m)
    return [x, a, b, c]


def tailpert(x, run):
    return np.stack([_quiet_all6(v, run["path"]) for v in _tail_variants(x)])


def fscache(x, run):
    import contextlib
    import io
    from pyspi.statistics.wavelet import PhaseSlopeIndex
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1 = PhaseSlopeIndex(fs=1, fmin=0, fmax=0.5, statistic="mean")
        s2 = PhaseSlopeIndex(fs=2, fmin=0, fmax=0.5, statistic="mean")
        with contextlib.redirect_stdout(io.StringIO()):
            d = _make_data(x, run["path"])
            d2 = _make_data(x, run["path"])
        A1 = np.array(s1.multivariate(d), dtype=float)
        A2_shared = np.array(s2.multivariate(d), dtype=float)
        A2_fresh = np.array(s2.multivariate(d2), dtype=float)
    return np.stack([A1, A2_shared, A2_fresh])

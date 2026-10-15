"""Probes for the wavelet_measures family (PROJECT venv; called by pyspi_outputs.py).

Units: the nine extra wavelet classes of ``pyspi/statistics/wavelet.py``
(CoherenceMagnitude ... debiased_weighted_squared_PhaseLagIndex) and the
infrastructure unit ``wavelet.morlet_cwt`` (the shared ``mne`` base class and
the mne / mne-connectivity Morlet CWT it calls).  Every function has the probe
signature ``fn(x, run) -> ndarray``; ``x`` is the raw dataset (P, T) after the
manifest's process selection, ``run`` the manifest entry.  The run's path
(kernel | pipeline | pipeline-detrend) is honoured through ``_make_data``.
Extra probe parameters live in ``run["probe"]`` (``variant`` = one of the six
band/statistic configs, default ``mean-full``).

Library intermediates are obtained with exactly the calls mne-connectivity
0.9.0 makes in ``spectral_connectivity_epochs(mode='cwt_morlet')``:
``mne.time_frequency.morlet(sfreq, freqs, n_cycles=freqs/7, zero_mean=True)``
(epochs.py:360-362) and ``mne.time_frequency.tfr.cwt(x, Ws, use_fft=True,
mode='same')`` (epochs.py:451-456); the cross-spectrum of a lower-triangle
connection (i > j) is ``X_i * conj(X_j)`` (epochs.py:652).

* ``six``      the unit's six identifiers on one Data (shared cache), (6, P, P).
* ``perm``     [A(x), A(x[::-1])[::-1, ::-1]] for the variant; for the phase unit
               a third slice = 2 pi n(Im csd == 0 and Re csd < 0) / N over the
               band (the predicted reversal offset), (2|3, P, P).
* ``ident``    phase / pli / wspli: [pyspi six, identity six, aux] (3, 6, P, P);
               identity from the library CWT: phase mean = pi (n_+pi - n_-pi)/N,
               max = pi if any angle near +pi; pli mean = 1 - frac(Im csd == 0),
               max = 1 if any Im csd != 0; wspli: identity = pyspi pli six.
               aux[0] max distance of an angle from {0, +pi, -pi}; aux[1]
               fraction of angles not exactly in that set; aux[2] fraction of
               points with Im csd == 0 and Re csd < 0; aux[3] fraction of raw
               csd with Im == -0.0; aux[4] fraction of coherency Im == -0.0;
               aux[5] fraction Im csd == 0 (all over the full band).
* ``roundoff`` [A(x), A(zscore(z))] where z is pyspi's z-scored input and the
               second Data z-scores it again (sample changes ~1e-16), plus a
               third slice filled with max |zscore(z) - z|, (3, P, P).
* ``pts``      ppc / dswpli per-point census of the cached tensor over the full
               band: (4, P, P) = counts of -inf, NaN, +inf, finite.
* ``ratio``    ppc: max |Im csd| / |Re csd| per lower pair (mirrored), (P, P).
* ``pinf``     ppc: mne-connectivity's _PPCEst on synthetic single-epoch csd =
               Re (1 + i r), r in linspace(0, 3e-8, 61), Re in (1.0, 0.37, 2.9e-7):
               (3, 61) con values.
* ``dsplisyn`` dswpli: mne-connectivity's _PLIUnbiasedEst on synthetic single-epoch
               csd, nanmean over 100 points: [all Im != 0, one Im == 0], (2,).
* ``wavelets`` morlet_cwt: (125, 4) = [length, Re W[0], Im W[0], sigma_t] of the
               125 wavelets mne builds for pyspi's grid.
* ``cwtx``     morlet_cwt: library CWT of x (use_fft=True, as the pipeline),
               (2, P, 125, T) = [Re X, Im X].
* ``cwtdirect`` morlet_cwt: library CWT with use_fft=False (direct convolution),
               (2, P, 125, T).
* ``mtadapt``  morlet_cwt: for the nine measures, max |conn(mt_adaptive=True) -
               conn(mt_adaptive=False)| (equal non-finite entries count as 0) and
               the number of entries that are not bitwise equal: (9, 2).
* ``freqs``    morlet_cwt: the frequency vector pyspi caches, (125,).
* ``short``    morlet_cwt: the six cohmag identifiers on x[:, :n] (n =
               run["probe"]["n"]); an identifier that raises becomes NaN; the
               exception texts and the cached freqs go to out/exchange/.
* ``orient7``  morlet_cwt: library cohy with n_cycles = 7 (otherwise pyspi's
               call), time-mean of the dense tensor (2, P, P, 125) = [Re, Im].
* ``orient1``  morlet_cwt: pyspi's cached cohy tensor (CoherencePhase): (2, P, P)
               = [mean over (f, t) of Re conn, max over (f, t) of |conn|] BEFORE
               pyspi's mirror (raw dense layout).
* ``fscache``  morlet_cwt: icoh mean-full with fs = 1, then fs = 10 on the SAME
               Data, and fs = 10 on a fresh Data: (3, P, P).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parent.parent
EXCHANGE = CHECKS / "out" / "exchange" / "wavelet_measures"
FREQS = np.linspace(0.2, 0.5, 125)
IDS = [("mean", 0, 0.5), ("mean", 0, 0.25), ("mean", 0.25, 0.5),
       ("max", 0, 0.5), ("max", 0, 0.25), ("max", 0.25, 0.5)]
VARIANTS = {"mean-full": 0, "mean-low": 1, "mean-high": 2, "max-full": 3, "max-low": 4, "max-high": 5}
CLASSES = {
    "wavelet.cohmag": ("CoherenceMagnitude", "coh"),
    "wavelet.phase": ("CoherencePhase", "cohy"),
    "wavelet.icoh": ("ImaginaryCoherence", "imcoh"),
    "wavelet.plv": ("PhaseLockingValue", "plv"),
    "wavelet.ppc": ("PairwisePhaseConsistency", "ppc"),
    "wavelet.pli": ("PhaseLagIndex", "pli"),
    "wavelet.dswpli": ("DebiasedSquaredWeightedPhaseLagIndex", "pli2_unbiased"),
    "wavelet.wspli": ("weighted_PhaseLagIndex", "wpli"),
    "wavelet.dwspli": ("debiased_weighted_squared_PhaseLagIndex", "wpli2_debiased"),
}


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


def _cls(unit):
    import pyspi.statistics.wavelet as w
    return getattr(w, CLASSES[unit][0])


def _spi(unit, k, fs=1):
    st, fmin, fmax = IDS[k]
    return _cls(unit)(fs=fs, fmin=fmin, fmax=fmax, statistic=st)


def _variant(run):
    return VARIANTS[(run.get("probe") or {}).get("variant", "mean-full")]


def _compute(spi, data):
    P = data.n_processes
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array(spi.multivariate(data), dtype=float), None
    except Exception as exc:  # Calculator.compute stores NaN for a failing SPI
        A = np.full((P, P), np.nan)
        return A, repr(exc)


def _six(data, unit):
    return np.stack([_compute(_spi(unit, k), data)[0] for k in range(6)])


def _sym(L):
    """Mirror the lower triangle to the upper one, NaN diagonal (pyspi wavelet.py:87-89)."""
    A = np.array(L, dtype=float, copy=True)
    ui = np.triu_indices(A.shape[0], 1)
    A[ui] = A.T[ui]
    np.fill_diagonal(A, np.nan)
    return A


def _lib_cwt(z, n_cycles=None, sfreq=1.0, use_fft=True):
    from mne.time_frequency import morlet
    from mne.time_frequency.tfr import cwt
    nc = FREQS / 7.0 if n_cycles is None else n_cycles
    Ws = morlet(sfreq, FREQS, n_cycles=nc, zero_mean=True)
    return cwt(np.asarray(z, dtype=np.float64), Ws, use_fft=use_fft, mode="same")


def _band(k):
    _, fmin, fmax = IDS[k]
    return (FREQS >= fmin) & (FREQS <= fmax)


def _write_exchange(run, obj):
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    name = run["run_id"].replace("|", "__").replace("/", "_") + ".json"
    (EXCHANGE / name).write_text(json.dumps(obj, indent=1, default=str) + "\n")


# ----------------------------------------------------------------- SPI-level probes
def six(x, run):
    return _six(_make_data(x, run["path"]), run["unit_id"])


def perm(x, run):
    unit, k = run["unit_id"], _variant(run)
    A = _compute(_spi(unit, k), _make_data(x, run["path"]))[0]
    Ar = _compute(_spi(unit, k), _make_data(np.asarray(x)[::-1], run["path"]))[0][::-1, ::-1]
    out = [A, Ar]
    if unit == "wavelet.phase":
        z = _make_data(x, run["path"]).to_numpy(squeeze=True)
        X = _lib_cwt(z)
        m = _band(k)
        P = z.shape[0]
        C = np.zeros((P, P))
        for i in range(P):
            for j in range(i):
                csd = (X[i] * X[j].conjugate())[m]
                C[i, j] = 2 * np.pi * np.count_nonzero((csd.imag == 0) & (csd.real < 0)) / csd.size
        out.append(_sym(C))
    return np.stack(out)


def ident(x, run):
    unit = run["unit_id"]
    data = _make_data(x, run["path"])
    pys = _six(data, unit)
    z = data.to_numpy(squeeze=True)
    P = z.shape[0]
    idn = np.zeros((6, P, P))
    aux = np.zeros((6, P, P))
    if unit == "wavelet.wspli":
        pli = _six(data, "wavelet.pli")
        return np.stack([pys, pli, pli])
    X = _lib_cwt(z)
    psd = (X * X.conj()).real
    for i in range(P):
        for j in range(i):
            csd = X[i] * X[j].conjugate()
            if unit == "wavelet.phase":
                acc = np.zeros(csd.shape, dtype=np.complex128)
                acc += csd                                   # epochs_bivariate.py:64-68
                coh = (acc / 1) / np.sqrt(psd[i] * psd[j])   # epochs_bivariate.py:90-91
                ang = np.angle(coh)
                dev = np.minimum(np.abs(ang), np.minimum(np.abs(ang - np.pi), np.abs(ang + np.pi)))
                npl, nmi, nze = ang > np.pi / 2, ang < -np.pi / 2, np.abs(ang) <= np.pi / 2
                for k in range(6):
                    m = _band(k)
                    N = npl[m].size
                    if IDS[k][0] == "mean":
                        idn[k, i, j] = np.pi * (npl[m].sum() - nmi[m].sum()) / N
                    else:
                        idn[k, i, j] = np.pi if npl[m].any() else (0.0 if nze[m].any() else -np.pi)
                aux[0, i, j] = dev.max()
                aux[1, i, j] = np.mean((ang != 0) & (ang != np.pi) & (ang != -np.pi))
                aux[2, i, j] = np.mean((csd.imag == 0) & (csd.real < 0))
                aux[3, i, j] = np.mean((csd.imag == 0) & np.signbit(csd.imag))
                aux[4, i, j] = np.mean((coh.imag == 0) & np.signbit(coh.imag))
                aux[5, i, j] = np.mean(csd.imag == 0)
            elif unit == "wavelet.pli":
                zero = csd.imag == 0
                for k in range(6):
                    m = _band(k)
                    if IDS[k][0] == "mean":
                        idn[k, i, j] = 1.0 - zero[m].mean()
                    else:
                        idn[k, i, j] = 1.0 if (~zero[m]).any() else 0.0
                aux[5, i, j] = zero.mean()
            else:
                raise ValueError(unit)
    idn = np.stack([_sym(a) for a in idn])
    aux = np.stack([_sym(a) for a in aux])
    return np.stack([pys, idn, aux])


def roundoff(x, run):
    from scipy.stats import zscore
    unit, k = run["unit_id"], _variant(run)
    data = _make_data(x, run["path"])
    A = _compute(_spi(unit, k), data)[0]
    z = data.to_numpy(squeeze=True)
    z2 = zscore(z, axis=1, ddof=1)
    A2 = _compute(_spi(unit, k), _make_data(z, run["path"]))[0]
    return np.stack([A, A2, np.full(A.shape, float(np.abs(z2 - z).max()))])


def pts(x, run):
    unit = run["unit_id"]
    data = _make_data(x, run["path"])
    _compute(_spi(unit, 0), data)
    conn, _ = data.mne[CLASSES[unit][1]]
    P = conn.shape[0]
    out = np.zeros((4, P, P))
    for i in range(P):
        for j in range(i):
            c = conn[i, j]
            out[0, i, j] = np.count_nonzero(c == -np.inf)
            out[1, i, j] = np.count_nonzero(np.isnan(c))
            out[2, i, j] = np.count_nonzero(c == np.inf)
            out[3, i, j] = np.count_nonzero(np.isfinite(c))
    return np.stack([_sym(a) for a in out])


def ratio(x, run):
    z = _make_data(x, run["path"]).to_numpy(squeeze=True)
    X = _lib_cwt(z)
    P = z.shape[0]
    R = np.zeros((P, P))
    for i in range(P):
        for j in range(i):
            csd = X[i] * X[j].conjugate()
            with np.errstate(divide="ignore", invalid="ignore"):
                R[i, j] = np.nanmax(np.abs(csd.imag) / np.abs(csd.real))
    return _sym(R)


def pinf(x, run):
    from mne_connectivity.spectral.epochs_bivariate import _PPCEst
    r = np.linspace(0.0, 3e-8, 61)
    out = []
    for re in (1.0, 0.37, 2.9e-7):
        csd = (re + 1j * re * r)[None, :]
        est = _PPCEst(1, r.size, 0)
        est.accumulate(slice(0, 1), csd.copy())
        with np.errstate(divide="ignore", invalid="ignore"):
            est.compute_con(slice(0, 1), 1)
        out.append(est.con_scores[0])
    return np.stack(out)


def dsplisyn(x, run):
    from mne_connectivity.spectral.epochs_bivariate import _PLIUnbiasedEst
    rng = np.random.default_rng(7)
    csd = rng.standard_normal(100) + 1j * (1e-16 * (1 + rng.random(100)))
    res = []
    for c in (csd, np.r_[csd[:-1], csd[-1].real + 0j]):
        est = _PLIUnbiasedEst(1, c.size, 0)
        est.accumulate(slice(0, 1), c[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            est.compute_con(slice(0, 1), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res.append(np.nanmean(est.con_scores[0]))
    return np.array(res)


# ----------------------------------------------------------------- infrastructure probes
def wavelets(x, run):
    from mne.time_frequency import morlet
    Ws = morlet(1, FREQS, n_cycles=FREQS / 7.0, zero_mean=True)
    sig = (FREQS / 7.0) / (2.0 * np.pi * FREQS)
    return np.array([[W.size, W[W.size // 2].real, W[W.size // 2].imag, s] for W, s in zip(Ws, sig)])


def cwtx(x, run):
    X = _lib_cwt(_make_data(x, run["path"]).to_numpy(squeeze=True))
    return np.stack([X.real, X.imag])


def cwtdirect(x, run):
    X = _lib_cwt(_make_data(x, run["path"]).to_numpy(squeeze=True), use_fft=False)
    return np.stack([X.real, X.imag])


def mtadapt(x, run):
    import mne_connectivity as mnec
    data = _make_data(x, run["path"])
    z = np.moveaxis(data.to_numpy(), 2, 0)
    out = []
    for unit in CLASSES:
        meas = CLASSES[unit][1]
        res = []
        for adapt in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = mnec.spectral_connectivity_epochs(
                    data=z, method=meas, mode="cwt_morlet", sfreq=1, mt_adaptive=adapt,
                    fmin=5 / data.n_observations, fmax=0.5, cwt_freqs=FREQS,
                    cwt_n_cycles=FREQS / 7.0, verbose="WARNING").get_data(output="dense")
            res.append(np.asarray(c))
        a, b = res
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        d = np.abs(a - b)
        d[same] = 0
        out.append([float(np.nanmax(d)), float((~same).sum())])
    return np.array(out)


def freqs(x, run):
    data = _make_data(x, run["path"])
    spi = _spi("wavelet.cohmag", 0)
    _compute(spi, data)
    return np.asarray(data.mne["coh"][1], dtype=float)


def short(x, run):
    n = int(run["probe"]["n"])
    data = _make_data(np.asarray(x)[:, :n], run["path"])
    out, errs = [], []
    for k in range(6):
        A, e = _compute(_spi("wavelet.cohmag", k), data)
        out.append(A)
        errs.append(e)
    fr = data.mne["coh"][1] if hasattr(data, "mne") and "coh" in data.mne else None
    _write_exchange(run, {"errors": errs, "n_freqs_cached": None if fr is None else int(len(fr)),
                          "freqs_cached_min": None if fr is None else float(np.min(fr))})
    return np.stack(out)


def orient7(x, run):
    import mne_connectivity as mnec
    data = _make_data(x, run["path"])
    z = np.moveaxis(data.to_numpy(), 2, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = mnec.spectral_connectivity_epochs(
            data=z, method="cohy", mode="cwt_morlet", sfreq=1, mt_adaptive=True,
            fmin=5 / data.n_observations, fmax=0.5, cwt_freqs=FREQS, cwt_n_cycles=7.0,
            verbose="WARNING").get_data(output="dense")
    m = np.asarray(c).mean(axis=-1)
    return np.stack([m.real, m.imag])


def orient1(x, run):
    data = _make_data(x, run["path"])
    _compute(_spi("wavelet.phase", 0), data)
    conn, _ = data.mne["cohy"]
    return np.stack([conn.real.mean(axis=(2, 3)), np.abs(conn).max(axis=(2, 3))])


def fscache(x, run):
    unit = "wavelet.icoh"
    d1 = _make_data(x, run["path"])
    A1 = _compute(_spi(unit, 0, fs=1), d1)[0]
    A10_shared = _compute(_spi(unit, 0, fs=10), d1)[0]
    A10_fresh = _compute(_spi(unit, 0, fs=10), _make_data(x, run["path"]))[0]
    return np.stack([A1, A10_shared, A10_fresh])

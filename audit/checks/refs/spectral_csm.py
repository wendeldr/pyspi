#!/usr/bin/env python3
"""References for the spectral_csm family (COMPARISON venv; no pyspi, no spectral_connectivity).

Units: spectral.multitaper_csm (the multitaper cross-spectral matrix every
multitaper spectral SPI is built on) and spectral.band_selection (the fmin/fmax
bin selection of pyspi's reducers, PSI and GD).  One reference per run_id of
manifest/spectral_csm.json, written to out/ref/<run_id>.npz (contract in
compare.py).  The run kind is the first ``-`` token of the last ``|`` field of
the run_id; a token ``n<int>`` there truncates the series to its first <int>
samples (as probes/spectral_csm_lib.prepare does).

Independence: the multitaper reference is ``_kernels_spectral`` (scipy.signal
dpss tapers + numpy FFT, conventions mirrored from spectral_connectivity 1.1.2
but not importing it); tapers are also rebuilt from a dense eigendecomposition of
the Slepian commuting tridiagonal matrix and their concentrations from the
dense sinc kernel; the external-package opinion on coherency is
mne_connectivity (multitaper mode).  Preprocessing for the pipeline paths is
re-implemented here (z-score with ddof=1; least-squares linear detrend), not
taken from pyspi.  Band rules are evaluated analytically on the grid
k / n_fft.  ``bvagg`` runs are same-lineage: the per-bin directed coherence /
DTF come from the library (through the probe); only the aggregation is checked.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.signal.windows
from scipy.stats import norm

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "spectral_csm"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
SELF = "refs/spectral_csm.py"

CSM_BANDS = [(0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.49)]   # = probes/spectral_csm_lib.CSM_BANDS
CONFIG_BANDS = [(0, 0.5), (0, 0.25), (0.25, 0.5)]
EEG_BANDS = [(0.00048828125, 0.001953125), (0.001953125, 0.00390625), (0.00390625, 0.005859375),
             (0.005859375, 0.0146484375), (0.0146484375, 0.0341796875), (0.0341796875, 0.0732421875),
             (0.0732421875, 0.1220703125), (0.00048828125, 0.0341796875), (0.0341796875, 0.1220703125),
             (0.00048828125, 0.1220703125)]                                   # pyspi/workflow/retained185.yaml
ALL_BANDS = CONFIG_BANDS + EEG_BANDS
NW, K = 3.0, 5
MC_SEED = 20261014


# ----------------------------------------------------------------------- helpers
def tag_tokens(run):
    return run["run_id"].split("|")[-1].split("=")[0].split("-")


def kind_of(run):
    return tag_tokens(run)[0]


def trunc_of(run):
    for t in tag_tokens(run):
        if t.startswith("n") and t[1:].isdigit():
            return int(t[1:])
    return None


def raw(run, dataset=None):
    ds = make_data.load(dataset or run["dataset"], run["T"])
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    n = trunc_of(run)
    return (x[:, :n] if n is not None else x), ds


def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def linear_detrend(x):
    t = np.arange(x.shape[1], dtype=np.float64)
    G = np.stack([t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(G, x.T, rcond=None)
    return x - (G @ coef).T


def prepare(x, path):
    if path == "kernel":
        return x
    if path == "pipeline":
        return zscore(x)
    if path == "pipeline-detrend":
        return zscore(linear_detrend(x))
    raise ValueError(path)


def grid(n):
    nfft = ks.next_fast_len_235(n)
    return np.arange((nfft + 1) // 2) / nfft, nfft


def csm(z, nonnegative=False):
    co, fr = ks.multitaper_fft(z)
    return ks.cross_spectral_matrix(co, nonnegative=nonnegative), fr


def ri(a):
    return np.stack([np.real(a), np.imag(a)])


def pyspi_value(rid):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z["value"], dtype=np.float64)


def offdiag(P):
    return ~np.eye(P, dtype=bool)


def slepian_tapers(T, nw=NW, k=K):
    """Top-k eigenvectors of the dense Slepian commuting tridiagonal matrix (Slepian 1978;
    Percival & Walden 1993 sec. 8.3) via scipy.linalg.eigh (dense LAPACK), signs fixed by the
    Percival & Walden convention (even tapers: positive sum; odd tapers: positive slope up to
    the largest peak of the first half).  Returns (k, T) unit-norm tapers."""
    W = nw / T
    n = np.arange(T, dtype=np.float64)
    M = np.diag(((T - 1 - 2 * n) / 2.0) ** 2 * np.cos(2 * np.pi * W))
    off = n[1:] * (T - n[1:]) / 2.0
    M += np.diag(off, 1) + np.diag(off, -1)
    _, V = scipy.linalg.eigh(M, subset_by_index=[T - k, T - 1])
    V = V[:, ::-1].T.copy()
    for i in range(k):
        if i % 2 == 0:
            if V[i].sum() < 0:
                V[i] *= -1
        else:
            pk = int(np.argmax(np.abs(V[i, : T // 2])))
            if V[i, :pk].sum() < 0:
                V[i] *= -1
    return V


def sinc_concentration(V, nw=NW):
    """Rayleigh quotients v^T A v with the dense sinc kernel A[m,n] = sin(2 pi W (m-n)) / (pi (m-n)),
    A[m,m] = 2W: the fraction of each taper's energy inside |f| <= W (Slepian 1978)."""
    T = V.shape[1]
    W = nw / T
    d = np.subtract.outer(np.arange(T), np.arange(T)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.sin(2 * np.pi * W * d) / (np.pi * d)
    A[np.arange(T), np.arange(T)] = 2 * W
    return np.einsum("kt,ts,ks->k", V, A, V)


def band_quantities(S, f):
    P = S.shape[-1]
    iu = [(i, j) for i in range(P) for j in range(i + 1, P)]
    out = []
    for lo, hi in CSM_BANDS:
        m = (f >= lo) & (f < hi)
        Sb = S[m].mean(axis=0)
        row = [Sb[i, i].real for i in range(P)]
        for i, j in iu:
            row += [Sb[i, j].real, Sb[i, j].imag]
        out.append(row)
    return np.array(out)


def simulate_var(A_list, Sigma, T, rng, burn=1000):
    P = Sigma.shape[0]
    e = rng.standard_normal((T + burn, P)) @ np.linalg.cholesky(Sigma).T
    x = np.zeros((T + burn, P))
    for t in range(len(A_list), T + burn):
        acc = e[t].copy()
        for k, A in enumerate(A_list, start=1):
            acc += A @ x[t - k]
        x[t] = acc
    return x[burn:].T


def phase_slope_matrix(z):
    S, fr = csm(z, nonnegative=True)
    f = ks.nonneg(fr)
    m = (f > 0) & (f < 0.1)
    P = z.shape[0]
    D = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i != j:
                D[i, j] = np.polyfit(f[m], np.unwrap(np.angle(S[m, i, j])), 1)[0] / (2 * np.pi)
    return D


def bonferroni_z(m, alpha=0.05):
    return float(norm.ppf(1 - alpha / (2 * m)))


# ------------------------------------------------------------- mne second opinion
def mne_coherency(x, patched):
    """Complex coherency from mne_connectivity.spectral_connectivity_epochs(method='cohy',
    mode='multitaper', sfreq=1, mt_bandwidth=2 NW / T, mt_adaptive=False, mt_low_bias=True) for
    every ordered pair (seed i, target j) -> C[f, i, j]; n_fft = T (equal to spectral_connectivity's
    next_fast_len(T) for the 5-smooth T used).  ``patched``: mne_connectivity's taper call
    (_compute_mt_params) is replaced by symmetric scipy DPSS (NW=3, K=5) with unit eigenvalue weights,
    i.e. the library's estimator; otherwise mne's defaults (periodic DPSS sym=False, sqrt(eigenvalue)
    taper weights)."""
    import mne
    import mne_connectivity
    import mne_connectivity.spectral.epochs as ep
    mne.set_log_level("ERROR")
    P, T = x.shape
    seeds = np.array([i for i in range(P) for j in range(P) if i != j])
    targets = np.array([j for i in range(P) for j in range(P) if i != j])
    orig = ep._compute_mt_params
    if patched:
        def fake(n_times, sfreq, bandwidth, low_bias, adaptive, verbose=None):
            w, _ = ks.dpss_tapers(n_times, NW, K)
            return w, np.ones(w.shape[0]), False
        ep._compute_mt_params = fake
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            con = mne_connectivity.spectral_connectivity_epochs(
                x[None], method="cohy", indices=(seeds, targets), sfreq=1.0, mode="multitaper",
                fmin=0.0, fmax=0.5, mt_bandwidth=2 * NW / T, mt_adaptive=False, mt_low_bias=True,
                faverage=False, verbose=False)
    finally:
        ep._compute_mt_params = orig
    d = con.get_data()
    f = np.asarray(con.freqs)
    C = np.full((len(f), P, P), np.nan, dtype=complex)
    for k, (i, j) in enumerate(zip(seeds, targets)):
        C[:, i, j] = d[k]
    return C, f


def lambda_weighted_coherency(x):
    """mne's default estimator rebuilt from scipy: periodic DPSS (sym=False, Kmax=int(2 NW)),
    low-bias (> 0.9), weights sqrt(lambda) -> S = sum lambda_k X_i conj X_j / sum lambda_k."""
    P, T = x.shape
    w, lam = scipy.signal.windows.dpss(T, NW, Kmax=int(2 * NW), sym=False, return_ratios=True)
    keep = lam > 0.9
    w, lam = w[keep], lam[keep]
    xc = x - x.mean(axis=1, keepdims=True)
    X = np.fft.rfft(w[:, None, :] * xc[None], axis=-1)            # (K, P, nf)
    S = np.einsum("k,kif,kjf->fij", lam, X, np.conj(X)) / lam.sum()
    return ks.coherency(S, nan_diagonal=True)


# ------------------------------------------------------------------ run handlers
def h_csm(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    S, _ = csm(z)
    sc = float(np.abs(S).max())
    return dict(value=ri(S), criterion="tolerance", rtol=1e-10, atol=1e-10 * sc,
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_csm -> _kernels_spectral.multitaper_fft + cross_spectral_matrix "
                                 f"(scipy {scipy.__version__} dpss NW=3 K=5, eig>0.9, constant detrend, "
                                 "n_fft=next_fast_len_235(T), two-sided, S_ij = mean_k X_i conj X_j); "
                                 "pipeline paths z-scored (ddof=1) / least-squares detrended here",
                notes="two-sided CSM stacked [Re, Im]; atol = 1e-10 * max|S|", max_abs_S=sc)


def h_coh(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    S, _ = csm(z, nonnegative=True)
    C = ks.coherency(S, nan_diagonal=True)
    Cp = pyspi_value(run["run_id"])
    Cp = Cp[0] + 1j * Cp[1]
    herm = float(np.nanmax(np.abs(Cp - np.conj(np.swapaxes(Cp, -1, -2)))))
    return dict(value=ri(C), criterion="tolerance", rtol=1e-10, atol=1e-10,
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_coh -> _kernels_spectral.coherency (nan diagonal) on the "
                                 "first (n_fft+1)//2 bins of the kernel CSM",
                notes="coherency on the non-negative grid stacked [Re, Im]",
                pyspi_hermitian_residual=herm)


def h_cohmne(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    C, f = mne_coherency(z, patched=True)
    fl, nfft = grid(z.shape[1])
    if nfft != z.shape[1]:
        raise RuntimeError("mne n_fft = T differs from next_fast_len(T)")
    import mne_connectivity
    return dict(value=ri(C[: len(fl)]), criterion="tolerance", rtol=1e-10, atol=1e-10,
                reference_kind="external-pkg",
                reference_detail=f"mne_connectivity=={mne_connectivity.__version__} spectral_connectivity_epochs("
                                 "method='cohy', mode='multitaper', sfreq=1, mt_bandwidth=6/T, mt_adaptive=False, "
                                 "mt_low_bias=True) with its _compute_mt_params patched to symmetric scipy DPSS "
                                 f"(NW=3, K=5) and unit weights ({SELF}:mne_coherency); mne's own rfft, mean "
                                 "removal, CSD assembly and coherency normalisation; Nyquist bin dropped",
                notes="all ordered pairs (seed i, target j) computed by mne, so both orientations are independent",
                mne_freq_max_kept=float(f[len(fl) - 1]))


def h_cohmnedef(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    C, f = mne_coherency(z, patched=False)
    fl, _ = grid(z.shape[1])
    C = C[: len(fl)]
    Cl = lambda_weighted_coherency(z)[: len(fl)]
    Ck = ks.coherency(csm(z, nonnegative=True)[0], nan_diagonal=True)
    Cp = pyspi_value(run["run_id"])
    Cp = Cp[0] + 1j * Cp[1]
    e_l = float(np.nanmax(np.abs(Cl - C)))
    e_k = float(np.nanmax(np.abs(Cp - Ck)))
    d_pm = float(np.nanmax(np.abs(Cp - C)))
    d_abs = float(np.nanmax(np.abs(np.abs(Cp) ** 2 - np.abs(C) ** 2)))
    import mne_connectivity
    return dict(value=ri(C), criterion="qualitative", pass_flag=int(e_l < 1e-10 and e_k < 1e-10),
                reference_kind="external-pkg",
                reference_detail=f"mne_connectivity=={mne_connectivity.__version__} defaults (periodic DPSS sym=False, "
                                 "Kmax=6 then low-bias -> 5, sqrt(eigenvalue) weights); pass = the difference "
                                 "pyspi - mne is fully explained: scipy rebuild of mne's estimator matches mne and "
                                 "pyspi matches the equal-weight symmetric-taper kernel",
                notes="value = mne default coherency [Re, Im]",
                max_abs_pyspi_vs_mne_default=d_pm, max_abs_msc_pyspi_vs_mne_default=d_abs,
                max_abs_lambda_rebuild_vs_mne=e_l, max_abs_pyspi_vs_kernel=e_k)


def h_grid(run):
    n = trunc_of(run)
    f, nfft = grid(n)
    return dict(value=f, criterion="exact", reference_kind="analytic",
                reference_detail=f"{SELF}:grid: n_fft = smallest 2^a 3^b 5^c >= T, f_k = k/n_fft, k < (n_fft+1)//2",
                notes=f"T={n} n_fft={nfft} (even n_fft: Nyquist dropped; odd: top bin (n_fft-1)/(2 n_fft))",
                n_fft=nfft, n_bins=len(f), top_bin=float(f[-1]))


def h_gridsum(run):
    n = trunc_of(run)
    f, nfft = grid(n)
    return dict(value=np.array([nfft, len(f), K, K], dtype=float), criterion="exact", reference_kind="analytic",
                reference_detail=f"{SELF}:h_gridsum: [n_fft, (n_fft+1)//2, K=floor(2NW-1)=5 tapers, n_observations = 1 trial x 5 tapers]",
                notes=f"T={n}")


def h_tapers(run):
    T = run["T"]
    V = slepian_tapers(T)
    return dict(value=V, criterion="tolerance", rtol=0.0, atol=1e-10 * float(np.abs(V).max()),
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:slepian_tapers: scipy.linalg.eigh (dense, subset) of the Slepian commuting "
                                 "tridiagonal matrix, W = 3/T, P&W sign convention",
                notes="library Multitaper.tapers (K, T), unit energy")


def h_eig(run):
    T = run["T"]
    lam = sinc_concentration(slepian_tapers(T))
    return dict(value=lam, criterion="tolerance", rtol=0.0, atol=1e-10, reference_kind="independent-impl",
                reference_detail=f"{SELF}:sinc_concentration: Rayleigh quotient of the dense sinc kernel (W=3/T) "
                                 "with the commuting-matrix eigenvectors",
                notes="concentration eigenvalues of the 5 DPSS; library keeps tapers with eigenvalue > 0.9",
                min_eigenvalue=float(lam.min()))


def h_ident(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    V = slepian_tapers(z.shape[1])
    zc = z - z.mean(axis=1, keepdims=True)
    rhs = ((V ** 2).mean(axis=0)[None, :] * zc ** 2).sum(axis=1)
    return dict(value=np.concatenate([[0.0, 0.0], rhs]), criterion="exact", reference_kind="analytic",
                reference_detail=f"{SELF}:h_ident: Hermitian S = S^H (0), Im S(0) = 0 for real input, Parseval: "
                                 "mean over the n_fft two-sided bins of S_ii = sum_t (1/K sum_k h_k(t)^2) x_c(t)^2 "
                                 "(h_k from slepian_tapers)",
                notes="[max|S - S^H|, max|Im S(0)|, Parseval per channel]")


def h_bandavg(run, nsim=300):
    x, ds = raw(run)
    A_list, Sig = [ds["A1"], ds["A2"]], ds["Sigma"]
    T = x.shape[1]
    f, nfft = grid(T)
    _, _, S_an = ks.var_transfer_function(A_list, f, Sig)
    truth = band_quantities(S_an, f)
    rng = np.random.default_rng(MC_SEED + T)
    sims = []
    for _ in range(nsim):
        xs = simulate_var(A_list, Sig, T, rng)
        S, _ = csm(xs, nonnegative=True)
        sims.append(band_quantities(S, f))
    sims = np.array(sims)
    sd = sims.std(axis=0, ddof=1)
    zb = bonferroni_z(truth.size)
    bias_z = np.abs(sims.mean(axis=0) - truth) / sd
    return dict(value=truth, se=sd * zb / 2.0, criterion="statistical", reference_kind="analytic",
                reference_detail=f"{SELF}:h_bandavg: S(f) = H Sigma H^H from the dataset's A1, A2, Sigma "
                                 "(_kernels_spectral.var_transfer_function == stored S_true) averaged over "
                                 f"bands {CSM_BANDS} (lo <= f < hi) of the library grid; quantities per band: S_ii, "
                                 "Re S_ij, Im S_ij (i<j); se = Monte-Carlo sd of the same band averages of the "
                                 f"kernel multitaper CSM over {nsim} fresh VAR draws, scaled so that 2 se = "
                                 f"Bonferroni z({truth.size} tests, family-wise 5%) * sd",
                notes="statistical: pass if every |pyspi - truth| <= z_bonf * sd",
                n_sim=nsim, z_bonferroni=zb, mc_bias_max_over_sd=float(bias_z.max()))


def _delays(ds):
    d = np.asarray(ds["delay"], dtype=float)
    return d[None, :] - d[:, None]          # D[i, j] = d_j - d_i


def h_phslope(run, nsim=300):
    x, ds = raw(run)
    truth = _delays(ds)
    np.fill_diagonal(truth, np.nan)
    P, T = x.shape
    rng = np.random.default_rng(MC_SEED + 7 * T)
    d = np.asarray(ds["delay"], dtype=int)
    m = int(d.max())
    sims = []
    for _ in range(nsim):
        base = rng.standard_normal(T + m)
        xs = np.empty((P, T))
        xs[0] = base[m:]
        for k in range(1, P):
            xs[k] = base[m - d[k]: m - d[k] + T] + 0.1 * rng.standard_normal(T)
        sims.append(phase_slope_matrix(prepare(xs, run["path"])))
    sims = np.array(sims)
    sd = np.nanstd(sims, axis=0, ddof=1)
    zb = bonferroni_z(P * (P - 1))
    off = offdiag(P)
    return dict(value=truth, se=np.where(off, sd * zb / 2.0, np.nan), criterion="statistical",
                reference_kind="analytic",
                reference_detail=f"{SELF}:h_phslope: x_k(t) = x0(t - d_k) => S_ij = E X_i conj X_j has phase "
                                 "2 pi f (d_j - d_i); slope/(2 pi) of the unwrapped phase over 0 < f < 0.1; se from "
                                 f"{nsim} Monte-Carlo draws of the dataset's generating model (kernel CSM), "
                                 "Bonferroni-scaled",
                notes="D[i, j] = d_j - d_i: positive when row i leads column j",
                mc_bias_max=float(np.nanmax(np.abs(sims.mean(axis=0) - truth))), z_bonferroni=zb)


def h_phdir(run):
    x, ds = raw(run)
    truth = _delays(ds)
    np.fill_diagonal(truth, np.nan)
    sign = np.sign(np.nan_to_num(truth))
    return dict(value=truth, expected_sign=sign, criterion="direction", reference_kind="analytic",
                reference_detail=f"{SELF}:h_phdir: sign of d_j - d_i (delayed_noise d = 0,1,3,5): the phase slope of "
                                 "S[i, j] is positive iff row i leads column j",
                notes="direction of the library CSM phase slope")


def parse_identifier(ident):
    m = re.match(r"^(\w+?)_multitaper_(mean|max)_fs-1_fmin-([0-9-]+)_fmax-([0-9-]+)$", ident)
    meas, stat, a, b = m.groups()
    return meas, stat, float(a.replace("-", ".")), float(b.replace("-", "."))


def per_bin_measure(meas, S):
    if meas == "cohmag":
        return ks.coherence_magnitude(S)                  # |C|^2, library form
    if meas == "icoh":
        return ks.imaginary_coherence(S)                  # |Im C|, library form
    raise ValueError(meas)


def h_spi(run, dataset=None):
    meas, stat, fmin, fmax = parse_identifier(run["identifier"])
    x, _ = raw(run, dataset)
    z = prepare(x, run["path"])
    S, fr = csm(z, nonnegative=True)
    f = ks.nonneg(fr)
    M = per_bin_measure(meas, S)
    fn = np.nanmean if stat == "mean" else np.nanmax
    P = z.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        incl = fn(M[ks.band_mask(f, fmin, fmax, True)], axis=0)
        strict = fn(M[ks.band_mask(f, fmin, fmax, False)], axis=0)
    np.fill_diagonal(incl, np.nan)
    np.fill_diagonal(strict, np.nan)
    extra = {}
    if dataset is None:
        Ap = pyspi_value(run["run_id"])
        extra["max_abs_pyspi_vs_strict_rule"] = float(np.nanmax(np.abs(Ap - strict)))
        extra["n_bins_inclusive"] = int(ks.band_mask(f, fmin, fmax, True).sum())
        extra["n_bins_strict"] = int(ks.band_mask(f, fmin, fmax, False).sum())
        if meas == "icoh" and stat == "mean" and fmin == 0:
            mk = ks.band_mask(f, fmin, fmax, True)
            n = int(mk.sum())
            no_dc = np.nanmean(M[mk & (f > 0)], axis=0)
            extra["icoh_dc_bin_max"] = float(np.nanmax(M[0]))
            extra["max_abs_mean_minus_(n-1)/n_mean_without_dc"] = float(np.nanmax(np.abs(Ap - (n - 1) / n * no_dc)[offdiag(P)]))
    detail = (f"{SELF}:h_spi: kernel CSM -> {'|C|^2' if meas == 'cohmag' else '|Im C|'} per bin -> "
              f"np.{fn.__name__} over the INCLUSIVE set fmin <= f_k <= fmax (documented closed band) of f_k = k/n_fft; "
              "z-score ddof=1 re-implemented")
    if dataset is not None:
        detail += f"; invariance: computed on {dataset} (scaled_offset = scale * {dataset} + offset)"
    return dict(value=incl, criterion="tolerance", rtol=1e-10, atol=1e-12,
                reference_kind="analytic" if dataset is not None else "independent-impl",
                reference_detail=detail, notes=f"{meas} {stat} [{fmin}, {fmax}]", **extra)


def h_invariance(run):
    return h_spi(run, dataset="bivariate_gaussian")


def lfq(z):
    S, fr = csm(z, nonnegative=True)
    f = ks.nonneg(fr)
    p = ks.power(S)
    return p[f < 6.0 / z.shape[1]].sum(axis=0) / p.sum(axis=0)


def h_lfq(run):
    x, _ = raw(run)
    return dict(value=lfq(prepare(x, run["path"])), criterion="tolerance", rtol=1e-10, atol=1e-12,
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:lfq: kernel power over f < 6/T divided by the power over the non-negative grid",
                notes="low-frequency power fraction per channel")


def h_lfqjudge(run):
    T = run["T"]
    base = pyspi_value(f"spectral.multitaper_csm|bivariate_gaussian|{T}|pipeline|lfq")
    tp = pyspi_value(f"spectral.multitaper_csm|trending|{T}|pipeline|lfq")
    td = pyspi_value(f"spectral.multitaper_csm|trending|{T}|pipeline-detrend|lfq")
    slope = np.asarray(make_data.load("trending", T)["slope"], dtype=float)
    tr = slope != 0
    ok = bool(np.all(tp[tr] > 5 * base[tr]) and np.allclose(tp[~tr], base[~tr], rtol=1e-12, atol=0)
              and np.all(td <= 2 * base) and np.all(td >= base / 2))
    return dict(value=tp, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                reference_detail=f"{SELF}:h_lfqjudge: pass if, without detrend, every channel with a nonzero slope "
                                 "puts > 5x the low-frequency (f < 6/T) power fraction of the same untrended draw into "
                                 "those bins and the zero-slope channel equals the untrended draw; and with "
                                 "pipeline-detrend every fraction is within a factor 2 of the untrended draw",
                notes="constant detrend in the library removes only the mean",
                frac_trending_pipeline=tp.tolist(), frac_trending_detrend=td.tolist(), frac_untrended=base.tolist())


def masks_for(n):
    f, nfft = grid(n)
    mv = np.stack([ks.band_mask(f, a, b, True) for a, b in ALL_BANDS]).astype(float)
    bv = np.stack([ks.band_mask(f, a, b, False) for a, b in ALL_BANDS]).astype(float)
    return f, nfft, mv, bv


def h_masks(run):
    n = trunc_of(run) or run["T"]
    f, nfft, mv, bv = masks_for(n)
    return dict(value=np.stack([mv, bv]), criterion="exact", reference_kind="analytic",
                reference_detail=f"{SELF}:h_masks: on f_k = k/n_fft, multivariate = inclusive fmin <= f <= fmax, "
                                 "bivariate = strict fmin < f < fmax; bands = 3 config + 10 EEG (retained185.yaml exact floats)",
                notes=f"T={n}",
                counts_inclusive=mv.sum(axis=1).astype(int).tolist(), counts_strict=bv.sum(axis=1).astype(int).tolist())


def spy_prediction(n):
    f, nfft = grid(n)
    df = f[1] - f[0]
    rows, nb = [], []
    for a, b in ALL_BANDS:
        res = (b - a) / 50
        step = int(np.ceil(res / df))
        strict = np.nonzero(ks.band_mask(f, a, b, False))[0]
        k = len(strict[::step])
        psi_code = 0 if k >= 2 else 1
        gd_code = 2 if len(strict) == 0 else 0
        rows.append([2, 0, a, b, res, k, psi_code, 2, 0, a, b, res, gd_code])
        nb.append(k)
    return np.array(rows, dtype=float), nb


def h_spy(run):
    n = trunc_of(run) or run["T"]
    pred, nb = spy_prediction(n)
    return dict(value=pred, criterion="exact", reference_kind="analytic",
                reference_detail=f"{SELF}:spy_prediction: pyspi first calls phase_slope_index()/group_delay() without "
                                 "arguments (TypeError), then with frequencies_of_interest=[fmin, fmax], "
                                 "frequency_resolution=(fmax-fmin)/50 (spectral.py:296-317); PSI sums over "
                                 "strict-band bins [::ceil(resolution * n_fft)] (connectivity.py:1086-1104); PSI "
                                 "fails (IndexError) with < 2 bins; GD raises ValueError with 0 strict bins",
                notes="per band: PSI [calls, kwargs#1, fmin, fmax, res, n_bins, code], GD [calls, kwargs#1, fmin, fmax, res, code]",
                psi_bins_used=nb)


SPI_NAMES = ["cohmag_mean", "cohmag_max", "psi", "gd_delay", "dcoh_mean"]


def h_eeg(run):
    n = trunc_of(run) or run["T"]
    x, _ = raw(run)
    P = x.shape[0]
    noff = P * (P - 1)
    f, nfft, mv, bv = masks_for(n)
    _, nb = spy_prediction(n)
    got = pyspi_value(run["run_id"])
    ok, lines = True, []
    for b in range(len(EEG_BANDS)):
        ni, ns, npsi = int(mv[3 + b].sum()), int(bv[3 + b].sum()), nb[3 + b]
        pred = [
            [0, noff if ni == 0 else 0, 0 if ni == 0 else noff],      # cohmag mean (inclusive)
            [2, noff, 0] if ni == 0 else [0, 0, noff],                # cohmag max: empty -> ValueError
            [1, noff, 0] if npsi < 2 else [0, 0, noff],               # psi: < 2 bins -> IndexError
            [2, noff, 0] if ns == 0 else ([0, noff, 0] if ns < 3 else [0, None, None]),  # gd
            [0, noff, 0] if ns == 0 else [0, 0, noff],                # dcoh mean (strict, bivariate)
        ]
        for s, p in enumerate(pred):
            g = got[b, s]
            match = all(pv is None or int(g[q]) == pv for q, pv in enumerate(p))
            ok &= match
            if not match or p[0] != 0 or (p[1] is not None and p[1] > 0) or (s == 3 and g[1] > 0):
                lines.append(f"band {EEG_BANDS[b][0] * 1024:g}-{EEG_BANDS[b][1] * 1024:g}/1024 {SPI_NAMES[s]}: "
                             f"bins incl/strict/psi {ni}/{ns}/{npsi}, code {int(g[0])}, NaN {int(g[1])}/{noff}"
                             f"{'' if match else ' MISMATCH predicted ' + str(p)}")
    return dict(value=got, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                reference_detail=f"{SELF}:h_eeg: outcome predicted from the bin counts on k/n_fft: inclusive "
                                 "count 0 -> nanmean NaN / nanmax ValueError; PSI < 2 bins -> IndexError; GD 0 strict "
                                 "bins -> ValueError, 1-2 -> all NaN (min_group_size 3), >= 3 -> data dependent "
                                 "(not predicted); dcoh 0 strict bins -> NaN",
                notes="codes: 0 ok, 1 IndexError, 2 ValueError (Calculator turns 1/2 into an all-NaN table); "
                      + " | ".join(lines))


def h_fftcount(run):
    x, _ = raw(run)
    P = x.shape[0]
    return dict(value=np.array([1, 1, 0, 1, 1, 1, 1, P * (P - 1)], dtype=float), criterion="exact",
                reference_kind="analytic",
                reference_detail=f"{SELF}:h_fftcount: expected from the cache logic of spectral.py:64-83 (first insert keyed "
                                 "by the string measure, later lookups by the tuple key; PSI/GD keyed per band) and "
                                 ":106-126 (one 2-channel transform per ordered pair)",
                notes="order: cohmag mean full, cohmag mean low, cohmag max full, icoh, psi full, psi low, gd, dcoh")


def h_id025(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    S, fr = csm(z, nonnegative=True)
    f = ks.nonneg(fr)
    k0 = int(np.argmin(np.abs(f - 0.25)))
    msc = ks.coherence_magnitude(S)[k0]
    nl, nh, nf_ = (int(ks.band_mask(f, a, b, True).sum()) for a, b in ((0, 0.25), (0.25, 0.5), (0, 0.5)))
    A = pyspi_value(run["run_id"])
    off = offdiag(z.shape[0])
    resid = (nl * A[0] + nh * A[1] - nf_ * A[2] - msc)[off]
    r = float(np.abs(resid).max())
    return dict(value=A, criterion="qualitative", pass_flag=int(r < 1e-10 and f[k0] == 0.25),
                reference_kind="analytic",
                reference_detail=f"{SELF}:h_id025: inclusive rule => n_low*mean_low + n_high*mean_high - n_full*mean_full "
                                 "= MSC(f=0.25) (the 0.25 bin counted in both half bands); MSC(0.25) from the kernel CSM",
                notes=f"counts low/high/full {nl}/{nh}/{nf_}; MSC(0.25) off-diagonal min {msc[off].min():.7f}",
                identity_residual_max=r, msc_025_min=float(msc[off].min()))


def h_bvagg(run):
    A = pyspi_value(run["run_id"])
    n = trunc_of(run) or run["T"]
    f, nfft = grid(n)
    nf = len(f)
    P = A.shape[-1]
    V = A.copy()
    per = [A[:nf], A[nf: 2 * nf]]
    k, maxdiff_incl = 2 * nf, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in range(2):
            for fn in (np.nanmean, np.nanmax):
                for a, b in CONFIG_BANDS:
                    v = fn(per[m][ks.band_mask(f, a, b, False)], axis=0)
                    vi = fn(per[m][ks.band_mask(f, a, b, True)], axis=0)
                    np.fill_diagonal(v, np.nan)
                    np.fill_diagonal(vi, np.nan)
                    V[k] = v
                    maxdiff_incl = max(maxdiff_incl, float(np.nanmax(np.abs(A[k] - vi))))
                    k += 1
    extra = {}
    if run["dataset"].startswith("narrowband_f0-0.25"):
        k0 = int(np.argmin(np.abs(f - 0.25)))
        ns = [int(ks.band_mask(f, a, b, False).sum()) for a, b in CONFIG_BANDS]
        mfull, mlow, mhigh = A[2 * nf], A[2 * nf + 1], A[2 * nf + 2]
        resid = ns[0] * mfull - ns[1] * mlow - ns[2] * mhigh - per[0][k0]
        extra["strict_identity_residual_max"] = float(np.nanmax(np.abs(resid[offdiag(P)])))
        extra["strict_counts_full_low_high"] = ns
    return dict(value=V, criterion="exact", reference_kind="same-lineage",
                reference_detail=f"{SELF}:h_bvagg: per-bin directed_coherence / directed_transfer_function from "
                                 "spectral_connectivity (2-channel [i, j] Connectivity, entry [0, 1], via the probe), "
                                 "aggregated here with np.nanmean / np.nanmax over the STRICT set fmin < f_k < fmax",
                notes="slices [0:2 nf] are the library per-bin values copied; the 12 summaries are checked",
                max_abs_pyspi_vs_inclusive_rule=maxdiff_incl, **extra)


HANDLERS = {"csm": h_csm, "coh": h_coh, "cohmne": h_cohmne, "cohmnedef": h_cohmnedef, "grid": h_grid,
            "gridsum": h_gridsum, "tapers": h_tapers, "eig": h_eig, "ident": h_ident, "bandavg": h_bandavg,
            "phslope": h_phslope, "phdir": h_phdir, "spi": h_spi, "invariance": h_invariance, "lfq": h_lfq,
            "lfqjudge": h_lfqjudge, "masks": h_masks, "spy": h_spy, "eeg": h_eeg, "fftcount": h_fftcount,
            "id025": h_id025, "bvagg": h_bvagg}


def save(rid, d):
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = np.array(v)
        elif isinstance(v, (list, tuple)):
            out[k] = np.asarray(v, dtype=np.float64)
        else:
            out[k] = np.asarray(v)
    np.savez_compressed(OUT_REF / f"{rid}.npz", **out)


def main():
    OUT_REF.mkdir(parents=True, exist_ok=True)
    runs = json.loads((CHECKS / "manifest" / f"{FAMILY}.json").read_text())["runs"]
    only = set(sys.argv[1:])
    n_ok = n_err = 0
    for run in runs:
        kind = kind_of(run)
        if only and kind not in only:
            continue
        rid = run["run_id"]
        try:
            save(rid, HANDLERS[kind](run))
            n_ok += 1
        except Exception as exc:  # report, keep going; a missing ref shows as "no reference"
            n_err += 1
            print(f"REF ERROR {rid}: {exc!r}", file=sys.stderr)
    print(f"{FAMILY} refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()

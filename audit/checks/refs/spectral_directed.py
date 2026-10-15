#!/usr/bin/env python3
"""References for the spectral_directed family (COMPARISON venv; no pyspi, no spectral_connectivity).

Units: spectral.dtf, spectral.dcoh, spectral.pdcoh, spectral.gpdcoh, spectral.ddtf (pyspi's
bivariate multitaper + Wilson directed measures, delegated to spectral_connectivity 1.1.2) and the
infrastructure unit spectral.wilson_factorisation.  One reference per run_id of
manifest/spectral_directed.json, written to out/ref/<run_id>.npz (contract in compare.py).

Run tag = last ``|`` field of the run_id: ``kind[-opt][:stat-band | :unit][~pIJ]``; ``~pIJ`` is
the manifest's process selection (for pair probes the selected pair IS the ordered array
[x_i, x_j] pyspi factorises for A[i, j]).

Independence: the estimator chain is re-implemented with ``_kernels_spectral`` (scipy dpss
tapers + numpy FFT multitaper, numpy Wilson 1972 factorisation in the Dhamala et al. 2008 form,
the canonical directed measures and literal transcriptions of the library formulas, none of which
import spectral_connectivity); preprocessing (z-score ddof=1, least-squares linear detrend) is
re-implemented here.  Analytic truths come from the VAR coefficients of make_data
(``var_transfer_function``: A(f) = I - sum_k A_k e^{-2 pi i f k}, H = A^-1, S = H Sigma H^H) and,
for marginal (bivariate) population values, from Wilson factorisation of the exact marginal
spectrum at tolerance 1e-13.  Statistical criteria use a Monte Carlo of the same independent
estimator on fresh draws of the generating model (seeded, cached under
out/exchange/spectral_directed/mc_*.npz); ``se`` is the Monte Carlo SD scaled by
z_Bonferroni(m)/2 over the m off-diagonal entries so that compare.py's 2*se rule is a familywise
5% test.  ``ident`` runs (pdcoh vs dtf) and the reuse of pyspi's own outputs are same-lineage.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks")
sys.path.insert(0, "/media/dan/Data/git/pyspi/audit/checks/refs")
import make_data  # noqa: E402
import _kernels_spectral as ks  # noqa: E402

CHECKS = Path("/media/dan/Data/git/pyspi/audit/checks")
FAMILY = "spectral_directed"
OUT_REF = CHECKS / "out" / "ref"
OUT_PY = CHECKS / "out" / "pyspi"
EXCHANGE = CHECKS / "out" / "exchange" / FAMILY
SELF = "refs/spectral_directed.py"
UNITS = ("dtf", "dcoh", "pdcoh", "gpdcoh", "ddtf")
BANDS = {"full": (0.0, 0.5), "low": (0.0, 0.25), "high": (0.25, 0.5)}
STATS = {"mean": np.nanmean, "max": np.nanmax}
MC_SEED = 20261015
MC_R = {64: 400, 256: 400, 1024: 200, 4096: 100}
W = "spectral.wilson_factorisation"
KDETAIL = ("_kernels_spectral.multitaper_fft (dpss NW=3, K=5, constant detrend, n_fft=next_fast_len) + "
           "cross_spectral_matrix + wilson_factorise (Wilson 1972 / Dhamala 2008, tol 1e-8, 60 it)")


# ----------------------------------------------------------------------- parsing / data
def parse(run):
    tag = run["run_id"].split("|")[-1]
    base = tag.split("~")[0]
    kind_full, _, sub = base.partition(":")
    kind, _, opt = kind_full.partition("-")
    stat, band = (sub.split("-") + [None])[:2] if "-" in sub else (None, None)
    return dict(kind=kind, opt=opt, sub=sub, stat=stat, band=band,
                unit=run["unit_id"].split(".")[1])


def ptag(p):
    return "" if p is None else "~p" + "".join(str(i) for i in p)


def safe_name(rid):
    return rid.replace("|", "__").replace(":", "_").replace("~", "_")


def raw(run, dataset=None, T=None):
    ds = make_data.load(dataset or run["dataset"], T or run["T"])
    x = np.asarray(ds["x"], dtype=np.float64)
    if run.get("processes") is not None:
        x = x[list(run["processes"])]
    return x, ds


def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)


def linear_detrend(x):
    t = np.arange(x.shape[1], dtype=np.float64)
    G = np.stack([t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(G, x.T, rcond=None)
    return x - (G @ coef).T


def prepare(x, path):
    if path == "kernel":
        return np.asarray(x, dtype=np.float64)
    if path == "pipeline":
        return zscore(x)
    if path == "pipeline-detrend":
        return zscore(linear_detrend(x))
    raise ValueError(path)


def pyspi_npz(rid, key="value"):
    with np.load(OUT_PY / f"{rid}.npz") as z:
        return np.asarray(z[key])


def offdiag(P):
    return ~np.eye(P, dtype=bool)


def bonferroni_z(m, alpha=0.05):
    return float(norm.ppf(1 - alpha / (2 * max(int(m), 1))))


def ri(a):
    return np.stack([np.real(a), np.imag(a)])


def from_ri(a):
    return np.asarray(a[0]) + 1j * np.asarray(a[1])


# ----------------------------------------------------------------------- measures
def lib_measure(u, Hn, Sigma):
    """Literal transcription of the spectral_connectivity 1.1.2 measure (row = target, col = source)."""
    if u == "dtf":
        return ks.dtf(Hn)                                          # SC/connectivity.py:758-779
    if u == "dcoh":
        return ks.directed_coherence_library_as_coded(Hn, Sigma)   # SC/connectivity.py:781-805
    if u == "pdcoh":
        return ks.pdc(np.linalg.inv(Hn))                           # SC/connectivity.py:807-844
    if u == "gpdcoh":
        return ks.gpdc(np.linalg.inv(Hn), Sigma)                   # SC/connectivity.py:846-875
    if u == "ddtf":
        return ks.ddtf_library_as_coded(Hn)                        # SC/connectivity.py:877-900
    raise ValueError(u)


def canon_measure(u, Hn, Sigma, Sn, form="squared"):
    """Canonical (paper) definition; ``form`` squared or modulus."""
    if u == "dtf":
        return ks.dtf(Hn, form)                                    # Kaminski & Blinowska 1991
    if u == "dcoh":
        return ks.directed_coherence(Hn, Sigma, form)              # Baccala et al. 1998
    if u == "pdcoh":
        return ks.pdc(np.linalg.inv(Hn), form)                     # Baccala & Sameshima 2001
    if u == "gpdcoh":
        return ks.gpdc(np.linalg.inv(Hn), Sigma, form)             # Baccala et al. 2007
    if u == "ddtf":
        return ks.ddtf(Hn, Sn, form)                               # Korzeniewska et al. 2003
    raise ValueError(u)


def grid(n_fft):
    return np.arange((n_fft + 1) // 2) / n_fft


def band_reduce(L, n_fft, stat, band, inclusive=False):
    """pyspi bivariate reducer (spectral.py:128-134): statfn(L[strict band, 0, 1])."""
    f = grid(n_fft)
    sel = ks.band_mask(f, *BANDS[band], inclusive=inclusive)
    return float(STATS[stat](L[sel, 0, 1])), int(sel.sum())


def kchain(z, n_fft=None):
    """Independent multitaper + Wilson on the (P, T) array z (row order = channel order)."""
    co, _ = ks.multitaper_fft(z, n_fft=n_fft)
    S = ks.cross_spectral_matrix(co)
    H, Sg, info = ks.wilson_factorise(S)
    return dict(S=S, Hn=ks.nonneg(H), Sn=ks.nonneg(S), Sigma=Sg, info=info, n_fft=S.shape[0])


_CHAIN = {}


def pair_chain(key, z, i, j):
    k = (key, i, j)
    if k not in _CHAIN:
        _CHAIN[k] = kchain(z[[i, j]])
    return _CHAIN[k]


def data_key(run, dataset=None, T=None):
    return (dataset or run["dataset"], T or run["T"], run["path"], str(run.get("processes")))


def replicate(z, u, stat, band, key, inclusive=False):
    """Pyspi matrix replicated with the independent chain: A[i, j] = stat over the band of
    L[0, 1] on [z_i, z_j].  Returns A, n_bins, max recon err, all converged."""
    P = z.shape[0]
    A = np.full((P, P), np.nan)
    nb, rec, conv = 0, 0.0, True
    for j in range(P):
        for i in range(P):
            if i == j:
                continue
            c = pair_chain(key, z, i, j)
            L = lib_measure(u, c["Hn"], c["Sigma"])
            A[i, j], nb = band_reduce(L, c["n_fft"], stat, band, inclusive)
            rec = max(rec, c["info"]["recon_err"])
            conv = conv and c["info"]["converged"]
    return A, nb, rec, conv


# ----------------------------------------------------------------------- models / truth
def var_model(name):
    d = make_data.load(name, 1024)
    if name == "gauss_ar1_pair":
        return [np.asarray(d["A"])], np.asarray(d["Sigma_E"]), np.asarray(d["Sigma_X"])
    return [np.asarray(d["A1"]), np.asarray(d["A2"])], np.asarray(d["Sigma"]), np.asarray(d["Gamma0"])


def exact_full(name, n, path):
    """Exact two-sided H, A(f), S and Sigma of the dataset model on fftfreq(n), scaled to unit
    marginal variance when the path z-scores (H' = D H D^-1, Sigma' = D Sigma D, S' = D S D)."""
    A_list, Sigma, G0 = var_model(name)
    H, Af, S = ks.var_transfer_function(A_list, np.fft.fftfreq(n), Sigma)
    if path != "kernel":
        d = 1.0 / np.sqrt(np.diag(G0))
        D, Di = np.diag(d), np.diag(1.0 / d)
        H, S, Sigma = D @ H @ Di, D @ S @ D, D @ Sigma @ D
    return H, S, Sigma


def delayed_exact(n, path):
    """Exact two-sided S of delayed_noise: X_k = e^{-2 pi i f d_k} X_0 + N_k, var N_k = 0.01 (k > 0)."""
    d = np.array(make_data.D6_DELAYS, dtype=float)
    nv = np.array([0.0] + [make_data.D6_NOISE ** 2] * (len(d) - 1))
    f = np.fft.fftfreq(n)
    E = np.exp(-2j * np.pi * f[:, None] * d[None, :])           # (n, P)
    S = E[:, :, None] * np.conj(E[:, None, :]) + np.diag(nv)[None]
    if path != "kernel":
        s = 1.0 / np.sqrt(1.0 + nv)
        S = S * s[None, :, None] * s[None, None, :]
    return S


def exact_S(name, n, path):
    if name == "delayed_noise":
        return delayed_exact(n, path)
    return exact_full(name, n, path)[1]


_POP = {}


def population(name, n, path, u, stat, band, processes=None):
    """Population value of pyspi's bivariate quantity: for each ordered pair the exact MARGINAL
    2-channel spectrum is Wilson-factorised (tol 1e-13) and the library formula is band-reduced."""
    key = (name, n, path, u, stat, band, str(processes))
    if key in _POP:
        return _POP[key]
    if name == "iid_gaussian":
        P = len(processes) if processes is not None else 5
        A = np.zeros((P, P))
        np.fill_diagonal(A, np.nan)
        _POP[key] = A
        return A
    S = exact_S(name, n, path)
    if processes is not None:
        S = S[:, processes][:, :, processes]
    P = S.shape[-1]
    A = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            idx = np.array([i, j])
            H, Sg, info = ks.wilson_factorise(S[:, idx[:, None], idx[None, :]], tol=1e-13, max_iter=2000)
            L = lib_measure(u, ks.nonneg(H), Sg)
            A[i, j], _ = band_reduce(L, n, stat, band)
    _POP[key] = A
    return A


def multivariate_truth(name, n, path, u, stat, band):
    """Full-model (P-channel, conditioned) value of the library formula from the analytic H."""
    H, S, Sigma = exact_full(name, n, path)
    L = lib_measure(u, ks.nonneg(H), Sigma)
    P = Sigma.shape[0]
    A = np.full((P, P), np.nan)
    f = grid(n)
    sel = ks.band_mask(f, *BANDS[band], inclusive=False)
    for i in range(P):
        for j in range(P):
            if i != j:
                A[i, j] = STATS[stat](L[sel, i, j])
    return A


def simulate(name, T, rng, processes=None):
    if name == "delayed_noise":
        m = max(make_data.D6_DELAYS)
        base = rng.standard_normal(T + m)
        x = np.empty((len(make_data.D6_DELAYS), T))
        x[0] = base[m:]
        for k, d in enumerate(make_data.D6_DELAYS[1:], start=1):
            x[k] = base[m - d:m - d + T] + make_data.D6_NOISE * rng.standard_normal(T)
        return x if processes is None else x[list(processes)]
    if name == "iid_gaussian":
        P = len(processes) if processes is not None else 5
        return rng.standard_normal((P, T))
    A_list, Sigma, _ = var_model(name)
    return make_data.simulate_var(A_list, Sigma, T, rng)


def mc(name, T, path, processes=None):
    """Monte Carlo of the independent estimator: returns {u: {(stat, band): (R, P, P)}} for the
    stat/band pairs used by truth/mc/chain runs.  Cached on disk."""
    R = MC_R[T]
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    f = EXCHANGE / f"mc_{name}_{T}_{path}_{'all' if processes is None else ''.join(map(str, processes))}_R{R}_s{MC_SEED}.npz"
    sbs = [("mean", "full"), ("mean", "low"), ("mean", "high")]
    if f.is_file():
        with np.load(f) as z:
            return {u: {sb: z[f"{u}_{sb[0]}_{sb[1]}"] for sb in sbs} for u in UNITS}
    seed0 = MC_SEED + 7919 * T + sum(ord(c) for c in name)
    out = {u: {sb: [] for sb in sbs} for u in UNITS}
    for r in range(R):
        x = simulate(name, T, np.random.default_rng(seed0 + r), processes)
        z = prepare(x, path)
        P = z.shape[0]
        mats = {u: {sb: np.full((P, P), np.nan) for sb in sbs} for u in UNITS}
        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                c = kchain(z[[i, j]])
                for u in UNITS:
                    L = lib_measure(u, c["Hn"], c["Sigma"])
                    for sb in sbs:
                        mats[u][sb][i, j], _ = band_reduce(L, c["n_fft"], *sb)
        for u in UNITS:
            for sb in sbs:
                out[u][sb].append(mats[u][sb])
    res = {u: {sb: np.stack(out[u][sb]) for sb in sbs} for u in UNITS}
    np.savez_compressed(f, **{f"{u}_{sb[0]}_{sb[1]}": res[u][sb] for u in UNITS for sb in sbs})
    return res


def mc_summary(name, T, path, u, stat, band, processes=None, value=None):
    """Monte Carlo mean, SD, R and (if ``value`` is given) the smallest per-entry empirical
    two-sided tail fraction of ``value`` among the draws (2 * min(P(X <= v), P(X >= v)))."""
    arr = mc(name, T, path, processes)[u][(stat, band)]
    mean, sd = np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)
    if value is None:
        return mean, sd, arr.shape[0]
    with np.errstate(invalid="ignore"):
        lo = (arr <= value[None]).mean(axis=0)
        hi = (arr >= value[None]).mean(axis=0)
    tail = 2 * np.minimum(lo, hi)
    np.fill_diagonal(tail, np.nan)
    return mean, sd, arr.shape[0], float(np.nanmin(tail))


# ----------------------------------------------------------------------- measure-unit handlers
def h_rep(run):
    p = parse(run)
    x, _ = raw(run)
    z = prepare(x, run["path"])
    key = data_key(run)
    A, nb, rec, conv = replicate(z, p["unit"], p["stat"], p["band"], key)
    Ai, nbi, _, _ = replicate(z, p["unit"], p["stat"], p["band"], key, inclusive=True)
    S = pyspi_npz(run["run_id"])
    off = offdiag(z.shape[0])
    extra = dict(n_bins_strict=nb, n_bins_inclusive=nbi,
                 max_abs_pyspi_vs_inclusive_rule=float(np.nanmax(np.abs(S - Ai)[off])),
                 max_wilson_recon_err=rec, all_wilson_converged=int(conv),
                 max_offdiag_value=float(np.nanmax(S[off])),
                 max_abs_asym=float(np.nanmax(np.abs(S - S.T)[off])))
    return dict(value=A, criterion="tolerance", rtol=1e-6, atol=1e-8, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_rep: z-score(ddof=1)/detrend re-implemented; per ordered pair "
                                 f"{KDETAIL}; library formula transcribed (lib_measure); strict band fmin<f<fmax "
                                 f"on k/n_fft; np.nan{p['stat']}",
                notes="pyspi matrix replicated independently; inclusive-rule diff shows which band rule pyspi uses",
                **extra)


def _formula_extras(u, Hn, Sigma, Sn, n_fft):
    """Canonical-vs-library numbers and the 2x2 identities, all from ONE set of factors."""
    lib = lib_measure(u, Hn, Sigma)
    can_sq = canon_measure(u, Hn, Sigma, Sn, "squared")
    can_mod = canon_measure(u, Hn, Sigma, Sn, "modulus")
    f = grid(n_fft)
    sel = ks.band_mask(f, 0.0, 0.5, inclusive=False)
    e = dict(max_abs_lib_vs_canonical_squared=float(np.abs(lib - can_sq)[:, [0, 1], [1, 0]].max()),
             max_abs_lib_vs_canonical_modulus=float(np.abs(lib - can_mod)[:, [0, 1], [1, 0]].max()),
             lib_band_mean_10=float(lib[sel, 1, 0].mean()), lib_band_mean_01=float(lib[sel, 0, 1].mean()),
             canonical_squared_band_mean_10=float(can_sq[sel, 1, 0].mean()),
             canonical_squared_band_mean_01=float(can_sq[sel, 0, 1].mean()),
             canonical_modulus_band_mean_10=float(can_mod[sel, 1, 0].mean()),
             canonical_modulus_band_mean_01=float(can_mod[sel, 0, 1].mean()),
             lib_max_offdiag=float(lib[:, [0, 1], [1, 0]].max()))
    Abar = np.linalg.inv(Hn)
    dtf_ = ks.dtf(Hn)
    pdc_ = ks.pdc(Abar)
    if u in ("dtf", "pdcoh"):
        det = np.linalg.det(Abar)[:, None, None]
        adj = np.stack([np.stack([Abar[:, 1, 1], -Abar[:, 0, 1]], -1),
                        np.stack([-Abar[:, 1, 0], Abar[:, 0, 0]], -1)], -2)
        e["max_abs_H_minus_adjA_over_detA"] = float(np.abs(Hn - adj / det).max())
        e["max_abs_offdiag_DTF_minus_PDC"] = float(np.abs(dtf_ - pdc_)[:, [0, 1], [1, 0]].max())
    if u == "dcoh":
        e["max_abs_lib_minus_absH_sqrtDTF"] = float(np.abs(lib - np.abs(Hn) * np.sqrt(dtf_))[:, [0, 1], [1, 0]].max())
        e["max_abs_lib_minus_DTF"] = float(np.abs(lib - dtf_)[:, [0, 1], [1, 0]].max())
    if u in ("dcoh", "gpdcoh"):
        dc = ks.directed_coherence(Hn, Sigma, "squared")
        gp = ks.gpdc(Abar, Sigma, "squared")
        e["max_abs_offdiag_canonicalDC2_minus_gPDC2"] = float(np.abs(dc - gp)[:, [0, 1], [1, 0]].max())
    if u == "ddtf":
        ff = ks.ddtf_ff(Hn)
        e["max_abs_lib_minus_sqrtff_sqrtDTF"] = float(np.abs(lib - np.sqrt(ff) * np.sqrt(dtf_))[:, [0, 1], [1, 0]].max())
        pc = ks.partial_coherence(Sn)
        coh = np.abs(Sn) ** 2 / (np.real(np.diagonal(Sn, axis1=1, axis2=2))[:, :, None]
                                 * np.real(np.diagonal(Sn, axis1=1, axis2=2))[:, None, :])
        e["max_abs_2x2_partialcoh_minus_coh"] = float(np.abs(pc - coh)[:, [0, 1], [1, 0]].max())
        e["max_abs_lib_vs_K2003_ffDTF2_pcoh2"] = e["max_abs_lib_vs_canonical_squared"]
        e["ff_band_sum_check"] = float(ff[:, 1, :].sum())
    return e


def h_formula(run):
    p = parse(run)
    with np.load(EXCHANGE / f"{safe_name(run['run_id'])}.npz") as z:
        S, Hn, Sigma = z["S"], z["H"], z["Sigma"]
    Sn = ks.nonneg(S)
    V = lib_measure(p["unit"], Hn, Sigma)
    recon = np.abs(Hn @ Sigma @ np.conj(np.swapaxes(Hn, -1, -2)) - Sn).max() / np.abs(S).max()
    extra = _formula_extras(p["unit"], Hn, Sigma, Sn, S.shape[0])
    extra["library_factor_recon_err_nonneg"] = float(recon)
    return dict(value=V, criterion="tolerance", rtol=1e-9, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_formula: measure formula transcribed in numpy (_kernels_spectral."
                                 "dtf/directed_coherence_library_as_coded/pdc/gpdc/ddtf_library_as_coded) applied to "
                                 "the library's OWN H and Sigma of the same pair (exchange npz from the probe)",
                notes="isolates the formula: same factors in, library per-bin values out; canonical-form and "
                      "2x2-identity numbers in the extra keys", **extra)


def h_indep(run):
    p = parse(run)
    x, _ = raw(run)
    z = prepare(x, run["path"])
    c = kchain(z)
    V = lib_measure(p["unit"], c["Hn"], c["Sigma"])
    with np.load(EXCHANGE / f"{safe_name(run['run_id'])}.npz") as zz:
        Hl, Sl = zz["H"], zz["Sigma"]
    return dict(value=V, criterion="tolerance", rtol=1e-6, atol=1e-8, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_indep: {KDETAIL} on the raw pair, library formula transcribed",
                notes="per-bin values, whole chain independent of spectral_connectivity",
                max_abs_H_kernel_vs_library=float(np.abs(c["Hn"] - Hl).max()),
                max_abs_Sigma_kernel_vs_library=float(np.abs(c["Sigma"] - Sl).max()),
                kernel_wilson_iters=c["info"]["n_iter"], kernel_recon_err=c["info"]["recon_err"])


def h_exactbin(run):
    p = parse(run)
    n = run["T"]
    H, S, Sigma = exact_full(run["dataset"], n, "kernel")
    Hn, Sn = ks.nonneg(H), ks.nonneg(S)
    V = lib_measure(p["unit"], Hn, Sigma)
    can = canon_measure(p["unit"], Hn, Sigma, Sn, "squared")
    P = Sigma.shape[0]
    off = offdiag(P)
    extra = dict(max_abs_lib_vs_canonical_squared=float(np.abs(V - can)[:, off].max()),
                 lib_max=float(V[:, off].max()))
    if run["dataset"] == "var2_chain":
        extra["lib_max_over_bins_20"] = float(V[:, 2, 0].max())
        extra["canonical_max_over_bins_20"] = float(can[:, 2, 0].max())
    else:
        extra["lib_max_over_bins_01"] = float(V[:, 0, 1].max())
    atol = 1e-8 if p["unit"] == "gpdcoh" else 1e-10
    return dict(value=V, criterion="tolerance", rtol=1e-9, atol=atol, reference_kind="analytic",
                reference_detail=f"{SELF}:h_exactbin: analytic VAR(2) H = A(f)^-1 on fftfreq({n}) "
                                 "(_kernels_spectral.var_transfer_function), true Sigma; library formula transcribed",
                notes="library Wilson on the exact P-channel spectrum (probe) vs analytic factors; gpdcoh atol 1e-8 "
                      "because the library Sigma is accurate only to its stopping tolerance 1e-8", **extra)


def h_dir(run):
    p = parse(run)
    name, T = run["dataset"], run["T"]
    if name.startswith("logistic_maps"):
        V = np.array([[np.nan, -1.0], [1.0, np.nan]])            # x (0) -> y (1) only
        sign = np.nan_to_num(V)
        detail = "generating model: logistic_maps_chaotic beta_yx = 0.1 (x -> y), beta_xy = 0"
        extra = {}
    else:
        M = population(name, T, run["path"], p["unit"], "mean", "full")
        V = M - M.T
        sign = np.where(np.abs(np.nan_to_num(V)) > 1e-6, np.sign(np.nan_to_num(V)), 0.0)
        detail = (f"population value of the bivariate quantity: exact marginal 2-channel spectrum of the "
                  f"{name} model (z-scaled), Wilson tol 1e-13, library formula, strict full band mean; "
                  f"value = M - M.T")
        extra = {"population_matrix": M}
    return dict(value=V, criterion="direction", expected_sign=sign, reference_kind="analytic",
                reference_detail=f"{SELF}:h_dir: {detail}",
                notes="pyspi value = A - A.T from the probe; expected sign + at [target, source]", **extra)


def _truth_like(run, mode):
    p = parse(run)
    name, T, path = run["dataset"], run["T"], run["path"]
    procs = run.get("processes")
    truth = population(name, T, path, p["unit"], p["stat"], p["band"], procs)
    S = pyspi_npz(run["run_id"])
    mean, sd, R, tail = mc_summary(name, T, path, p["unit"], p["stat"], p["band"], procs, value=S)
    P = truth.shape[0]
    m = P * (P - 1)
    zb = bonferroni_z(m)
    off = offdiag(P)
    V = truth if mode == "truth" else mean
    with np.errstate(invalid="ignore", divide="ignore"):
        zt = (S - truth) / sd
        zm = (S - mean) / sd
    extra = dict(mc_R=R, bonferroni_m=m, bonferroni_z=zb, mc_mean=mean, mc_sd=sd, truth=truth,
                 mc_bias_vs_truth=mean - truth,
                 max_abs_z_pyspi_vs_truth=float(np.nanmax(np.abs(zt[off]))),
                 max_abs_z_pyspi_vs_mc=float(np.nanmax(np.abs(zm[off]))),
                 max_mc_bias=float(np.nanmax(np.abs(mean - truth)[off])),
                 min_empirical_two_sided_tail=tail)
    kind = "analytic" if mode == "truth" else "independent-impl"
    return dict(value=V, criterion="statistical", se=sd * zb / 2.0, reference_kind=kind,
                reference_detail=f"{SELF}:h_{mode}: truth = population value of the bivariate quantity "
                                 f"(exact marginal spectra, Wilson tol 1e-13, library formula, strict band); "
                                 f"Monte Carlo: {R} fresh draws of the {name} model through {KDETAIL}; "
                                 f"se = MC SD * z_Bonf({m})/2",
                notes=("pyspi vs analytic truth, tolerance = multitaper sampling SD (finite-sample bias of the "
                       "5-taper estimator shows as failure)" if mode == "truth" else
                       "pyspi vs Monte Carlo mean of the same estimator (is pyspi a typical draw?)"), **extra)


def h_truth(run):
    return _truth_like(run, "truth")


def h_mc(run):
    return _truth_like(run, "mc")


def h_chain(run):
    p = parse(run)
    u, T, path = p["unit"], run["T"], run["path"]
    mv = multivariate_truth("var2_chain", T, path, u, "mean", "full")
    bv = population("var2_chain", T, path, u, "mean", "full")
    mean, sd, R = mc_summary("var2_chain", T, path, u, "mean", "full")
    S = pyspi_npz(run["run_id"])
    off = offdiag(3)
    ok = bool(abs(bv[2, 0] - mv[2, 0]) <= 1e-3)
    return dict(value=mv, criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                reference_detail=f"{SELF}:h_chain: multivariate = library formula on the analytic 3-channel H "
                                 "(conditioned on all channels); bivariate = population value on the exact marginal "
                                 f"2-channel spectra; Monte Carlo {R} draws of the independent estimator",
                notes="pass iff the bivariate path preserves the full-model value of the 0 -> 2 entry (A[2,0]) "
                      "within 1e-3; for PDC/gPDC/dDTF the full-model value is 0 (no direct link)",
                multivariate_20=float(mv[2, 0]), bivariate_population_20=float(bv[2, 0]),
                pyspi_20=float(S[2, 0]), mc_mean_20=float(mean[2, 0]), mc_sd_20=float(sd[2, 0]),
                pyspi_20_z_vs_mc=float((S[2, 0] - mean[2, 0]) / sd[2, 0]),
                pyspi_minus_multivariate_20=float(S[2, 0] - mv[2, 0]),
                null_entry_02_pyspi=float(S[0, 2]), null_entry_02_mc_mean=float(mean[0, 2]),
                multivariate_matrix=mv, bivariate_population_matrix=bv,
                max_abs_bivariate_minus_multivariate=float(np.nanmax(np.abs(bv - mv)[off])))


def h_scale(run):
    """Channel-scaling invariance.  Canonical gPDC and Baccala DC are invariant to x_k -> c_k x_k
    (the offset is removed by the constant detrend); DTF/PDC/dDTF are not.  The 5-taper CSM itself
    is exactly equivariant (D S D), but the step-converged Wilson factor of a CSM that is not exactly
    factorisable on the n_fft = T grid depends on the starting point (Cholesky of the scaled lag-0
    covariance), so even an invariant formula changes by the estimator's non-equivariance (same
    mechanism and size as the channel-order dependence, order runs: 1e-3..2e-2 per bin).  Pass
    criterion: the band-mean change is below 1e-2 for an invariant formula and above it for a
    non-invariant one; the exact-spectrum population value (no estimation) is reported as well."""
    p = parse(run)
    u = p["unit"]
    x, ds = raw(run, dataset="bivariate_gaussian")
    key = data_key(run, dataset="bivariate_gaussian")
    A, _, _, _ = replicate(prepare(x, "kernel"), u, "mean", "full", key)
    S = pyspi_npz(run["run_id"])
    off = offdiag(A.shape[0])
    diff = float(np.nanmax(np.abs(S - A)[off]))
    # population level: exact var2_bivariate spectrum vs the same spectrum scaled by diag(0.2, 2)
    Sx = exact_S("var2_bivariate", 1024, "kernel")
    D = np.diag([0.2, 2.0])
    vals = []
    for SS in (Sx, D @ Sx @ D):
        H, Sg, _ = ks.wilson_factorise(SS, tol=1e-13, max_iter=2000)
        L = lib_measure(u, ks.nonneg(H), Sg)
        vals.append(np.array([band_reduce(L, 1024, "mean", "full")[0],
                              band_reduce(L[:, ::-1, ::-1], 1024, "mean", "full")[0]]))
    pop_diff = float(np.abs(vals[0] - vals[1]).max())
    invariant = diff < 1e-2
    expected = u in ("gpdcoh", "dcoh")       # canonical gPDC and canonical DC are scale invariant
    return dict(value=A, criterion="qualitative", pass_flag=int(invariant == expected),
                reference_kind="analytic",
                reference_detail=f"{SELF}:h_scale: independent chain on the UNSCALED D2 draw (scaled_offset = "
                                 "scale*D2 + offset, same draw); canonical invariance: gPDC and Baccala DC invariant "
                                 "to channel scaling, DTF/PDC/dDTF not; population check on the exact var2 spectrum "
                                 "scaled by diag(0.2, 2)",
                notes="pass iff (band-mean change < 1e-2) equals the canonical expectation; threshold set by the "
                      "Wilson estimator's own non-equivariance (see docstring and order runs)",
                max_abs_scaled_vs_unscaled=diff, observed_invariant=int(invariant),
                canonical_expected_invariant=int(expected), population_scaled_vs_unscaled=pop_diff)


def h_range(run):
    S = pyspi_npz(run["run_id"])
    off = offdiag(S.shape[0])
    mx, mn = float(np.nanmax(S[off])), float(np.nanmin(S[off]))
    ok = mn >= 0 and mx <= 1 + 1e-12
    return dict(value=np.array(1.0), criterion="qualitative", pass_flag=int(ok), reference_kind="analytic",
                reference_detail=f"{SELF}:h_range: canonical range of every coherence-type directed measure "
                                 "(squared or modulus) is [0, 1]",
                notes="pass iff all off-diagonal band maxima lie in [0, 1]", max_value=mx, min_value=mn)


def h_ident(run):
    p = parse(run)
    rid = f"spectral.dtf|{run['dataset']}|{run['T']}|{run['path']}|rep:{p['sub']}{ptag(run.get('processes'))}"
    return dict(value=pyspi_npz(rid), criterion="exact", reference_kind="same-lineage",
                reference_detail=f"{SELF}:h_ident: pyspi dtf output {rid}",
                notes="2x2 identity H = adj(A)/det(A) makes off-diagonal |PDC|^2 = DTF (tests/README.md worth knowing)")


def h_ddtfscale(run):
    out, canon, canon_low = [], [], []
    for n in (64, 256, 1024, 4096):
        H, S, Sigma = exact_full("var2_bivariate", n, "kernel")
        L = ks.ddtf_library_as_coded(ks.nonneg(H))
        f = grid(n)
        sel = ks.band_mask(f, 0.0, 0.5, inclusive=False)
        out.append([L[sel, 1, 0].mean(), L[sel, 0, 1].mean()])
        # canonical Korzeniewska form on the same grid: ffDTF^2 x partial coherence^2, ffDTF normalised over
        # the frequencies supplied (all non-negative bins, as in the library), so its T dependence is part of
        # the definition; canon_low renormalises over the low half-band only (band-restricted ffDTF)
        Hn, Sn = ks.nonneg(H), ks.nonneg(S)
        K = ks.ddtf(Hn, Sn, "squared")
        canon.append(K[sel, 1, 0].mean())
        low = ks.band_mask(f, 0.0, 0.25, inclusive=False)
        canon_low.append(ks.ddtf(Hn[low], Sn[low], "squared")[:, 1, 0].mean())
    V = np.array(out)
    Sp = pyspi_npz(run["run_id"])
    ns = np.array([64, 256, 1024, 4096], dtype=float)
    scaled = Sp[:, 0] * np.sqrt(ns / 2)
    slopes = np.diff(np.log(Sp[:, 0])) / np.diff(np.log(ns))
    canon = np.array(canon)
    canon_low = np.array(canon_low)
    return dict(value=V, criterion="tolerance", rtol=1e-8, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:h_ddtfscale: library-as-coded dDTF (|ffDTF| |PDC|) on the analytic "
                                 "var2_bivariate H at n = 64, 256, 1024, 4096; strict full band mean of [1,0], [0,1]",
                notes="part (a) of the T^-1/2 scaling check: exact spectrum, no estimation",
                scaled_10=scaled, spread_all=float((scaled.max() - scaled.min()) / scaled.min()),
                spread_n256=float((scaled[1:].max() - scaled[1:].min()) / scaled[1:].min()),
                slopes_10=slopes,
                canonical_K2003_mean_10=canon, canonical_K2003_scaled_by_n2_10=canon * (ns / 2),
                canonical_K2003_slopes_10=np.diff(np.log(canon)) / np.diff(np.log(ns)),
                canonical_K2003_lowband_renorm_mean_10=canon_low,
                canonical_K2003_lowband_renorm_slopes_10=np.diff(np.log(canon_low)) / np.diff(np.log(ns)))


def h_ddtfslope(run):
    vals = []
    for T in (64, 256, 1024, 4096):
        x, _ = raw(run, T=T)
        z = prepare(x, run["path"])
        A, _, _, _ = replicate(z, "ddtf", "mean", "full", data_key(run, T=T))
        vals.append(A)
    V = np.stack(vals)
    Sp = pyspi_npz(run["run_id"])
    Ts = np.array([64, 256, 1024, 4096], dtype=float)
    P = V.shape[1]
    off = offdiag(P)
    lT = np.log(Ts)
    offmean = np.array([Sp[k][off].mean() for k in range(4)])
    extra = dict(slopes_offdiag_mean=np.diff(np.log(offmean)) / np.diff(lT))
    if P == 2:
        extra["slopes_10"] = np.diff(np.log(Sp[:, 1, 0])) / np.diff(lT)
        extra["slopes_01"] = np.diff(np.log(Sp[:, 0, 1])) / np.diff(lT)
        inband = lambda s: bool(np.all((s >= -0.70) & (s <= -0.40)))  # noqa: E731
        extra["part_b_10_in_band"] = int(inband(extra["slopes_10"]))
        extra["part_b_offmean_in_band"] = int(inband(extra["slopes_offdiag_mean"]))
    else:
        sl = np.diff(np.log(Sp[:, off]), axis=0) / np.diff(lT)[:, None]
        extra["per_entry_slope_min"] = sl.min(axis=1)
        extra["per_entry_slope_max"] = sl.max(axis=1)
    return dict(value=V, criterion="tolerance", rtol=1e-6, atol=1e-9, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_ddtfslope: independent chain ({KDETAIL}) at T = 64..4096, "
                                 "library-as-coded dDTF, strict full band mean",
                notes="part (b): pyspi pipeline values per T; slopes are report-only except where stated", **extra)


# ----------------------------------------------------------------------- Wilson handlers
def _wrun(run, kind):
    return f"{W}|{run['dataset']}|{run['T']}|{run['path']}|{kind}{ptag(run.get('processes'))}"


def lib_csm(run):
    return from_ri(pyspi_npz(_wrun(run, "csm")))


def h_csm(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    co, _ = ks.multitaper_fft(z)
    S = ks.cross_spectral_matrix(co)
    G0 = np.fft.ifft(S, axis=0)[0].real
    ev = np.linalg.eigvalsh(G0)
    return dict(value=ri(S), criterion="tolerance", rtol=1e-8, atol=1e-12, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_csm: _kernels_spectral.multitaper_fft + cross_spectral_matrix (two-sided)",
                notes="library CSM of the pair vs independent multitaper",
                lag0_cov_min_eig=float(ev.min()), lag0_cov_max_eig=float(ev.max()))


def _kernel_on_lib(run):
    S = lib_csm(run)
    H, Sg, info = ks.wilson_factorise(S)
    return S, ks.nonneg(H), Sg, info


def h_H(run):
    S, Hn, Sg, info = _kernel_on_lib(run)
    Hl = from_ri(pyspi_npz(run["run_id"]))
    Sl = pyspi_npz(_wrun(run, "Sigma"))
    Sn = ks.nonneg(S)
    rec_lib = np.abs(Hl @ Sl @ np.conj(np.swapaxes(Hl, -1, -2)) - Sn).max() / np.abs(S).max()
    rec_k = np.abs(Hn @ Sg @ np.conj(np.swapaxes(Hn, -1, -2)) - Sn).max() / np.abs(S).max()
    x, _ = raw(run)
    own = kchain(prepare(x, run["path"]))
    return dict(value=ri(Hn), criterion="tolerance", rtol=1e-6, atol=1e-6, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_H: _kernels_spectral.wilson_factorise on the LIBRARY's exported CSM",
                notes="library H vs independent Wilson factor of the same CSM; residual S - H Sigma H^H in extras",
                library_recon_err_nonneg=float(rec_lib), kernel_recon_err_nonneg=float(rec_k),
                kernel_iters=info["n_iter"], kernel_converged=int(info["converged"]),
                max_abs_H_own_csm_vs_library=float(np.abs(own["Hn"] - Hl).max()))


def h_Sigma(run):
    S, Hn, Sg, info = _kernel_on_lib(run)
    return dict(value=Sg, criterion="tolerance", rtol=1e-6, atol=1e-6, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_Sigma: _kernels_spectral.wilson_factorise on the LIBRARY's exported CSM",
                notes="library Sigma = A0 A0^T vs independent factor", kernel_iters=info["n_iter"])


def h_conv(run):
    S, Hn, Sg, info = _kernel_on_lib(run)
    Sn = ks.nonneg(S)
    recn = np.abs(Hn @ Sg @ np.conj(np.swapaxes(Hn, -1, -2)) - Sn).max() / np.abs(S).max()
    V = np.array([info["n_iter"], float(info["converged"]), info["recon_err"], recn])
    Sp = pyspi_npz(run["run_id"])
    unreliable = (Sp[1] < 1) or (Sp[3] > 0.1)
    return dict(value=V, criterion="tolerance", rtol=1e-3, atol=1e-6, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_conv: _kernels_spectral.wilson_factorise info on the library CSM "
                                 "[n_iter, converged, max|psi psi^H - S|/max|S|, max|H Sigma H^H - S|/max|S| (f>=0)]",
                notes="library diagnostics reproduced; 'unreliable' = not converged or residual > 0.1",
                library_unreliable=int(unreliable), library_iters=float(Sp[0]), library_recon_twosided=float(Sp[2]))


def h_exactH(run):
    H, S, Sigma = exact_full(run["dataset"], run["T"], "kernel")
    return dict(value=ri(ks.nonneg(H)), criterion="exact", rtol=1e-10, atol=1e-12, reference_kind="analytic",
                reference_detail=f"{SELF}:h_exactH: analytic H = A(f)^-1 (var_transfer_function) on fftfreq(T)",
                notes="library Wilson (shipped tol 1e-8) on the exact spectrum")


def h_exactSigma(run):
    p = parse(run)
    _, _, Sigma = exact_full(run["dataset"], run["T"], "kernel")
    atol = 1e-11 if p["opt"] == "tol12" else 1e-8
    Sp = pyspi_npz(run["run_id"])
    return dict(value=Sigma, criterion="tolerance", rtol=0.0, atol=atol, reference_kind="analytic",
                reference_detail=f"{SELF}:h_exactSigma: true innovation covariance of the dataset",
                notes=f"library Wilson tolerance {'1e-12' if p['opt'] == 'tol12' else '1e-8 (shipped)'}",
                max_abs_err=float(np.abs(Sp - Sigma).max()))


def h_nfft2(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    T = z.shape[1]
    rec, d10, d01 = [], [], []
    for n in (T, 2 * T):
        c = kchain(z, n_fft=n)
        rec.append(c["info"]["recon_err"])
        L = ks.dtf(c["Hn"])
        sel = ks.band_mask(grid(n), 0.0, 0.5, inclusive=False)
        d10.append(L[sel, 1, 0].mean())
        d01.append(L[sel, 0, 1].mean())
    return dict(value=np.array(rec + d10 + d01), criterion="tolerance", rtol=1e-3, atol=1e-6,
                reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_nfft2: independent chain with n_fft = T and 2T (zero padding)",
                notes="[rec(T), rec(2T), dtf10(T), dtf10(2T), dtf01(T), dtf01(2T)]")


def h_order(run):
    x, _ = raw(run)
    z = prepare(x, run["path"])
    c1, c2 = kchain(z), kchain(z[::-1].copy())
    V = [np.abs(lib_measure(u, c1["Hn"], c1["Sigma"])[:, 0, 1] - lib_measure(u, c2["Hn"], c2["Sigma"])[:, 1, 0]).max()
         for u in UNITS]
    return dict(value=np.array(V), criterion="tolerance", rtol=1e-3, atol=1e-7, reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_order: independent chain on [x_i, x_j] and [x_j, x_i]",
                notes="max over bins |L01(i,j) - L10(j,i)| for dtf, dcoh, pdcoh, gpdcoh, ddtf; exact factor is "
                      "order-equivariant, the estimate is not")


def h_calls(run):
    P = 3
    pairs = P * (P - 1)
    V = np.array([2 * pairs, 3 * pairs, 2 * pairs, 3 * pairs, 4 * pairs, 2 * pairs, 0.0])
    return dict(value=V, criterion="exact", reference_kind="analytic",
                reference_detail=f"{SELF}:h_calls: property-access count in SC/connectivity.py (dtf 2, dcoh 3, pdcoh 2, "
                                 "gpdcoh 3, ddtf 4 Wilson runs per call) x P(P-1) ordered pairs; six dtf identifiers on "
                                 "one Data share the (measure, i, j) cache; one shared factorisation is bit-identical",
                notes="[dtf, dcoh, pdcoh, gpdcoh, ddtf, six dtf ids on one Data, max diff shared vs uncached]")


def h_singular(run):
    """Duplicated channel (shared_envelope x2 == x0): the (0, 2) pair's lag-0 covariance is singular
    in exact arithmetic.  Whether the Cholesky start of Wilson fails depends on rounding.  The
    independent kernel (same Cholesky start, numpy.linalg) predicts the branch: if it raises
    LinAlgError, the library is expected to hit mpd:48 ``xp.linalg.linalg.LinAlgError`` (absent in
    numpy 2) -> AttributeError -> whole SPI matrix NaN; if not, the library is expected to return a
    finite value for the degenerate pair equal to the kernel's.  pass = library follows the
    predicted branch.  Neither branch is the safe behaviour (NaN for the degenerate pair only)."""
    p = parse(run)
    rec = json.loads((EXCHANGE / f"{safe_name(run['run_id'])}.json").read_text())
    Sp = pyspi_npz(run["run_id"])
    we = rec.get("whole_error") or ""
    x, _ = raw(run)
    z = prepare(x, run["path"])
    try:
        A, _, _, _ = replicate(z[[0, 2]], p["sub"], "mean", "full", data_key(run) + ("02",))
        kernel_raises, kval = False, float(A[0, 1])
    except np.linalg.LinAlgError:
        kernel_raises, kval = True, float("nan")
    lib_val = rec["pair_values"].get("0,2", float("nan"))
    if kernel_raises:
        ok = ("AttributeError" in we) and ("linalg" in we) and bool(np.all(np.isnan(Sp)))
    else:
        ok = (not we) and np.isfinite(lib_val) and abs(lib_val - kval) <= 1e-6 * max(1.0, abs(kval))
    bad = sorted(rec["pair_errors"])
    return dict(value=np.array(1.0), criterion="qualitative", pass_flag=int(ok), reference_kind="independent-impl",
                reference_detail=f"{SELF}:h_singular: _kernels_spectral.wilson_factorise on the degenerate pair "
                                 "predicts the branch (LinAlgError -> library AttributeError, whole matrix NaN; "
                                 "else a finite value for the degenerate pair)",
                notes=f"whole-matrix error: {we[:120] or 'none'}; failing pairs {bad}; finite pairs "
                      f"{len(rec['pair_values'])}",
                kernel_raises=int(kernel_raises), kernel_value_02=kval, library_value_02=lib_val,
                n_failing_pairs=len(bad), n_finite_pairs=len(rec["pair_values"]),
                all_nan=int(bool(np.all(np.isnan(Sp)))))


HANDLERS = {"rep": h_rep, "formula": h_formula, "indep": h_indep, "exactbin": h_exactbin, "dir": h_dir,
            "truth": h_truth, "mc": h_mc, "chain": h_chain, "scale": h_scale, "range": h_range,
            "ident": h_ident, "ddtfscale": h_ddtfscale, "ddtfslope": h_ddtfslope,
            "csm": h_csm, "H": h_H, "Sigma": h_Sigma, "conv": h_conv, "exactH": h_exactH,
            "exactSigma": h_exactSigma, "nfft2": h_nfft2, "order": h_order, "calls": h_calls,
            "singular": h_singular}


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
        kind = parse(run)["kind"]
        if only and kind not in only:
            continue
        rid = run["run_id"]
        if not (OUT_PY / f"{rid}.npz").is_file():
            print(f"REF SKIP {rid}: no pyspi output", file=sys.stderr)
            continue
        try:
            save(rid, HANDLERS[kind](run))
            n_ok += 1
        except Exception as exc:  # report, keep going; a missing ref shows as "no reference"
            n_err += 1
            import traceback
            print(f"REF ERROR {rid}: {exc!r}\n{traceback.format_exc()[-600:]}", file=sys.stderr)
    print(f"{FAMILY} refs: {n_ok} written, {n_err} errors")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
